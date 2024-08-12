""" A simplified MQTT client for Braingeneers specific connections """
import redis
import logging
import os
import re
import io
import configparser
import threading
import queue
import uuid
import random
import json
import braingeneers.iot.shadows as sh
import pickle
import importlib
import datetime

from typing import Callable, Tuple, List, Dict, Union
from deprecated import deprecated
from paho.mqtt import client as mqtt_client
from paho.mqtt.enums import CallbackAPIVersion


AWS_REGION = 'us-west-2'
AWS_PROFILE = 'aws-braingeneers-iot'
REDIS_HOST = 'redis.braingeneers.gi.ucsc.edu'
REDIS_PORT = 6379


class MQTTError(RuntimeError):
    """Exception raised for errors during MQTT operations."""

    def __init__(self, message_info):
        mqtt_error_codes = {
            0: "No error. The message was published successfully.",
            1: "Out of memory. The client failed to allocate memory for the message.",
            2: "A network error occurred.",
            3: "Invalid function arguments were provided.",
            4: "The client is not currently connected.",
            5: "The server refused our connection, the client failed to authenticate.",
            6: "Message not found (internal error).",
            7: "The connection to the server was lost.",
            8: "A TLS error occurred.",
            9: "Payload size is too large.",
            10: "This feature has not been implemented.",
            11: "Problem with authentication.",
            12: "Access denied by ACL.",
            13: "An unknown error occurred.",
            14: "Check 'errno' for the error code.",
            15: "The queue size has been exceeded."
        }

        err_message = mqtt_error_codes.get(message_info.rc, 'Unknown error code.')
        super().__init__(f"MQTT Error {message_info.rc}: {err_message}")


class CallableQueue(queue.Queue):
    def __call__(self, *args):
        self.put(args)


class MessageBroker:
    """
    This class provides a simplified API for interacting with the AWS MQTT service and Redis service
    for Braingeneers. It assumes all possible defaults specific to the Braingeneers use of Redis and MQTT,
    handling details like AWS region, endpoints, etc. When instantiated the class will
    automatically connect to AWS IoT.

    See documentation at: https://github.com/braingeneers/wiki/blob/main/shared/mqtt.md

    Assumes the following:
        - `~/.aws/credentials_file` file has a profile [aws-braingeneers-iot] defined with AWS credentials_file
        - Python dependencies: `awsiotsdk, awscrt, boto3, redis`

    Public functions:
        #
        # Publish/subscribe short messages
        #
        publish_message(topic: str, message: (dict, list, str)))  # publish a message to a topic
        subscribe_message(topic: str, callback: Callable)  # subscribe to a topic, callable is a function with signature (topic: str, message: str)

        #
        # Publish/subscribe to data streams (can be large chunks of data)
        #
        publish_data_stream(stream_name: str, data: dict, stream_range: int)  # publish large data to a stream.
        subscribe_data_stream(stream_name: str, callback: Callable)  # subscribe to data on a raw data stream.
        poll_data_stream(stream_name: str, last_update_timestamp: str)  # returns a list of (time_str, data_dict) tuples since the last time stamp, non blocking. It's preferrable to use subscribe_data_stream unless polling is required, see function docs.

        #
        # List and register IoT devices
        #
        list_devices(**filters)  # list connected devices, filter by one or more state variables.
        create_device(device_name: str, device_type: str)  # create a new device if it doesn't already exist

        #
        # Get/set/update/subscribe to device state variables
        #
        get_device_state(device_name: str)  # returns the device shadow file as a dictionary.
        update_device_state(device: str, device_state: dict)  # updates one or more state variables for a registered device.
        set_device_state(device_name: str, state: dict)  # saves the shadow file, a dict that is JSON serializable.
        subscribe_device_state_change(device: str, device_state_keys: List[str], callback: Callable)  # subscribe to notifications when a device state changes.

    Useful documentation references:
        https://github.com/braingeneers/wiki/blob/main/shared/mqtt.md
        https://aws.github.io/aws-iot-device-sdk-python-v2/
        https://awslabs.github.io/aws-crt-python/
    """

    def __init__(self, name: str = None, credentials_file: (str, io.IOBase) = None, logger: logging.Logger = None):
        """
        Typical usage example:
            mb = MessageBroker()
            mb.publish_message('test/test', {"START_EXPERIMENT": None, "UUID": "2020-11-27-e-primary-axion-morning"})

        Example with debug logging enabled:
            import logging
            logging.basicConfig(level=logging.DEBUG)
            mb = MessageBroker(logger=logging.getLogger())
            mb.publish_message('test/test', {"START_EXPERIMENT": None, "UUID": "2020-11-27-e-primary-axion-morning"})

        :param name: name of device or client, must be a globally unique string ID.
        :param endpoint: optional AWS endpoint, defaults to Braingeneers standard us-west-2
        :param credentials_file: optional file path string or file-like object containing the
            standard `~/.aws/credentials` file. See https://github.com/braingeneers/wiki/blob/main/shared/permissions.md
            defaults to looking in `~/.aws/credentials` if left as None. This file expects to find profiles named
            'aws-braingeneers-iot' and 'redis' in it.
        :param logger: optional logger object, defaults to a new logger with the name of this class.
        """
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.name = name if name is not None else str(uuid.uuid4())

        if credentials_file is None:
            credentials_file = os.path.expanduser('~/.aws/credentials')  # default credentials_file location

        if isinstance(credentials_file, str):
            with open(credentials_file, 'r') as f:
                self._credentials = f.read()
        else:
            assert hasattr(credentials_file, 'read'), 'credentials_file parameter must be a filename string or file-like object.'
            self._credentials = credentials_file.read()

        config = configparser.ConfigParser()
        config.read_file(io.StringIO(self._credentials))

        assert 'braingeneers-mqtt' in config, \
            'Your AWS credentials_file file is missing a section [braingeneers-mqtt], you may have the wrong ' \
            'version of the credentials_file file.'
        assert 'profile-id' in config['braingeneers-mqtt'], \
            'Your AWS credentials_file file is malformed, profile-id is missing from the [braingeneers-mqtt] section.'
        assert 'profile-key' in config['braingeneers-mqtt'], \
            'Your AWS credentials_file file is malformed, profile-key was not found under the [braingeneers-mqtt] section.'
        assert 'endpoint' in config['braingeneers-mqtt'], \
            'Your AWS credentials_file file is malformed, endpoint was not found under the [braingeneers-mqtt] section.'
        assert 'port' in config['braingeneers-mqtt'], \
            'Your AWS credentials_file file is malformed, ' \
                                                      'port was not found under the [braingeneers-mqtt] section.'

        self.certs_temp_dir = None
        self._mqtt_connection = None
        self._mqtt_profile_id = config['braingeneers-mqtt']['profile-id']
        self._mqtt_profile_key = config['braingeneers-mqtt']['profile-key']
        self._mqtt_endpoint = config['braingeneers-mqtt']['endpoint']
        self._mqtt_port = int(config['braingeneers-mqtt']['port'])
        self._boto_iot_client = None
        self._boto_iot_data_client = None
        self._redis_client = None
        self._jwt_service_account_token = None

        self.shadow_interface = sh.DatabaseInteractor(jwt_service_token=self.jwt_service_account_token)

        self._subscribed_data_streams = set()  # keep track of subscribed data streams
        self._subscribed_message_callback_map = {}  # keep track of subscribed message callbacks, key is regex, value is tuple of (callback, topic)
        self._subscribe_message_lock = threading.Lock()  # lock for updating self._subscribed_message_callback_map

    class NamedQueue:
        """ Internal class, use: MessageBroker.get_queue() """
        NAME_PREFIX = 'bpy-queue-'
        TOKEN_PREFIX = 'bpy-queue-tokens-'
        TASK_COUNT_PREFIX = 'bpy-queue-task-count-'
        JOIN_CHANNEL_PREFIX = 'bpy-queue-join-channel-'

        def __init__(self, mb_instance, name: str, maxsize: int, cleanup_sec: int):
            self.key = f'{self.NAME_PREFIX}{name}'
            # to implement blocking
            self.token_key = None if maxsize is None else f'{self.TOKEN_PREFIX}{name}'
            self.task_count_key = f'{self.TASK_COUNT_PREFIX}{name}'
            self.join_channel_key = f'{self.JOIN_CHANNEL_PREFIX}{name}'

            self.mb_instance = mb_instance
            self.redis_client = mb_instance.redis_client
            self.maxsize = maxsize
            self.cleanup_sec = cleanup_sec
            self.redis_client.set(self.TASK_COUNT_PREFIX, 0)

            self.pubsub = self.redis_client.pubsub()

            if maxsize > 0:
                # create tokens for maxsize so BRPOPLPUSH can be used for blocking
                # this is a bit hacky, but the only way to implement the blocking in redis
                # because redis only supports blocking on GET operations not PUT operations.
                self.redis_client.lpush(self.token_key, *[b''] * maxsize)

        def qsize(self) -> int:
            return self.redis_client.llen(self.key)

        def empty(self) -> bool:
            return self.qsize() == 0

        def full(self):
            return self.qsize() >= self.maxsize

        def put(self, item, block=True, timeout=None):
            # For explanation of this implementation see: https://stackoverflow.com/a/76057700/4790871
            timeout = 1e-12 if timeout is None or block is False else timeout

            if self.maxsize > 0:
                # blocking happens here if we're out of $maxsize tokens
                # if block = False but self.maxsize > 0 then the timeout will be effectively 0 but
                # this function will return a nil that we can use to identify the queue full condition without
                # risking a race condition that would occur with check-and-push
                result = self.redis_client.brpop(self.token_key, timeout=timeout)
                if result is None:
                    raise queue.Full(f'Queue size {self.maxsize} is full or timeout exceeded, unable to push to queue.')

            # no blocking on the push to the main queue because tokens keep count for us, so this is safe at this point
            # if it wouldn't be safe to push here we would have already blocked or raised an exception above
            blob = pickle.dumps(item)

            pipe = self.redis_client.pipeline()
            pipe.lpush(self.key, blob)
            # reset expiration on each get or put
            for key in [self.key, self.token_key, self.task_count_key]:
                pipe.expire(key, self.cleanup_sec)
            pipe.incr(self.task_count_key)
            pipe.execute()

        def put_nowait(self, item):
            self.put(item, block=False)

        def get(self, block=True, timeout=None):
            timeout = 1e-12 if timeout is None else timeout

            pipe = self.redis_client.pipeline()
            # reset expiration on each get or put
            for key in [self.key, self.token_key, self.task_count_key]:
                pipe.expire(key, self.cleanup_sec)

            if self.maxsize > 0:
                pipe.lpush(self.token_key, b'')

            if block is True:
                pipe.brpop(self.key, timeout=timeout)
            else:
                pipe.rpop(self.key)

            pipe_result = pipe.execute()
            blob = None if pipe_result[-1] is None else pipe_result[-1][1] if isinstance(pipe_result[-1], tuple) else pipe_result[-1]

            if blob is None:
                raise queue.Empty('Queue is empty, ' + f'timeout exceeded.' if block else f'non-blocking mode.')

            o = pickle.loads(blob)
            return o

        def get_nowait(self):
            self.get(False)

        def task_done(self):
            task_count = self.redis_client.decr(self.task_count_key)
            print('DEBUG> TASK_COUNT: ' + str(task_count))
            if task_count == 0:
                self.redis_client.publish(self.join_channel_key, b'JOIN')
            elif task_count < 0:
                raise ValueError('task_done() was called more times than items placed in the queue.')

        def join(self):
            p = self.redis_client.pubsub()
            p.subscribe(self.join_channel_key)

            # resolve the race condition that would exist otherwise
            if int(self.redis_client.get(self.task_count_key)) == 0:
                self.redis_client.publish(self.join_channel_key, b'JOIN')

            # "subscribe" message types will be seen on the channel and are discarded
            # by the loop here, which only breaks when it sees the JOIN message issued
            # in this function or in task_done.
            for message in p.listen():
                if message['type'] == 'message' and message['data'] == b'JOIN':
                    break

    class NamedLock:
        """ Internal class, use: MessageBroker.get_lock() """
        NAME_PREFIX = 'bpy-lock-'

        def __init__(self, mb_instance, name: str, cleanup_sec: int):
            self.redis_client = mb_instance.redis_client
            self.key = f'{self.NAME_PREFIX}{name}'
            self.cleanup_sec = cleanup_sec
            self.lock = self.redis_client.lock(self.key)
            self.redis_client.expire(self.key, self.cleanup_sec)

        def __enter__(self):
            self.acquire()

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.release()

        def acquire(self):
            self.lock.acquire()
            self.redis_client.expire(self.key, self.cleanup_sec)

        def release(self):
            self.lock.release()
            self.redis_client.expire(self.key, self.cleanup_sec)

    def create_device(self, device_name: str, device_type: str) -> None:
        """
        This function creates a new device in the Shadows database

        Creating a device is not necessary to communicate over MQTT, if you do so when you
        connect to MQTT with the device name that device will be searchable using list_devices.

        It may be called once or multiple times, when called multiple times only the first call
        has any effect, subsequent calls will identify that the device already exists and do nothing.

        Will throw an exception if you try to create an existing device_name with a device_type that
        doesn't match the existing device.

        :param device_name: Name of the device, for example 'marvin'
        :param device_type: Device type as defined in AWS, standard device types are ['ephys', 'picroscope', 'feeding']
        """
        self.shadow_interface.create_interaction_thing(device_type, device_name)

    def publish_message(self, topic: str, message: (object, str) = None, confirm_receipt: bool = False) -> None:
        """
        Publish a message on a topic. Example:
            publish('devices/ephys/marvin', {"START_EXPERIMENT": None, "UUID": "2020-11-27-e-primary-axion-morning"})

        :param topic: an MQTT topic as documented at https://github.com/braingeneers/wiki/blob/main/shared/mqtt.md
        :param message: a JSON serializable object or string. It may be None.
        :param confirm_receipt: blocks until the message send is confirmed. This will cause the `publish_message`
            function to block for a network delay, if your application exits before the message
            can be sent this may be necessary.
        """
        barrier = None

        if confirm_receipt is True:
            barrier = threading.Barrier(2, timeout=10)
            original_callback = self.mqtt_connection.on_publish

            def on_publish_callback(_client, _userdata, _msg):
                barrier.wait()
                self.mqtt_connection.on_publish = original_callback

            self.mqtt_connection.on_publish = on_publish_callback

        payload = json.dumps(message) if not isinstance(message, str) else message
        assert len(payload.encode('utf-8')) < 268435455, 'Message payloads are limited to 256MB'
        message_info = self.mqtt_connection.publish(
            topic=topic,
            payload=payload,
            qos=2,
        )

        if confirm_receipt is True:
            barrier.wait()

        if message_info.rc != 0:
            raise MQTTError(message_info)

    def subscribe_message(self, topic: str, callback: Callable) -> Union[Callable, CallableQueue]:
        """
        Subscribes to receive messages on a given topic. When providing a topic you will be
        subscribing to all messages on that topic and any sub topic. For example, subscribing to
        '/devices' would get messages on all devices, subscribing on 'devices/ephys' would subscribe
        to all messages on all ephys devices, and 'devices/ephys/marvin' would subscribe to messages
        to the marvin ephys device only.

        Note that callbacks to your function `callable` will be made in a separate thread.

        Example:
            def my_callback(topic: str, message: dict):
                print(f'Received message {message} on topic {topic}')  # Print message

            mb = MessageBroker('test')  # device named test
            mb.subscribe(topic='test', my_callback)  # subscribe to all topics under test

        Polling messages instead of subscribing to push:
            You can poll for new messages instead of subscribing to push notifications (which happen
            in a separate thread) using the following example:

                mb = MessageBroker()  # an unnamed connection
                # returns a queue.Queue object that stores (topic, message) tuples because callable=None
                q = mb.subscribe_message(topic='test', callback=None)  # subscribe to all topics under test
                topic, message = q.get()
                print(f'Topic {topic} received message {message}')  # Print message

            The above code can be shortened to:
                mb = MessageBroker()
                q = mb.subscribe_message('test')
                topic, message = q.get()
                print(f'Topic {topic} received message {message}')  # Print message

        :param topic: an MQTT topic as documented at
            https://github.com/braingeneers/wiki/blob/main/shared/mqtt.md
        :param callback: a function with the signature mycallbackfunction(topic: str, message),
            where message is a JSON object serialized to python format. If callback is None then
            a CallableQueue object is created and returned and all messages go into that queue.
        :return: the original callable, this is returned for convenience only, it's not altered in any way.
        """
        callback = callback if callback is not None else CallableQueue()
        assert isinstance(callback, Callable), 'callback must be a callable function (or CallableQueue object) ' \
                                               'with a signature of (topic, message)'

        def on_message(_client, _userdata, msg):
            try:
                message = json.loads(msg.payload.decode())
            except ValueError:
                message = msg.payload.decode()

            # Test each regex in the callback map to see if it matches the topic and call
            # the appropriate callback function, we test all regexes because a single topic may
            # match multiple subscriptions.
            with self._subscribe_message_lock:
                matched_callbacks = [
                    callback_loop
                    for topic_regex_loop, (callback_loop, _) in self._subscribed_message_callback_map.items()
                    if re.match(topic_regex_loop, msg.topic)
                ]
            for callback_loop in matched_callbacks:
                callback_loop(msg.topic, message)

        # set on_message callback if it's not already set
        if len(self._subscribed_message_callback_map) == 0:
            self.mqtt_connection.on_message = on_message

        topic_regex = _mqtt_topic_regex(topic)
        with self._subscribe_message_lock:
            self._subscribed_message_callback_map[topic_regex] = (callback, topic)
        self.mqtt_connection.subscribe(topic=topic, qos=2)

        return callback

    def unsubscribe_message(self, topic: str) -> None:
        """
        Unsubscribes from a topic, this will stop receiving messages on that topic.

        :param topic: an MQTT topic as documented at
        """
        self.mqtt_connection.unsubscribe(topic=topic)
        topic_regex = _mqtt_topic_regex(topic)
        del self._subscribed_message_callback_map[topic_regex]

    def publish_data_stream(self, stream_name: str, data: Dict[Union[str, bytes], bytes], stream_size: int) -> None:
        """
        Publish (potentially large) data to a stream.

        :param stream_name: string name of the stream.
        :param data: a dictionary of data, may contain large data, keys are string or bytes and values must bytes type.
        :param stream_size: the maximum published data points to keep on the stream (technically approximate for efficiency)
        :return:
        """
        data_bytes_type = {k.encode('utf-8') if isinstance(k, str) else k: v for k, v in data.items()}

        self.redis_client.xadd(
            name=stream_name,
            fields=data_bytes_type,
            maxlen=stream_size,
            approximate=True
        )

    def subscribe_data_stream(self, stream_name: (str, List[str]), callback: Callable,
                              include_existing_stream_data: bool = True) -> Callable:
        """
        Data streams are intended for streaming larger chunks of data vs. small messages.

        Subscribes to all new data on one or more stream(s). callback is a function with signature

          def mycallback(stream_name:str, data: Dict[bytes, bytes])

        data is a BLOBs (binary data). uuid_timestamp is the update time (note: not a standard unix timestamp format)

        This function asynchronously calls callback each time new data is available. If you prefer to poll for
        new data use the `poll_data_stream` function instead. Note that if you subscribe to a stream
        multiple times you will get multiple callbacks, this function should only be called once.

        To poll for updates instead of receiving an asynchronous push use the following example:

            client = MqttClient('test')  # device named test
            q = messaging.CallableQueue()  # a queue.Queue object that stores (stream_name, data) tuples
            client.subscribe_data_stream('test', q)  # subscribe to all topics under test
            topic, message = q.get()
            print(f'Topic {topic} received message {message}')  # Print message

        :param stream_name: name of the data stream, see standard naming convention documentation at:
            https://github.com/braingeneers/redis
        :param callback: a self defined function with signature defined above.
        :param include_existing_stream_data: sends all data in the stream if True (default), otherwise
            only sends new data on the stream after the subscribe_data_stream function was called.
        :return: the original callable, this is returned for convenience sake only, it's not altered in any way.
        """
        stream_names = [stream_name] if isinstance(stream_name, str) else stream_name

        for stream_name in stream_names:
            if stream_name in self._subscribed_data_streams:
                raise AttributeError(f'Stream {stream_name} is already subscribed, can\'t '
                                     f'subscribe to the same stream twice.')

        logging.debug(f'Subscribing to data stream {stream_name} using callback {callback}')
        t = threading.Thread(
            target=self._redis_xread_thread,
            args=(stream_names, callback, include_existing_stream_data)
        )
        t.name = f'redis_listener_{stream_name}'
        t.daemon = True
        self._subscribed_data_streams.add(stream_name)
        t.start()
        return callback

    def poll_data_streams(self, stream_names_to_timestamps_dict: dict, count: int = -1) \
            -> List[List[Union[bytes, List[Tuple[bytes, Dict[bytes, bytes]]]]]]:
        """
        subscribe_data_stream is preferred to poll_data_stream in most cases, this function is included for specific
        uses cases when maintaining an open connection is infeasible, such as with individual worker processes. It's
        recommended to use subscribe_data_stream unless you have a specific reason to use poll_data_stream.

        :param stream_names_to_timestamps_dict: dictionary of {stream_name: last_update_timestamp} for 1 or more streams.
            The last timestamp received in the previous call, will be a string. Use '-' to
            get all available data on the stream, subsequent calls should use the timestamp received in
            the previous call.
        :param count: maximum number of entries to return, -1 for as many as are available.
        :return: a list of (last_update_timestamp, data_dict) pairs. Empty list if no data is available
            after waiting block_ms milliseconds or block_ms is None.
        """
        # update timestamps to be exclusive (required for a pre Redis 6.2 version only)
        streams_exclusive = {k: self._update_timestamp_exclusive(v) for k, v in stream_names_to_timestamps_dict.items()}

        result = self.redis_client.xread(streams=streams_exclusive, count=count if count >= 1 else None)
        return result

    def list_devices_by_type(self, thing_type_name) -> List[str]:
        """
        Lists devices, filtered by thing_type_name. Returns
        a list of device names in string format along with the device id in the database.

        Example usage:
        list_devices_by_type("BioPlateScope")

        This is a wrapper for the function located in the shadows interface, provided here for legacy compatibility.

        """
        
        return self.shadow_interface.list_devices_by_type(thing_type_name)
    
    def list_devices(self, **filters) -> List[str]:
        """
        Lists active/connected devices, filtered by one or more state variables. Returns
        a list of device names in string format.

        This function supports both exact matches and ranges. You can specify multiple conditions
        which are joined by a logical AND.

        The function doesn't currently support OR conditions which
        are technically possible to do through the underlying client API and omitted for simplicity and
        lack of a use case.

        Example usage:
          list_devices()  # list all connected devices
          list_devices(sample_rate=25000)  # list connected devices with sample_rate of 25,000
          list_devices(sample_rate=(12500, 25000)  # list devices with a sample rate between 12,500 and 25,000 inclusive
          list_devices(sample_rate=(12500, 25000), num_channels=32)  # list devices within the range of sample_rates
                                                                     # and state variable num_channels set to 32.

        :param filters:
        :return: a list of device names that match the filtering criteria.
        """
        raise NotImplementedError("This function is not implemented yet")

    def get_device_state(self, device_name: str) -> dict:
        """
        Get a dictionary of the devices state. State is a dict of key:value pairs.
        :param device_name: The devices name, example: "marvin"
        :return: a dictionary of {key: value, ...} state key: value pairs.

        wrapper for function in shadows interface, provided here for legacy compatibility.
        """

        return self.shadow_interface.get_device_state_by_name(device_name)

    def update_device_state(self, device_name: str, state: dict) -> None:
        """
        Update the state of a device. State is a dict of key:value pairs.
        :param device_name: The devices name, example: "marvin"
        :param state: a dictionary of {key: value, ...} state key: value pairs.

        wrapper for function in shadows interface, provided here for legacy compatibility.
        """

        thing = self.shadow_interface.get_device(name=device_name)
        thing.add_to_shadow(state)

    def delete_device_state(self, device_name: str, device_state_keys: List[str] = None) -> None:
        """
        Delete one or more state variables.

        :param device_name: device name
        :param device_state_keys: a List of one or more state variables to delete. None to delete all.
        :return:
        """
        thing = self.shadow_interface.get_device(name=device_name)
        if device_state_keys is None:
            thing.attributes["shadow"] = {}
            # Delete the whole shadow file
        else:
            # Delete specific keys from the shadow file
            state = self.get_device_state(device_name)
            for key in device_state_keys:
                state.pop(key,None)

            thing.attributes["shadow"] = state
            thing.push()

    def get_lock(self, name: str, cleanup_sec: int = 604800) -> NamedLock:
        """
        Get an instance of named lock or creates one if it doesn't already exist.

        An inter-process named lock will block as long as the lock is acquired and held by
        another process. Supports with blocks. Usage example:

            import braingeneers.iot.messaging.MessageBroker as MessageBroker

            mb = MessageBroker()
            with mb.get_lock('spikesorting/9999-00-00-e-test'):
                do_something()

        :param name: A globally unique name for the lock, you can get the same lock from multiple devices by this name.
            It's recommended to prefix lock names with something specific to your process.
        :param cleanup_sec: Locks will be automatically removed this many seconds after their last use, defaults to 1 week.
            This ensures the server is not contaminated with stale locks that were never deleted.
        """
        return MessageBroker.NamedLock(self, name, cleanup_sec)

    def get_queue(self, name: str, maxsize: int = 0, cleanup_sec: int = 604800) -> NamedQueue:
        """
        Inter-process named queue
        This queue functions across network connected devices.

        This data structure implements the python standard queue.Queue interface.
        See Python queue docs: https://docs.python.org/3/library/queue.html#queue.Queue

        Usage example:
            import braingeneers.iot.messaging.MessageBroker as MessageBroker

            mb = MessageBroker()
            q = mb.get_queue('spikesorting/9999-00-00-e-test')
            q.put({'serializable': 'objects only'})  # objects must serialize with Python pickle
            task = q.get()                           # may be running on a different computer or process

        :param name: queue name, use the same name across devices to access a common queue object.
        :param maxsize: maximum size of the queue as defined in Python standard queue.Queue
        :param cleanup_sec: deletes the queue after inactivity, defaults to 1 week.
            This ensures the server is not contaminated with stale locks that were never deleted.
        """
        return MessageBroker.NamedQueue(self, name, maxsize, cleanup_sec)

    def delete_lock(self, name: str):
        """ Deletes a named lock, this will delete the lock regardless of whether it's held on not. """
        key = f'{MessageBroker.NamedLock.NAME_PREFIX}{name}'
        self.redis_client.delete(key)

    def delete_queue(self, name: str):
        """ Deletes a named queue, this will delete the queue regardless of its state. """
        keys = [
            f'{MessageBroker.NamedQueue.NAME_PREFIX}{name}',
            f'{MessageBroker.NamedQueue.TOKEN_PREFIX}{name}',
            f'{MessageBroker.NamedQueue.TASK_COUNT_PREFIX}{name}',
            f'{MessageBroker.NamedQueue.JOIN_CHANNEL_PREFIX}{name}',
        ]
        self.redis_client.delete(*keys)

    @deprecated
    def subscribe_device_state_change(self, device_name: str, device_state_keys: List[str], callback: Callable) -> None:
        """
        Subscribe to be notified if one or more state variables changes.

        Callback is a function with the following signature:
          def mycallback(device_name: str, device_state_key: str, new_value)

        There is one built-in state variable named 'connectivity.connected' which fires
        if the device connected status changes, value will be True or False. All other
        state variables are user defined as specified in [get|update|delete]_device_state methods.

        :param device_name:
        :param device_state_keys:
        :param callback:
        :return:
        """
        raise NotImplementedError("This function is not supported by the braingeneers shadows interface, "
                                  "it is a holdover from the original implementation using AWS IoT")

    @staticmethod
    def _callback_subscribe_device_state_change(callback: Callable,
                                                device_name: str, device_state_keys: List[str],
                                                topic: str, message: dict):
        """
        Not in use with current implementation of shadows interface
        
        """

        # Call users callback once for each updated key
        for k in set(device_state_keys).intersection(message['state']['reported'].keys()):
            callback(device_name, k, message['state']['reported'][k])

    def _redis_xread_thread(self, stream_names, callback, include_existing):
        """ Performs blocking Redis XREAD operations in a continuous loop. """
        # last_timestamps = ['0-0' if include_existing else '$' for _ in range(len(stream_names))]
        streams = {
            s: '0-0' if include_existing else '$'
            for i, s in enumerate(stream_names)
        }

        while True:
            response = self.redis_client.xread(streams=streams, block=0)
            for item_stream in response:
                stream_name = item_stream[0].decode('utf-8')
                for item in item_stream[1]:
                    timestamp = item[0]
                    data_dict = item[1]

                    streams[stream_name] = timestamp
                    callback(stream_name, data_dict)

    @staticmethod
    def _update_timestamp_exclusive(timestamp: (str, bytes)):
        # based on SO article this hacky method of incrementing the timestamp is necessary until Redis 6.2:
        # https://stackoverflow.com/questions/66035607/redis-xrange-err-invalid-stream-id-specified-as-stream-command-argument
        if timestamp not in ['-', b'-', '0-0']:
            last_update_str = timestamp.decode('utf-8') if isinstance(timestamp, bytes) else timestamp
            last_update_exclusive = last_update_str[:-1] + str(int(last_update_str[-1]) + 1)
        else:
            last_update_exclusive = '0-0'
        return last_update_exclusive

    @property
    def mqtt_connection(self):
        """ Lazy initialization of mqtt connection. """
        if self._mqtt_connection is None:
            '''
            root certs only required for https connection our current mqtt broker does not have this yet
            '''
            # MQTT connection callbacks
            def on_connect(client, userdata, flags, rc):
                if rc == 0:
                    self.logger.info('MQTT connected: client:' + str(client) + ' userdata:' + str(userdata) + ' flags:' + str(flags) + ' rc:' + str(rc))
                    if len(self._subscribed_message_callback_map) > 0:
                        with self._subscribe_message_lock:
                            reconnect_topics = [topic for _, topic in self._subscribed_message_callback_map.values()]
                        for topic in reconnect_topics:
                            self.logger.info('Re-subscribing to topic: ' + topic)
                            self.mqtt_connection.subscribe(topic, qos=2)
                else:
                    self.logger.error("Failed to connect to MQTT, return code %d\n", rc)

            def on_log(client, userdata, level, buf):
                self.logger.debug("MQTT log: %s", buf)

            client_id = f'braingeneerspy-{random.randint(0, 1000)}'
            self._mqtt_connection = mqtt_client.Client(CallbackAPIVersion.VERSION1, client_id)
            self._mqtt_connection.username_pw_set(self._mqtt_profile_id, self._mqtt_profile_key)
            self._mqtt_connection.on_connect = on_connect
            self._mqtt_connection.on_log = on_log
            self._mqtt_connection.reconnect_delay_set(min_delay=1, max_delay=60)
            self._mqtt_connection.connect(host=self._mqtt_endpoint, port=self._mqtt_port, keepalive=15)
            self._mqtt_connection.loop_start()

        return self._mqtt_connection

    @property
    def redis_client(self) -> redis.Redis:
        """ Lazy initialization of the redis client. """
        if self._redis_client is None:
            config = configparser.ConfigParser()
            config.read_file(io.StringIO(self._credentials))
            assert 'redis' in config, 'Your AWS credentials_file file is missing a section [redis], ' \
                                      'you may have the wrong version of the credentials_file file.'
            assert 'redis_password' in config['redis'], 'Your AWS credentials_file file is malformed, ' \
                                                  'password was not found under the [redis] section.'
            self._redis_client = redis.Redis(
                host=REDIS_HOST, port=REDIS_PORT, password=config['redis']['redis_password']
            )
            self._redis_client.config_set(name='notify-keyspace-events', value='t')

        return self._redis_client

    @property
    def jwt_service_account_token(self) -> str:
        """ Lazy initialization of the JWT service account token. """
        PACKAGE_NAME = "braingeneers.iot"
        config_dir = os.path.join(importlib.resources.files(PACKAGE_NAME), 'service_account')
        config_file = os.path.join(config_dir, 'config.json')

        if self._jwt_service_account_token is None:
            # Check if the JWT token exists
            # This token is required for all operations that require web services.
            # The token is a (json) dict of form {'access_token': '----', 'expires_at': '2024-11-07 23:39:42 UTC'}
            os.makedirs(config_dir, exist_ok=True)

            # Try to load an existing JWT token locally if it exists
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    self._jwt_service_account_token = json.load(f)

            if self._jwt_service_account_token is None:
                raise PermissionError('JWT service account token not found, please generate one using: python -m braingeneers.iot.messaging authenticate')

        # Check if the token is still valid, this happens on every access, but takes no action while it's still valid.
        # If the token has less than 3 month left, refresh it, default tokens have 30 days at issuance.
        expires_at = datetime.datetime.fromisoformat(self._jwt_service_account_token['expires_at'].replace(' UTC', ''))
        if (expires_at - datetime.datetime.now()).days < 90:
            GENERATE_TOKEN_URL = 'https://service-accounts.braingeneers.gi.ucsc.edu/generate_token'
            self._jwt_service_account_token = requests.get(GENERATE_TOKEN_URL).json()
            with open(config_file, 'w') as f:
                json.dump(self._jwt_service_account_token, f)

        return self._jwt_service_account_token

    def shutdown(self):
        """ Release resources and shutdown connections as needed. """
        if self.certs_temp_dir is not None:
            self.certs_temp_dir.cleanup()


class TemporaryEnvironment:
    """ Sets an environment variable temporarily using python `with` syntax. """
    def __init__(self, env, value):
        self.env = env
        self.value = value
        self.save_original_env_value = None

    def __enter__(self):
        self.save_aws_profile = os.environ.get(self.env)
        os.environ[self.env] = self.value

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.save_original_env_value is None:
            del os.environ[self.env]
        else:
            os.environ[self.env] = self.save_original_env_value


def _mqtt_topic_regex(topic: str) -> str:
    """ Converts a topic string with wildcards to a regex string """
    return "^" + topic.replace("+", "[^/]+").replace("#", ".*").replace("$", "\\$") + "$"
