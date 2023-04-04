""" A simplified MQTT client for Braingeneers specific connections """

# import awsiot.mqtt_connection_builder
import redis
import tempfile
import functools
import json
import inspect
import logging
import os
import io
import configparser
import threading
import queue
import uuid
from typing import Callable, Tuple, List, Dict, Union
import random
import json
import braingeneers.iot.shadows as sh
from paho.mqtt import client as mqtt_client
from deprecated import deprecated


AWS_REGION = 'us-west-2'
PRP_ENDPOINT = 'https://s3.nautilus.optiputer.net'
AWS_PROFILE = 'aws-braingeneers-iot'
REDIS_HOST = 'redis.braingeneers.gi.ucsc.edu'
REDIS_PORT = 6379
logger = logging.getLogger()
logger.level = logging.INFO


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
        - `~/.aws/credentials` file has a profile [aws-braingeneers-iot] defined with AWS credentials
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

    def __init__(self, name: str = None, endpoint: str = AWS_REGION, credentials: (str, io.IOBase) = None):
        """
        :param name: name of device or client, must be a globally unique string ID.
        :param endpoint: optional AWS endpoint, defaults to Braingeneers standard us-west-2
        :param credentials: optional file path string or file-like object containing the
            standard `~/.aws/credentials` file. See https://github.com/braingeneers/wiki/blob/main/shared/permissions.md
            defaults to looking in `~/.aws/credentials` if left as None. This file expects to find profiles named
            'aws-braingeneers-iot' and 'redis' in it.
        """
        self.name = name if name is not None else str(uuid.uuid4())
        self.endpoint = endpoint

        if credentials is None:
            credentials = os.path.expanduser('~/.aws/credentials')  # default credentials location

        if isinstance(credentials, str):
            with open(credentials, 'r') as f:
                self._credentials = f.read()
        else:
            assert hasattr(credentials, 'read'), 'credentials parameter must be a filename string or file-like object.'
            self._credentials = credentials.read()

        config = configparser.ConfigParser()
        config.read_file(io.StringIO(self._credentials))

        assert 'braingeneers-mqtt' in config, \
            'Your AWS credentials file is missing a section [braingeneers-mqtt], you may have the wrong ' \
            'version of the credentials file.'
        assert 'profile-id' in config['braingeneers-mqtt'], \
            'Your AWS credentials file is malformed, profile-id is missing from the [braingeneers-mqtt] section.'
        assert 'profile-key' in config['braingeneers-mqtt'], \
            'Your AWS credentials file is malformed, profile-key was not found under the [braingeneers-mqtt] section.'
        assert 'endpoint' in config['braingeneers-mqtt'], \
            'Your AWS credentials file is malformed, endpoint was not found under the [braingeneers-mqtt] section.'
        assert 'port' in config['braingeneers-mqtt'], \
            'Your AWS credentials file is malformed, ' \
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

        self.shadow_interface = sh.DatabaseInteractor()

        self._subscribed_data_streams = set()  # keep track of subscribed data streams

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

    def publish_message(self, topic: str, message: (dict, list, str), confirm_receipt: bool = False) -> None:
        """
        Publish a message on a topic. Example:
            publish('/devices/ephys/marvin', '{"START_EXPERIMENT":None, "UUID":"2020-11-27-e-primary-axion-morning"}')

        :param topic: an MQTT topic as documented at https://github.com/braingeneers/wiki/blob/main/shared/mqtt.md
        :param message: a message in dictionary/list format, JSON serializable, or a JSON string. May be None.
        :param confirm_receipt: blocks until the message send is confirmed. This will cause the `publish_message` function
            to block for a network delay
        """
        if confirm_receipt is True:
            barrier = threading.Barrier(2, timeout=10)
            original_callback = self.mqtt_connection.on_publish

            def on_publish_callback(client, userdata, msg):
                barrier.wait()
                self.mqtt_connection.on_publish = original_callback

            self.mqtt_connection.on_publish = on_publish_callback

        payload = json.dumps(message) if not isinstance(message, str) else message
        result = self.mqtt_connection.publish(
            topic=topic,
            payload=payload,
            qos=2,
        )

        if confirm_receipt:
            barrier.wait()

        return result

    def subscribe_message(self, topic: str, callback: Callable) -> \
            Union[Callable, CallableQueue]:
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
            mb.subscribe('test', my_callback)  # subscribe to all topics under test

        Polling messages instead of subscribing to push:
            You can poll for new messages instead of subscribing to push notifications (which happen
            in a separate thread) using the following example:

            mb = MessageBroker('test')  # device named test
            q = messaging.CallableQueue()  # a queue.Queue object that stores (topic, message) tuples
            mb.subscribe_message('test', q)  # subscribe to all topics under test
            topic, message = q.get()
            print(f'Topic {topic} received message {message}')  # Print message

        :param topic: an MQTT topic as documented at
            https://github.com/braingeneers/wiki/blob/main/shared/mqtt.md
        :param callback: a function with the signature mycallbackfunction(topic: str, message),
            where message is a JSON object serialized to python format.
        :param timeout_sec: number of seconds to wait to verify connection successful.
        :return: the original callable, this is returned for convenience sake only, it's not altered in any way.
        """
        def on_message(client, userdata, msg):
            # this modifies callback for compatibility with code written for the AWS SDK
            callback(msg.topic, json.loads(msg.payload.decode()))

        self.mqtt_connection.subscribe(topic, qos=2)
        self.mqtt_connection.on_message = on_message

        return callback

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
        # query_string = 'connectivity.connected:true'  # always query for connected devices
        # for k, v in filters.items():
        #     if isinstance(v, (tuple, list)):
        #         assert len(v) == 2, f'One or two values expected for filter {k}, found {v}'
        #         query_string = f'{query_string}'
        #     else:
        #         query_string = f'{query_string} AND shadow.{k}:{v}'

        # response = self.boto_iot_client.search_index(queryString=query_string)
        # ret_val = [thing['thingName'] for thing in response['things']]
        # return ret_val

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
        print('')
        print(f'_callback_subscribe_device_state_change\n\tdevice_name: {device_name}\n\ttopic: {topic}\n\tmessage: {message}')  # todo debug step, remove

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

    @staticmethod
    def _callback_handler(topic, payload, callback):
        """ Dispatches callbacks after performing json deserialization """
        message = json.loads(payload)
        callback(topic, message)

    @staticmethod
    def _on_connection_interrupted(*args, **kwargs):
        logger.error(f'Connection interrupted: {args}, {kwargs}')

    @staticmethod
    def _on_connection_resumed(*args, **kwargs):
        logger.error(f'Connection resumed: {args}, {kwargs}')

    @property
    def mqtt_connection(self):
        """ Lazy initialization of mqtt connection. """
        if self._mqtt_connection is None:
            '''
            root certs only required for https connection our current mqtt broker does not have this yet
            '''
            def on_connect(client, userdata, flags, rc):
                if rc == 0:
                    logger.info('MQTT connected: client:' + str(client) + ' userdata:' + str(userdata) + ' flags:' + str(flags) + ' rc:' + str(rc))
                else:
                    logger.error("Failed to connect to MQTT, return code %d\n", rc)

            client_id = f'braingeneerspy-{random.randint(0, 1000)}'

            self._mqtt_connection = mqtt_client.Client(client_id)
            self._mqtt_connection.username_pw_set(self._mqtt_profile_id, self._mqtt_profile_key)
            self._mqtt_connection.on_connect = on_connect
            self._mqtt_connection.connect(self._mqtt_endpoint, self._mqtt_port)
            self._mqtt_connection.loop_start()

        return self._mqtt_connection

    @property
    def boto_iot_data_client(self):
        """ Lazy initialization of boto3 client """
        if self._boto_iot_data_client is None:
            boto_session = boto3.Session(profile_name=AWS_PROFILE)
            self._boto_iot_data_client = boto_session.client('iot-data', region_name=AWS_REGION, verify=False)
        return self._boto_iot_data_client

    @property
    def boto_iot_client(self):
        """ Lazy initialization of boto3 client """
        if self._boto_iot_client is None:
            boto_session = boto3.Session(profile_name=AWS_PROFILE)
            self._boto_iot_client = boto_session.client('iot', region_name=AWS_REGION)
        return self._boto_iot_client

    @property
    def redis_client(self):
        """ Lazy initialization of the redis client. """
        if self._redis_client is None:
            config = configparser.ConfigParser()
            config.read_file(io.StringIO(self._credentials))
            assert 'redis' in config, 'Your AWS credentials file is missing a section [redis], ' \
                                      'you may have the wrong version of the credentials file.'
            assert 'password' in config['redis'], 'Your AWS credentials file is malformed, ' \
                                                  'password was not found under the [redis] section.'
            self._redis_client = redis.Redis(
                host=REDIS_HOST, port=REDIS_PORT, password=config['redis']['password']
            )
            self._redis_client.config_set(name='notify-keyspace-events', value='t')

        return self._redis_client

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
