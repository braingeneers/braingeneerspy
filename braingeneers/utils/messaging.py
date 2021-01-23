""" A simplified MQTT client for Braingeneers specific connections """
import boto3
import awsiot
import awscrt
import redis
import tempfile
import functools
import json
import inspect
import logging
import os
import io
import configparser
from typing import Callable, Tuple, List


MQTT_ENDPOINT = 'ahp00abmtph4i-ats.iot.us-west-2.amazonaws.com'
AWS_REGION = 'us-west-2'
PRP_ENDPOINT = 'https://s3.nautilus.optiputer.net'
AWS_PROFILE = 'aws-braingeneers-iot'
REDIS_HOST = 'redis.braingeneers.gi.ucsc.edu'
REDIS_PORT = 6379
logger = logging.getLogger()


# todo
# class DeviceState(dict):
#     device_name
#     device_type
#     status
#     samples
#     sampling_rate

class MessageBroker:
    """
    This class provides a simplified API for interacting with the AWS MQTT service and Redis service
    for Braingeneers. It assumes all possible defaults specific to the Braingeneers use of MQTT,
    handling details like AWS region, ca-certificates, etc. When instantiated the class will
    automatically connect to AWS IoT.

    See documentation at: https://github.com/braingeneers/wiki/blob/main/shared/mqtt.md

    Assumes the following:
        - `~/.aws/credentials` file has a profile [aws-braingeneers-iot] defined with AWS credentials
        - Python dependencies: `awsiotsdk, boto3, smart_open`

    Public functions:
        publish_message(topic: str, message:dict)  # publish a message to a topic
        publish_data_stream(stream_name: str, data: bytes, stream_size: int)  # publish large data to a stream

        subscribe_topic(topic: str, callback: Callable)  # subscribe to a topic, callable is a function with signature (topic: str, message: str)
        subscribe_data_stream(stream_name: str, callback: Callable)  # subscribe to data on a raw data stream

        list_devices(filter: dict=None)
        create_device(device_name: str, device_type: str)  # create a new device if it doesn't already exist

        get_device_state(device_name: str)  # returns the device shadow file as a dictionary.
        set_device_state(device_name: str, state: dict)  # saves the shadow file, a dict that is JSON serializable.

    Useful documentation references:
        https://github.com/braingeneers/wiki/blob/main/shared/mqtt.md
        https://aws.github.io/aws-iot-device-sdk-python-v2/
        https://awslabs.github.io/aws-crt-python/
    """
    def create_device(self, device_name: str, device_type: str) -> bool:
        """
        This function creates a new device in AWS IoT.

        It may be called once or multiple times, when called multiple times only the first call
        has any effect, subsequent calls will identify that the device already exists and do nothing.

        The first call to this function will create the device and copy the ca-certificates to S3 in the
        standard location at s3://braingeneers/ca-certificates/$DEVICE_NAME/*

        :param device_name: Name of the device, for example 'marvin'
        :param device_type: Device type as defined in AWS, standard device types are
            ['ephys', 'picroscope', 'feeding', 'client']
        :return: True if a new device was created, False if the device already existed and no action was taken.
        """
        pass

    def publish_message(self, topic: str, message: (dict, list, str)) -> None:
        """
        Publish a message on a topic. Example:
            publish('/devices/ephys/marvin', '{"START_EXPERIMENT":None, "UUID":"2020-11-27-e-primary-axion-morning"}')

        :param topic: an MQTT topic as documented at https://github.com/braingeneers/wiki/blob/main/shared/mqtt.md
        :param message: a message in dictionary/list format, JSON serializable, or a JSON string. May be None.
        """
        payload = json.dumps(message) if not isinstance(message, str) else message
        publish_future, packet_id = self.mqtt_connection.publish_message(
            topic=topic,
            payload=payload,
            qos=awscrt.mqtt.QoS.AT_LEAST_ONCE
        )
        publish_future.result()

    def publish_data_stream(self, stream_name: str, data: bytes, stream_size: int):
        pass

    def subscribe_message(self, topic: str, callback: Callable) -> None:
        """
        Subscribes to receive messages on a given topic. When providing a topic you will be
        subscribing to all messages on that topic and any sub topic. For example, subscribing to
        '/devices' would get messages on all devices, subscribing on '/devices/ephys' would subscribe
        to all messages on all ephys devices, and '/devices/ephys/marvin' would subscribe to messages
        to the marvin ephys device only.

        Note that callbacks to your function `callable` will be made in a separate thread.

        Example:
            def my_callback(topic: str, message):
                print(f'Received message {message} on topic {topic}')  # Print message

            client = MqttClient('test')  # device named test
            client.subscribe('/test', my_callback)  # subscribe to all topics under /test

        :param topic: topic: an MQTT topic as documented at
            https://github.com/braingeneers/wiki/blob/main/shared/mqtt.md
        :param callback: a function with the signature mycallbackfunction(topic: str, message),
            where message is a JSON object serialized to python format.
        """
        subscribe_future, packet_id = self.mqtt_connection.subscribe_message(
            topic=topic,
            callback=functools.partial(self._callback_handler, callback=callback),
            qos=awscrt.mqtt.QoS.AT_LEAST_ONCE,
        )
        subscribe_future.result()

    def subscribe_data_stream(self, stream_name: str, callback: Callable) -> None:
        """
        Subscribes to all new data on a stream. callback is a function with signature

          def mycallback(data: bytes, timestamp: int)

        data is a BLOBs (binary data). timestamps is the update timestamp (note: not a standard unix timestamp format)

        This function asynchronously calls callback each time new data is available. If you prefer to poll for
        new data use the `poll_data_stream` function instead.

        :param stream_name: name of the data stream, see standard naming convention documentation at:
            https://github.com/braingeneers/redis
        :param callback: a self defined function with signature defined above.
        """
        pass

    def poll_data_stream(self, stream_name: str, last_timestamp: int) -> Tuple[List[bytes], List[int]]:
        """
        Polls for new
        :param stream_name: name of the data stream, see standard naming convention documentation at:
            https://github.com/braingeneers/redis
        :param last_timestamp: the last timestamp to get data from the stream, can be 0 to get the full stream,
            pass it the last timestamp from the previous call on subsequent calls.
        :return: a tuple of two lists, (data_list, timestamps_list) which contain 0 or more data blocks in bytes
            format and an equal number of timestamps for each of those blocks. When no new data exists a tuple of empty
            lists is returned: ([], [])
        """
        pass

    def get_device_state(self, device: str) -> dict:
        """

        :param device:
        :return:
        """
        pass

    def set_device_state(self, device: str, device_state: dict) -> None:
        pass

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
            # write the aws root cert to a temp location, doing this to avoid configuration dependencies, for simplicity
            self.certs_temp_dir = tempfile.TemporaryDirectory()  # cleans up automatically on exit
            with open(f'{self.certs_temp_dir.name}/AmazonRootCA1.pem', 'wb') as f:
                f.write(AWS_ROOT_CA1.encode('utf-8'))

            event_loop_group = awscrt.io.EventLoopGroup(1)
            host_resolver = awscrt.io.DefaultHostResolver(event_loop_group)
            client_bootstrap = awscrt.io.ClientBootstrap(event_loop_group, host_resolver)
            credentials_provider = awscrt.auth.AwsCredentialsProvider.new_default_chain(client_bootstrap)

            self._mqtt_connection = awsiot.mqtt_connection_builder.websockets_with_default_aws_signing(
                endpoint=MQTT_ENDPOINT,
                client_bootstrap=client_bootstrap,
                region=AWS_REGION,
                credentials_provider=credentials_provider,
                ca_filepath=f'{self.certs_temp_dir.name}/AmazonRootCA1.pem',
                on_connection_interrupted=self._on_connection_interrupted,
                on_connection_resumed=self._on_connection_resumed,
                client_id=self.name,
                clean_session=False,
                keep_alive_secs=6
            )

            connect_future = self.mqtt_connection.connect()
            logger.info('MQTT connected: ', connect_future.result())

        return self._mqtt_connection

    @property
    def redis_client(self):
        """ Lazy initialization of the redis client. """
        if self._redis_client is None:
            config = configparser.ConfigParser()
            config.read_file(io.StringIO(self._credentials))
            self._redis_client = redis.Redis(
                host=REDIS_HOST, port=REDIS_PORT, password=config['redis']['password']
            )

        return self._redis_client

    def __init__(self, name, endpoint='us-west-2', credentials=None):
        """

        :param name: name of device or client, must be a globally unique string ID.
        :param endpoint: optional AWS endpoint, defaults to Braingeneers standard us-west-2
        :param credentials: optional file path string or file-like object containing the
            standard `~/.aws/credentials` file. See https://github.com/braingeneers/wiki/blob/main/shared/permissions.md
            defaults to looking in `~/.aws/credentials` if left as None. This file expects to find profiles named
            'aws-braingeneers-iot' and 'redis' in it.
        """
        os.environ['AWS_PROFILE'] = AWS_PROFILE  # sets the AWS profile name for credentials
        self.name = name
        self.endpoint = endpoint

        if credentials is None:
            credentials = os.path.expanduser('~/.aws/credentials')  # default credentials location

        if isinstance(credentials, str):
            with open(credentials, 'r') as f:
                self._credentials = f.read()
        else:
            self._credentials = credentials.read()

        self.certs_temp_dir = None
        self._mqtt_connection = None

        self._redis_client = None


# The AWS root certificate. Embedded here to avoid requiring installing it as a dependency.
AWS_ROOT_CA1 = inspect.cleandoc("""
    -----BEGIN CERTIFICATE-----
    MIIDQTCCAimgAwIBAgITBmyfz5m/jAo54vB4ikPmljZbyjANBgkqhkiG9w0BAQsF
    ADA5MQswCQYDVQQGEwJVUzEPMA0GA1UEChMGQW1hem9uMRkwFwYDVQQDExBBbWF6
    b24gUm9vdCBDQSAxMB4XDTE1MDUyNjAwMDAwMFoXDTM4MDExNzAwMDAwMFowOTEL
    MAkGA1UEBhMCVVMxDzANBgNVBAoTBkFtYXpvbjEZMBcGA1UEAxMQQW1hem9uIFJv
    b3QgQ0EgMTCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBALJ4gHHKeNXj
    ca9HgFB0fW7Y14h29Jlo91ghYPl0hAEvrAIthtOgQ3pOsqTQNroBvo3bSMgHFzZM
    9O6II8c+6zf1tRn4SWiw3te5djgdYZ6k/oI2peVKVuRF4fn9tBb6dNqcmzU5L/qw
    IFAGbHrQgLKm+a/sRxmPUDgH3KKHOVj4utWp+UhnMJbulHheb4mjUcAwhmahRWa6
    VOujw5H5SNz/0egwLX0tdHA114gk957EWW67c4cX8jJGKLhD+rcdqsq08p8kDi1L
    93FcXmn/6pUCyziKrlA4b9v7LWIbxcceVOF34GfID5yHI9Y/QCB/IIDEgEw+OyQm
    jgSubJrIqg0CAwEAAaNCMEAwDwYDVR0TAQH/BAUwAwEB/zAOBgNVHQ8BAf8EBAMC
    AYYwHQYDVR0OBBYEFIQYzIU07LwMlJQuCFmcx7IQTgoIMA0GCSqGSIb3DQEBCwUA
    A4IBAQCY8jdaQZChGsV2USggNiMOruYou6r4lK5IpDB/G/wkjUu0yKGX9rbxenDI
    U5PMCCjjmCXPI6T53iHTfIUJrU6adTrCC2qJeHZERxhlbI1Bjjt/msv0tadQ1wUs
    N+gDS63pYaACbvXy8MWy7Vu33PqUXHeeE6V/Uq2V8viTO96LXFvKWlJbYK8U90vv
    o/ufQJVtMVT8QtPHRh8jrdkPSHCa2XV4cdFyQzR1bldZwgJcJmApzyMZFo6IQ6XU
    5MsI+yMRQ+hDKXJioaldXgjUkK642M4UwtBV8ob2xJNDd2ZhwLnoQdeXeGADbkpy
    rqXRfboQnoZsG4q5WTP468SQvvG5
    -----END CERTIFICATE-----
""")
