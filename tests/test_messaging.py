""" Unit test for BraingeneersMqttClient """
import unittest
from messaging import MessageBroker
import threading


class TestBraingeneersMqttClient(unittest.TestCase):
    def setUp(self) -> None:
        self.mqtt_client = MessageBroker('test')

    def test_publish_subscribe(self):
        message_received_barrier = threading.Barrier(2, timeout=7)

        def unittest_mqtt_subscriber(topic, message, **kwargs):
            print('In unittest_mqtt_subscriber')
            self.assertEqual(topic, 'test/unittest')
            self.assertEqual(message, {'test': 'true'})
            message_received_barrier.wait()  # synchronize between threads

        self.mqtt_client.subscribe_message('test/unittest', unittest_mqtt_subscriber)
        self.mqtt_client.publish_message('test/unittest', message={'test': 'true'})

        # message_received_barrier.wait()  # will throw BrokenBarrierError if timeout
        import time; time.sleep(5)

    def test_subscribe(self):
        self.fail()

    def test_get_shadow(self):
        self.fail()

    def test_put_shadow(self):
        self.fail()


if __name__ == '__main__':
    mqtt_client = MessageBroker('test')

    message_received_barrier = threading.Barrier(2, timeout=20)

    def unittest_mqtt_subscriber(topic, message, **kwargs):
        print(f'Topic {topic} received message: {message}')
        message_received_barrier.wait()  # synchronize between threads

    mqtt_client.subscribe_message('test/unittest', unittest_mqtt_subscriber)
    mqtt_client.publish_message('test/unittest', message={'test': 'true'})

    message_received_barrier.wait()  # will throw BrokenBarrierError if timeout
