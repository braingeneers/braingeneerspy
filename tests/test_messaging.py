""" Unit test for BraingeneersMqttClient, assumes Braingeneers ~/.aws/credentials file exists """
import unittest
import braingeneers.utils.messaging as messaging
import threading


class TestBraingeneersMessageBroker(unittest.TestCase):
    def setUp(self) -> None:
        self.mb = messaging.MessageBroker('test')

    def test_publish_subscribe_message(self):
        """ Uses a custom callback to test publish subscribe messages """
        message_received_barrier = threading.Barrier(2, timeout=7)

        def unittest_subscriber(topic, message):
            self.assertEqual(topic, 'test/unittest')
            self.assertEqual(message, {'test': 'true'})
            message_received_barrier.wait()  # synchronize between threads

        self.mb.subscribe_message('test/unittest', unittest_subscriber)
        self.mb.publish_message('test/unittest', message={'test': 'true'})

        message_received_barrier.wait()  # will throw BrokenBarrierError if timeout

    def test_publish_subscribe_data_stream(self):
        """ Uses polling method to test publish/subscribe data streams """
        q = messaging.CallableQueue(1)
        self.mb.subscribe_data_stream(stream_name='unittest', callback=q)
        self.mb.publish_data_stream(stream_name='unittest', data=b'42', stream_size=1)
        result_stream_name, result_data = q.get(timeout=7)
        self.assertEqual(result_stream_name, 'unittest')
        self.assertEqual(result_data, b'42')

    def test_delete_device_state(self):
        self.mb.delete_device_state('test')
        self.mb.update_device_state('test', {'x': 42, 'y': 24})
        state = self.mb.get_device_state('test')
        self.assertTrue('x' in state)
        self.assertTrue(state['x'] == 42)
        self.assertTrue('y' in state)
        self.assertTrue(state['y'] == 24)
        self.mb.delete_device_state('test', ['x'])
        state_after_del = self.mb.get_device_state('test')
        self.assertTrue('x' not in state_after_del)
        self.assertTrue('y' in state)
        self.assertTrue(state['y'] == 24)

    def test_get_set_device_state(self):
        self.mb.delete_device_state('test')
        self.mb.update_device_state('test', {'x': 42})
        state = self.mb.get_device_state('test')
        self.assertTrue('x' in state)
        self.assertEqual(state['x'], 42)
        self.mb.delete_device_state('test')
