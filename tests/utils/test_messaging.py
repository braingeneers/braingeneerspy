""" Unit test for BraingeneersMqttClient, assumes Braingeneers ~/.aws/credentials file exists """
import unittest
import unittest.mock
import braingeneers.utils.messaging as messaging
import threading
import awscrt.io


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
        """ Uses queue method to test publish/subscribe data streams """
        q = messaging.CallableQueue(1)
        self.mb.subscribe_data_stream(stream_name='unittest', callback=q)
        self.mb.publish_data_stream(stream_name='unittest', data={b'x': b'42'}, stream_size=1)
        result_stream_name, result_data = q.get(timeout=7)
        self.assertEqual(result_stream_name, 'unittest')
        self.assertDictEqual(result_data, {b'x': b'42'})

    def test_publish_subscribe_multiple_data_streams(self):
        self.mb.redis_client.delete('unittest1', 'unittest2')
        q = messaging.CallableQueue()
        self.mb.subscribe_data_stream(stream_name=['unittest1', 'unittest2'], callback=q)
        self.mb.publish_data_stream(stream_name='unittest1', data={b'x': b'42'}, stream_size=1)
        self.mb.publish_data_stream(stream_name='unittest2', data={b'x': b'43'}, stream_size=1)
        self.mb.publish_data_stream(stream_name='unittest2', data={b'x': b'44'}, stream_size=1)

        result_stream_name, result_data = q.get(timeout=7)
        self.assertEqual(result_stream_name, 'unittest1')
        self.assertDictEqual(result_data, {b'x': b'42'})

        result_stream_name, result_data = q.get(timeout=7)
        self.assertEqual(result_stream_name, 'unittest2')
        self.assertDictEqual(result_data, {b'x': b'43'})

        result_stream_name, result_data = q.get(timeout=7)
        self.assertEqual(result_stream_name, 'unittest2')
        self.assertDictEqual(result_data, {b'x': b'44'})

    def test_poll_data_stream(self):
        """ Uses more advanced poll_data_stream function """
        self.mb.redis_client.delete('unittest')

        self.mb.publish_data_stream(stream_name='unittest', data={b'x': b'42'}, stream_size=1)
        self.mb.publish_data_stream(stream_name='unittest', data={b'x': b'43'}, stream_size=1)
        self.mb.publish_data_stream(stream_name='unittest', data={b'x': b'44'}, stream_size=1)

        result1 = self.mb.poll_data_streams({'unittest': '-'}, count=1)
        self.assertEqual(len(result1[0][1]), 1)
        self.assertDictEqual(result1[0][1][0][1], {b'x': b'42'})

        result2 = self.mb.poll_data_streams({'unittest': result1[0][1][0][0]}, count=2)
        self.assertEqual(len(result2[0][1]), 2)
        self.assertDictEqual(result2[0][1][0][1], {b'x': b'43'})
        self.assertDictEqual(result2[0][1][1][1], {b'x': b'44'})

        result3 = self.mb.poll_data_streams({'unittest': '-'})
        self.assertEqual(len(result3[0][1]), 3)
        self.assertDictEqual(result3[0][1][0][1], {b'x': b'42'})
        self.assertDictEqual(result3[0][1][1][1], {b'x': b'43'})
        self.assertDictEqual(result3[0][1][2][1], {b'x': b'44'})

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

    def test_list_devices_basic(self):
        awscrt.io.init_logging(awscrt.io.LogLevel.Warn, 'stderr')
        self.mb.subscribe_message('devices/test', callback=unittest.mock.Mock())
        devices_online = self.mb.list_devices()
        assert len(devices_online) > 0
