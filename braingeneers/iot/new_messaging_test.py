""" Unit test for BraingeneersMqttClient, assumes Braingeneers ~/.aws/credentials file exists """
import datetime
import time
import unittest.mock
import new_messaging as messaging
import threading
import uuid
import warnings
import functools
# import awscrt



class TestBraingeneersMessageBroker(unittest.TestCase):
    def setUp(self) -> None:
        warnings.filterwarnings("ignore", category=ResourceWarning, message="unclosed.*<ssl.SSLSocket.*>")
        self.mb = messaging.MessageBroker(f'test-{uuid.uuid4()}')
        self.mb_test_device = messaging.MessageBroker('unittest')
        # awscrt.io.init_logging(awscrt.io.LogLevel.Trace, 'stderr')  # enable Trace logging of AWS IOT

    # def tearDown(self) -> None:
    #     self.mb.shutdown()
    #     self.mb_test_device.shutdown()

    # def test_publish(self):
    #     """ Test publish function """
    #     self.mb.publish_message('test/unittest', message={'test': 'true'})

    def test_publish_subscribe_message(self):
        """ Uses a custom callback to test publish subscribe messages """
        message_received_barrier = threading.Barrier(2, timeout=5)

        def unittest_subscriber(topic, message):
            self.assertEqual(topic, 'test/unittest')
            self.assertEqual(message, {'test': 'true'})
            message_received_barrier.wait()  # synchronize between threads

        self.mb.subscribe_message('test/unittest', unittest_subscriber)
        self.mb.publish_message('test/unittest', message={'test': 'true'})

        message_received_barrier.wait()  # will throw BrokenBarrierError if timeout

#     def test_publish_subscribe_data_stream(self):
#         """ Uses queue method to test publish/subscribe data streams """
#         q = messaging.CallableQueue(1)
#         self.mb.subscribe_data_stream(stream_name='unittest', callback=q)
#         self.mb.publish_data_stream(stream_name='unittest', data={b'x': b'42'}, stream_size=1)
#         result_stream_name, result_data = q.get(timeout=15)
#         self.assertEqual(result_stream_name, 'unittest')
#         self.assertDictEqual(result_data, {b'x': b'42'})

#     def test_publish_subscribe_multiple_data_streams(self):
#         self.mb.redis_client.delete('unittest1', 'unittest2')
#         q = messaging.CallableQueue()
#         self.mb.subscribe_data_stream(stream_name=['unittest1', 'unittest2'], callback=q)
#         self.mb.publish_data_stream(stream_name='unittest1', data={b'x': b'42'}, stream_size=1)
#         self.mb.publish_data_stream(stream_name='unittest2', data={b'x': b'43'}, stream_size=1)
#         self.mb.publish_data_stream(stream_name='unittest2', data={b'x': b'44'}, stream_size=1)

#         result_stream_name, result_data = q.get(timeout=15)
#         self.assertEqual(result_stream_name, 'unittest1')
#         self.assertDictEqual(result_data, {b'x': b'42'})

#         result_stream_name, result_data = q.get(timeout=15)
#         self.assertEqual(result_stream_name, 'unittest2')
#         self.assertDictEqual(result_data, {b'x': b'43'})

#         result_stream_name, result_data = q.get(timeout=15)
#         self.assertEqual(result_stream_name, 'unittest2')
#         self.assertDictEqual(result_data, {b'x': b'44'})

#     def test_poll_data_stream(self):
#         """ Uses more advanced poll_data_stream function """
#         self.mb.redis_client.delete('unittest')

#         self.mb.publish_data_stream(stream_name='unittest', data={b'x': b'42'}, stream_size=1)
#         self.mb.publish_data_stream(stream_name='unittest', data={b'x': b'43'}, stream_size=1)
#         self.mb.publish_data_stream(stream_name='unittest', data={b'x': b'44'}, stream_size=1)

#         result1 = self.mb.poll_data_streams({'unittest': '-'}, count=1)
#         self.assertEqual(len(result1[0][1]), 1)
#         self.assertDictEqual(result1[0][1][0][1], {b'x': b'42'})

#         result2 = self.mb.poll_data_streams({'unittest': result1[0][1][0][0]}, count=2)
#         self.assertEqual(len(result2[0][1]), 2)
#         self.assertDictEqual(result2[0][1][0][1], {b'x': b'43'})
#         self.assertDictEqual(result2[0][1][1][1], {b'x': b'44'})

#         result3 = self.mb.poll_data_streams({'unittest': '-'})
#         self.assertEqual(len(result3[0][1]), 3)
#         self.assertDictEqual(result3[0][1][0][1], {b'x': b'42'})
#         self.assertDictEqual(result3[0][1][1][1], {b'x': b'43'})
#         self.assertDictEqual(result3[0][1][2][1], {b'x': b'44'})

#     def test_delete_device_state(self):
#         self.mb.delete_device_state('test')
#         self.mb.update_device_state('test', {'x': 42, 'y': 24})
#         state = self.mb.get_device_state('test')
#         self.assertTrue('x' in state)
#         self.assertTrue(state['x'] == 42)
#         self.assertTrue('y' in state)
#         self.assertTrue(state['y'] == 24)
#         self.mb.delete_device_state('test', ['x'])
#         state_after_del = self.mb.get_device_state('test')
#         self.assertTrue('x' not in state_after_del)
#         self.assertTrue('y' in state)
#         self.assertTrue(state['y'] == 24)

#     def test_get_update_device_state(self):
#         self.mb_test_device.delete_device_state('test')
#         self.mb_test_device.update_device_state('test', {'x': 42})
#         state = self.mb_test_device.get_device_state('test')
#         self.assertTrue('x' in state)
#         self.assertEqual(state['x'], 42)
#         self.mb_test_device.delete_device_state('test')

#     def test_list_devices_basic(self):
#         q = self.mb_test_device.subscribe_message('test/unittest', callback=messaging.CallableQueue())
#         self.mb_test_device.publish_message('test/unittest', message={'test': 'true'})
#         q.get()  # waits for the message to be published and received before moving on to check the online devices

#         time.sleep(20)  # Due to issue: https://stackoverflow.com/questions/72564492
#         devices_online = self.mb_test_device.list_devices()
#         self.assertTrue(len(devices_online) > 0)

#     @staticmethod
#     def callback_device_state_change(barrier: threading.Barrier, result: dict,
#                                      device_name: str, device_state_key: str, new_value):
#         print('')
#         print(f'unittest callback - device_name: {device_name}, device_state_key: {device_state_key}, new_value: {new_value}')
#         result['device_name'] = device_name
#         result['device_state_key'] = device_state_key
#         result['new_value'] = new_value
#         barrier.wait()

#     def test_subscribe_device_state_change(self):
#         result = {}
#         t = str(datetime.datetime.today())
#         self.mb_test_device.update_device_state('unittest', {'unchanging_key': 'static'})
#         barrier = threading.Barrier(2)
#         func = functools.partial(self.callback_device_state_change, barrier, result)
#         self.mb_test_device.subscribe_device_state_change(
#             device_name='unittest', device_state_keys=['test_key'], callback=func
#         )
#         self.mb_test_device.update_device_state('unittest', {'test_key': t})
#         try:
#             barrier.wait(timeout=5)
#         except threading.BrokenBarrierError:
#             self.fail(msg='Barrier timeout')

#         self.assertEqual(result['device_name'], 'unittest')
#         self.assertEqual(result['device_state_key'], 'test_key')
#         self.assertEqual(result['new_value'], t)


# def callback_device_state_change(barrier: threading.Barrier, result: dict,
#                                  device_name: str, device_state_key: str, new_value):
#     print('')
#     print(f'UNITTEST CALLBACK\n\tdevice_name: {device_name}\n\tdevice_state_key: {device_state_key}\n\tnew_value: {new_value}')
#     result['device_name'] = device_name
#     result['device_state_key'] = device_state_key
#     result['new_value'] = new_value
#     barrier.wait()


if __name__ == '__main__':
    mb = messaging.MessageBroker('unittest')
    # print(TestBraingeneersMessageBroker('setUp'))
    # unittest.main()


    message_received_barrier = threading.Barrier(2, timeout=5)
    def on_message(client, userdata, msg):
        print(f"Received `{msg.payload.decode()}` from `{msg.topic}` topic")

    mb.subscribe_message('test/unittest', on_message)
    mb.publish_message('test/unittest', message="booooof")

    message_received_barrier.wait()  # will throw BrokenBarrierError if timeout
    # print(mb_test_device.publish_message('test/unittest', message={'test': 'true'}))

#     # awscrt.io.init_logging(awscrt.io.LogLevel.Trace, 'stderr')
#
#     result = {}
#     print('a')
#     t = str(datetime.datetime.today())
#     print('b')
#     mb_test_device.update_device_state('unittest', {'unchanging_key': 'static'})
#     print('c')
#     barrier = threading.Barrier(2)
#     print('d')
#     func = functools.partial(callback_device_state_change, barrier, result)
#     print('e')
#     mb_test_device.subscribe_device_state_change(
#         device_name='unittest', device_state_keys=['test_key'], callback=func
#     )
#     print('f')
#     mb_test_device.update_device_state('unittest', {'test_key': t})
#     try:
#         barrier.wait(timeout=5)
#     except threading.BrokenBarrierError:
#         print(msg='Barrier timeout')
#
#     assert result['device_name'] == 'unittest'
#     assert result['device_state_key'] == 'test_key'
#     assert result['new_value'] == t
