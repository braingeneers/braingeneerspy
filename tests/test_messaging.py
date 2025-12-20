""" Unit test for BraingeneersMqttClient, assumes Braingeneers ~/.aws/credentials file exists """
import logging
import queue
import threading
import time
import unittest
import uuid
import warnings

from unittest.mock import MagicMock
from unittest.mock import patch
from tenacity import retry, stop_after_attempt

import braingeneers.iot.messaging as messaging


def _make_message_broker():
    try:
        return messaging.MessageBroker(f"unittest-{uuid.uuid4()}")
    except PermissionError:
        raise unittest.SkipTest("Can't test messaging without access token.") from None


class TestBraingeneersMessageBroker(unittest.TestCase):
    def setUp(self) -> None:
        warnings.filterwarnings(
            "ignore", category=ResourceWarning, message="unclosed.*<ssl.SSLSocket.*>"
        )
        self.mb = _make_message_broker()
        self.mb_test_device = messaging.MessageBroker("unittest")
        self.mb.create_device("test", "Other")


    def tearDown(self) -> None:
        self.mb.shutdown()
        self.mb_test_device.shutdown()

    def test_publish_message_error(self):
        self.mb._mqtt_connection = MagicMock()

        # Mock a failed publish_message attempt
        self.mb._mqtt_connection.publish.return_value.rc = 1

        with self.assertRaises(messaging.MQTTError):
            self.mb.publish_message("test", "message")

    def test_subscribe_system_messages(self):
        q = self.mb.subscribe_message("$SYS/#", callback=None)
        self.mb.publish_message("test/unittest", message={"test": "true"})

        t0 = time.time()
        while time.time() - t0 < 5:
            topic, message = q.get(timeout=5)
            print(f"DEBUG TEST> {topic}")
            if topic.startswith("$SYS"):
                self.assertTrue(True)
                break

    def test_two_message_broker_objects(self):
        """Tests that two message broker objects can successfully publish and subscribe messages"""
        mb1 = messaging.MessageBroker()
        mb2 = messaging.MessageBroker()
        q1 = messaging.CallableQueue()
        q2 = messaging.CallableQueue()
        mb1.subscribe_message("test/unittest1", q1)
        mb2.subscribe_message("test/unittest2", q2)
        mb1.publish_message("test/unittest1", message={"test": "true"})
        mb2.publish_message("test/unittest2", message={"test": "true"})
        topic, message = q1.get()
        self.assertEqual(topic, "test/unittest1")
        self.assertEqual(message, {"test": "true"})
        topic, message = q2.get()
        self.assertEqual(topic, "test/unittest2")
        self.assertEqual(message, {"test": "true"})
        mb1.shutdown()
        mb2.shutdown()

    def test_publish_subscribe_message(self):
        """Uses a custom callback to test publish subscribe messages"""
        message_received_barrier = threading.Barrier(2, timeout=30)

        def unittest_subscriber(topic, message):
            print(f"DEBUG> {topic}: {message}")
            self.assertEqual(topic, "test/unittest")
            self.assertEqual(message, {"test": "true"})
            message_received_barrier.wait()  # synchronize between threads

        self.mb.subscribe_message("test/unittest", unittest_subscriber)
        self.mb.publish_message("test/unittest", message={"test": "true"})

        message_received_barrier.wait()  # will throw BrokenBarrierError if timeout

    def test_publish_subscribe_message_with_confirm_receipt(self):
        q = messaging.CallableQueue()
        self.mb.subscribe_message("test/unittest", q)
        self.mb.publish_message(
            "test/unittest", message={"test": "true"}, confirm_receipt=True
        )
        topic, message = q.get()
        self.assertEqual(topic, "test/unittest")
        self.assertEqual(message, {"test": "true"})

    def test_publish_subscribe_data_stream(self):
        """Uses queue method to test publish/subscribe data streams"""
        q = messaging.CallableQueue(1)
        self.mb.subscribe_data_stream(stream_name="unittest", callback=q)
        self.mb.publish_data_stream(
            stream_name="unittest", data={b"x": b"42"}, stream_size=1
        )
        result_stream_name, result_data = q.get(timeout=15)
        self.assertEqual(result_stream_name, "unittest")
        self.assertDictEqual(result_data, {b"x": b"42"})

    def test_publish_subscribe_multiple_data_streams(self):
        self.mb.redis_client.delete("unittest1", "unittest2")
        q = messaging.CallableQueue()
        self.mb.subscribe_data_stream(
            stream_name=["unittest1", "unittest2"], callback=q
        )
        self.mb.publish_data_stream(
            stream_name="unittest1", data={b"x": b"42"}, stream_size=1
        )
        self.mb.publish_data_stream(
            stream_name="unittest2", data={b"x": b"43"}, stream_size=1
        )
        self.mb.publish_data_stream(
            stream_name="unittest2", data={b"x": b"44"}, stream_size=1
        )

        result_stream_name, result_data = q.get(timeout=15)
        self.assertEqual(result_stream_name, "unittest1")
        self.assertDictEqual(result_data, {b"x": b"42"})

        result_stream_name, result_data = q.get(timeout=15)
        self.assertEqual(result_stream_name, "unittest2")
        self.assertDictEqual(result_data, {b"x": b"43"})

        result_stream_name, result_data = q.get(timeout=15)
        self.assertEqual(result_stream_name, "unittest2")
        self.assertDictEqual(result_data, {b"x": b"44"})

    @retry(stop=stop_after_attempt(3))
    def test_poll_data_stream(self):
        """Uses more advanced poll_data_stream function"""
        self.mb.redis_client.delete("unittest")

        self.mb.publish_data_stream(
            stream_name="unittest", data={b"x": b"42"}, stream_size=1
        )
        self.mb.publish_data_stream(
            stream_name="unittest", data={b"x": b"43"}, stream_size=1
        )
        self.mb.publish_data_stream(
            stream_name="unittest", data={b"x": b"44"}, stream_size=1
        )

        result1 = self.mb.poll_data_streams({"unittest": "-"}, count=1)
        self.assertEqual(len(result1[0][1]), 1)
        self.assertDictEqual(result1[0][1][0][1], {b"x": b"42"})

        result2 = self.mb.poll_data_streams({"unittest": result1[0][1][0][0]}, count=2)
        self.assertEqual(len(result2[0][1]), 2)
        self.assertDictEqual(result2[0][1][0][1], {b"x": b"43"})
        self.assertDictEqual(result2[0][1][1][1], {b"x": b"44"})

        result3 = self.mb.poll_data_streams({"unittest": "-"})
        self.assertEqual(len(result3[0][1]), 3)
        self.assertDictEqual(result3[0][1][0][1], {b"x": b"42"})
        self.assertDictEqual(result3[0][1][1][1], {b"x": b"43"})
        self.assertDictEqual(result3[0][1][2][1], {b"x": b"44"})

    # TypeError: 'NoneType' object is not subscriptable
    @unittest.expectedFailure
    def test_delete_device_state(self):
        self.mb.delete_device_state("test")
        self.mb.update_device_state("test", {"x": 42, "y": 24})
        state = self.mb.get_device_state("test")
        self.assertTrue("x" in state)
        self.assertTrue(state["x"] == 42)
        self.assertTrue("y" in state)
        self.assertTrue(state["y"] == 24)
        self.mb.delete_device_state("test", ["x"])
        state_after_del = self.mb.get_device_state("test")
        self.assertTrue("x" not in state_after_del)
        self.assertTrue("y" in state)
        self.assertTrue(state["y"] == 24)

    # TypeError: 'NoneType' object is not subscriptable
    @unittest.expectedFailure
    def test_get_update_device_state(self):
        self.mb_test_device.delete_device_state("test")
        self.mb_test_device.update_device_state("test", {"x": 42})
        state = self.mb_test_device.get_device_state("test")
        self.assertTrue("x" in state)
        self.assertEqual(state["x"], 42)
        self.mb_test_device.delete_device_state("test")

    def test_lock(self):
        with self.mb.get_lock("unittest"):
            print("lock granted")

    def test_unsubscribe(self):
        q = messaging.CallableQueue()
        self.mb.subscribe_message("test/unittest", callback=q)
        self.mb.unsubscribe_message("test/unittest")
        self.mb.publish_message("test/unittest", message={"test": 1})
        with self.assertRaises(queue.Empty):
            q.get(timeout=3)

    def test_two_subscribers(self):
        q1 = messaging.CallableQueue()
        q2 = messaging.CallableQueue()
        self.mb.subscribe_message("test/unittest1", callback=q1)
        self.mb.subscribe_message("test/unittest2", callback=q2)
        self.mb.publish_message("test/unittest1", message={"test": 1})
        self.mb.publish_message("test/unittest2", message={"test": 2})
        topic1, message1 = q1.get(timeout=5)
        topic2, message2 = q2.get(timeout=5)
        self.assertDictEqual(message1, {"test": 1})
        self.assertDictEqual(message2, {"test": 2})


class TestMqttDisconnect(unittest.TestCase):
    def setUp(self):
        self.config = """
[braingeneers-mqtt]
profile-id = test-id
profile-key = test-key
endpoint = localhost
port = 1883
"""

        # Manually construct MessageBroker without invoking __init__ to avoid
        # external dependencies such as Redis and JWT credentials.
        self.broker = messaging.MessageBroker.__new__(messaging.MessageBroker)
        self.broker.logger = logging.getLogger(__name__)
        self.broker.name = "test-broker"
        self.broker._credentials = self.config
        self.broker._mqtt_connection = None
        self.broker._mqtt_profile_id = "test-id"
        self.broker._mqtt_profile_key = "test-key"
        self.broker._mqtt_endpoint = "localhost"
        self.broker._mqtt_port = 1883
        self.broker._boto_iot_client = None
        self.broker._boto_iot_data_client = None
        self.broker._redis_client = None
        self.broker._jwt_service_account_token = None
        self.broker.certs_temp_dir = None
        self.broker._subscribed_data_streams = set()
        self.broker._subscribed_message_callback_map = {}
        self.broker._subscribe_message_lock = threading.Lock()

    @staticmethod
    def _make_fake_client_class(event_queue):
        class FakeMessageInfo:
            def __init__(self, rc=0):
                self.rc = rc

        class FakeClient:
            def __init__(self, _callback_version, _client_id):
                self.connected = False
                self.loop_running = False
                self.on_connect = None
                self.on_disconnect = None
                self.on_message = None
                self.published_payloads = []

            def username_pw_set(self, *_args, **_kwargs):
                pass

            def reconnect_delay_set(self, *_args, **_kwargs):
                pass

            def connect(self, *_, **__):
                self.connected = True
                if self.on_connect:
                    self.on_connect(self, None, None, 0)

            def loop_start(self):
                self.loop_running = True

            def subscribe(self, *_args, **_kwargs):
                event_queue.put("subscribed")

            def unsubscribe(self, *_args, **_kwargs):
                pass

            def publish(self, *_args, **_kwargs):
                info = FakeMessageInfo(rc=0 if self.connected else 1)
                self.published_payloads.append(kwargs.get("payload"))
                return info

            def simulate_disconnect(self):
                self.connected = False
                self.loop_running = False
                if self.on_disconnect:
                    self.on_disconnect(self, None, 1)

            def simulate_reconnect(self):
                self.connected = True
                if self.on_connect:
                    self.on_connect(self, None, None, 0)

            def trigger_message(self, topic, payload):
                if self.connected and self.loop_running and self.on_message:
                    class Msg:
                        def __init__(self, t, p):
                            self.topic = t
                            self.payload = p

                    self.on_message(self, None, Msg(topic, payload))

        return FakeClient

    def test_callbacks_continue_after_disconnect_and_reconnect(self):
        """Expose regression where callbacks stop firing after reconnect."""

        event_queue = queue.Queue()
        FakeClient = self._make_fake_client_class(event_queue)

        with patch.object(messaging.mqtt_client, "Client", FakeClient):
            received = queue.Queue()

            # Initialize connection and subscription
            self.broker.subscribe_message("test/topic", lambda t, m: received.put((t, m)))
            self.assertEqual(event_queue.get(timeout=1), "subscribed")

            # Deliver a message before disconnect
            self.broker.mqtt_connection.trigger_message("test/topic", b'{"ok": true}')
            topic, message = received.get(timeout=1)
            self.assertEqual(topic, "test/topic")
            self.assertEqual(message, {"ok": True})

            # Simulate connection drop
            self.broker.mqtt_connection.simulate_disconnect()

            # Simulate broker reconnecting without restarting the loop thread
            self.broker.mqtt_connection.simulate_reconnect()

            # Messages after a reconnect should still invoke callbacks.
            self.broker.mqtt_connection.trigger_message("test/topic", b'{"after": true}')

            # Expected behavior: callbacks keep working and deliver the message.
            topic, message = received.get(timeout=0.5)
            self.assertEqual(topic, "test/topic")
            self.assertEqual(message, {"after": True})


class TestInterprocessQueue(unittest.TestCase):
    def setUp(self) -> None:
        self.mb = _make_message_broker()
        self.mb.delete_queue("unittest")

    @retry(stop=stop_after_attempt(3))
    def test_get_put_defaults(self):
        q = self.mb.get_queue("unittest")
        q.put("some-value")
        result = q.get("some-value")
        self.assertEqual(result, "some-value")

    def test_get_put_nonblocking_without_maxsize(self):
        q = self.mb.get_queue("unittest")
        q.put("some-value", block=False)
        result = q.get(block=False)
        self.assertEqual(result, "some-value")

    @retry(stop=stop_after_attempt(3))
    def test_maxsize(self):
        q = self.mb.get_queue("unittest", maxsize=1)
        q.put("some-value")
        result = q.get()
        self.assertEqual(result, "some-value")

    @retry(stop=stop_after_attempt(3))
    def test_timeout_put(self):
        q = self.mb.get_queue("unittest", maxsize=1)
        q.put("some-value-1")
        with self.assertRaises(queue.Full):
            q.put("some-value-2", timeout=0.1)
            time.sleep(1)
            self.fail(
                "Queue failed to throw an expected exception after 0.1s timeout period."
            )

    def test_timeout_get(self):
        q = self.mb.get_queue("unittest", maxsize=1)
        with self.assertRaises(queue.Empty):
            q.get(timeout=0.1)
            time.sleep(1)
            self.fail(
                "Queue failed to throw an expected exception after 0.1s timeout period."
            )

    def test_task_done_join(self):
        """Test that task_done and join work as expected."""

        def f(ql, jl, bl):
            t0 = time.time()
            ql.join()
            jl["join_time"] = time.time() - t0
            b.wait()

        b = threading.Barrier(2)
        join_time = {"join_time": 0}  # a mutable datastructure

        q = self.mb.get_queue("unittest")
        q.put("some-value")
        threading.Thread(target=f, args=(q, join_time, b)).start()
        time.sleep(0.1)
        q.get()
        q.task_done()
        q.join()
        b.wait()

        t = join_time["join_time"]
        self.assertTrue(t >= 0.1, msg=f"Join time {t} less than expected 0.1 sec.")


class TestNamedLock(unittest.TestCase):
    def setUp(self) -> None:
        self.mb = _make_message_broker()
        self.mb.delete_lock("unittest")

    def tearDown(self) -> None:
        self.mb.delete_lock("unittest")

    def test_enter_exit(self):
        with self.mb.get_lock("unittest"):
            self.assertTrue(True)

    def test_acquire_release(self):
        lock = self.mb.get_lock("unittest")
        lock.acquire()
        lock.release()


if __name__ == "__main__":
    unittest.main()
