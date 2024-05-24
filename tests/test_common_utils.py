import io
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import braingeneers.utils.smart_open_braingeneers as smart_open
from braingeneers.utils import common_utils
from braingeneers.utils.common_utils import checkout, map2


class TestFileListFunction(unittest.TestCase):
    @patch(
        "braingeneers.utils.common_utils._lazy_init_s3_client"
    )  # Updated to common_utils
    def test_s3_files_exist(self, mock_s3_client):
        # Mock S3 client response
        mock_response = {
            "Contents": [
                {"Key": "file1.txt", "LastModified": "2023-01-01", "Size": 123},
                {"Key": "file2.txt", "LastModified": "2023-01-02", "Size": 456},
            ]
        }
        mock_s3_client.return_value.list_objects.return_value = mock_response

        result = common_utils.file_list("s3://test-bucket/")  # Updated to common_utils
        expected = [("file2.txt", "2023-01-02", 456), ("file1.txt", "2023-01-01", 123)]
        self.assertEqual(result, expected)

    @patch(
        "braingeneers.utils.common_utils._lazy_init_s3_client"
    )  # Updated to common_utils
    def test_s3_no_files(self, mock_s3_client):
        # Mock S3 client response for no files
        mock_s3_client.return_value.list_objects.return_value = {}
        result = common_utils.file_list("s3://test-bucket/")  # Updated to common_utils
        self.assertEqual(result, [])

    def test_local_files_exist(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            for f in ["tempfile1.txt", "tempfile2.txt"]:
                with open(os.path.join(temp_dir, f), "w") as w:
                    w.write("nothing")

            result = common_utils.file_list(temp_dir)  # Updated to common_utils
            # The result should contain two files with their details
            self.assertEqual(len(result), 2)

    def test_local_no_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = common_utils.file_list(temp_dir)  # Updated to common_utils
            self.assertEqual(result, [])


class TestCheckout(unittest.TestCase):
    def setUp(self):
        # Setup mock for smart_open and MessageBroker
        self.message_broker_patch = patch("braingeneers.iot.messaging.MessageBroker")

        # Start the patches
        self.mock_message_broker = self.message_broker_patch.start()

        # Mock the message broker's get_lock and delete_lock methods
        self.mock_message_broker.return_value.get_lock.return_value = MagicMock()
        self.mock_message_broker.return_value.delete_lock = MagicMock()

        self.mock_file = MagicMock(spec=io.StringIO)
        self.mock_file.read.return_value = (
            "Test data"  # Ensure this is correctly setting the return value for read
        )
        self.mock_file.__enter__.return_value = self.mock_file
        self.mock_file.__exit__.return_value = None
        self.smart_open_mock = MagicMock(spec=smart_open)
        self.smart_open_mock.open.return_value = self.mock_file

        common_utils.smart_open = self.smart_open_mock

    def tearDown(self):
        # Stop all patches
        self.message_broker_patch.stop()

    def test_checkout_context_manager_read(self):
        # Test the reading functionality
        with checkout("s3://test-bucket/test-file.txt", isbinary=False) as locked_obj:
            data = locked_obj.get_value()
            self.assertEqual(data, "Test data")

    def test_checkout_context_manager_write_text(self):
        # Test the writing functionality for text mode
        test_data = "New test data"
        self.mock_file.write.reset_mock()  # Reset mock to ensure clean state for the test
        with checkout("s3://test-bucket/test-file.txt", isbinary=False) as locked_obj:
            locked_obj.checkin(test_data)
            self.mock_file.write.assert_called_once_with(test_data)

    def test_checkout_context_manager_write_binary(self):
        # Test the writing functionality for binary mode
        test_data = b"New binary data"
        self.mock_file.write.reset_mock()  # Reset mock to ensure clean state for the test
        with checkout("s3://test-bucket/test-file.bin", isbinary=True) as locked_obj:
            locked_obj.checkin(test_data)
            self.mock_file.write.assert_called_once_with(test_data)


class TestMap2Function(unittest.TestCase):
    def test_with_pass_through_kwargs_handling(self):
        """Test map2 with a function accepting dynamic kwargs, specifically to check the handling of 'experiment_name'
        passed through **kwargs, using the original signature for f_with_kwargs."""

        def f_with_kwargs(cache_path: str, max_size_gb: int = 10, **kwargs):
            # Simulate loading data where 'experiment_name' and other parameters are expected to come through **kwargs
            self.assertTrue(isinstance(kwargs, dict), "kwargs should be a dict")
            self.assertFalse("kwargs" in kwargs)
            return "some data"

        experiments = [
            {"experiment": "exp1"},
            {"experiment": "exp2"},
        ]  # List of experiment names to be passed as individual kwargs
        fixed_values = {
            "cache_path": "/tmp/ephys_cache",
            "max_size_gb": 50,
            "metadata": {"some": "metadata"},
            "channels": ["channel1"],
            "length": -1,
        }

        # Execute the test under the assumption that map2 is supposed to handle 'experiment_name' in **kwargs correctly
        map2(
            f_with_kwargs,
            kwargs=experiments,
            fixed_values=fixed_values,
            parallelism=False,
        )
        self.assertTrue(True)  # If the test reaches this point, it has passed

    def test_with_kwargs_function_parallelism_false(self):
        # Define a test function that takes a positional argument and arbitrary kwargs
        def test_func(a, **kwargs):
            return a + kwargs.get("increment", 0)

        # Define the arguments and kwargs to pass to map2
        args = [(1,), (2,), (3,)]  # positional arguments
        kwargs = [
            {"increment": 10},
            {"increment": 20},
            {"increment": 30},
        ]  # kwargs for each call

        # Call map2 with the test function, args, kwargs, and parallelism=False
        result = map2(func=test_func, args=args, kwargs=kwargs, parallelism=False)

        # Expected results after applying the function with the given args and kwargs
        expected_results = [11, 22, 33]

        # Assert that the actual result matches the expected result
        self.assertEqual(result, expected_results)

    def test_with_fixed_values_and_variable_kwargs_parallelism_false(self):
        # Define a test function that takes fixed positional argument and arbitrary kwargs
        def test_func(a, **kwargs):
            return a + kwargs.get("increment", 0)

        # Define the kwargs to pass to map2, each dict represents kwargs for one call
        kwargs = [{"increment": 10}, {"increment": 20}, {"increment": 30}]

        # Call map2 with the test function, no args, variable kwargs, fixed_values containing 'a', and parallelism=False
        result = map2(
            func=test_func,
            kwargs=kwargs,
            fixed_values={"a": 1},  # 'a' is fixed for all calls
            parallelism=False,
        )

        # Expected results after applying the function with the fixed 'a' and given kwargs
        expected_results = [11, 21, 31]

        # Assert that the actual result matches the expected result
        self.assertEqual(result, expected_results)

    def test_with_no_kwargs(self):
        # Define a test function that takes a positional argument and no kwargs
        def test_func(a):
            return a + 1

        # While we're at it, also test the pathway that normalizes the args.
        args = range(1, 4)
        result = map2(
            func=test_func,
            args=args,
            parallelism=False,
        )

        self.assertEqual(result, [2, 3, 4])


if __name__ == "__main__":
    unittest.main()
