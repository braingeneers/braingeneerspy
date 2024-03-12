import io
import unittest
from unittest.mock import patch, MagicMock
import common_utils
from common_utils import checkout, force_release_checkout
from braingeneers.iot import messaging
import os
import tempfile
import braingeneers.utils.smart_open_braingeneers as smart_open
from typing import Union


class TestFileListFunction(unittest.TestCase):

    @patch('common_utils._lazy_init_s3_client')  # Updated to common_utils
    def test_s3_files_exist(self, mock_s3_client):
        # Mock S3 client response
        mock_response = {
            'Contents': [
                {'Key': 'file1.txt', 'LastModified': '2023-01-01', 'Size': 123},
                {'Key': 'file2.txt', 'LastModified': '2023-01-02', 'Size': 456}
            ]
        }
        mock_s3_client.return_value.list_objects.return_value = mock_response

        result = common_utils.file_list('s3://test-bucket/')  # Updated to common_utils
        expected = [('file2.txt', '2023-01-02', 456), ('file1.txt', '2023-01-01', 123)]
        self.assertEqual(result, expected)

    @patch('common_utils._lazy_init_s3_client')  # Updated to common_utils
    def test_s3_no_files(self, mock_s3_client):
        # Mock S3 client response for no files
        mock_s3_client.return_value.list_objects.return_value = {}
        result = common_utils.file_list('s3://test-bucket/')  # Updated to common_utils
        self.assertEqual(result, [])

    def test_local_files_exist(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            for f in ['tempfile1.txt', 'tempfile2.txt']:
                with open(os.path.join(temp_dir, f), 'w') as w:
                    w.write('nothing')

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
        self.message_broker_patch = patch('braingeneers.iot.messaging.MessageBroker')

        # Start the patches
        self.mock_message_broker = self.message_broker_patch.start()

        # Mock the message broker's get_lock and delete_lock methods
        self.mock_message_broker.return_value.get_lock.return_value = MagicMock()
        self.mock_message_broker.return_value.delete_lock = MagicMock()

        self.mock_file = MagicMock(spec=io.StringIO)
        self.mock_file.read.return_value = 'Test data'  # Ensure this is correctly setting the return value for read
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
        with checkout('s3://test-bucket/test-file.txt', isbinary=False) as locked_obj:
            data = locked_obj.get_value()
            self.assertEqual(data, 'Test data')

    def test_checkout_context_manager_write_text(self):
        # Test the writing functionality for text mode
        test_data = 'New test data'
        self.mock_file.write.reset_mock()  # Reset mock to ensure clean state for the test
        with checkout('s3://test-bucket/test-file.txt', isbinary=False) as locked_obj:
            locked_obj.checkin(test_data)
            self.mock_file.write.assert_called_once_with(test_data)

    def test_checkout_context_manager_write_binary(self):
        # Test the writing functionality for binary mode
        test_data = b'New binary data'
        self.mock_file.write.reset_mock()  # Reset mock to ensure clean state for the test
        with checkout('s3://test-bucket/test-file.bin', isbinary=True) as locked_obj:
            locked_obj.checkin(test_data)
            self.mock_file.write.assert_called_once_with(test_data)

    def test_with_pass_through_kwargs_handling(self):
        """Test map2 with a function accepting dynamic kwargs, specifically to check the handling of 'experiment_name'
        passed through **kwargs, using the original signature for f_with_kwargs."""

        def f_with_kwargs(cache_path: str, max_size_gb: int = 10, **kwargs):
            # Simulate loading data where 'experiment_name' and other parameters are expected to come through **kwargs
            self.assertTrue(isinstance(kwargs, dict), 'kwargs should be a dict')
            self.assertFalse('kwargs' in kwargs)
            return 'some data'

        experiments = [{'experiment': 'exp1'}, {'experiment': 'exp2'}]  # List of experiment names to be passed as individual kwargs
        fixed_values = {
            "cache_path": '/tmp/ephys_cache',
            "max_size_gb": 50,
            "metadata": {"some": "metadata"},
            "channels": ["channel1"],
            "length": -1,
        }

        # Execute the test under the assumption that map2 is supposed to handle 'experiment_name' in **kwargs correctly
        map2(f_with_kwargs, kwargs=experiments, fixed_values=fixed_values, parallelism=False)
        self.assertTrue(True)  # If the test reaches this point, it has passed


if __name__ == '__main__':
    unittest.main()
