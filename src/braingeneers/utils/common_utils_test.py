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


if __name__ == '__main__':
    unittest.main()
