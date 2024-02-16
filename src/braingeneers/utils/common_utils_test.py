import unittest
from unittest.mock import patch, MagicMock
from common_utils import checkout, checkin, force_release_checkout
from braingeneers.iot import messaging
import os
import tempfile
import braingeneers.utils.smart_open_braingeneers as smart_open


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


class TestCheckingCheckout(unittest.TestCase):
    def setUp(self) -> None:
        self.text_value = 'unittest1'
        self.filepath = 's3://braingeneersdev/unittest/test.txt'
        force_release_checkout(self.filepath)

        with smart_open.open(self.filepath, 'w') as f:
            f.write(self.text_value)

    def test_checkout_checkin(self):
        f = checkout(self.filepath)
        self.assertEqual(f.read(), self.text_value)
        checkin(self.filepath, f)

# class TestAtomicGetSetEphysMetadata(unittest.TestCase):
#     def setUp(self) -> None:
#         self.mb = messaging.MessageBroker()
#         # Delete any previously held lock
#         AtomicGetSetEphysMetadata('2020-03-25-e-testit').force_release()
#
#     def test_noop(self):
#         """ Very trivial exercise of the code. """
#
#         with AtomicGetSetEphysMetadata('2020-03-25-e-testit') as metadata:
#             self.assertTrue(metadata is not None)
#             self.assertTrue(isinstance(metadata, dict))
#             self.assertTrue(len(metadata) > 0)


if __name__ == '__main__':
    unittest.main()
