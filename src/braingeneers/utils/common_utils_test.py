import unittest
from unittest.mock import patch, MagicMock
import common_utils  # Updated import statement
import os
import tempfile


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


if __name__ == '__main__':
    unittest.main()
