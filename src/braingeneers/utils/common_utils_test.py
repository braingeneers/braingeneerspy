import unittest
from unittest.mock import patch, MagicMock
import file_utils
import os
import tempfile
from pathlib import Path


class TestFileListFunction(unittest.TestCase):

    @patch('file_utils._lazy_init_s3_client')
    def test_s3_files_exist(self, mock_s3_client):
        # Mock S3 client response
        mock_response = {
            'Contents': [
                {'Key': 'file1.txt', 'LastModified': '2023-01-01', 'Size': 123},
                {'Key': 'file2.txt', 'LastModified': '2023-01-02', 'Size': 456}
            ]
        }
        mock_s3_client.return_value.list_objects.return_value = mock_response

        result = file_utils.file_list('s3://test-bucket/')
        expected = [('file1.txt', '2023-01-01', 123), ('file2.txt', '2023-01-02', 456)]
        self.assertEqual(result, expected)

    @patch('file_utils._lazy_init_s3_client')
    def test_s3_no_files(self, mock_s3_client):
        # Mock S3 client response for no files
        mock_s3_client.return_value.list_objects.return_value = {}
        result = file_utils.file_list('s3://test-bucket/')
        self.assertEqual(result, [])

    def test_local_files_exist(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create temporary files
            Path(temp_dir, 'tempfile1.txt').touch()
            Path(temp_dir, 'tempfile2.txt').touch()

            result = file_utils.file_list(temp_dir)
            # The result should contain two files with their details
            self.assertEqual(len(result), 2)

    def test_local_no_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = file_utils.file_list(temp_dir)
            self.assertEqual(result, [])


if __name__ == '__main__':
    unittest.main()
