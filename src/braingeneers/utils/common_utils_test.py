import unittest
from unittest.mock import patch, MagicMock
import common_utils
from common_utils import map2
import os
import tempfile


def multiply(x, y):
    return x * y


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


class TestMap2(unittest.TestCase):
    def test_basic_functionality(self):
        """Test map2 with a simple function, no fixed values, no parallelism."""

        def simple_add(x, y):
            return x + y

        args = [(1, 2), (2, 3), (3, 4)]
        expected = [3, 5, 7]
        result = map2(simple_add, args=args, parallelism=False)
        self.assertEqual(result, expected)

    def test_with_fixed_values(self):
        """Test map2 with fixed values."""

        def f(a, b, c):
            return f'{a} {b} {c}'

        args = [2, 20, 200]
        expected = ['1 2 3', '1 20 3', '1 200 3']
        result = map2(func=f, args=args, fixed_values=dict(a=1, c=3), parallelism=False)
        self.assertEqual(result, expected)

    def test_with_parallelism(self):
        """Test map2 with parallelism enabled (assuming the environment supports it)."""
        args = [(1, 2), (2, 3), (3, 4)]
        expected = [2, 6, 12]
        result = map2(multiply, args=args, parallelism=True)
        self.assertEqual(result, expected)

    def test_with_invalid_args(self):
        """Test map2 with invalid args to ensure it raises the correct exceptions."""

        def simple_subtract(x, y):
            return x - y

        with self.assertRaises(AssertionError):
            map2(simple_subtract, args=[1], parallelism="invalid")


if __name__ == '__main__':
    unittest.main()
