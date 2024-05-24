import unittest

from braingeneers.utils import s3wrangler


class S3WranglerUnitTest(unittest.TestCase):
    def test_online_s3wrangler(self):
        dir_list = s3wrangler.list_directories("s3://braingeneers/")
        self.assertTrue("s3://braingeneers/ephys/" in dir_list)


if __name__ == "__main__":
    unittest.main()
