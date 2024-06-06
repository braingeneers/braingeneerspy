import sys
import tempfile
import unittest

import braingeneers
import braingeneers.utils.smart_open_braingeneers as smart_open


class SmartOpenTestCase(unittest.TestCase):
    test_bucket = "braingeneersdev"
    test_file = "test_file.txt"

    def test_online_smart_open_read(self):
        """Tests that a simple file open and read operation succeeds"""
        braingeneers.set_default_endpoint()  # sets the default PRP endpoint
        s3_url = f"s3://{self.test_bucket}/{self.test_file}"
        with smart_open.open(s3_url, "r") as f:
            txt = f.read()

        self.assertEqual(txt, "Don't panic\n")

    @unittest.skipIf(sys.platform.startswith("win"), "TODO: Test is broken on Windows.")
    def test_local_path_endpoint(self):
        with tempfile.TemporaryDirectory(prefix="smart_open_unittest_") as tmp_dirname:
            with tempfile.NamedTemporaryFile(
                dir=tmp_dirname, prefix="temp_unittest"
            ) as tmp_file:
                tmp_file_name = tmp_file.name
                tmp_file.write(b"unittest")
                tmp_file.flush()

                braingeneers.set_default_endpoint(f"{tmp_dirname}/")
                with smart_open.open(tmp_file_name, mode="rb") as tmp_file_smart_open:
                    self.assertEqual(tmp_file_smart_open.read(), b"unittest")


if __name__ == "__main__":
    unittest.main()
