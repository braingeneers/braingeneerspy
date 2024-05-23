import unittest
from unittest import mock

from botocore.exceptions import ClientError

from .configure import skip_unittest_if_offline
from .memoize_s3 import memoize


class TestMemoizeS3(unittest.TestCase):
    @skip_unittest_if_offline
    def test(self):
        # Run these checks in a context where S3_USER is set.
        with mock.patch.dict("os.environ", {"S3_USER": "unittest"}):
            # Memoize a function that counts its calls.
            @memoize()(ignore=["y"])
            def foo(x, y):
                nonlocal side_effect
                side_effect += 1
                return x

            self.assertEqual(
                foo.store_backend.location, "s3://braingeneersdev/unittest/cache/joblib"
            )

            # Call it a few times and make sure it only runs once.
            foo.clear()
            side_effect = 0
            for i in range(3):
                self.assertEqual(foo("bar", i), "bar")
            self.assertEqual(side_effect, 1)

            # Force it to run again and make sure it happens.
            foo("baz", 1)
            self.assertEqual(side_effect, 2)

            # Clean up by reaching into the cache and clearing the directory
            # without recreating the cache.
            foo.store_backend.clear()

    @skip_unittest_if_offline
    def test_uri_validation(self):
        # Our backend only supports S3 URIs.
        with self.assertRaises(ValueError):

            @memoize("this has to start with s3://")
            def foo(x):
                return x

    @skip_unittest_if_offline
    def test_cant_mmap(self):
        # We have to fail if memory mapping is requested because it's
        # impossible on S3.
        with self.assertRaises(ValueError):

            @memoize("s3://this-uri-doesnt-matter/", mmap_mode=True)
            def foo(x):
                return x

    @skip_unittest_if_offline
    def test_bucket_existence(self):
        # Bucket existence should be checked at creation.
        with self.assertRaises(ClientError):

            @memoize("s3://i-sure-hope-this-crazy-bucket-doesnt-exist/")
            def foo(x):
                return x

    @skip_unittest_if_offline
    def test_default_location(self):
        # Make sure a default location is correctly set when S3_USER is not.
        with mock.patch.dict("os.environ", {}, clear=True):

            @memoize()
            def foo(x):
                return x

            self.assertEqual(
                foo.store_backend.location, "s3://braingeneersdev/common/cache/joblib"
            )

    @skip_unittest_if_offline
    def test_custom_location(self):
        # Make sure custom locations get set correctly.
        @memoize("s3://braingeneersdev/unittest/cache")
        def foo(x):
            return x

        self.assertEqual(
            foo.store_backend.location, "s3://braingeneersdev/unittest/cache/joblib"
        )
