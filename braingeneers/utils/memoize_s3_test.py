import os
import unittest
from botocore.exceptions import ClientError
from unittest import mock
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
            def foo(x): return x

    @skip_unittest_if_offline
    def test_cant_mmap(self):
        # We have to fail if memory mapping is requested because it's
        # impossible on S3.
        with self.assertRaises(ValueError):
            @memoize("s3://this-uri-doesnt-matter/", mmap_mode=True)
            def foo(x): return x

    @skip_unittest_if_offline
    def test_bucket_existence(self):
        # Bucket existence should be checked at creation.
        with self.assertRaises(ClientError):
            @memoize("s3://i-sure-hope-this-crazy-bucket-doesnt-exist/")
            def foo(x): return x

    @skip_unittest_if_offline
    def test_default_location_requires_s3_user_but_custom_doesnt(self):
        # The default location should require S3_USER to be set, but a custom
        # location shouldn't ever check the value of the variable.
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(KeyError):
                @memoize()
                def foo(x): return x

            @memoize("s3://braingeneersdev/unittest/cache")
            def foo(x): return x

            # Get rid of the directory which will have been created by the
            # successful creation of that memoized function.
            foo.store_backend.clear()
