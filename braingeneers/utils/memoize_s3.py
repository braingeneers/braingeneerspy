import os
import glob
import boto3
import awswrangler as wr
from smart_open.s3 import parse_uri
from joblib import register_store_backend, Memory
from joblib._store_backends import StoreBackendBase, StoreBackendMixin, CacheItemInfo
from .smart_open_braingeneers import open


def s3_isdir(path):
    """
    S3 doesn't support directories, so to check whether some path "exists",
    instead check whether it is a prefix of at least one object.
    """
    try:
        next(wr.s3.list_objects(glob.escape(path), chunked=True))
        return True
    except StopIteration:
        return False



class S3StoreBackend(StoreBackendBase, StoreBackendMixin):
    _open_item = staticmethod(open)

    def _item_exists(self, location):
        return wr.s3.does_object_exist(location) or s3_isdir(location)

    def _move_item(self, src_uri, dst_uri):
        # awswrangler only includes a fancy move/rename method that actually
        # makes it pretty hard to just do a simple move.
        src, dst = [parse_uri(x) for x in (src_uri, dst_uri)]
        self.client.copy_object(
            Bucket=dst["bucket_id"],
            Key=dst["key_id"],
            CopySource=f"{src['bucket_id']}/{src['key_id']}",
        )
        self.client.delete_object(Bucket=src["bucket_id"], Key=src["key_id"])

    def create_location(self, location):
        # Actually don't do anything. There are no locations on S3.
        pass

    def clear_location(self, location):
        # Recursive delete.
        wr.s3.delete_objects(glob.escape(location))

    def get_items(self):
        return [
            CacheItemInfo(
                key,
                item["ContentLength"],
                # This is supposed to be an access date, but it only gets used
                # this way for LRU caching, so it's not a big deal to use the
                # modified time instead.
                item["LastModified"],
            )
            for key, item in wr.s3.describe_objects(self.location).items()
        ]

    def configure(self, location, verbose, backend_options={}):
        # We don't do any logging yet, but `configure()` must accept this
        # argument, so store it for forwards compatibility.
        self.verbose = verbose

        # We have to save this on the backend because joblib queries it, but
        # default to True instead of joblib's usual False because S3 is not
        # local disk and compression can really help.
        self.compress = backend_options.get("compress", True)

        # This option is available by default but we can't accept it because
        # there's no reasonable way to make joblib use NumpyS3Memmap.
        self.mmap_mode = backend_options.get("mmap_mode")
        if self.mmap_mode is not None:
            raise ValueError("impossible to mmap on S3.")

        # Don't attempt to handle local files, just use the default backend
        # for that!
        if not location.startswith("s3://"):
            raise ValueError("location must be an s3:// URI")

        # We don't have to check that the bucket exists because joblib
        # performs a `list_objects()` in it, but note that this doesn't
        # actually check whether we can write to it!
        self.location = location

        # We need a boto3 client, so create it using the endpoint which was
        # configured in awswrangler by importing smart_open_braingeneers.
        self.client = boto3.Session().client(
            "s3", endpoint_url=wr.config.s3_endpoint_url
        )


def memoize(location=None, backend="s3", **kwargs):
    """
    Memoize a function to S3 using joblib.Memory. By default, saves to
    `s3://braingeneersdev/$S3_USER/cache`, but this can be configured by
    explicitly providing a cache directory.

    Also accepts all the same keyword arguments as `joblib.Memory`, including
    `backend` which can be set to "local" to recover default behavior. The
    keyword arguments of `joblib.Memory.cache()` can also be used by doubly
    invoking the decorator. Usage examples:

    ```
    from braingeneers.utils.memoize_s3 import memoize

    # Cache to the default location on NRP S3.
    @memoize
    def foo(x):
        return x

    # Cache to a different NRP S3 location.
    @memoize("s3://braingeneers/someplace/else/idk")
    def bar(x):
        return x

    # Ignore some parameters when deciding which cache entry to check.
    @memoize()(ignore=["verbose"])
    def plover(x, verbose):
        if verbose: ...
        return x
    ```

    If the bucket doesn't exist, an error will be raised, but if the only
    problem is permissions, silent failure to cache may be all that occurs
    depending on the verbosity setting.
    """
    if callable(location):
        # This case probably means the @memoize decorator was used without
        # arguments, but pass the kwargs on anyway just in case.
        return memoize(**kwargs)(location)

    if location is None and backend == "s3":
        location = f"s3://braingeneersdev/{os.environ['S3_USER']}/cache"

    return Memory(location, backend=backend, **kwargs).cache


register_store_backend("s3", S3StoreBackend)
