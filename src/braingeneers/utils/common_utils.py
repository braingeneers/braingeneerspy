""" Common utility functions """
import io
import urllib
import boto3
from botocore.exceptions import ClientError
import os
import braingeneers
import braingeneers.utils.smart_open_braingeneers as smart_open
from typing import List, Tuple, Union, Callable, Iterable
import functools
import inspect
import multiprocessing
import posixpath
import itertools
import pathlib
import json
import hashlib

_s3_client = None  # S3 client for boto3, lazy initialization performed in _lazy_init_s3_client()
_message_broker = None  # Lazy initialization of the message broker
_named_locks = {}  # Named locks for checkout and checkin


def _lazy_init_s3_client():
    """
    This function lazy inits the s3 client, it can be called repeatedly and will work with multiprocessing.
    This function is for internal use and is automatically called by functions in this class that use the boto3 client.
    """
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client('s3', endpoint_url=braingeneers.get_default_endpoint())
    return _s3_client


def get_basepath() -> str:
    """
    Returns a local or S3 path so URLs or local paths can be appropriate prefixed.
    """
    if braingeneers.get_default_endpoint().startswith('http'):
        return 's3://braingeneers/'
    else:
        return braingeneers.get_default_endpoint()


def path_join(*args) -> str:
    """
    Joins the basepath from get_basepath() to a set of paths. Example:

    path_join('2020-01-01-e-test', 'original', experiment_name, data_file)

    That would produce one of the following depending on the user:
        /some/local/path/2020-01-01-e-test/original/experiment1/data.bin
        s3://braingeneers/ephys/2020-01-01-e-test/original/experiment1/data.bin
    """
    return posixpath.join(get_basepath(), *args)


def file_exists(filename: str) -> bool:
    """
    Simple test for whether a file exists or not, supports local and S3.
    This is implemented as a utility function because supporting multiple platforms (s3 and local) is not trivial.
    Issue history:
        - Using tensorflow for this functionality failed when libcurl rejected an expired cert.
        - Using PyFilesystem is a bad choice because it doesn't support streaming
        - Using smart_open supports streaming but not basic file operations like size and exists

    :param filename: file path + name, local or S3
    :return: boolean exists|not_exists
    """
    if filename.startswith('s3://'):
        s3_client = _lazy_init_s3_client()
        o = urllib.parse.urlparse(filename)
        try:
            s3_client.head_object(Bucket=o.netloc, Key=o.path[1:])
            exists = True
        except ClientError:
            exists = False
    else:
        exists = os.path.isfile(filename)

    return exists


def file_size(filename: str) -> int:
    """
    Gets file size, supports local and S3 files, same issues as documented in function file_exists
    :param filename: file path + name, local or S3
    :return: int file size in bytes
    """
    if filename.startswith('s3://'):
        s3_client = _lazy_init_s3_client()
        o = urllib.parse.urlparse(filename)
        try:
            sz = s3_client.head_object(Bucket=o.netloc, Key=o.path[1:])['ContentLength']
        except ClientError as e:
            # noinspection PyProtectedMember
            raise Exception(f'S3 ClientError using endpoint {s3_client._endpoint} for file {filename}.') from e
    else:
        sz = os.path.getsize(filename)

    return sz


def file_list(filepath: str) -> List[Tuple[str, str, int]]:
    """
    Returns a list of files, last modified time, and size on local or S3 in descending order of last modified time

    :param filepath: Local or S3 file path to list, example: "local/dir/" or "s3://bucket/prefix/"
    :return: A list of tuples of [('fileA', 'last_modified_A', size), ('fileB', 'last_modified_B', size), ...]
    """
    files_and_details = []

    if filepath.startswith('s3://'):
        s3_client = _lazy_init_s3_client()
        o = urllib.parse.urlparse(filepath)
        response = s3_client.list_objects(Bucket=o.netloc, Prefix=o.path[1:])

        if 'Contents' in response:
            files_and_details = [
                (f['Key'].split('/')[-1], str(f['LastModified']), int(f['Size']))
                for f in sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True)
            ]
    elif os.path.exists(filepath):
        files = sorted(pathlib.Path(filepath).iterdir(), key=os.path.getmtime, reverse=True)
        files_and_details = [(f.name, str(f.stat().st_mtime), f.stat().st_size) for f in files]

    return files_and_details


def map2(func: Callable,
         args: Iterable[Union[Tuple, object]] = None,
         fixed_values: dict = None,
         parallelism: (bool, int) = True,
         use_multithreading: bool = False) -> List[object]:
    """
    A universal multiprocessing version of the map function to simplify parallelizing code.
    The function provides a simple method for parallelizing operations while making debugging easy.

    Combines functionality of: map, itertools.starmap, and pool.map

    Eliminates the need to understand functools, multiprocessing.Pool, and
    argument unpacking operators which should be unnecessary to accomplish a simple
    multi-threaded function mapping operation.

    Usage example:
        def f(x, y):
            print(x, y)

        common_py_utils.map2(
            func=f,
            args=[(1, 'yellow'), (2, 'yarn'), (3, 'yack')],  # (x, y) arguments
            parallelism=3,                                   # use a 3 process multiprocessing pool
        )

        common_py_utils.map2(
            func=f,
            args=[1, 2, 3],  # x arguments
            fixed_values=dict(y='yellow'),  # y always is 'yellow'
            parallelism=False,              # Runs without parallelism which makes debugging exceptions easier
        )

    :param func: a callable function
    :param args: a list of arguments (if only 1 argument is left after fixed_values) or a list of tuples
        (if multiple arguments are left after fixed_values)
    :param fixed_values: a dictionary with parameters that will stay the same for each call to func
    :param parallelism: number of processes to use or boolean, default is # of CPU cores.
        When parallelism==False or 1, this maps to itertools.starmap and does not use multiprocessing.
        If parallelism >1 then multiprocessing.pool.starmap will be used with this number of worker processes.
        If parallelism == True then multiprocessing.pool will be used with multiprocessing.cpu_count() processes.
    :param use_multithreading: advanced option, use the default (False) in most cases. Parallelizes using
        threads instead of multiprocessing. Multiprocessing should be used if more than one CPU core is needed
        due to the GIL, threads are lighter weight than processes for some non cpu-intensive tasks.
    :return: a list of the return values of func
    """
    assert isinstance(fixed_values, (dict, type(None)))
    assert isinstance(parallelism, int)
    parallelism = multiprocessing.cpu_count() if parallelism is True else 1 if parallelism is False else parallelism
    assert isinstance(parallelism, int)

    func_partial = functools.partial(func, **(fixed_values or {}))
    n_required_params = sum([p.default == inspect.Parameter.empty for p in inspect.signature(func).parameters.values()])
    n_fixed_values = len(fixed_values or {})
    args_list = list(args or [])
    args_tuples = args \
        if len(args_list) > 0 \
           and isinstance(args_list[0], tuple) \
           and len(args_list[0]) >= n_required_params - n_fixed_values \
        else [(a,) for a in args_list]

    if parallelism == 1:
        result_iterator = itertools.starmap(func_partial, args_tuples)
    else:
        # noinspection PyPep8Naming
        ProcessOrThreadPool = multiprocessing.pool.ThreadPool if use_multithreading is True else multiprocessing.Pool
        with ProcessOrThreadPool(parallelism) as pool:
            result_iterator = pool.starmap(func_partial, args_tuples)

    return list(result_iterator)


# class AtomicGetSetEphysMetadata:
#     """
#     This class allows multiple devices/processes/threads to safely read and write to the ephys metadata file.
#
#     This is a context manager, used with the `with` statement.
#
#     It will acquire a lock on the metadata file, read the metadata, and return it. When the context manager
#     exits it will release the lock and write the metadata back to the file if it has changed.
#
#     The with block below in example usage is a guaranteed critical section by UUID,
#     any other code using AtomicGetSetEphysMetadata will wait to
#     enter the with block, if any other code, on any hosts, is already in the critical section (`with` block).
#     on any host. Upon exiting the block the `metadata` object is updated on S3 safely.
#
#     Example usage:
#         from braingeneers.utils.common_utils import AtomicGetSetEphysMetadata
#
#         with AtomicGetSetEphysMetadata(uuid) as metadata:
#             metadata['new_key'] = 'new_value'
#
#     Notes:
#     An exception within the `with` block will release the lock and not write the metadata back to the file.
#     The only time a dangling lock is possible is if code execution stops during the `with` block.
#     To manually clear a dangling lock call:
#
#         from braingeneers.utils.common_utils import AtomicGetSetEphysMetadata
#         AtomicGetSetEphysMetadata(uuid).force_release()
#     """
#     def __init__(self, batch_uuid: str):
#         from braingeneers.iot.messaging import MessageBroker
#         self.batch_uuid = batch_uuid
#         self.lock_str = f'atomic-metadata-lock-{batch_uuid}'
#         self.mb = MessageBroker()
#
#         self.named_lock = None
#         self.metadata = None
#         self.metadata_md5_hash = None
#
#     @staticmethod
#     def _md5_hash(data: dict) -> str:
#         return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()
#
#     def __enter__(self):
#         self.named_lock = self.mb.get_lock(self.lock_str)
#         self.named_lock.acquire()
#         self.metadata_filepath = posixpath.join(get_basepath(), 'ephys', self.batch_uuid, 'metadata.json')
#         self.metadata = json.loads(smart_open.open(self.metadata_filepath, 'r').read())
#         self.metadata_md5_hash = self._md5_hash(self.metadata)
#         return self.metadata
#
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         try:
#             if self.metadata_md5_hash == self._md5_hash(self.metadata):
#                 print('Warning: metadata was not changed, not saving.')
#             else:
#                 smart_open.open(self.metadata_filepath, 'w').write(
#                     json.dumps(self.metadata, indent=2)
#                 )
#         finally:
#             self.named_lock.release()
#
#     def force_release(self):
#         """
#         Force release the lock, use with caution.
#         If a lock is created but not released this function can be used to
#         force its release. This is not recommended for normal use.
#         """
#         self.mb.delete_lock(self.lock_str)


def checkout(s3_file: str, mode: str = 'r') -> io.IOBase:
    """
    Check out a file from S3 for reading or writing, use checkin to release the file.
    Any subsequent calls to checkout will block until the file is returned with checkin(s3_file).

    Example usage:
        f = checkout('s3://braingeneersdev/test/test_file.bin', mode='rb')
        new_bytes = do_something(f.read())
        checkin('s3://braingeneersdev/test/test_file.bin', new_bytes)

    Example usage to update metadata:
        f = checkout('s3://braingeneersdev/test/metadata.json')
        metadata_dict = json.loads(f.read())
        metadata_dict['new_key'] = 'new_value'
        metadata_updated_str = json.dumps(metadata_dict, indent=2)
        checkin('s3://braingeneersdev/test/metadata.json', updated_metadata_str)

    :param s3_file: The S3 file path to check out.
    :param mode: The mode to open the file in, 'r' or 'rb' for reading, 'w' or 'wb' for writing.
    """
    # Avoid circular import
    from braingeneers.iot.messaging import MessageBroker

    global _message_broker, _named_locks
    if _message_broker is None:
        print('creating message broker')
        _message_broker = MessageBroker()
    mb = _message_broker

    lock_str = f'common-utils-checkout-{s3_file}'
    named_lock = mb.get_lock(lock_str)
    named_lock.acquire()
    _named_locks[s3_file] = named_lock
    f = smart_open.open(s3_file, mode)
    return f


def checkin(s3_file: str, file: Union[str, bytes, io.IOBase]):
    """
    Releases a file that was checked out with checkout.

    :param s3_file: The S3 file path, must match checkout.
    :param file: The string, bytes, or file object to write back to S3.
    """
    assert isinstance(file, (str, bytes, io.IOBase)), 'file must be a string, bytes, or file object.'

    with smart_open.open(s3_file, 'wb') as f:
        if isinstance(file, str):
            f.write(file.encode())
        elif isinstance(file, bytes):
            f.write(file)
        else:
            file.seek(0)
            data = file.read()
            f.write(data if isinstance(data, bytes) else data.encode())

    global _named_locks
    named_lock = _named_locks[s3_file]
    named_lock.release()


def force_release_checkout(s3_file: str):
    """
    Force release the lock on a file that was checked out with checkout.
    """
    # Avoid circular import
    from braingeneers.iot.messaging import MessageBroker

    global _message_broker
    if _message_broker is None:
        _message_broker = MessageBroker()
    mb = _message_broker

    lock_str = f'common-utils-checkout-{s3_file}'
    mb.delete_lock(lock_str)
