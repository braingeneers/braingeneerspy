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


class checkout:
    """
    A context manager for atomically checking out a file from S3 for reading or writing.

    Example usage:

    # Read-then-update metadata.json (or any text based file on S3)
    with checkout('s3://braingeneers/ephys/9999-0-0-e-test/metadata.json', isbinary=False) as locked_obj:
        metadata_dict = json.loads(locked_obj.get_value())
        metadata_dict['new_key'] = 'new_value'
        metadata_updated_str = json.dumps(metadata_dict, indent=2)
        locked_obj.checkin(metadata_updated_str)

    # Read-then-update data.npy (or any binary file on S3)
    with checkout('s3://braingeneersdev/test/data.npy', isbinary=True) as locked_obj:
        file_obj = locked_obj.get_file()
        ndarray = np.load(file_obj)
        ndarray[3, 3] = 42
        locked_obj.checkin(ndarray.tobytes())

    # Edit a file in place, note checkin is not needed, the file is updated when the context manager exits
    with checkout('s3://braingeneersdev/test/test_file.bin', isbinary=True) as locked_obj:
        with zipfile.ZipFile(locked_obj.get_file(), 'a') as z:
            z.writestr('new_file.txt', 'new file contents')

    locked_obj functions:
       get_value()  # returns a string or bytes object (depending on isbinary)
       get_file()   # returns a file-like object akin to open()
       checkin()    # updates the file, accepts string, bytes, or file like objects
    """
    class LockedObject:
        def __init__(self, s3_file_object, s3_path_str, isbinary):
            self.s3_path_str = s3_path_str
            self.s3_file_object = s3_file_object  # underlying file object
            self.isbinary = isbinary  # binary or text mode

        def get_value(self):
            # Read file object from outer class s3_file_object
            self.s3_file_object.seek(0)
            return self.s3_file_object.read()

        def get_file(self):
            # Return file object from outer class s3_file_object
            self.s3_file_object.seek(0)
            return self.s3_file_object

        def checkin(self, update_file: Union[str, bytes, io.IOBase]):
            # Validate input
            if not isinstance(update_file, (str, bytes, io.IOBase)):
                raise TypeError('file must be a string, bytes, or file object.')
            if isinstance(update_file, str) or (
                    isinstance(update_file, io.IOBase) and isinstance(update_file, io.StringIO)):
                if self.isbinary:
                    raise ValueError(
                        'Cannot check in a string or text file when checkout is specified for binary mode.')
            if isinstance(update_file, bytes) or (
                    isinstance(update_file, io.IOBase) and isinstance(update_file, io.BytesIO)):
                if not self.isbinary:
                    raise ValueError('Cannot check in bytes or a binary file when checkout is specified for text mode.')

            mode = 'w' if not self.isbinary else 'wb'
            with smart_open.open(self.s3_path_str, mode=mode) as f:
                f.write(update_file if not isinstance(update_file, io.IOBase) else update_file.read())

    def __init__(self, s3_path_str: str, isbinary: bool = False):
        # Avoid circular import
        from braingeneers.iot.messaging import MessageBroker

        self.s3_path_str = s3_path_str
        self.isbinary = isbinary
        self.mb = MessageBroker()
        self.named_lock = None  # message broker lock
        self.locked_obj = None  # user facing locked object

    def __enter__(self):
        lock_str = f'common-utils-checkout-{self.s3_path_str}'
        named_lock = self.mb.get_lock(lock_str)
        named_lock.acquire()
        self.named_lock = named_lock
        f = smart_open.open(self.s3_path_str, 'rb' if self.isbinary else 'r')
        self.locked_obj = checkout.LockedObject(f, self.s3_path_str, self.isbinary)
        return self.locked_obj

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.named_lock.release()
        return None  # Do not suppress exceptions


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


def pretty_print(data, n=10, indent=0):
    """
    Custom pretty print function that uniformly truncates any collection (list or dictionary)
    longer than `n` values, showing the first `n` values and a summary of omitted items.
    Ensures mapping sections and similar are displayed compactly.

    Example usage (to display metadata.json):

      from braingeneers.utils.common_utils import pretty_print
      from braingeneers.data import datasets_electrophysiology as de

      metadata = de.load_metadata('2023-04-17-e-connectoid16235_CCH')
      pretty_print(metadata)

    Parameters:
    - data: The data to pretty print, either a list or a dictionary.
    - n: Maximum number of elements or items to display before truncation.
    - indent: Don't use this. Current indentation level for formatting, used during recursion.
    """
    indent_space = ' ' * indent
    if isinstance(data, dict):
        keys = list(data.keys())
        if len(keys) > n:
            truncated_keys = keys[:n]
            omitted_keys = len(keys) - n
        else:
            truncated_keys = keys
            omitted_keys = None

        print('{')
        for key in truncated_keys:
            value = data[key]
            print(f"{indent_space}    '{key}': ", end='')
            if isinstance(value, dict):
                pretty_print(value, n, indent + 4)
                print()
            elif isinstance(value, list) and all(isinstance(x, (list, tuple)) and len(x) == 4 for x in value):
                # Compact display for lists of tuples/lists of length 4.
                print('[', end='')
                if len(value) > n:
                    for item in value[:n]:
                        print(f"{item}, ", end='')
                    print(f"... (+{len(value) - n} more items)", end='')
                else:
                    print(', '.join(map(str, value)), end='')
                print('],')
            else:
                print(f"{value},")
        if omitted_keys:
            print(f"{indent_space}    ... (+{omitted_keys} more items)")
        print(f"{indent_space}}}", end='')
    elif isinstance(data, list):
        print('[')
        for item in data[:n]:
            pretty_print(item, n, indent + 4)
            print(',')
        if len(data) > n:
            print(f"{indent_space}    ... (+{len(data) - n} more items)")
        print(f"{indent_space}]", end='')

