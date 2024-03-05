""" Common utility functions """
import io
import urllib
import boto3
from botocore.exceptions import ClientError
import os
import braingeneers
import braingeneers.utils.smart_open_braingeneers as smart_open
from typing import Callable, Iterable, Union, List, Tuple, Dict, Any
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


# Define the wrapper function as a top-level function
def _map2_wrapper(fixed_values: Dict[str, Any], required_params: List[str], func: Callable, args: Tuple) -> Any:
    """Internal wrapper function for map2 to handle fixed values and dynamic arguments."""
    # Merge fixed_values with provided arguments, aligning provided args with required_params
    call_args = {**fixed_values, **dict(zip(required_params, args))}
    return func(**call_args)


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
    assert parallelism is False or isinstance(parallelism, (bool, int)), "parallelism must be a boolean or an integer"
    parallelism = multiprocessing.cpu_count() if parallelism is True else 1 if parallelism is False else parallelism
    assert isinstance(parallelism, int), "parallelism must be resolved to an integer"

    fixed_values = fixed_values or {}
    func_signature = inspect.signature(func)
    required_params = [p.name for p in func_signature.parameters.values() if
                       p.default == inspect.Parameter.empty and p.name not in fixed_values]

    args_list = list(args or [])
    args_tuples = args_list if all(isinstance(a, tuple) for a in args_list) else [(a,) for a in args_list]

    if parallelism == 1:
        result_iterator = map(lambda args: _map2_wrapper(fixed_values, required_params, func, args), args_tuples)
    else:
        ProcessOrThreadPool = multiprocessing.pool.ThreadPool if use_multithreading else multiprocessing.Pool
        with ProcessOrThreadPool(parallelism) as pool:
            result_iterator = pool.starmap(_map2_wrapper,
                                           [(fixed_values, required_params, func, args) for args in args_tuples])

    return list(result_iterator)


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
    :param mode: The mode to open the file in, 'r' (text mode) or 'rb' (binary mode), analogous to system open(filename, mode)
    """
    # Avoid circular import
    from braingeneers.iot.messaging import MessageBroker

    assert mode in ('r', 'rb'), 'Use "r" (text) or "rb" (binary) mode only. File changes are applied at checkout(...)'

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

