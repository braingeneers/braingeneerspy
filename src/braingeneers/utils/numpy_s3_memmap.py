import sys
import ast
import boto3
import os
import logging
import urllib.parse
import numpy as np
from tenacity import *
import braingeneers

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
logger = logging.getLogger(__name__)
_s3client = None
braingeneers.set_default_endpoint()


# todo list:
#   1) switch to using smart_open and support local or remote files


class NumpyS3Memmap:
    """
    Provides array slicing over a numpy file on S3
    Uses os.environ['ENDPOINT_URL'] and default boto3 credential lookup

    The class

    Example use:
        # Open a remote numpy file and query it's size
        > from apps.utils import NumpyS3Memmap
        > data = NumpyS3Memmap('s3://braingeneersdev/dfparks/test/test.npy')
        > data.shape
        (2, 3)

        # Read a slice using standard numpy slicing syntax
        > data[0, :]
        array([1., 2., 3.], dtype=float32)

        # If your slice is not contiguous multiple HTTP requests will be made but you'll be warned
        > data[:, [1,2]]
        WARNING:apps.utils:2 separate requests to S3 are being performed for this slice.
        array([[2., 3.],
               [5., 6.]], dtype=float32)

        # Read the full ndarray using [:], this does not flatten the data as it does in normal numpy indexing
        > data[:]
        array([[1., 2., 3.],
               [4., 5., 6.]], dtype=float32)

    Available properties:
        bucket          S3 bucket name from URL
        key             S3 key from URL
        dtype           dtype of numpy file
        shape           shape of numpy file
        fortran_order   boolean, whether the array data is Fortran-contiguous or not
    """

    def __init__(self, s3url, warning_s3_calls=1):
        """
        :param s3url: An S3 url, example: s3://braingeneersdev/dfparks/test/test.npy
        :param warning_s3_calls: max number of calls to S3 before a warning is issued
        """
        self.s3client = boto3.client(
            's3', endpoint_url=os.environ.get('ENDPOINT_URL', 'https://s3.nautilus.optiputer.net')
        )
        self.warning_s3_calls = warning_s3_calls

        # Parse S3 URL
        o = urllib.parse.urlparse(s3url)
        self.bucket = o.netloc
        self.key = o.path[1:]

        # Read numpy header, get shape, dtype, and order
        numpy_header = read_s3_bytes(self.bucket, self.key, 0, 128)  # initial guess at size, likely correct.
        assert numpy_header[:6] == b'\x93NUMPY', 'File {} not in numpy format.'.format(self.key)
        self.header_size = np.frombuffer(numpy_header[8:10], dtype=np.int16)[0] + 10
        if self.header_size != 128:  # re-read numpy header if we guessed wrong on the size
            numpy_header = read_s3_bytes(self.bucket, self.key, 0, self.header_size)
        header_dict = ast.literal_eval(numpy_header[10:].decode('utf-8'))  # parse the header information
        self.dtype = np.dtype(header_dict['descr'])
        self.fortran_order = header_dict['fortran_order']
        self.shape = header_dict['shape']

    def __getitem__(self, item):
        # This is a naive indexing approach but it works.
        # Better options seem to involve reimplementing all numpy indexing options which is quite a rat hole to go down.
        dummy = np.arange(np.prod(self.shape), dtype=np.int64).reshape(
            self.shape, order='F' if self.fortran_order else 'C'
        )
        ixs_tensor = dummy[item]
        shape = ixs_tensor.shape
        ixs = ixs_tensor.T.flatten() if self.fortran_order else ixs_tensor.flatten()
        splits = np.where(ixs[1:] - ixs[:-1] > 1)[0] + 1
        read_sets = np.split(ixs, splits)

        # Compute raw byte offsets, adjusting for data size and header length
        read_from_to = [
            (
                r[0] * self.dtype.itemsize + self.header_size,
                (r[-1] + 1) * self.dtype.itemsize + self.header_size
            )
            for r in read_sets
        ]

        # Warn if too many requests are made to S3
        if len(read_from_to) > self.warning_s3_calls:
            logger.warning('{} separate requests to S3 are being performed for this slice.'.format(len(read_from_to)))

        # Read raw bytes and concatenate
        b = b''.join([
            read_s3_bytes(self.bucket, self.key, offset_from, offset_to)
            for offset_from, offset_to in read_from_to
        ])

        # Convert bytes to numpy dtype and reshape
        arr = np.frombuffer(b, dtype=self.dtype).reshape(shape, order='F' if self.fortran_order else 'C')

        return arr


# @retry(wait=wait_exponential(multiplier=1/(2**5), max=30), after=after_log(logger, logging.WARNING))
def read_s3_bytes(bucket, key, bytes_from=None, bytes_to=None, s3client=None):
    """
    Performs S3 request, bytes_to is exclusive, using python standard indexing.

    :param bucket: S3 bucket name, example: 'briangeneersdev'
    :param key: S3 key, example 'somepath/somefile.txt'
    :param bytes_from: starting read byte, or None for the full file. Must be 0 to read from 0 to part of a file.
    :param bytes_to: ending read byte + 1 (python standard indexing),
        example: (0, 10) reads the first 10 bytes, (10, 20) would read the next 10 bytes of a file.
    :param s3client: If s3client is not passed a single client will be instantiated lazily at the
        module level and re-used for all requests.
    :return: raw bytes as a byte array (b'').
    """
    if s3client is None:
        global _s3client
        if _s3client is None:
            _s3client = boto3.client('s3', endpoint_url=braingeneers.get_default_endpoint())
        s3client = _s3client

    rng = '' if bytes_from is None else 'bytes={}-{}'.format(bytes_from, bytes_to - 1)
    s3obj = s3client.get_object(Bucket=bucket, Key=key, Range=rng)
    b = s3obj['Body'].read()
    return b
