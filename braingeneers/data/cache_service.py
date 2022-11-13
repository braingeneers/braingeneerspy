from __future__ import annotations  # https://stackoverflow.com/questions/33533148
import datasets_electrophysiology as de
import braingeneers.utils.smart_open_braingeneers as smart_open
from typing import List, Tuple, Union, Iterable
import json
from datetime import datetime, timedelta
import re
import os
from concurrent.futures import Future
from functools import lru_cache
import kubernetes


# todo how to handle variable memory requirement (answer: calculate the memory requirement!)

class CacheConfig:
    """
    See documentation on cache_service.ensure_cache(...) for usage examples.

    CacheConfig provides configuration for a read-optimal caching of a dataset or transformation of the dataset.

    This configuration can be imported/exported to/from json for convenience (both S3 and local paths supported).
    """
    def __init__(self):
        self.uuid: str = None
        self.data_ranges: List[Tuple[int, int]] = []
        self.channels: List[Tuple[int]] = []  # tuples contain one or more channels to be read as a group
        self.cache_until_iso: str = None  # ISO format datetime string: 2022-11-09T16:36:54.066051
        self.local_cache = None
        self.transformations = []

    def save_json(self, filename_or_s3url: str, indent: int = 2) -> None:
        """
        Saves the cache config JSON to a file, S3 or local path.

        Example usage:
            cache_config.save_json('s3://braingeneersdev/personal/me/my_cache_config.json')
        """
        with smart_open.open(filename_or_s3url, 'w') as f:
            f.write(self.to_json(indent=indent))

    def to_json(self, indent: int = 2) -> str:
        """ Creates a JSON representation of the CacheConfig object which can be reloaded using from_json. """
        return json.dumps(vars(self), indent=indent)

    @staticmethod
    def load_json(filename_or_s3url: str) -> CacheConfig:
        """
        Loads a CacheConfig JSON from file, S3 or local path.

        Example usage:
            cache_config = CacheConfig.load_json('s3://braingeneersdev/personal/me/my_cache_config.json')
        """
        with smart_open.open(filename_or_s3url, 'r') as f:
            return CacheConfig.from_json(f.read())

    @staticmethod
    def from_json(json_str: str) -> CacheConfig:
        """ Returns a CacheConfig object from the JSON representation as created by to_json """
        cache_config = CacheConfig()
        config_variables_dict = json.loads(json_str)

        for key, value in config_variables_dict.items():
            assert hasattr(cache_config, key), f'Invalid JSON, found key: {key}, which does not belong to CacheConfig'
            setattr(cache_config, key, value)

        return cache_config

    def set_uuid(self, uuid: str) -> CacheConfig:
        self.uuid = uuid
        return self

    def set_data_range(self, offset: int = 0, size: int = -1) -> CacheConfig:
        """
        Set the range of data to be read. By default offset == 0 (beginning of file) and size == -1 (full file) is used.

        For example, the offset units are in data points. E.g. to cache the 10th to 30th second of a
        recording @ 20,000 hz recording rate the offset would be 200000 and size would be 600000.
        The values are the same as is used in the load_data(...) function.

        Multiple discontinuous ranges can be set by calling `set_data_range` multiple times.

        :param offset: an int, the data point offset signifying the start of the range.
            This is the same value as used in `load_data(...)`.
        :param size: the number of data points to read from offset. This is the same value as used in `load_data(...)`.
        """
        # validate input
        if not isinstance(offset, int):
            raise ValueError(f'offset must be an int, got {type(offset)}')
        if not offset >= 0:
            raise ValueError(f'offset must be >= 0, got {offset}')
        if not isinstance(size, int):
            raise ValueError(f'size must be an int, got {type(size)}')
        if not (size == -1 or size > 0):
            raise ValueError(f'size must be -1 or >0, got {size}')

        self.data_ranges.append((offset, size))
        return self

    def set_channels_individual(self, channels: Union[int, Iterable[int]]) -> CacheConfig:
        """
        Channels defined in a single call to `set_channels_individual` will organize the data so that any
        single channel can be streamed efficiently by itself. This means that a separate read will be used
        for each channel that is being read.

        Example usage:
            cache_config = CacheConfig.set_channels_individual(0)  # Caches a single channel
            cache_config = CacheConfig.set_channels_individual([10,20,21,22])  # Caches an multiple channels to be read individually

        Performance note:
            Since multiple reads are parallelized, reading a few individually packed channels will still be efficient,
            however with 100's or 1000's of channels individual channels would be slower than grouped.

        :param channels: Specific channel(s) to be streamed individually, see example above
        :return: CacheConfig with options set
        """
        if isinstance(channels, int):
            self.channels.append((channels,))
        else:
            if not hasattr(channels, '__iter__'):
                raise ValueError(f'channels is not iterable, please use a list, tuple, or equivalent iterable object. '
                                 f'Got object of type: {type(channels)}')
            for c in channels:
                if not isinstance(c, int):
                    raise ValueError(f'channels must be an iterable of int(s), got an iterable of: {type(c)}')
                self.channels.append((c,))

        self.channels.sort()
        return self

    def set_channels_grouped(self, channels: Iterable[int]):
        """
        Channels defined in a single call to `set_channels_grouped` will organize the data to be efficiently streamed
        together as a group of channels. This means that a single read can be used to stream a block of channels.

        Example usage:  note: ranges are [inclusive, exclusive):
            cache_config = CacheConfig.set_channels_grouped(range(64, 128))  # Caches a block of 64 channels, 64 to 128
            cache_config = CacheConfig.set_channels_grouped([0,20,21,40])  # Caches an arbitrary set of channels

        Performance note:
            Since multiple reads are parallelized, reading a few individually packed channels will still be efficient,
            however with 100's or 1000's of channels individual channels would be slower than grouped.

        Multiple groups can be defined by calling set_channels_grouped multiple times. When doing this each
        group will be stored as a separate unit for efficient streaming.

        :param channels: Specific channel(s) to be streamed as a group, see example above
        :return: CacheConfig with options set
        """
        channels_list = list(channels)

        if not all([type(c) == int for c in channels_list]):
            raise ValueError(f'channels must a an Iterable of int objects. Found non int objects.')

        self.channels.append(tuple(channels_list))
        return self

    def set_cache_until(self, until: str) -> CacheConfig:
        """
        Set the length of time since the cache was last accessed until it is automatically cleaned up. Default: "3d"

        Note that the last access time is calculated from the last time that `ensure_cache` was called.

        Example usage:
            cache_config = CacheConfig().set_cache_until('2w')  # ensures the cache is stored for at least 2 weeks
            cache_config = CacheConfig().set_cache_until('1m 2w')  # ensures the cache is stored for at least 1 month 2 weeks

        :param until: Relative times as string: "xd" (x days), "xw" (x weeks), "xm" (x months).
            Can specify multiple times, as in "1m 2w" which is equivalent to "6w"
        """
        convert_days = {'d': 1, 'w': 7, 'm': 31}
        delimiter_regex = r'\d+[dwm]'
        td = timedelta()
        for u in re.findall(delimiter_regex, until):  # example: ["1m", "7d"]
            assert u[-1] in convert_days, f'Relative time string is an unsupported format, got: {until}'
            td += timedelta(days=int(u[:-1]) * convert_days[u[-1]])

        self.cache_until_iso = (datetime.now() + td).isoformat()
        return self

    def set_local_cache(self, cache_dir: str) -> CacheConfig:
        """
        Syncs the files to a local cache at the directory specified.
        :param cache_dir:
        :return:
        """
        if not os.path.exists(cache_dir):
            raise NotADirectoryError(cache_dir)

        self.local_cache = cache_dir
        return self

    # This function may be implemented later.
    # def set_transformation(self, dataset_transformation: DatasetTransformation) -> CacheConfig:
    #     """
    #     Performs a data transformation as part of the process.
    #
    #     Can be called multiple times to perform multiple transformations in
    #     which case they are processed in order submitted.
    #     """
    #     raise NotImplementedError('This functionality has not yet been implemented, see author David Parks')

# class DataTransformation:
#     """
#     Extend this class and implement its methods to apply a custom dataset transformation.
#
#     Example/standard DataTransformation classes can be found in this module.
#
#     ISSUES:
#      - The dataset transformation may not be able to function on a subset of the data, in which case the
#        user needs the ability to ensure the full dataset is processed in a single container.
#      - This may have memory implications.
#     """
#
#     def __init__(self, uuid: str, metadata: dict):
#         self.uuid = uuid
#         self.metadata = metadata
#
#     def transform_chunk(self, data_chunk: np.ndarray, eof: bool = False) -> np.ndarray:
#         """
#         Implement this function to perform a dataset transformation.
#         Accepts an arbitrarily sized chunk of data and returns a transformation of that data.
#
#         eof will be set to True on the last call to transform_chunk
#         """
#         raise NotImplementedError('This class must be implemented by the subclass.')
#
#
# class BandpassTransformation(DataTransformation):
#     """
#     Performs a bandpass filter of the data. By default, the bandpass is applied per segment of data so the operation
#     can still be parallelized, that can be overridden with
#     """
#     def __init__(self, uuid: str, metadata: dict, bandpass_hz_low: int, bandpass_hz_high: int):
#         self.bandpass_hz_low = bandpass_hz_low
#         self.bandpass_hz_high = bandpass_hz_high
#         super().__init__(uuid, metadata)
#
#     def transform_chunk(self, data_chunk: np.ndarray, eof: bool = False):
#         raise NotImplementedError('To be implemented by David Parks')


def ensure_cache(cache_config: CacheConfig, wait: bool = True) -> Future:
    """
    Ensures that data is available in optimal format for fast IO reads from S3.
    Also provides the ability to make a copy of the cache locally for faster processing.

    This function can be called multiple times. There is a small network delay when calling it to validate an
    existing cache.

    Basic usage example:

        from braingeneers.data.cache_service import ensure_cache, CacheConfig
        from braingeneers.data.datasets_electrophysiology import load_data

        # Build the CacheConfig object (this can also be loaded/saved from file using `CacheConfig.load_json(filepath)`)
        cache_config = CacheConfig()\
            .set_uuid('9999-00-00-e-test')\
            .set_channels([0,1,2,3])\
            .set_local_cache('/tmp/cache/')

        # Validates that the cached files exist, if not it creates them, this may take a long time to run.
        ensure_cache(cache_config)

        # When calling load_data to read data pass the CacheConfig object to load_data.
        data = load_data(offset=200000, size=400000, use_cache=cache_config)

    :param cache_config: The CacheConfig object. This can be created programmatically or loaded from a file
        using CacheConfig.load_json(filepath).
    :param wait: wait for the job to complete before returning, this can take minutes or hours depending on
        the size of the cache. Defaults to True. If False a Future object will be returned.
        See: https://docs.python.org/3/library/concurrent.futures.html#future-objects
    :return: None if the cache already existed, else a Future object which provides the status of the
        job that creates the cache. See: https://docs.python.org/3/library/concurrent.futures.html#future-objects
    """
    # Check if the data exists on S3 already
    if _cache_exists(cache_config):
        result = None
    else:
        result = _prepare_cache(cache_config)
        if wait:
            result.wait()

    return result


@lru_cache()
def _cache_exists(cache_config: CacheConfig) -> bool:
    """ Internal function which verifies if the data exists in the cache already. """
    return true  # todo


def _prepare_cache(cache_config: CacheConfig) -> Future:
    """ Internal function which submits the jobs to prepare the cache. """

    # Load kube config file
    kubernetes.config.load_kube_config()
    # todo temp test
    v1 = kubernetes.client.CoreV1Api()
    print("Listing pods with their IPs:")
    ret = v1.list_namespaced_pod(namespace='braingeneers')
    for i in ret.items:
        print("%s\t%s\t%s" % (i.status.pod_ip, i.metadata.namespace, i.metadata.name))


@lru_cache
def _ensure_cleanup_cronjob_running() -> None:
    """
    Internal function which validates that the cleanup cronjob is running.
    The @lru_cache will ensure this function is only called once per process to avoid the network
    round trip if it happens to be called in a loop.
    """
    pass  # todo


# Kubernetes yaml config for JOB which creates a cache segment
JOB_YAML_CREATE_CACHE = """
TODO
"""

# Kubernetes yaml config for CRONJOB which cleans up old cache files
CRONJOB_YAML_CACHE_CLEANUP = """
TODO
"""
