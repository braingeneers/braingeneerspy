import datasets_electrophysiology as de


class CacheConfig:
    """
    See documentation on cache_service.ensure_cache(...) for usage examples.

    CacheConfig provides configuration for a read-optimal caching of a dataset or transformation of the dataset.

    This configuration can be imported/exported to/from json for convenience (both S3 and local paths supported).
    """
    def __init__(self):
        self.uuid = None
        self.channels = None
        self.cache_until = None
        self.local_cache = None
        self.transformations = []

    def save_json(self, filename_or_s3url: str):
        pass

    def load_json(self, filename_or_s3url: str) -> CacheConfig:
        pass

    def to_json(self) -> str:
        pass

    def from_json(self, json_str: str) -> CacheConfig:
        pass

    def set_uuid(self, uuid: str) -> CacheConfig:
        self.uuid = uuid
        return self

    def set_range(self, offset: int = 0, size: int = -1) -> CacheConfig:
        """
        Set the range of data to be read. By default offset == 0 (beginning of file) and size == -1 (full file) is used.

        For example, the offset units are in data points. E.g. to cache the 10th to 30th second of a
        recording @ 20,000 hz recording rate the offset would be 200000 and size would be 600000.
        The values are the same as is used in the load_data(...) function.

        :param offset: an int, the data point offset signifying the start of the range.
            This is the same value as used in `load_data(...)`.
        :param size: the number of data points to read from offset. This is the same value as used in `load_data(...)`.
        """
        pass

    def set_channels(self, channels: Union[int, List[int], Tuple[int], List[Tuple[int]]],
                     per_channel: bool = False) -> CacheConfig:
        """
        Your data will be grouped by this set of channels or range of channels so that the whole group of channels
        can be read together efficiently in blocks of time, or if per_channel == True, each channel will be
        packed individually so single channel reads are efficient.

        If multiple groups of channels need to be read create and submit multiple CacheConfig objects to
        ensure_cache(...)

        If channels is None or this function is not called then all channels will be assumed.

        Example usage:
            cache_config = CacheConfig.set_channels(0)  # Caches a single channel
            cache_config = CacheConfig.set_channels([10,20,21,22])  # Caches a group of 4 channels together
            cache_config = CacheConfig.set_channels((0, 64))  # Caches the first 64 channels together
            cache_config = CacheConfig.set_channels([(0, 64), (128, 192)])  # Caches 128 channels together using two ranges
            cache_config = CacheConfig.set_channels([0,1,2,3], per_channel=True)  # Caches 4 channels separately so individual channel reads are efficient.

        :param channels: Specific channel(s), or range(s) of channels.
            Form 1: int - a single channel, zero-indexed
            Form 2: List[int] - a list of one or more channels
            Form 3: Tuple[int] - a range of channels in Python-standard [inclusive, exclusive) form.
            Form 4: List[Tuple[int]] - a list of multiple channel ranges from Form 3 above.
        :param per_channel: True if data should be packed per each channel so single-channel reads are efficient
            but reading multiple channels will be in-efficient.
        :return:
        """
        pass

    def set_cache_until(self, until: Union[str, datetime]) -> CacheConfig:
        """
        Set the length of time since the cache was last accessed until it is automatically cleaned up. Default: "3d"

        Note that the last access time is calculated from the last time that `ensure_cache` was called.

        :param until: Relative times supported as string "xd" (x days), "xw" (x weeks), "xm" (x months), or
            you may specify a datetime objects.
        """

    def set_local_cache(self, cache_dir: str) -> CacheConfig:
        """
        Syncs the files to a local cache at the directory specified.
        :param cache_dir:
        :return:
        """

    def set_transformation(self) -> CacheConfig:
        """
        Performs a data transformation as part of the process.
        Can be called multiple times to perform multiple transformations.
        """
        pass


def ensure_cache(cache_config: CacheConfig, wait: bool = True):
    """
    Ensures that data is available in optimal format for fast IO reads from S3. Also provides the ability to:
     - cache locally
     - preprocess data

    This function can be called multiple times. There is a small network delay when calling it to validate an
    existing cache, so it should not be called repeatedly in a high performance read loop, but it can be called
    at initialization time to check or re-check that the cache is available.

    See detailed documentation and usage examples in CacheConfig

    Basic usage example:
        from braingeneers.data.dataset_electrophysiology import ensure_cache, CacheConfig

        cache_config = CacheConfig().\
            set_uuid('9999-00-00-e-test').\
            set_channels([0,1,2,3]).\
            set_local_cache('/tmp/cache/')

        ensure_cache(cache_config)

        data = load_data(offset=200000, size=400000, use_cache=cache_config)
    """
    raise NotImplementedError('David Parks to implement this')
