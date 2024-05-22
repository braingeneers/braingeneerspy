from __future__ import annotations

import os
import sys
import json
import warnings
import copy
import diskcache
import matplotlib.pyplot as plt
import numpy as np
import shutil
import h5py
import braingeneers.utils.smart_open_braingeneers as smart_open
from collections import namedtuple
import collections.abc
import time
from braingeneers.utils import s3wrangler
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union, Iterable, Optional
from nptyping import NDArray, Int16, Float16, Float32, Float64
import io
import braingeneers
from braingeneers.utils import common_utils
from braingeneers.utils.common_utils import get_basepath, map2
import itertools
import posixpath
import pandas as pd
from datetime import datetime
import requests
import re
from types import ModuleType, SimpleNamespace
import bisect
from deprecated import deprecated
import threading
from pynwb import NWBHDF5IO


VALID_LOAD_DATA_DTYPES = [np.int16, np.float16, np.float32, np.float64]
load_data_cache = dict()  # minimal cache for avoiding looping lookups of metadata in load_data


def list_uuids():
    """
    List all UUIDs in the archive
    Returns
    -------
    uuids : list
        List of UUIDs
    """
    # return common_utils.file_list('ephys/')
    if braingeneers.get_default_endpoint().startswith('http'):
        return [s.split('/')[-2] for s in s3wrangler.list_directories('s3://braingeneers' + '/ephys/')]
    else:
        # list file locally
        return os.listdir(braingeneers.get_default_endpoint() + '/ephys/')


def save_metadata(metadata: dict):
    """
    Saves a metadata file back to S3. This is not multi-writer safe, you can use:
        braingeneers.utils.common_utils.checkout
        braingeneers.utils.common_utils.checkin
    to lock the file while you are writing to it.

    :param metadata: the metadata dictionary as obtained from load_metadata(uuid)
    """
    batch_uuid = metadata['uuid']
    save_path = posixpath.join(
        braingeneers.utils.common_utils.get_basepath(),
        'ephys',
        batch_uuid,
        'metadata.json'
    )
    with smart_open.open(save_path, 'w') as f:
        f.write(json.dumps(metadata, indent=2))


@deprecated("This function is deprecated since NWB format transition, load_data has a cache option built in.")
def cached_load_data(cache_path: str, max_size_gb: int = 10, **kwargs):
    """
    Wraps a call to load_data with a diskcache at path `cache_path`.
    This is multiprocessing/thread safe.
    All arguments after the cache_path are passed to load_data (see load_data docs)
    You must specify the load_data argument names to avoid ambiguity with the cached_load_data parameters.

    When reading data from S3 (or even a compressed local file), this can provide a significant speedup by
    storing the results of load_data in a local (uncompressed) cache.

    Example usage:
        from braingeneers.data.datasets_electrophysiology import load_metadata, cached_load_data

        metadata = load_metadata('9999-00-00-e-test')
        data = cached_load_data(cache_path='/tmp/cache-dir', metadata=metadata, experiment=0, offset=0, length=1000)

    Note: this can safely be used with `map2` from `braingeneers.utils.common_utils` to parallelize calls to load_data.

    :param cache_path: str, path to the cache directory.
    :param max_size_gb: int, maximum size of the cache in GB (10 GB default). If the cache exceeds this size, the oldest items will be removed.
    :param kwargs: keyword arguments to pass to load_data, see load_data documentation.
    """
    cache = diskcache.Cache(cache_path, size_limit=10 ** 9 * max_size_gb)
    key = json.dumps(kwargs)
    if key in cache:
        return cache[key]
    else:
        data = load_data(**kwargs)
        cache[key] = data
        return data


def load_metadata(batch_uuid: str) -> dict:
    """
    Loads the batch UUID metadata.
    Metadata structure documentation:
        https://github.com/braingeneers/wiki/blob/main/shared/organizing-data.md#metadata-json-file
    Example usage:
        metadata_json = load_metadata('2020-03-10-e-128silicon-mouse-p35')
    :param batch_uuid: Dataset UUID, example: 2020-03-10-e-128silicon-mouse-p35
    :return: A single dict containing the contents of metadata.json. See wiki for further documentation:
        https://github.com/braingeneers/wiki/blob/main/shared/organizing-data.md
    """
    base_path = 's3://braingeneers/' \
        if braingeneers.get_default_endpoint().startswith('http') \
        else braingeneers.get_default_endpoint()

    metadata_full_path = posixpath.join(base_path, 'ephys', batch_uuid, 'metadata.json')
    with smart_open.open(metadata_full_path, 'r') as f:
        metadata = json.load(f)

    return metadata


def load_data(metadata: dict,
              experiment: Union[str, int, Iterable[str], Iterable[int], None],
              offset: int = 0,
              length: int = None,
              channels: Union[Iterable[int], int] = None,
              dtype: Union[str, NDArray[Int16, Float16, Float32, Float64]] = 'float32',
              cache_path: str = None,
              max_size_gb: int = 30,
              parallelism: (bool, int) = True) -> NDArray[Int16, Float16, Float32, Float64]:
    """
    This is the primary way to read raw ephys data from the Braingeneers archive.
    It assumes your data is in the standard HDF5 format and in row-major order
    (this typically happens automatically when data is uploaded).

    Example usage:
        metadata = load_metadata('2020-03-10-e-128silicon-mouse-p35')
        data = load_data(metadata, 'experiment1', offset=0, length=20000)  # 1s of data @ 20kHz
        print(data.shape)
        > (1024, 20000)

    Special notes:
     - The data is returned in the format [channels, time] where channels are the rows and time is the columns.
     - The data is returned in float32 format by default, but you can specify int16 to get the raw unscaled data.
     - The data is scaled by the voltage_scaling_factor if it is present in the metadata.
     - Offset and length are in units of samples (which depend on the sampling rate, available in the metadata).
     - Multiple channels can be read at once by specifying a list of channels.
     - Multiple experiments can be spanned by specifying a list of experiments by name or index
       (when specified by index, experiments are sorted by time). None can be used to read all experiments.
     - Data can be local or on S3, if the data is local it must mirror the S3 directory structure, use
       braingeneers.set_default_endpoint('path/to/ephys/folder') to read locally. Reads from S3 are default.
     - If the cache_path is set, the results of load_data will be cached in a diskcache at that path. This is
       helpful to reduce loading times when reading data from S3 or a compressed local file.

    As a note, the user MUST provide the offset (offset) and the length of frames to take. This function reads
    across as many blocks as the length specifies.

    :param metadata: result of load_metadata, dict
    :param experiment: Which experiment in the batch to load, by name (string) or index location (int),
        examples: "experiment1" or 0
    :param channels: Channel(s) to obtain data from (default None for all channels) list of ints or single int
    :param offset: int, Starting datapoint of interest, 0-indexed from start of
        recording across all data files and channels-agnostic
    :param length: int, required, number of data points to return
        (-1 meaning all data points across the full experiment recording)
    :param dtype: default "float32", a numpy dtype or string equivalent, valid options:
        np.float16, 32, or 64 or np.int16 to return the raw unscaled values.
    :param cache_path: str, path to a local directory where data will be cache.
    :param max_size_gb: int, maximum size of the cache in GB (30 GB default).
        If the cache exceeds this size, the oldest items will be removed.
    :param parallelism: (default == True) to enable parallel read operations using the number of CPUs
        to determine number of threads to use.
        False will disable parallelism and read using only the Main thread/process. Note, if you are reading
        data using a multiprocess data loader such as PyTorch then disabling parallelism is recommended.
        Debugging issues is much easier when parallelism is disabled.
        2+ will dictate the number of threads to use.
    :return: ndarray array of chosen data points in [channels, time] format.
    """
    assert 'uuid' in metadata, \
        'Metadata file is invalid, it does not contain required uuid field.'
    assert 'ephys_experiments' in metadata or 'experiments' in metadata, \
        'Metadata file is invalid, it does not contain required ephys_experiments field.'
    assert isinstance(experiment, (str, int, type(None))) or (isinstance(experiment, Iterable) and not isinstance(experiment, (str, bytes)) and all(isinstance(exp, (str, int)) for exp in experiment)), \
        f'Parameter experiment must be an int index, experiment name string, or an iterable of these (excluding strings and bytes as iterables). Got: {experiment}'
    assert length is not None and isinstance(length, int) and length >= -1, \
        f'Length parameter must be set explicitly, use -1 for the full experiment dataset ' \
        f'(across all files, warning, this can be a very large amount of data)'
    assert isinstance(parallelism, (bool, int)) and parallelism >= 0, \
        'Parallelism must be a boolean or an integer, got: {parallelism}'
    assert np.dtype(dtype) in VALID_LOAD_DATA_DTYPES, \
        'dtype must be one of np.int16 (unscaled raw data), np.float16, np.float32, or np.float64 (floats are in originally recorded units)'
    assert isinstance(offset, int) and isinstance(length, int), \
        f'offset and length must be integers, got offset is {type(offset)}, length is {type(length)}'

    n_threads = os.cpu_count() if parallelism is True else 1 if parallelism is False else parallelism
    channels = [channels] if isinstance(channels, int) else channels

    # look up experiment(s) by experiment name, index, list of experiment names, or list of indexes
    # make experiment iterable
    experiments = [experiment] if isinstance(experiment, (str, int)) \
        else list(experiment) if experiment is not None \
        else sorted(metadata['ephys_experiments'].keys(), key=lambda x: metadata['ephys_experiments'][x]['timestamp'])
    assert all([(type(e) in [str, int]) and isinstance(e, type(experiments[0])) for e in experiments]), \
        'Invalid experiment type, must be a string, int, or list of all strings or all integers.'

    # verify experiments and convert indexes to named experiments for consistent handling
    # experiments is a list of experiment names or indexes
    if isinstance(experiments[0], int):
        # experiments are indexed by number, verify indexing and convert to named experiments
        experiment_exists = [exp_ix < len(metadata['ephys_experiments']) for exp_ix in experiments]
        if not all(experiment_exists):
            missing_experiments = [experiments[i] for i in range(len(experiments)) if not experiment_exists[i]]
            raise ValueError(f'Experiment(s) {missing_experiments} not found in metadata.json')
        experiments_sorted = sorted(metadata['ephys_experiments'].keys(), key=lambda x: metadata['ephys_experiments'][x]['timestamp'])
        experiments_selected = [experiments_sorted[exp_ix] for exp_ix in experiments]
    else:
        # experiments are indexed by name, verify that all experiments exist
        experiment_exists = [exp_name in metadata['ephys_experiments'] for exp_name in experiments]
        if not all(experiment_exists):
            missing_experiments = [experiments[i] for i in range(len(experiments)) if not experiment_exists[i]]
            raise ValueError(f'Experiment(s) {missing_experiments} not found in metadata.json')
        experiments_selected = experiments

    # get metadata for selected experiments
    experiments_metadata = [metadata['ephys_experiments'][e] for e in experiments_selected]

    # verify that ephys_experiments are in timestamp sorted order
    is_sorted_order = all(
        datetime.strptime(experiments_metadata[i]['timestamp'], '%Y-%m-%dT%H:%M:%S') <=
        datetime.strptime(experiments_metadata[i + 1]['timestamp'], '%Y-%m-%dT%H:%M:%S')
        for i in range(len(experiments_metadata) - 1)
    )
    if not is_sorted_order:
        raise ValueError('When listing multiple experiments, the `experiment` parameter must be in '
                         'order of timestamp. This exception is raised to avoid unexpected behavior and risk of bugs.')

    dataset_size = sum([block['num_frames'] for exp in experiments_metadata for block in exp['blocks']])

    if offset + length > dataset_size:
        raise IndexError(
            f'Dataset size is {dataset_size} (across all selected experiments), but parameters offset + length '
            f'is {offset + length} which exceeds dataset size. Use -1 to read to the end.'
        )

    batch_uuid = metadata['uuid']

    # load data from cache if cache is specified and data is present
    data = None
    cache = None
    key = None
    if cache_path is not None:
        cache = diskcache.Cache(cache_path, size_limit=10 ** 9 * max_size_gb)
        key = json.dumps([batch_uuid, experiments_selected, offset, length, channels])
        if key in cache:
            data = cache[key]

    # load data if it was not found in cache
    if data is None:
        ephys_experiments = [metadata['ephys_experiments'][e] for e in experiments_selected]
        data = _NWBParallelReader(
            batch_uuid=batch_uuid,
            n_threads=n_threads,
            ephys_experiments=ephys_experiments,
            channels=channels,
            offset=offset,
            length=length,
            dtype=dtype,
            voltage_scaling_factor=metadata.get('voltage_scaling_factor')
        ).read_data()
        if cache_path is not None:
            cache[key] = data

    return data


class _NWBParallelReader:
    """
    Not a public class, this is used internally by load_data.

    This class provides a greedy generator that implements a work-stealing queue to keep all threads fully utilized.
    The use of a generator allows the read patterns to adjust dynamically based on the progress of threads.
    This class contains all logic governing the order data is read in.
    A simple greedy algorithm enforced here keeps all threads fully utilized with a simple executor.

    Notes about h5py:
     - When reading contiguous data (as defined by the user defined chunks size)
       from an open h5py file, multiple requests for sequential data will be
       efficiently read without extra indexing lookups. Therefore, this method is
       efficient over the network.
    """
    def __init__(self,
                 batch_uuid,
                 n_threads,
                 ephys_experiments : List[dict],
                 channels: List[int],
                 offset,
                 length,
                 dtype,
                 voltage_scaling_factor: float = None):
        """
        :param batch_uuid: UUID of the batch
        :param n_threads: number of threads to use
        :param ephys_experiments: list of experiments, experiments is a full experiment metadata dictionary
        :param channels: list of channels to read
        :param offset: offset in frames
        :param length: length in frames
        :param dtype: data type to return
        :param voltage_scaling_factor: voltage scaling factor
        """

        self.batch_uuid = batch_uuid
        self.n_threads = n_threads
        self.voltage_scaling_factor = np.array(voltage_scaling_factor, dtype=dtype) if voltage_scaling_factor is not None else None
        self.lock = threading.Lock()
        self.nwb_files = {}  # {worker_ix: {filepath: h5py.File, ...}, ...}

        # a list (contiguous blocks) of lists (individual blocks) [[block0, block1, ...], [block0, block1, ...], ...]
        self.contiguous_blocks, length_computed = self.compute_contiguous_blocks(
            ephys_experiments,
            channels,
            offset,
            length,
        )
        # pad self.contiguous_blocks with empty lists to match n_threads
        self.contiguous_blocks.extend([[] for _ in range(n_threads - len(self.contiguous_blocks))])
        self.balance_blocks()

        num_channels = ephys_experiments[0]['num_channels'] if channels is None else len(channels)
        self.data = np.empty((num_channels, length_computed), dtype=dtype)

    def read_data(self):
        """ Starts the worker threads """
        map2(func=self.worker, args=list(range(self.n_threads)), parallelism=self.n_threads, use_multithreading=True)
        return self.data

    def worker(self, worker_ix: int):
        """ Picks the next block to read and reads it """
        while True:
            with self.lock:
                if len(self.contiguous_blocks[worker_ix]) == 0:
                    self.balance_blocks()
                    if len(self.contiguous_blocks[worker_ix]) == 0:
                        return
                block_params = self.contiguous_blocks[worker_ix].pop(0)

            self.read_block(worker_ix, *block_params)

    def read_block(self,
                   worker_ix: int,
                   metadata_block: dict,
                   channel: int,
                   offset: int,
                   length: int,
                   data_channel: int,
                   data_offset: int) -> NDArray[Int16]:
        """
        Reads a single block directly into the appropriate place in self.data,
        handles voltage scaling if specified.
        """
        filepath = posixpath.join(get_basepath(), 'ephys', self.batch_uuid, metadata_block['path'])
        data_path = 'acquisition/ElectricalSeries/data'

        if filepath not in self.nwb_files.get(worker_ix, {}):
            worker_files = self.nwb_files.setdefault(worker_ix, {})
            worker_files[filepath] = h5py.File(filepath, 'r')
            assert data_path in worker_files[filepath], \
                f'Expected data path {data_path} not found in {filepath}'

        # Read data from NWB file
        nwb_file = self.nwb_files[worker_ix][filepath]
        data_slice = nwb_file[data_path][channel, offset:offset+length]

        # apply voltage scaling factor
        data_slice = data_slice * self.voltage_scaling_factor if self.voltage_scaling_factor is not None else data_slice

        # update self.data
        self.data[data_channel, data_offset:data_offset+length] = data_slice

    def balance_blocks(self):
        """
        Rebalances the blocks so there are at least n_threads blocks to read.
        This is performed when a thread finishes a contiguous segment.
        This function is responsible for deleting completed segments (inner list in self.contiguous_blocks).
        worker threads will continue to pull the next block from their index in self.contiguous_blocks[worker_ix].

        This is not thread safe, so a lock must be held when calling this function.
        """
        # While there are idle threads...
        while any(len(b) == 0 for b in self.contiguous_blocks[:self.n_threads]):
            # Get the index of the first idle worker with a 0 length segment
            first_idle_worker = next(i for i, b in enumerate(self.contiguous_blocks[:self.n_threads]) if len(b) == 0)

            # Get the index of the longest segment in self.contiguous_blocks
            longest_segment_ix, _ = max(enumerate(self.contiguous_blocks), key=lambda x: len(x[1]))

            # Move the second half of the longest segment to the first_idle_worker and
            # remove the second half of the longest segment
            mid_point = len(self.contiguous_blocks[longest_segment_ix]) // 2
            self.contiguous_blocks[first_idle_worker], self.contiguous_blocks[longest_segment_ix] = (
                self.contiguous_blocks[longest_segment_ix][mid_point:],  # Assign second half
                self.contiguous_blocks[longest_segment_ix][:mid_point]  # Retain first half
            )

            # Early exit if the moved segment was already the smallest possible
            if not self.contiguous_blocks[longest_segment_ix]:
                break

    @staticmethod
    def compute_contiguous_blocks(experiments: List[dict], channels: List[int], offset: int, length: int) -> List[List]:
        """
        Returns a list of lists where the outer list is contiguous blocks,
        and the inner list is each chunk of a block.
        Results are sorted by length of segments with longest segments first.

        offset and length are stitched across experiments

        Example return:
            [
                [(metadata_block, channel, offset, length, data_channel, data_array_ix), ...],  # contiguous segment of multiple blocks
                [(metadata_block, channel, offset, length, data_channel, data_array_ix), ...],  # contiguous segment of multiple blocks
                ...  # more contiguous segments
            ],
            data_length  # total length of data in frames

        :param experiments: list of experiments, experiments is a full experiment metadata dictionary
        :param channels: list of channels to read
        :param offset: offset in frames
        :param length: length in frames
        """
        # Input parameter validation
        assert len(experiments) > 0, 'No experiments found in metadata.'
        assert all([e['num_channels'] == experiments[0]['num_channels'] for e in experiments]), \
            'All experiments must have the same number of channels.'
        assert all([(('data_chunk_size' in b) and (b['data_chunk_size'][0] == 1)) for e in experiments for b in e['blocks']]), \
            (f'Only channel-major order is supported. data_chunk_size must be [1, n], '
             f'got: {experiments[0]["blocks"][0]["data_chunk_size"]} from metadata.json.')

        num_channels = experiments[0]['num_channels']
        length = length if length != -1 else np.inf

        # Determine the contiguous blocks to read.
        segments = []
        # offset is used to keep location within the current block (it's reset to 0 every block)
        # data_array_ix is used to keep location within the data array, self.data (it's not reset)
        data_array_ix = 0
        for exp in experiments:
            for block in exp['blocks']:
                num_frames = min(block['num_frames'] - offset, length)
                data_chunk_size = block['data_chunk_size'][1]

                # Skip blocks that are before the offset
                if offset >= block['num_frames']:
                    offset -= block['num_frames']
                    continue

                # Record the contiguous segments
                channel_segments = []
                for data_channel, channel in enumerate(channels if channels is not None else range(num_channels)):
                    for i in range(0, num_frames, data_chunk_size):
                        segment = (
                            block,  # block metadata
                            channel,  # channel number, int
                            i + offset,  # offset within the block
                            min(data_chunk_size, num_frames - i),  # length of the segment
                            data_channel,  # channel index in self.data[ch_ix]
                            data_array_ix + i,  # index in self.data[ch_ix, data_array_ix:data_array_ix+length]
                        )
                        channel_segments.append(segment)
                    segments.append(channel_segments)

                # Update offset and length to reflect the remaining data to read
                data_array_ix += num_frames
                offset = 0  # reset offset to 0 after the first block, offset is used to keep location within the current block
                length -= num_frames

        # Sort by length of contiguous segments with longest segments first
        segments.sort(key=lambda channel_segment: len(channel_segment), reverse=True)

        return segments, data_array_ix


def load_window(metadata, exp, window, dtype=np.float32, channels=None):
    """Loads a window of data from an experiment
    window is in frames
    Parameters
    ----------
    metadata : dict
        metadata dictionary
    exp : str
        experiment name
    window : tuple
        window start and end in frames
    dtype : np.dtype
        data type to return
    
    Returns
    -------
    data : np.array (n_channels, window_sz)
    """
    assert type(window[0]) == int, "Window start must be an integer"
    assert type(window[1]) == int, "Window end must be an integer"
    assert window[0] < window[1], "Window start must be less than window end"

    lsb = 6.294*10**-6
    gain = 512
    sig_offset = 512
    
    # print("Loading window: ", window)
    data = load_data(metadata, exp, offset=window[0],
                            length=window[1]-window[0], dtype= dtype,
                            channels = channels)
    return (data -sig_offset) * lsb * gain*1000 # last is for V to mV


def load_windows(metadata, exp, window_centers, window_sz, dtype=np.float16,
                channels=None):
    """Loads a window of data from an experiment
    window is in frames
    Parameters
    ----------
    metadata : dict
        metadata dictionary
    exp : str
        experiment name
    window_centers : list
        list of window centers in frames
    window_sz : int
        window size in frames
    dtype : np.dtype
        data type to load
    
    Returns
    -------
    data : np.array (n_windows, n_channels, window_sz)

    """
    data = []
    dataset_length = metadata['ephys_experiments'][exp]['blocks'][0]['num_frames']


    for i,center in enumerate(window_centers):
        # window is (start, end)
        window = (center - window_sz//2, center + window_sz//2)

        # Check if window is out of bounds
        if window[0] < 0 or window[1] > dataset_length:
            print("Window out of bounds, inserting zeros for window", window)
            try:
                data_temp = np.zeros((data_temp.shape[0],window_sz),dtype=dtype)
            except Exception as e:
                print(e, file=sys.stderr)
                data_temp = load_window(metadata, exp, window, dtype=dtype, channels=channels)
        else:
            data_temp = load_window(metadata, exp, window, dtype=dtype, channels=channels)
        
        # Check if window is the right size
        if data_temp.shape[1] != window_sz:
            print("Data shape mismatch, inserting zeros for window",window)
            data_temp = np.zeros((data_temp.shape[0],window_sz),dtype=dtype)
        
        data.append(data_temp)
    return np.stack(data, axis=0)


def load_data_raspi(metadata, batch_uuid, experiment: str, offset, length) -> NDArray[Int16]:
    """
    :param metadata:
    :param batch_uuid:
    :param experiment:
    :param offset:
    :param length:
    :return:
    """
    raise NotImplementedError


def load_data_axion(metadata: dict, batch_uuid: str, experiment: str,
                    channels: Iterable[int], offset: int, length: int) -> NDArray[Int16]:
    """
    Reads from Axion raw data format using Python directly (not going through matlab scripts first).
    Reader originally written by Dylan Yong, updated by David Parks.
    Limitations:
     - Axion reader must read all channels, it then filters those to selected channels after reading all channel data.
    :param metadata: result of load_experiment
    :param batch_uuid: uuid, example: 2020-07-06-e-MGK-76-2614-Wash
    :param experiment: experiment name under 'ephys_experiments'
    :param channels: a list of channels
    :param offset: data offset (spans full experiment, across files)
    :param length: data read length in frames (e.g. data points, agnostic of channel count)
    :return: ndarray in (channels, frames) format where channels 0:64 are well A1, 64:128 are well A2, etc.
    """

    data_multi_file = []

    # get subset of files to read from metadata.blocks
    metadata_offsets_cumsum = np.cumsum([
        block['num_frames'] for block in metadata['ephys_experiments'][experiment]['blocks']
    ])
    block_ixs_range = np.minimum(
        len(metadata_offsets_cumsum),
        np.searchsorted(metadata_offsets_cumsum, [offset, offset + length - 1], side='right')
    )
    block_ixs = list(range(block_ixs_range[0], block_ixs_range[1] + 1))
    assert len(block_ixs) > 0, \
        f'No data file found starting from offset {offset}, is this past the end of the data file?'

    # this is a back reference which was constructed this way to avoid duplicating the large channel map many times
    channel_map_key = metadata['ephys_experiments'][experiment]['blocks'][0]['axion_channel_map_key']

    # perform N read operations accumulating results in data_multi_file
    frame_read_count = 0  # counter to track number of frames read across multiple files
    for block_ix in block_ixs:
        experiment = metadata['ephys_experiments'][experiment]
        block = experiment['blocks'][block_ix]
        file_name = block['path']
        full_file_path = posixpath.join(get_basepath(), 'ephys', batch_uuid, 'original', 'data', file_name)
        sample_start = max(0, offset - metadata_offsets_cumsum[block_ix] + block['num_frames'])
        data_length = min(block['num_frames'], length - frame_read_count)
        data_ndarray = _axion_get_data(
            file_name=full_file_path,
            file_data_start_position=block['axion_data_start_position'],
            sample_offset=sample_start,
            sample_length=data_length,
            num_channels=experiment['num_channels'],
            corrected_map=metadata[channel_map_key],
        )

        num_channels_per_well = np.prod(experiment['axion_electrode_layout_row_col'])

        # select channels (None for all channels)
        c = np.array(channels) if isinstance(channels, (Iterable, np.ndarray)) \
            else np.arange(num_channels_per_well) if isinstance(channels, type(None)) \
            else np.array([channels])
        c += experiment['axion_channel_offset']
        data_ndarray_select_channels = data_ndarray[c, :]

        # append data from this block/file to list
        data_multi_file.append(data_ndarray_select_channels)
        frame_read_count += data_ndarray_select_channels.shape[1]

    # concatenate the results and return
    data_concat = np.concatenate(data_multi_file, axis=1)

    return data_concat


def load_data_intan(metadata, batch_uuid, experiment: str, offset, length) -> NDArray[Int16]:
    """
    :param metadata:
    :param batch_uuid:
    :param experiment:
    :param offset:
    :param length
    """
    raise NotImplementedError


def load_data_maxwell(metadata, batch_uuid, experiment: str, channels, start, length) -> NDArray[Int16]:
    """
    Loads specified amount of data from one block
    :param metadata: json file
        JSON file containing metadata of experiment
    :param batch_uuid: str
        UUID of batch of experiments within the Braingeneers's archive
    :param experiment: str
        Experiment name as a int
    :param channels: list of int
        Channels of interest
    :param start: int
        Starting frame (offset) of the datapoints to use
    :param length: int
        Length of datapoints to take
    :return:
    dataset: nparray
        Dataset of datapoints.
    """
    # TODO: Check the length and see if there are enough blocks to even support it
    # NOTE: Blocks (right now) are worthless to me

    experiment_stem = posixpath.basename(metadata['ephys_experiments'][experiment]['blocks'][0]['path'])

    # if length == -1:
    #     print(
    #         f"Loading file Maxwell, UUID {batch_uuid}, {experiment}: {experiment_stem}, frame {start} to end of file....")
    # else:
    #     print(
    #         f"Loading file Maxwell, UUID {batch_uuid}, {experiment}: {experiment_stem}, frame {start} to {start + length}....")
    # get datafile

    filename = metadata['ephys_experiments'][experiment]['blocks'][0]['path'].split('/')[-1]
    datafile = posixpath.join(get_basepath(), 'ephys', batch_uuid, 'original', 'data', filename)

   
    frame_end = start + length

    # open file
    with smart_open.open(datafile, 'rb') as file:
        with h5py.File(file, 'r', libver='latest', rdcc_nbytes=2 ** 25) as h5file:
            # know that there are 1028 channels which all record and make 'num_frames'
            # lsb = np.float32(h5file['settings']['lsb'][0]*1000) #1000 for uv to mv  # voltage scaling factor is not currently implemented properly in maxwell reader
            table = 'sig' if 'sig' in h5file.keys() else '/data_store/data0000/groups/routed/raw'
            dataset = h5file[table]
            if channels is not None:
                sorted_channels = np.sort(channels)
                undo_sort_channels = np.argsort(np.argsort(channels))

                dataset = dataset[sorted_channels, start:frame_end]
            else:
                dataset = dataset[:, start:frame_end]
    
    if channels is not None:
        # Unsort data
        dataset = dataset[undo_sort_channels, :]

    return dataset


def load_data_maxwell_parallel(metadata: dict, batch_uuid: str, experiment: str,
                               channels: Iterable[int], offset: int, length: int) -> NDArray[Int16]:
    """
    High-performance version of load_data_maxwell that expects raw data in rows-first format rather than the default
    column-first format. This version of load_data encapsulates parallelism to make read calls efficient over S3
    """
    # filter blocks based on offset and length
    blocks, first_block_offset, first_block_read_offset = \
        get_blocks_for_load_data(metadata, experiment, offset, length)

    # construct arguments for _load_data_maxwell_per_channel per block per channel
    num_frames = [b['num_frames'] for b in blocks]
    starts_readlengths_per_block = [
        (first_block_read_offset, min(length, num_frames[0] - first_block_read_offset)),  # first block
        *[(0, nf) for nf in num_frames[1:-1]],  # middle blocks between first & last
    ]
    sum_readlengths = sum([rl for s, rl in starts_readlengths_per_block])
    if len(num_frames) >= 3:
        starts_readlengths_per_block.append((0, min(length - sum_readlengths, length - (num_frames[0] - first_block_offset + sum(num_frames[1:-1])))))  # last block
    filepaths_channels_starts_lengths = [
        (common_utils.path_join('ephys', batch_uuid, b['path']), c, s, l)
        for (b, (s, l)), c in itertools.product(zip(blocks, starts_readlengths_per_block), channels)
    ]

    # Parallel read from each channel using separate processes (necessary so HDF5 doesn't
    # step on its own toes as it would do if threads were used). If multiple files exist
    # then the read will be per channel per each block (aka file).
    data_per_block_per_channel = common_utils.map2(
        func=_load_data_maxwell_per_channel,
        args=filepaths_channels_starts_lengths,
        parallelism=False,
    )
    data = np.vstack(data_per_block_per_channel)

    return data


def _load_data_maxwell_per_channel(filepath: str, channel: int, start: int, length: int):
    """
    Internal function used by load_data_maxwell to read each channel. This function is
    called in worker sub-processes.
    """
    with smart_open.open(filepath, 'rb') as raw_file:
        with h5py.File(raw_file) as f:
            table = 'sig' if 'sig' in f.keys() else '/data_store/data0000/groups/routed/raw'  # 'recordings/rec0000/well000/groups/routed/raw'
            data = f[table][channel, start:start+length]
            return data


def compute_cumsum_num_frames(metadata: dict, experiment: str) -> List[int]:
    """
    Intended to be an internal function primarily.
    Computes and caches the cumulative sum of num_frames, this operation will be repeated
    for each call to load_data which may occur in a high performance loop, hence the caching.
    """
    cache_key = ('_compute_cumsum_num_frames_', metadata['uuid'], experiment)
    if cache_key not in load_data_cache:
        csum = np.cumsum([block['num_frames'] for block in metadata['ephys_experiments'][experiment]['blocks']]).tolist()
        load_data_cache[cache_key] = csum
    return load_data_cache[cache_key]


def get_blocks_for_load_data(metadata: dict, experiment: str, offset: int, length: int) -> List:
    """
    Intended to be an internal function primarily.
    Returns a subset of metadata.ephys_experiments.blocks required to
    perform the read from offset to offset + length, this will return a list of one or more
    elements from metadata.ephys_experiments.blocks.
    This method can be re-used by different data format readers assuming all use the same
    metadata format.
    This method is efficient for high performance operations due to caching
    of the call to compute_cumsum_num_frames.
    :param metadata: result of load_metadata(...)
    :param experiment: experiment index number as passed to load_data function
    :param offset: offset as passed to load_data function
    :param length: length as passed to load_data function
    :return: 1) a list of one or more metadata.ephys_experiments.blocks;
             2) the offset of the first block (globally across all blocks);
             3) the offset to start reading from in the first block;
    """
    # This step is cached in memory (lru_cache) after the first call, cum_sum_blocks ex: [100, 200, 300, ...]
    cum_sum_blocks = compute_cumsum_num_frames(metadata, experiment)

    ix_a = bisect.bisect_right(cum_sum_blocks, offset)
    ix_b = bisect.bisect_left(cum_sum_blocks, offset + length) + 1

    blocks = metadata['ephys_experiments'][experiment]['blocks'][ix_a:ix_b]

    cum_sum_offset = [0] + cum_sum_blocks
    first_block_offset = cum_sum_offset[ix_a]
    first_block_read_offset = offset - first_block_offset

    return blocks, first_block_offset, first_block_read_offset


def load_data_hengenlab(metadata: dict, batch_uuid: str, experiment: str,
                        channels: Iterable[int], offset: int, length: int) -> NDArray[Int16]:
    # filter blocks based on offset and length
    blocks, first_block_offset, _ = get_blocks_for_load_data(metadata, experiment, offset, length)
    n_channels = metadata['ephys_experiments'][experiment]['num_channels']

    # pre-allocate memory for data
    block_offset = offset - first_block_offset
    b = bytearray(length * n_channels * 2)  # pre-allocate byte array for data

    # read (length) bytes from one or more blocks
    pos = 0
    for block in blocks:
        file_or_url = common_utils.path_join('ephys', batch_uuid, block['path'])
        with smart_open.open(file_or_url, 'rb') as f:
            f.seek(8 + block_offset * n_channels * 2)  # 8 bytes ecube timestamp at front of file
            v = memoryview(b)
            pos += f.readinto(v[pos:])  # read until b is full
            block_offset = 0  # reset to 0 because the next file we read will start at the beginning

    # convert to ndarray
    data = np.frombuffer(b, dtype=np.int16).reshape((n_channels, length), order='F')

    channel_map = np.array(metadata['channel_map'])

    data_selected = data[channel_map, :] if channels is None else data[channel_map[channels], :]
    return data_selected


def load_mapping_maxwell(uuid: str, metadata_ephys_exp: dict, channels: list = None) -> pd.DataFrame:
    """
    Loads the mapping of maxwell array from hdf5 file
    
    :param uuid: uuid of the experiment
        UUID of batch of experiments
    :param metadata_ephys_exp: metadata of the experiment for one recording
        This must look like metadata['ephys_experiments']['experiment1']
        from the normal metadata loading function
    :param channels:
    :return: mapping of maxwell array as a dataframe
    """
    exp_path = metadata_ephys_exp['blocks'][0]['path']
    exp_filename = posixpath.basename(exp_path)
    DATA_PATH = 'original/data/'

    file_path = posixpath.join(common_utils.get_basepath(), 
            'ephys',uuid, DATA_PATH, exp_filename)

    print('Loading mapping from UUID: {}, experiment: {}'.format(uuid, exp_filename))

    with smart_open.open(file_path, 'rb') as f:
        with h5py.File(f, 'r') as h5:
            # version is 20160704 - ish?, old format
            if 'mapping' in h5:
                mapping = np.array(h5['mapping']) #ch, elec, x, y
                mapping = pd.DataFrame(mapping)
            # version is 20190530 - ish?
            else:
                mapping = np.array(h5['data_store/data0000/settings/mapping'])
                mapping = pd.DataFrame(mapping)

    if channels is not None:
        return mapping[mapping['channel'].isin(channels)]
    else:
        return mapping


def load_stims_maxwell(uuid: str, metadata_ephys_exp: dict = None, experiment_stem: str = None):
    '''
    Loads the stim log files for a given experiment.
    
    :param uuid: uuid of the experiment
        UUID of batch of experiments
    :param metadata_ephys_exp: metadata of the experiment for one recording
        This must look like metadata['ephys_experiments']['experiment1']
        from the normal metadata loading function
    :param experiment_stem: file basename of the experiment,
        Used in place of the dict to load the stim data
    :return: dataframe of stim logs
    '''
    if metadata_ephys_exp is not None:
        exp_path = metadata_ephys_exp['blocks'][0]['path']
        # This is gross, we have to split off the .raw.h5, requiring 2 splits
        exp_stem = os.path.splitext(posixpath.basename(exp_path))[0]
        exp_stem = os.path.splitext(exp_stem)[0]
    elif experiment_stem is not None:
        exp_stem = experiment_stem

    exp_stim_log = exp_stem + 'log.csv' 
    DATA_PATH = 'original/log/'


    stim_path = posixpath.join(get_basepath(), 'ephys', uuid, DATA_PATH, 
                                exp_stim_log)
    import io
    print('Loading stim log from UUID: {}, log: {}'.format(uuid, exp_stim_log))
    print(stim_path)
    try:
        with smart_open.open(stim_path, 'rb') as f:
            # read the csv into dataframe
            f = io.TextIOWrapper(f, encoding='utf-8')
            df = pd.read_csv(f, header=0)#, index_col=0)
        return df
        
    except FileNotFoundError:
        print(f'\tThere seems to be no stim log file for this experiment! :(', file=sys.stderr)
        return None
    except OSError:
        print(f'\tThere seems to be no stim log file (on s3) for this experiment! :(', file=sys.stderr)
        return None

   
def load_gpio_maxwell(dataset_path, fs=20000.0):
    """
    Loads the GPIO events for optogenetics stimulation.
    :param dataset_path: a local or a s3 path
    :param fs: sample rate
    :return: an array of opto stimulation pairs as [[start, end]] in seconds
    """
    with smart_open.open(dataset_path, 'rb') as f:
        with h5py.File(f, 'r') as dataset:
            if 'bits' not in dataset.keys():
                print('No GPIO event in the dataset!', file=sys.stderr)
                return np.array([])
            bits_dataset = list(dataset['bits'])
            bits_dataframe = [bits_dataset[i][0] for i in range(len(bits_dataset))]
            rec_startframe = dataset['sig'][-1, 0] << 16 | dataset['sig'][-2, 0]
    if len(bits_dataframe) % 2 == 0:
        stim_pairs = (np.array(bits_dataframe) - rec_startframe).reshape(len(bits_dataframe) // 2, 2)
        return stim_pairs / fs
    else:
        print("Odd number of GPIO events can't be paired. Here returns all the events.")
        return (np.array(bits_dataframe) - rec_startframe)/fs


def compute_milliseconds(num_frames, sampling_rate):
    """
    :param num_frames: int
        Number of frames to convert to milliseconds
    :param sampling_rate: int
        Sampling rate in units of Hz
    :return:
        A string detailing how many ms of recording are there
    """
    return f'{(num_frames / sampling_rate) * 1000} ms of total recording'


def load_spikes(batch_uuid, experiment_num):
    batch = load_batch(batch_uuid)
    experiment_name_with_json = batch['experiments'][experiment_num]
    experiment_name = experiment_name_with_json[:-5].rsplit('/', 1)[-1]
    path_of_firings = '/public/groups/braingeneers/ephys/' + batch_uuid + '/nico_spikes/' + experiment_name + '_spikes.npy'
    print(path_of_firings)

    try:
        firings = np.load(path_of_firings)
        spike_times = firings[1]
        return spike_times
    except:
        path_of_firings_on_prp = get_archive_url() + '/' + batch_uuid + '/nico_spikes/' + experiment_name + '_spikes.npy'
        response = requests.get(path_of_firings_on_prp, stream=True)

        with open('firings.npy', 'wb') as fin:
            shutil.copyfileobj(response.raw, fin)

        firings = np.load('firings.npy')
        spikes = firings[1]
        return spikes


def load_firings(batch_uuid, experiment_num, sorting_type):  # sorting type is "ms4" or "klusta" etc
    batch = load_batch(batch_uuid)
    experiment_name_with_json = batch['experiments'][experiment_num]
    experiment_name = experiment_name_with_json[:-5].rsplit('/', 1)[-1]
    if (sorting_type == "ms4"):
        path_of_firings = '/public/groups/braingeneers/ephys/' + batch_uuid + '/nico_spikes/' + experiment_name + '_spikes.npy'
    if (sorting_type == "klusta"):
        path_of_firings = '/public/groups/braingeneers/ephys/' + batch_uuid + '/klusta_spikes/' + experiment_name + '_spikes.npy'
    print(path_of_firings)

    try:
        firings = np.load(path_of_firings)
        return firings
    except:
        if (sorting_type == "ms4"):
            path_of_firings_on_prp = get_archive_url() + '/' + batch_uuid + '/nico_spikes/' + experiment_name + '_spikes.npy'
        if (sorting_type == "klusta"):
            path_of_firings_on_prp = get_archive_url() + '/' + batch_uuid + '/klusta_spikes/' + experiment_name + '_spikes.npy'
        response = requests.get(path_of_firings_on_prp, stream=True)

        with open('firings.npy', 'wb') as fin:
            shutil.copyfileobj(response.raw, fin)

        firings = np.load('firings.npy')
        return firings


def min_max_blocks(experiment, batch_uuid):
    batch = load_batch(batch_uuid)
    index = batch['experiments'].index("{}.json".format(experiment['name']))
    for i in range(len(experiment["blocks"])):
        print("Computing Block: ", str(i))
        X, t, fs = load_blocks(batch_uuid, index, i, i + 1)
        X = np.transpose(X)
        X = X[:int(experiment['num_voltage_channels'])]
        step = int(fs / 1000)
        yield np.array([[
            np.amin(X[:, j:min(j + step, X.shape[1] - 1)]),
            np.amax(X[:, j:min(j + step, X.shape[1] - 1)])]
            for j in range(0, X.shape[1], step)])


def create_overview(batch_uuid, experiment_num, with_spikes=True):
    # batch_uuid = '2020-02-06-kvoitiuk'

    batch = load_batch(batch_uuid)

    experiment = load_experiment(batch_uuid, experiment_num)
    index = batch['experiments'].index("{}.json".format(experiment['name']))
    plt.figure(figsize=(15, 5))

    overview = np.concatenate(list(min_max_blocks(experiment, batch_uuid)))

    print('Overview Shape:', overview.shape)

    plt.title("Overview for Batch: {} Experiment: {}".format(batch_uuid, experiment["name"]))
    plt.fill_between(range(0, overview.shape[0]), overview[:, 0], overview[:, 1])

    blocks = load_blocks(batch_uuid, experiment_num, 0)

    if with_spikes:

        spikes = load_spikes(batch_uuid, experiment_num)

        fs = blocks[2]

        step = int(fs / 1000)

        spikes_in_correct_units = spikes / step

        for i in spikes_in_correct_units:
            plt.axvline(i, .1, .2, color='r', linewidth=.8, linestyle='-', alpha=.05)

    plt.show()
    # path = "archive/features/overviews/{}/{}.npy".format(batch["uuid"], experiment["name"])
    # print(path)


# Next 4 fcns are for loading data quickly from the maxwell,
# assigning a path to a local kach dir "./ephys" if batch is synced
# generate and return metadata, h5file, exp_jsons from UUID and possible recording locations
# if theres a dir for the specified uuid, assign local experiment path, otherwise assign s3 experiment path
def fast_batch_path(uuid):
    if os.path.exists("/home/jovyan/Projects/maxwell_analysis/ephys/" + uuid):
        uuid = "/home/jovyan/Projects/maxwell_analysis/ephys/" + uuid
        metadata = json.load(smart_open.open(uuid + 'metadata.json', 'r'))
    else:
        uuid = "s3://braingeneers/ephys/" + uuid
    print(uuid)
    return uuid


# get actual path to recording, not the name of the json from metadata
def paths_2_each_exp(data_dir):
    try:
        objs = s3wrangler.list_objects(data_dir)  # if on s3
    except:
        objs = next(os.walk(data_dir), (None, None, []))[2]  # if on local dir
    return objs


def connect_kach_dir():
    global maxone_wetai_kach_dir
    maxone_wetai_kach_dir = "/home/jovyan/Projects/maxwell_analysis/ephys/"
    if not os.path.exists(maxone_wetai_kach_dir):
        os.makedirs(maxone_wetai_kach_dir)
    return maxone_wetai_kach_dir


# will download nonexistent batches into the ./ephys dir on the wetAI ephys maxwell analyzer
# Usage. Supply the batch_name for the batch you would like to download.
# If not downloaded into the local kach dir, it will be downloaded to improve loading time
def sync_s3_to_kach(batch_name):
    sync_command = f"aws --endpoint $ENDPOINT_URL s3 sync s3://braingeneers/ephys/{batch_name} /home/jovyan/Projects/maxwell_analysis/ephys/{batch_name}"
    os.system(sync_command)


def _read_hengenlab_ecube_timestamp(filepath: str) -> int:
    with smart_open.open(filepath, 'rb') as f:
        return int(np.frombuffer(f.read(8), dtype=np.uint64))


# --- AXION READER -----------------------------
def from_uint64(all_values):
    """
    FromUint64: Deserializes an entry record from its native 64
    bit format in AxIS files.
    ----------------64 Bits--------------
    | ID (1 Byte) |   Length (7 Bytes)  |
    -------------------------------------
    Note that only the last entry in a file amy have length ==
    (0x00ff ffff ffff  ffff) which denotes an entry that reads to
    the end of the file. These entries have a length field == inf
    when deserialized
    :param all_values:
    :return:
    """

    class EntryRecord:
        def __init__(self, type, length):
            self.type = type
            self.length = length

        LENGTH_MASK_HIGH = np.uint64(int('ffffff', 16))
        LENGTH_MASK_LOW = np.uint64(int('ffffffff', 16))

    return_list = []

    # verify allValues is uint64
    if type(all_values[0]) is not np.uint64:
        print("allValues must be of type uint64")

    # for every record in the list
    for obj in all_values:

        # this is missing a check to make sure the entry record is valid
        # read the upper word (with ID field)
        # have to cast the shift value to uint64 as well
        fid = obj >> np.uint64(64-8)
        # shift right 4 bytes and mask with LENGTH_MASK_HIGH
        f_length = (obj >> np.uint64(64-32)) & EntryRecord.LENGTH_MASK_HIGH  # right 4 bytes

        # start the check to see if this may be a 'read to the end'
        f_is_inf = f_length == EntryRecord.LENGTH_MASK_HIGH
        # shift left 4 bytes to be ANDed with lower word
        f_length = f_length << np.uint64(32)
        f_low_word = obj & EntryRecord.LENGTH_MASK_LOW        # read the lower word
        # finish the check to see if this may be a 'read to the end'
        f_is_inf = f_is_inf & (f_low_word == EntryRecord.LENGTH_MASK_LOW)
        # recombine the upper and lower length portions
        f_length = f_length | f_low_word

        # create record fid and add to list of records
        record = EntryRecord(fid, f_length)
        return_list.append(record)

    return return_list


def generate_metadata_axion(batch_uuid: str, experiment_prefix: str = '',
                            n_threads: int = 16, save: bool = False):
    """
    Generates metadata.json raw Axion data files on S3 from a standard UUID. Assumes raw data files are stored in:
        ${ENDPOINT}/ephys/YYYY-MM-DD-e-[descriptor]/original/experiments/*.raw
        (ENDPOINT defaults to s3://braingeneers)
    Raises an exception if no data files were found at the expected location.
    All Axion recordings are assumed to be a single experiment.
    Limitations:
     - timestamps are not taken from the original data files, the current time is used.
    :param batch_uuid: standard ephys UUID
    :param experiment_prefix: Experiments are named "A1", "A2", ..., "B1", "B2", etc. If multiple recordings are
        included in a UUID the experiment name can be prefixed, for example "recording1_A1", "recording2_A1"
        for separate recordings. It is suggested to end the prefix with "_" for readability.
    :param n_threads: number of concurrent file reads (useful for parsing many network based files)
    :param save: bool (default == False) saves the generated metadata file back to S3/ENDPOINT at batch_uuid
    :return: (metadata_json: dict, ephys_experiments: dict) a tuple of two dictionaries which are
        json serializable to metadata.json and experiment1.json.
    """
    metadata_json = {}
    ephys_experiments = {}
    current_timestamp = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%dT%H:%M:%S')

    # construct metadata_json
    metadata_json['issue'] = ''
    metadata_json['notes'] = ''
    metadata_json['timestamp'] = current_timestamp
    metadata_json['uuid'] = batch_uuid

    # list raw data files at batch_uuid
    # noinspection PyArgumentList
    list_of_raw_data_files = sorted(s3wrangler.list_objects(
        path=posixpath.join(get_basepath(), 'ephys', batch_uuid, 'original/data/'),
        suffix='.raw',
    ))
    if len(list_of_raw_data_files) == 0:
        raise FileNotFoundError(f'No raw data files found at {get_basepath()}/ephys/{batch_uuid}/original/data/*.raw')

    with ThreadPoolExecutor(n_threads) as pool:
        metadata_tuples_per_data_file = list(pool.map(_axion_generate_per_block_metadata, list_of_raw_data_files))

    for metadata_tuple, raw_data_file in zip(metadata_tuples_per_data_file, list_of_raw_data_files):
        data_start = metadata_tuple[0]
        data_length_bytes = metadata_tuple[1]
        num_channels = metadata_tuple[2]
        corrected_map = metadata_tuple[3]
        sampling_frequency = metadata_tuple[4]
        voltage_scale = metadata_tuple[5]
        plate_layout_row_col = metadata_tuple[6]
        electrode_layout_row_col = metadata_tuple[7]

        # the same map & experiment data comes from every file in a recording so some of these operations are redundant
        for well_row in range(plate_layout_row_col[0]):
            for well_col in range(plate_layout_row_col[1]):
                # Produces: A1, A2, ..., B1, B2, ... etc. Prepends experiment_prefix
                well_name = experiment_prefix + chr(well_row + ord('A')) + str(well_col + 1)
                n_channels_per_well = int(np.prod(electrode_layout_row_col))
                well_index = well_row * plate_layout_row_col[1] + well_col

                metadata_json[f'{experiment_prefix}axion_per_well_channel_map'] = corrected_map

                assert data_length_bytes % num_channels % 2 == 0, \
                    f'Encountered an unexpected data_length of {data_length_bytes} ' \
                    f'which is not evenly divisible by {num_channels} channels'

                data_length = data_length_bytes // num_channels // 2

                experiment = ephys_experiments.setdefault(well_name, {})
                experiment['name'] = well_name
                experiment['hardware'] = 'Axion BioSystems'
                experiment['notes'] = ''
                experiment['num_channels'] = num_channels
                experiment['num_current_input_channels'] = 0
                experiment['num_voltage_channels'] = num_channels
                experiment['offset'] = 0
                experiment['sample_rate'] = int(sampling_frequency)
                experiment['voltage_scaling_factor'] = float(voltage_scale)
                experiment['timestamp'] = current_timestamp
                experiment['units'] = '\u00b5V'
                experiment['version'] = '1.0.0'
                experiment['axion_plate_layout_row_col'] = plate_layout_row_col
                experiment['axion_electrode_layout_row_col'] = electrode_layout_row_col
                experiment['axion_channel_offset'] = well_index * n_channels_per_well

                block = dict()
                block['num_frames'] = data_length
                block['path'] = os.path.basename(raw_data_file)
                block['timestamp'] = current_timestamp
                block['axion_data_start_position'] = data_start
                block['axion_channel_map_key'] = f'{experiment_prefix}axion_per_well_channel_map'

                experiment.setdefault('blocks', [])
                experiment['blocks'].append(block)

    metadata_json['ephys_experiments'] = ephys_experiments

    if save:
        with smart_open.open(posixpath.join(get_basepath(), 'ephys', batch_uuid, 'metadata.json'), 'w') as f:
            json.dump(metadata_json, f, indent=2)

    return metadata_json


def _axion_generate_per_block_metadata(filename: str):
    """
    Internal function
    Generates the information necessary to create a metadata file for an Axion dataset.
    This function should only be called when uploading the dataset to Axion, under normal
    circumstances the metadata file is generated once at upload time and stored on S3.
    :param filename: S3 or local axion raw data file, example:
        "s3://braingeneers/ephys/2020-07-06-e-MGK-76-2614-Wash/raw/H28126_WK27_010320_Cohort_202000706_Wash(214).raw"
    :return: start of data position in file, length of data section, number of channels,
        mapping to correctly rearrange columns, sampling frequency, voltage scaling factor
    """
    ChannelData = namedtuple('ChannelData', 'wCol wRow eCol eRow')

    with smart_open.open(filename, 'rb') as fid:

        # replace frombuffer with 1 seek
        fid.seek(26, 0)

        # mark start for entries and get record list
        buff = fid.read(8 * 124)  # replace two read calls below with this one
        # buff = fid.read(8)
        entries_start = np.frombuffer(buff[:8], dtype=np.uint64, count=1)
        # buff = fid.read(8 * 123)
        entry_slots = np.frombuffer(buff[8:], dtype=np.uint64, count=123)
        record_list = from_uint64(entry_slots)

        # % Start Reading Entries
        fid.seek(entries_start[0], 0)
        channel_map = []
        corrected_map = []

        # retrieve metadata from records
        for obj in record_list:  # order is 1, 2, 7, 4, 6 - 6
            if obj.type == 1:
                start = fid.tell()
                fid.seek(start + int(obj.length.item()), 0)

            # map column correction
            elif obj.type == 2:
                start = fid.tell()
                fid.seek(4, 1)

                buff = fid.read(4)
                num_channels = int(np.frombuffer(buff, dtype=np.uint32, count=1))

                # determine the column ordering characteristics
                # single read all channels, replaces read+seek calls commented below
                buff = fid.read(8 * num_channels)
                for i in range(num_channels):
                    buff_ix = i * 8
                    tw_col = np.frombuffer(buff[buff_ix + 0:buff_ix + 1], dtype=np.uint8, count=1)
                    tw_row = np.frombuffer(buff[buff_ix + 1:buff_ix + 2], dtype=np.uint8, count=1)
                    te_col = np.frombuffer(buff[buff_ix + 2:buff_ix + 3], dtype=np.uint8, count=1)
                    te_row = np.frombuffer(buff[buff_ix + 3:buff_ix + 4], dtype=np.uint8, count=1)

                    channel_map_node = ChannelData(int(tw_col), int(tw_row), int(te_col), int(te_row))
                    channel_map.append(channel_map_node)
                    plate_layout_row_col = (max(c.wRow for c in channel_map), max(c.wCol for c in channel_map))
                    n_wells = int(np.product(plate_layout_row_col))
                    electrode_layout_row_col = (max(c.eRow for c in channel_map), max(c.eCol for c in channel_map))

                for i in range(n_wells):
                    mini_map = [None] * int((num_channels / n_wells))
                    corrected_map.append(mini_map)

                # well = (row, col)
                # A1 = (1,1) 1
                # A2 = (1,2) 2
                # A3 = (1,3) 3
                # B1 = (2,1) 4
                # B2 = (2,2) 5
                # B3 = (2,3) 6
                # determine what well the data is corresponding to
                for idx, item in enumerate(channel_map):
                    well = ((item.wRow - 1) * plate_layout_row_col[1]) + (item.wCol - 1)

                    # well = None
                    # if item.wRow == 1 and item.wCol == 1:
                    #     well = 0
                    # elif item.wRow == 1 and item.wCol == 2:
                    #     well = 1
                    # elif item.wRow == 1 and item.wCol == 3:
                    #     well = 2
                    # elif item.wRow == 2 and item.wCol == 1:
                    #     well = 3
                    # elif item.wRow == 2 and item.wCol == 2:
                    #     well = 4
                    # elif item.wRow == 2 and item.wCol == 3:
                    #     well = 5

                    # need electrode layout in rows and columns
                    corrected_idx = ((item.eRow - 1) * electrode_layout_row_col[0]) + (item.eCol - 1)
                    assert corrected_idx is not None and well is not None and isinstance(corrected_idx, int)
                    corrected_map[well][corrected_idx] = idx

                fid.seek(start + int(obj.length.item()), 0)
                if fid.tell() != start + obj.length:
                    print('Unexpected Channel array length', file=sys.stderr)

            elif obj.type == 3:
                continue

            elif obj.type == 4:
                start = fid.tell()
                data_start = fid.tell()
                fid.seek(start + int(obj.length.item()), 0)
                data_length = int(obj.length.item())

            elif obj.type == 5:
                continue

            elif obj.type == 6:
                continue

            elif obj.type == 7:
                start = fid.tell()
                fid.seek(8, 1)

                # sampling frequency
                buff = fid.read(8)
                sampling_frequency = np.frombuffer(buff, dtype=np.double, count=1)

                # voltage scale
                buff = fid.read(8)
                voltage_scale = np.frombuffer(buff, dtype=np.double, count=1)

                fid.seek(start + int(obj.length.item()), 0)

    return data_start, data_length, num_channels, corrected_map, sampling_frequency, voltage_scale, \
        plate_layout_row_col, electrode_layout_row_col


def _axion_get_data(file_name, file_data_start_position,
                    sample_offset, sample_length,
                    num_channels, corrected_map):
    """
    :param: file_name: file name
    :param: file_data_start_position: data start position in file as returned by metadata
    :param: sample_offset: number of frames to skip from beginning of file
    :param: sample_length: length of data section in frames (agnostic of channel count)
    :param: num_channels:
    :param: corrected_map: mapping to correctly rearrange columns
    :return:
    """
    with smart_open.open(file_name, 'rb', compression='disable') as fid:
        # --- GET DATA ---
        total_num_samples = sample_length * num_channels

        fid.seek(file_data_start_position, io.SEEK_SET)
        fid.seek(2 * num_channels * sample_offset, io.SEEK_CUR)
        buff = fid.read(2 * int(total_num_samples))
        temp_raw_data = np.frombuffer(buff, dtype=np.int16, count=int(total_num_samples))

        f_remainder_count = len(temp_raw_data) % num_channels
        if f_remainder_count > 0:
            raise AttributeError(f'Wrong number of samples for {num_channels} channels')

        temp_raw_data_reshaped = np.reshape(temp_raw_data, (num_channels, -1), order='F')
        final_raw_data_reshaped = temp_raw_data_reshaped[list(itertools.chain(*corrected_map))]

        # A1 = (1,1) 0-63
        # A2 = (1,2) 64-127
        # A3 = (1,3) 128-191
        # B1 = (2,1) 192-255
        # B2 = (2,2) 256-319
        # B3 = (2,3) 320-383
        return final_raw_data_reshaped


def get_mearec_h5_recordings_file(batch_uuid: str):
    """
    Returns the filepath to the MEArec .h5/.hdf5 recordings file for the given UUID.

    Assumes (and enforces) that exactly one data file is stored:
        ${ENDPOINT}/ephys/YYYY-MM-DD-e-[descriptor]/original/experiments/recordings_*.h5 ( or "recordings_*.hdf5")
        (ENDPOINT defaults to s3://braingeneers)
    """
    path = posixpath.join(get_basepath(), 'ephys', batch_uuid, 'original/data/')
    data_files = s3wrangler.list_objects(path=path, suffix=['.h5', '.hdf5'])
    # filter out the "templates_" prefix and any other unneeded provenance files
    h5_files = [f for f in data_files if f[len(path):].startswith('recordings_')]

    if len(h5_files) == 0:
        raise FileNotFoundError(f'No recordings_*.h5 / recordings_*.hdf5 files '
                                f'found in {get_basepath()}/ephys/{batch_uuid}/original/data/ !')

    if len(h5_files) > 1:
        raise FileNotFoundError(f'More than one recordings_*.h5 / recordings_*.hdf5 file was '
                                f'found in {get_basepath()}/ephys/{batch_uuid}/original/data/ !  '
                                f'Only one recordings_*.h5 / recordings_*.hdf5 file (and exactly one) '
                                f'per UUID for MEArec is currently supported.')
    return h5_files[0]


def generate_metadata_mearec(batch_uuid: str, n_threads: int = 16, save: bool = False):
    """
    Generates metadata.json from MEArec data on S3 from a standard UUID.

    Limitations:
     - timestamps are not taken from the original data files, the current time is used.

    :param batch_uuid: standard ephys UUID
    :param n_threads: Currently unused; number of concurrent file reads (useful for parsing many network based files)
    :param save: bool (default == False) saves the generated metadata file back to S3/ENDPOINT at batch_uuid
    :return: metadata_json: dict
    """
    h5_file = get_mearec_h5_recordings_file(batch_uuid)
    current_time = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%dT%H:%M:%S')

    with smart_open.open(h5_file, 'rb') as fid:
        with h5py.File(fid, "r") as f:
            sampling_frequency = f["info"]["recordings"]["fs"][()]
            if isinstance(sampling_frequency, bytes):
                sampling_frequency = sampling_frequency.decode("utf-8")
            elif isinstance(sampling_frequency, np.generic):
                sampling_frequency = sampling_frequency.item()

            num_channels = f['channel_positions'].shape[0]
            num_input_channels = num_channels - f['recordings'].shape[0]
            num_voltage_channels = num_channels - num_input_channels

            metadata = {
                'uuid': batch_uuid,
                'timestamp': current_time,
                'notes': {
                    'comments': 'This data is a simulated recording generated by MEArec.'
                },
                'ephys_experiments': {
                    "experiment0": {
                        "name": "experiment0",
                        "hardware": 'MEArec Simulated Recording',
                        "notes": 'This data is a simulated recording generated by MEArec.',
                        "timestamp": current_time,
                        'sample_rate': int(sampling_frequency),
                        'num_channels': num_channels,
                        'num_current_input_channels': num_input_channels,
                        'num_voltage_channels': num_voltage_channels,
                        'channels': list(range(num_channels)),
                        # the values "offset", "voltage_scaling_factor"/"gain, and "units" don't change in MEArec,
                        # and are currently hard-coded in their rawIO reader (so we do the same):
                        # units: https://github.com/NeuralEnsemble/python-neo/blob/354c8d9d5fbc4daad3547773d2f281f8c163d208/neo/rawio/mearecrawio.py#L97
                        # gain: https://github.com/NeuralEnsemble/python-neo/blob/354c8d9d5fbc4daad3547773d2f281f8c163d208/neo/rawio/mearecrawio.py#L98
                        # offset: https://github.com/NeuralEnsemble/python-neo/blob/354c8d9d5fbc4daad3547773d2f281f8c163d208/neo/rawio/mearecrawio.py#L99
                        'offset': 0,
                        'voltage_scaling_factor': 1,
                        'units': '\u00b5V',
                        'version': f.attrs.get("mearec_version", "0.0.0"),  # only included in MEArec since v1.5.0
                        'blocks': [{
                            "num_frames": f['recordings'].shape[1],
                            "path": h5_file,
                            "timestamp": current_time,
                            "data_order": "rowmajor"
                        }]
                    }
                }
            }

    if save:
        with smart_open.open(posixpath.join(get_basepath(), 'ephys', batch_uuid, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

    return metadata


def load_data_mearec(
        metadata: dict,
        batch_uuid: str,
        channels: Optional[Iterable[int]] = None,
        length: Optional[int] = None
) -> NDArray[Int16]:
    """
    Reads the MEArec data format and returns data for the selected channels, at the selected length.

    Note: MEArec data always has an offset of 0.
    Note: We currently assume (and enforce) that there is only one file (and this experiment) per MEArec UUID.

    :param batch_uuid: uuid, example: 2023-08-29-e-mearec-6cells-tetrode
    :param channels: a list of channels
    :param length: data read length in frames (e.g. data points, agnostic of channel count)
    """
    h5_file = metadata['ephys_experiments']['experiment0']['blocks'][0]['path']
    with smart_open.open(h5_file, 'rb') as fid:
        with h5py.File(fid, "r") as f:
            if channels is not None and length is not None:
                return np.array(f['recordings'][channels, :length])
            elif channels is not None:
                return np.array(f['recordings'][channels, :])
            else:
                return np.array(f['recordings'])


def modify_metadata_maxwell_raw_to_nwb(metadata_json: dict):
    """
    Given a Maxwell-based metadata dictionary, update key values to the current metadata structure, and replace
    raw Maxwell file paths (".raw.h5") with NWB file paths, if they exist.

    For example, the input:
      {"timestamp": "2023-08-12 T15:01:19",
       "ephys_experiments": {"experiment name 1": {"hardware": "Maxwell",
                                                   "blocks": [{"path": "original/data/data_GABA_BL_20325.raw.h5"}]}},
                            {"experiment name 2": {"hardware": "Maxwell",
                                                   "blocks": [{"path": "original/data/data_GABA_BL_20326.raw.h5"}]}}}

      would return:

      {"timestamp": "2023-08-12 T15:01:19",
       "hardware": "Maxwell",
       "ephys_experiments": {"experiment name 1": {"data_format": "NeurodataWithoutBorders",
                                                   "blocks": [{"path": "shared/data_GABA_BL_20325.nwb"}]}},
                            {"experiment name 2": {"data_format": "NeurodataWithoutBorders",
                                                   "blocks": [{"path": "shared/data_GABA_BL_20326.nwb"}]}}}

    Assuming that both "shared/data_GABA_BL_20325.nwb" and "shared/data_GABA_BL_20326.nwb" exist.
    """
    current_timestamp = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%dT%H:%M:%S')
    new_metadata_json = copy.deepcopy(metadata_json)
    new_metadata_json['timestamp'] = current_timestamp  # TODO: replace all timestamps?  i.e. experiments, blocks...

    for experiment_name, experiment_data in metadata_json['ephys_experiments'].items():
        if not new_metadata_json.get('hardware'):
            new_metadata_json['hardware'] = experiment_data['hardware']

        if experiment_data.get('hardware'):
            assert new_metadata_json['hardware'] == experiment_data['hardware']
            del new_metadata_json['ephys_experiments'][experiment_name]['hardware']

        for i, block in enumerate(experiment_data['blocks']):
            if block['path'].endswith('.raw.h5') and 'original/data/' in block['path']:
                nwb_filepath = block['path'][:-len('.raw.h5')] + '.nwb'
                nwb_filepath = 'shared/' + nwb_filepath.split('original/data/')[-1]
                full_s3_nwb_filepath = posixpath.join(get_basepath(), 'ephys', metadata_json['uuid'], nwb_filepath)
                if s3wrangler.does_object_exist(full_s3_nwb_filepath):
                    new_metadata_json['ephys_experiments'][experiment_name]['blocks'][i]['path'] = nwb_filepath
                    new_metadata_json['ephys_experiments'][experiment_name]['data_format'] = 'NeurodataWithoutBorders'
    return new_metadata_json


def generate_metadata_maxwell(batch_uuid: str, experiment_prefix: Optional[str] = None, n_threads: int = 16, save: bool = False):
    """
    Currently modifies metadata.json, if it already exists and is found for an associated Maxwell UUID.

    TODO: Generates a new metadata.json if it doesn't already exist from raw Maxwell data files on S3 from a standard UUID.
      Assume raw data files are stored in:
        ${ENDPOINT}/ephys/YYYY-MM-DD-e-[descriptor]/original/experiments/*.raw.h5
        (ENDPOINT defaults to s3://braingeneers)

    Raise a NotImplemented exception if no metadata.json is found.

    Limitations:
     - timestamps are not taken from the original data files, the current time is used.

    :param batch_uuid: standard ephys UUID
    :param experiment_prefix: Unused currently.
    :param n_threads: Unused currently.  Number of concurrent file reads (useful for parsing many network based files).
    :param save: bool (default == False) saves the generated metadata file back to S3/ENDPOINT at batch_uuid

    :return: (metadata_json: dict, ephys_experiments: dict) a tuple of two dictionaries which are
        json serializable to metadata.json and experiment1.json.
    """
    try:
        with smart_open.open(posixpath.join(get_basepath(), 'ephys', batch_uuid, 'metadata.json'), 'r') as f:
            metadata_json = json.load(f)
    except OSError as e:
        if 'error occurred (NoSuchKey)' in str(e) or '[Errno 2] No such file or directory' in str(e):
            raise NotImplementedError(f'This function did not find a metadata.json for {batch_uuid}, and '
                                      f'can only modify an existing metadata.json.')
        raise

    metadata_json = modify_metadata_maxwell_raw_to_nwb(metadata_json)

    if save:
        with smart_open.open(posixpath.join(get_basepath(), 'ephys', batch_uuid, 'metadata.json'), 'w') as f:
            json.dump(metadata_json, f, indent=2)

    return metadata_json
