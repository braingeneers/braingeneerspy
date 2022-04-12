import os
import json
import requests
import matplotlib.pyplot as plt
import numpy as np
import shutil
import h5py
import braingeneers.utils.smart_open_braingeneers as smart_open
from os import walk
from deprecated import deprecated
from collections import namedtuple
import datetime
import time
from braingeneers.utils import s3wrangler
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Union, Iterable, Iterator
import io
import braingeneers
from collections import OrderedDict
import sortedcontainers
import itertools


# todo braingeneerspy won't install because of setup.py
# todo make all tests cases work
# todo implement hengenlab metadata generator
# todo implement load_data for hengenlab
# todo name based or 0-based indexing for access to experimentN.json
# todo update existing datasets metadata json files on S3


@deprecated('Will be removed in the future, use braingeneers.utils.smart_open_braingeneers instead')
def get_archive_path():
    """/public/groups/braingeneers/ephys  Return path to archive on the GI public server """
    return os.getenv("BRAINGENEERS_ARCHIVE_PATH", "/public/groups/braingeneers/ephys")


@deprecated('Will be removed in the future, use braingeneers.utils.smart_open_braingeneers instead')
def get_archive_url():
    """  https://s3.nautilus.optiputer.net/braingeneers/ephys     Return URL to archive on PRP """
    return "{}/ephys".format(os.getenv("BRAINGENEERS_ARCHIVE_URL", "s3://braingeneers"))


@deprecated('Use load_metadata(batch_uuid, experiment_nums) instead.')
def load_batch(batch_uuid):
    """
    Load the metadata for a batch of experiments and return as a dict
    Parameters
    ----------
    batch_uuid : str
        UUID of batch of experiments within the Braingeneer's archive'
        Example: 2019-02-15, or d820d4a6-f59a-4565-bcd1-6469228e8e64
    """
    try:
        full_path = "{}/{}/metadata.json".format(get_archive_path(), batch_uuid)
        if not os.path.exists(full_path):
            full_path = "{}/{}/metadata.json".format(get_archive_url(), batch_uuid)

        with smart_open.open(full_path, "r") as f:
            return json.load(f)
    except OSError:
        raise OSError('Are you sure ' + batch_uuid + ' is the right uuid?')


@deprecated('Use load_metadata(batch_uuid, experiment_nums) instead.')
def load_experiment(batch_uuid, experiment_num):
    """
    Load metadata from PRP S3 for a single experiment
    Parameters
    ----------
    batch_uuid : str
        UUID of batch of experiments within the Braingeneer's archive'
    experiment_num : int
        Which experiment in the batch to load
    Returns
    -------
    metadata : dict
        All of the metadata associated with this experiment
    """
    batch = load_batch(batch_uuid)
    try:
        exp_full_path = "{}/{}/original/{}".format(get_archive_path(), batch_uuid, batch['experiments'][experiment_num])
        if not os.path.exists(exp_full_path):
            exp_full_path = "{}/{}/original/{}".format(get_archive_url(), batch_uuid,
                                                       batch['experiments'][experiment_num])

        with smart_open.open(exp_full_path, "r") as f:
            return json.load(f)
    except OSError:
        raise OSError('Are you sure experiment number ' + str(experiment_num) + ' exists?')


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

    metadata_full_path = os.path.join(base_path, 'ephys', batch_uuid, 'metadata.json')
    with smart_open.open(metadata_full_path, 'r') as f:
        metadata = json.load(f)

    # make 'ephys-experiments' a sorted dict indexable by experiment name if the key exists
    if 'ephys-experiments' in metadata:
        metadata['ephys-experiments'] = {e['name']: e for e in metadata['ephys-experiments']}

    return metadata


@deprecated(reason='Deprecated as a result of deprecating load_blocks, use load_data')
def load_files_raspi(metadata, batch_uuid, experiment_num, start, stop):
    """
    Load signal blocks of data from a single experiment
    Parameters
    ----------
    batch_uuid : str
        UUID of batch of experiments within the Braingeneer's archive'
    experiment_num : int
        Which experiment in the batch to load
    start : int, optional
        First rhd data block to return
    stop : int, optional
        Last-1 rhd data block to return
    Returns
    -------
    X : ndarray
        Numpy matrix of shape frames, channels
    t : ndarray
        Numpy array with time in milliseconds for each frame
    fs : float
        Sample rate in Hz
    """

    def _load_path(path):
        with open(path, "rb") as f:
            # f.seek(8, os.SEEK_SET)
            return np.fromfile(f, dtype='>i2', count=-1, offset=int(metadata['offset'] / 32))

    def _load_url(url):
        with np.DataSource(None).open(url, "rb") as f:
            # f.seek(8, os.SEEK_SET)
            return np.fromfile(f, dtype='>i2', count=-1, offset=int(metadata['offset'] / 32))

    print("Loading file Raspi... start:", start, " stop:", stop)
    # Load all the raw files into a single matrix
    if os.path.exists("{}/{}/derived/".format(get_archive_path(), batch_uuid)):
        # Load from local archive
        # filename = "{}/{}/derived/{}".format(get_archive_path(), batch_uuid, metadata["blocks"][0]["path"])
        # raw_data = np.fromfile(filename, dtype='>i2', count=-1, offset=metadata['offset']/32)
        raw_data = np.concatenate([
            _load_path("{}/{}/derived/{}".format(get_archive_path(), batch_uuid, s["path"]))
            for s in metadata["blocks"][start:stop]], axis=0)
    else:
        # Load from PRP S3
        # url = "{}/{}/derived/{}".format(get_archive_url(), batch_uuid, metadata["blocks"][0]["path"])
        # with np.DataSource(None).open(url, "rb") as f:
        #        raw_data = np.fromfile(f, dtype='>i2', count=-1, offset=metadata['offset']/32)
        raw_data = np.concatenate([
            _load_url("{}/{}/derived/{}".format(get_archive_url(), batch_uuid, s["path"]))
            for s in metadata["blocks"][start:stop]], axis=0)

    # throw away last partial frame
    max_index = (len(raw_data) // metadata["num_channels"]) * metadata["num_channels"]
    raw_data = raw_data[:max_index]

    # scale the data
    if "scaler" in metadata:
        scale = metadata["scaler"]  # legacy
    else:
        scale = metadata["voltage_scaling_factor"]
    uV_data = raw_data * scale

    # shape data into (frames, channels) (x, y)
    X = np.reshape(uV_data, (-1, metadata["num_channels"]), order="C")

    # S ampling rate
    fs = metadata["sample_rate"]  # Hz

    # Time
    T = (1.0 / fs) * 1000  # ms (given fs in Hz)
    start_t = 0
    end_t = start_t + (T * X.shape[0])
    t = np.linspace(start_t, end_t, X.shape[0], endpoint=False)  # MILISECOND_TIMESCALE
    assert t.shape[0] == X.shape[0]

    return X, t, fs


@deprecated(reason='Deprecated as a result of deprecating load_blocks, use load_data')
def load_files_intan(metadata, batch_uuid, experiment_num, start, stop):
    """
    Load signal blocks of data from a single experiment
    Parameters
    ----------
    batch_uuid : str
        UUID of batch of experiments within the Braingeneer's archive'
    experiment_num : int
        Which experiment in the batch to load
    start : int, optional
        First rhd data block to return
    stop : int, optional
        Last-1 rhd data block to return
    Returns
    -------
    X : ndarray
        Numpy matrix of shape frames, channels
    t : ndarray
        Numpy array with time in milliseconds for each frame
    fs : float
        Sample rate in Hz
    """

    def _load_path(path):
        with open(path, "rb") as f:
            f.seek(8, os.SEEK_SET)
            return np.fromfile(f, dtype=np.int16)

    def _load_url(url):
        with np.DataSource(None).open(url, "rb") as f:
            f.seek(8, os.SEEK_SET)
            return np.fromfile(f, dtype=np.int16)

    print("Loading file Intan... start:", start, " stop:", stop)
    # Load all the raw files into a single matrix
    if os.path.exists("{}/{}/derived/".format(get_archive_path(), batch_uuid)):
        # Load from local archive
        raw = np.concatenate([
            _load_path("{}/{}/derived/{}".format(get_archive_path(), batch_uuid, s["path"]))
            for s in metadata["blocks"][start:stop]], axis=0)
    else:
        # Load from PRP S3
        raw = np.concatenate([
            _load_url("{}/{}/derived/{}".format(get_archive_url(), batch_uuid, s["path"]))
            for s in metadata["blocks"][start:stop]], axis=0)
        print('Just ignore all the stuff in the pink rectangle.')

    # Get scale value for data from new or legacy format
    if "scaler" in metadata:
        scale = metadata["scaler"]  # legacy
    else:
        scale = metadata["voltage_scaling_factor"]

    # Reshape interpreting as row major
    X = raw.reshape((-1, metadata["num_channels"]), order="C")
    # Convert from the raw uint16 into float "units" via "offset" and "scale"
    X = np.multiply(scale, (X.astype(np.float32) - metadata["offset"]))

    # Extract sample rate for first channel and construct a time axis in ms
    fs = metadata["sample_rate"]

    start_t = (1000 / fs) * sum([s["num_frames"] for s in metadata["blocks"][0:start]])
    end_t = (1000 / fs) * sum([s["num_frames"] for s in metadata["blocks"][0:stop]])
    t = np.linspace(start_t, end_t, X.shape[0], endpoint=False)
    assert t.shape[0] == X.shape[0]

    return X, t, fs


@deprecated(reason='Deprecated as a result of deprecating load_blocks, use load_data')
def load_files_maxwell(metadata, batch_uuid, experiment_num, channels, start, stop):
    """
        Load signal blocks of data from a single experiment
        Parameters
        ----------
        metadata : json object
            Holds the json that details a specific experiment.
        batch_uuid : str
            UUID of batch of experiments within the Braingeneer's archive'
        channels : list of int
            List of picked channels
        start : int, optional
            First rhd data block to return
        stop : int, optional
            Last-1 rhd data block to return
        f_start : int
            starting frame to take data from (default 0)
        f_end : int
            ending frame to take data from (default None)
        Returns
        -------
        X : ndarray
            Numpy matrix of shape frames, channels
        t : ndarray
            Numpy array with time in milliseconds for each frame
        fs : float
            Sample rate in Hz
        """
    # show what's happening first
    print("Loading file Maxwell... start:", start, " stop:", stop)

    # as a note, load_experiment must have run successfully for execution to get here.
    # the json is stored in metadata, which was passed in already.
    # for robustness, maybe show all the h5 files present so it can be picked?
    # make sure start/stop have a non-None value

    # should this whole process be in a loop to get every file?

    # get name of experiment
    filename = metadata['blocks'][0]['path'].split('/')[-1]
    # datafile is the data we're interested in
    datafile = '{}/{}/original/data/{}'.format(get_archive_url(), batch_uuid, filename)

    # ask David Parks about this part, and if it should be edited throughout
    if "scaler" in metadata:
        scale = metadata["scaler"]  # legacy
    else:
        scale = metadata["voltage_scaling_factor"]

    with smart_open.open(datafile, 'rb') as file:
        with h5py.File(file, 'r', libver='latest', rdcc_nbytes=2 ** 25) as h5file:
            print("Keys: %s" % h5file.keys())
            print(h5file['sig'].shape)
            # know that there are 1028 channels which all record and make 'num_frames'
            # thus, cannot use the values passed in for start and stop because they're indicating blocks. it would conflict
            # with the assert
            sig = h5file['sig']
            # if stop is unset, have it be the end of all the blocks
            if stop is None:
                stop = len(metadata['blocks'])
            # if channels has values, pick only those.
            if channels is not None:
                data = sig[channels, :]
            # otherwise, pick them all
            else:
                data = sig[:, :]

    X = np.transpose(data)
    X = np.multiply(scale, X.astype(np.float32))

    fs = metadata["sample_rate"]

    # to get time in milliseconds for each frame, it would just be total num_frames / channel number / sampling rate

    start_t = (1000 / fs) * sum([s["num_frames"] for s in metadata["blocks"][0:start]])
    print('start_t: ', start_t)
    end_t = (1000 / fs) * sum([s["num_frames"] for s in metadata["blocks"][0:stop]])
    print('end_t: ', end_t)
    t = np.linspace(start_t, end_t, X.shape[0], endpoint=False)

    return X, t, fs


@deprecated(reason='Use load_data function instead.')
def load_blocks(batch_uuid, experiment_num, channels=None, start=0, stop=None):
    """
    Load signal blocks of data from a single experiment for Axion, Intan, Raspi, and Maxwell
    Parameters
    ----------
    batch_uuid : str
        UUID of batch of experiments within the Braingeneer's archive'
    experiment_num : int
        Which experiment in the batch to load
    channels : list of ints
        A list of channel numbers to return
    start : int, optional
        First rhd data block to return
    stop : int, optional
        Last-1 rhd data block to return
    Returns
    -------
    X : ndarray
        Numpy matrix of shape frames, channels. Raw electrical voltages being measured.
    t : ndarray
        Numpy array with timestamps in milliseconds for each frame.
    fs : float
        Sample rate in Hz. How quickly the samples are taken.
    """
    metadata = load_experiment(batch_uuid, experiment_num)
    assert start >= 0 and start < len(metadata["blocks"])
    assert not stop or stop >= 0 and stop <= len(metadata["blocks"])
    assert not stop or stop > start
    # if channels is 'None', return all channels
    if ("hardware" in metadata):
        if ("Raspi" in metadata["hardware"]):
            X, t, fs = load_files_raspi(metadata, batch_uuid, experiment_num, start, stop)
        elif ("Axion" in metadata["hardware"]):
            X, t, fs = load_files_axion(metadata, batch_uuid, experiment_num, start, stop)
        elif ("Intan" in metadata["hardware"]):
            X, t, fs = load_files_intan(metadata, batch_uuid, experiment_num, start, stop)
        elif ("Maxwell" in metadata["hardware"]):
            X, t, fs = load_files_maxwell(metadata, batch_uuid, experiment_num, channels, start, stop)
        else:
            raise Exception('hardware field in metadata.json must contain keyword Axion, Raspi, Intan, or Maxwell')
    else:  # assume intan
        X, t, fs = load_files_intan(metadata, batch_uuid, experiment_num, start, stop)

    return X, t, fs


def load_data(metadata: dict, experiment: Union[str, int], offset: int = 0, length: int = None,
              channels: Union[List[int], int] = None):
    """
    This function loads arbitrarily specified amounts of data from an experiment for Axion, Intan, Raspi, and Maxwell.
    As a note, the user MUST provide the offset (offset) and the length of frames to take. This function reads
    across as many blocks as the length specifies.

    :param metadata: result of load_metadata, dict
    :param experiment: Which experiment in the batch to load, by name (string) or index location (int),
        examples: "experiment1" or 0
    :param channels: Channels of interest to obtain data from (default None for all channels) list of ints or single int
    :param offset: int Starting datapoint of interest, 0-indexed from start of
        recording across all data files and channels-agnostic
    :param length: int, required, number of data points to return (-1 meaning all data points)
    :return: ndarray array of chosen data points in [channels, time] format.
    """
    assert 'uuid' in metadata, \
        'Metadata file is invalid, it does not contain required uuid field.'
    assert 'ephys-experiments' in metadata, \
        'Metadata file is invalid, it does not contain required ephys-experiments field.'
    assert isinstance(experiment, (str, int)), \
        f'Parameter experiment must be an int index or experiment name string. Got: {experiment}'

    experiment_name = list(metadata['ephys-experiments'].keys())[experiment] if isinstance(experiment, int) else experiment
    batch_uuid = metadata['uuid']
    hardware = metadata['ephys-experiments'][experiment_name]['hardware']

    # hand off to appropriate load_data function
    if 'Raspi' in hardware:
        data = load_data_raspi(metadata, batch_uuid, experiment_name, offset, length)  # todo
    elif 'Axion' in hardware:
        data = load_data_axion(metadata, batch_uuid, experiment_name, channels, offset, length)
    elif 'Intan' in hardware:
        data = load_data_intan(metadata, batch_uuid, experiment_name, offset, length)  # todo
    elif 'Maxwell' in hardware:
        data = load_data_maxwell(metadata, batch_uuid, experiment_name, channels, offset, length)
    elif 'Hengenlab' in hardware:
        data = load_data_hengenlab(metadata, batch_uuid, experiment_name, channels, offset, length)  # todo
    else:
        raise AttributeError(f'Metadata file contains invalid hardware field: {metadata["hardware"]}')

    return data


def load_data_raspi(metadata, batch_uuid, experiment_num, offset, length):
    """

    :param metadata:
    :param batch_uuid:
    :param experiment_num:
    :param offset:
    :param length:
    :return:
    """
    raise NotImplementedError


def load_data_axion(metadata: dict, batch_uuid: str, experiment_name: int,
                    channels: Iterable[int], offset: int, length: int):
    """
    Reads from Axion raw data format using Python directly (not going through matlab scripts first).
    Reader originally written by Dylan Yong, updated by David Parks.

    Limitations:
     - Axion reader must read all channels, it then filters those to selected channels after reading all channel data.

    :param metadata: result of load_experiment
    :param batch_uuid: uuid, example: 2020-07-06-e-MGK-76-2614-Wash
    :param experiment_name: experiment name under 'ephys-experiments'
    :param channels: a list of channels
    :param offset: data offset (spans full experiment, across files)
    :param length: data read length in frames (e.g. data points, agnostic of channel count)
    :return: ndarray in (channels, frames) format where channels 0:64 are well A1, 64:128 are well A2, etc.
    """
    data_multi_file = []

    # get subset of files to read from metadata.blocks
    metadata_offsets_cumsum = np.cumsum([
        block['num_frames'] for block in metadata['ephys-experiments'][experiment_name]['blocks']
    ])
    block_ixs_range = np.minimum(
        len(metadata_offsets_cumsum),
        np.searchsorted(metadata_offsets_cumsum, [offset, offset + length], side='right')
    )
    block_ixs = list(range(block_ixs_range[0], block_ixs_range[1] + 1))
    assert len(block_ixs) > 0, \
        f'No data file found starting from offset {offset}, is this past the end of the data file?'

    # this is a back reference which was constructed this way to avoid duplicating the large channel map many times
    channel_map_key = metadata['ephys-experiments'][experiment_name]['blocks'][0]['axion_channel_map_key']

    # perform N read operations accumulating results in data_multi_file
    frame_read_count = 0  # counter to track number of frames read across multiple files
    for block_ix in block_ixs:
        block = metadata['ephys-experiments'][experiment_name]['blocks'][block_ix]
        file_name = block['path']
        full_file_path = f's3://braingeneers/ephys/{batch_uuid}/original/data/{file_name}'
        sample_start = max(0, offset - metadata_offsets_cumsum[block_ix] + block['num_frames'])
        data_length = min(block['num_frames'], length - frame_read_count)
        data_ndarray = _axion_get_data(
            file_name=full_file_path,
            file_data_start_position=block['axion_data_start_position'],
            sample_offset=sample_start,
            sample_length=data_length,
            num_channels=metadata['ephys-experiments'][experiment_name]['num_channels'],
            corrected_map=metadata[channel_map_key],
        )

        # select channels
        channels = list(channels) if isinstance(channels, Iterable) else [channels]
        data_ndarray_select_channels = data_ndarray[channels, :] if channels is not None else data_ndarray

        # append data from this block/file to list
        data_multi_file.append(data_ndarray_select_channels)
        frame_read_count += data_ndarray_select_channels.shape[1]

    # concatenate the results and return
    data_concat = np.concatenate(data_multi_file, axis=1)

    # apply scaling factor
    voltage_scaling_factor = metadata['ephys-experiments'][experiment_name]['voltage_scaling_factor']
    data_scaled = data_concat * voltage_scaling_factor

    return data_scaled


def load_data_intan(metadata, batch_uuid, experiment_num, offset, length):
    """

    :param metadata:
    :param batch_uuid:
    :param experiment_num:
    :param offset:
    :param length
    """
    raise NotImplementedError


def load_data_maxwell(metadata, batch_uuid, experiment_num, channels, offset, length):
    """
    Loads specified amount of data from one block
    :param metadata: json file
        JSON file containing metadata of experiment
    :param batch_uuid: str
        UUID of batch of experiments within the Braingeneers's archive
    :param experiment_num: int
        Number of experiment
    :param channels: list of int
        Channels of interest
    :param offset: int
        Starting frame (offset) of the datapoints to use
    :param length: int
        Length of datapoints to take
    :return:
    dataset: nparray
        Dataset of datapoints.
    fs : float
        Sample rate in Hz. How quickly the samples are taken.
    num_frames : int
        number of frames being taken
    """
    if length == -1:
        print(
            f"Loading file Maxwell, UUID {batch_uuid}, experiment number {experiment_num}, frame {offset} to end of file....")
    else:
        print(
            f"Loading file Maxwell, UUID {batch_uuid}, experiment number {experiment_num}, frame {offset} to {offset + length}....")
    # get datafile

    filename = metadata['blocks'][0]['path'].split('/')[-1]
    datafile = '{}/{}/original/data/{}'.format(get_archive_url(), batch_uuid, filename)
    fs = metadata['sample_rate']

    # keep in mind that the range is across all channels. So, num_frames from the metadata is NOT the correct range.
    # Finding the block where the datapoints start
    start_block = 0
    # end_block = len(metadata['blocks']) - 1
    for index in range(len(metadata['blocks'])):
        # if the offset is lower than the upper range of frames in a block, break out
        if offset < metadata['blocks'][index]['num_frames'] / metadata['num_channels']:
            start_block = index
            break
        # otherwise, keep finding the block where the offset lies
        else:
            offset -= metadata['blocks'][index]['num_frames'] / metadata['num_channels']
    # Finding block where the datapoints end
    # if length is -1, read in all the frames from all blocks
    if length == -1:
        end_block = len(metadata['blocks']) - 1
        frame_end = 0
        # add up all the frames divided by their channel number
        for block in metadata['blocks']:
            frame_end += block['num_frames'] / metadata['num_channels']
        frame_end = int(frame_end)
        #frame_end /= metadata['num_channels']
    else:
        frame_end = offset + length
        for index in range(start_block, len(metadata['blocks'])):
            if (offset + length) < metadata['blocks'][index]['num_frames'] / metadata['num_channels']:
                end_block = index
                break
            else:
                offset -= metadata['blocks'][index]['num_frames'] / metadata['num_channels']
    assert end_block < len(metadata['blocks'])
    # now, with the starting block, ending block, and frames to take, take those frames and put into nparray.
    # open file
    with smart_open.open(datafile, 'rb') as file:
        with h5py.File(file, 'r', libver='latest', rdcc_nbytes=2 ** 25) as h5file:
            # know that there are 1028 channels which all record and make 'num_frames'
            sig = h5file['sig']
            # make dataset of chosen frames
            if channels is not None:
                num_frames = len(channels) * (frame_end - offset)
                dataset = sig[channels, offset:frame_end]
            else:
                num_frames = metadata['num_channels'] * (frame_end - offset)
                dataset = sig[:, offset:frame_end]
    dataset = dataset.astype(np.float32)
    return dataset


def load_data_hengenlab(metadata: dict, batch_uuid: str, experiment_num: int,
                        channels: Iterable[int], offset: int, length: int):
    pass  # todo


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
        objs = next(walk(data_dir), (None, None, []))[2]  # if on local dir
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
    sync_command = f"aws --endpoint $ENDPOINT_URL s3 sync s3://braingeneers/ephys/" + batch_name + f" /home/jovyan/Projects/maxwell_analysis/ephys/" + batch_name
    os.system(sync_command)


def generate_metadata_hengenlab(dataset_name: str):
    """
    Generates a metadata json and experiment1...experimentN jsons for a hengenlab dataset upload.

    File locations in S3 for hengenlab neural data files:
        s3://braingeneers/ephys/YYYY-MM-DD-e-${DATASET_NAME}/original/data/*.bin

    Contiguous recording periods

    :param dataset_name: the dataset_name as defined in `neuraltoolkit`. Metadata will be pulled from `neuraltoolkit`.
    :return:
    """
    pass  # todo


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


def generate_metadata_axion(batch_uuid: str, experiment_prefix: str = '', n_threads: int = 16):
    """
    Generates metadata.json raw Axion data files on S3 from a standard UUID. Assumes raw data files are stored in:

        s3://braingeneers/ephys/YYYY-MM-DD-e-[descriptor]/original/experiments/*.raw

    Raises an exception if no data files were found at the expected location.

    All Axion recordings are assumed to be a single experiment.

    Limitations:
     - timestamps are not taken from the original data files, the current time is used.

    :param batch_uuid: standard ephys UUID
    :param experiment_prefix: Experiments are named "A1", "A2", ..., "B1", "B2", etc. If multiple recordings are
        included in a UUID the experiment name can be prefixed, for example "recording1_A1", "recording2_A1"
        for separate recordings. It is suggested to end the prefix with "_" for readability.
    :param n_threads: number of concurrent file reads (useful for parsing many network based files)
    :return: (metadata_json: dict, ephys_experiments: dict) a tuple of two dictionaries which are
        json serializable to metadata.json and experiment1.json.
    """
    metadata_json = {}
    ephys_experiments = OrderedDict()
    current_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%dT%H:%M:%S')

    # construct metadata_json
    metadata_json['issue'] = ''
    metadata_json['notes'] = ''
    metadata_json['timestamp'] = current_timestamp
    metadata_json['uuid'] = batch_uuid

    # list raw data files at batch_uuid
    list_of_raw_data_files = sorted(s3wrangler.list_objects(
        path=f's3://braingeneers/ephys/{batch_uuid}/original/data/',
        suffix='.raw',
    ))
    if len(list_of_raw_data_files) == 0:
        raise FileNotFoundError(f'No raw data files found at s3://braingeneers/ephys/{batch_uuid}/original/data/*.raw')

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

                block = dict()
                block['num_frames'] = data_length
                block['path'] = os.path.basename(raw_data_file)
                block['timestamp'] = current_timestamp
                block['axion_data_start_position'] = data_start
                block['axion_channel_map_key'] = f'{experiment_prefix}axion_per_well_channel_map'

                experiment.setdefault('blocks', [])
                experiment['blocks'].append(block)

    metadata_json['ephys-experiments'] = list(ephys_experiments.values())

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
                    print('Unexpected Channel array length')

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
    :param file_name: file name
    :param file_data_start_position: data start position in file as returned by metadata
    :param sample_offset: number of frames to skip from beginning of file
    :param sample_length: length of data section in frames (agnostic of channel count)
    :param num_channels:
    :param corrected_map: mapping to correctly rearrange columns
    :param well_layout: row, col of well layout
    :param electrode_layout: row, col of electrode layout
    :return:
    """
    with smart_open.open(file_name, 'rb') as fid:
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

        # # print(temp_raw_data_reshaped.shape)
        # final_raw_data_reshaped = temp_raw_data_reshaped.copy()
        #
        # # reformatting the data to match the data map
        # # variable is called correctedMap
        # for wellIdx, well in enumerate(corrected_map):
        #     for colIdx, column in enumerate(well):
        #         final_idx = wellIdx * 64 + colIdx  # todo calc 64 from electrode layout
        #         final_raw_data_reshaped[final_idx] = temp_raw_data_reshaped[column]

        # A1 = (1,1) 0-63
        # A2 = (1,2) 64-127
        # A3 = (1,3) 128-191
        # B1 = (2,1) 192-255
        # B2 = (2,2) 256-319
        # B3 = (2,3) 320-383
        return final_raw_data_reshaped


# class IndexedList(list):
#     """
#     A variant of OrderedDict indexable by index (int) or name (str).
#     This class forces ints to represent index by location, else index by name/object.
#     Example usages:
#         metadata['ephys-experiments']['experiment0']    # index by name (must use str type)
#         metadata['ephys-experiments'][0]                # index by location (must use int type)
#     """
#
#     def __init__(self, original_list: list, key: callable):
#         self.keys_ordered = [key(v) for v in original_list]
#         self.dict = {key(v): v for v in original_list}
#         super().__init__()
#
#     def __getitem__(self, key):
#         print(key)
#         if isinstance(key, int):
#             return self.dict[self.keys_ordered[key]]
#         elif isinstance(key, str):
#             return self.dict[key]
#         else:
#             raise KeyError(f'Key must be type int (index by location) or str (index by name), got type: {type(key)}')
#
#     def __iter__(self) -> Iterator:
#         def g():
#             for k in self.keys_ordered:
#                 yield self.dict[k]
#
#         return g()
#
#     def __hash__(self):
#         return self.dict.__hash__()
#
#     def __eq__(self, other):
#         return isinstance(other, IndexedList) and self.dict.__eq__(other.dict)
#
#     def __add__(self, value):
#         self.keys_ordered.append(value)
#         self.dict[value] = value
