import os
import json
import requests
import matplotlib.pyplot as plt
import numpy as np
import shutil
import h5py
import scipy.io
import braingeneers.utils.smart_open as smart_open
from os import walk
import braingeneers.utils.s3wrangler as wr
from deprecated import deprecated


def get_archive_path():
    """/public/groups/braingeneers/ephys  Return path to archive on the GI public server """
    return os.getenv("BRAINGENEERS_ARCHIVE_PATH", "/public/groups/braingeneers/ephys")


def get_archive_url():
    """  https://s3.nautilus.optiputer.net/braingeneers/ephys     Return URL to archive on PRP """
    return "{}/ephys".format(os.getenv("BRAINGENEERS_ARCHIVE_URL", "s3://braingeneers"))


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


def load_files_axion(metadata, batch_uuid, experiment_num, start, stop):
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
    # scipy.io.loadmat('axion_data_file')
    raise Exception('Axion data loading function not implemented')
    return


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


@deprecated(reason='Deprecated as a result of deprecating load_blocks')
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


def load_data(batch_uuid, experiment_num, offset=0, length=-1, channels=None):
    """
    This function loads arbitrarily specified amounts of data from an experiment for Axion, Intan, Raspi, and Maxwell.
    As a note, the user MUST provide the offset (offset) and the length of frames to take. This function reads
    across as many blocks as the length specifies.
    :param batch_uuid: str
        UUID of batch of experiments within the Braingeneers's archive
    :param experiment_num: int
        Which experiment in the batch to load.
    :param channels: list of int
        Channels of interest to obtain data from (default None)
    :param offset: int
        Starting datapoint of interest
    :param length: int
        Range indicator of number of datapoints to take (default -1 meaning all datapoints)
    :return:
    dataset: nparray
        Array of chosen datapoints.
    """
    metadata = load_experiment(batch_uuid, experiment_num)

    # hand off to correct load_data helper
    if "hardware" in metadata:
        if "Raspi" in metadata["hardware"]:
            dataset = load_data_raspi(metadata, batch_uuid, experiment_num, offset, length)
        elif "Axion" in metadata["hardware"]:
            dataset = load_data_axion(metadata, batch_uuid, experiment_num, offset, length)
        elif "Intan" in metadata["hardware"]:
            dataset = load_data_intan(metadata, batch_uuid, experiment_num, offset, length)
        elif "Maxwell" in metadata["hardware"]:
            dataset = load_data_maxwell(metadata, batch_uuid, experiment_num, channels, offset,
                                                        length)
        else:
            raise Exception('hardware field in metadata.json must contain keyword Axion, Raspi, Intan, or Maxwell')
    else:  # assume intan
        dataset = load_data_intan(metadata, batch_uuid, experiment_num, offset, length)

    return dataset


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


def load_data_axion(metadata, batch_uuid, experiment_num, offset, length):
    """

    :param metadata:
    :param batch_uuid:
    :param experiment_num:
    :param offset:
    :param length:
    :return:
    """
    raise NotImplementedError


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
        objs = wr.list_objects(data_dir)  # if on s3
    except:
        objs = next(walk(data_dir), (None, None, []))[2]  # if on local dir
    return objs


def connect_kach_dir():
    global maxone_wetai_kach_dir
    maxone_wetai_kach_dir = "/home/jovyan/Projects/maxwell_analysis/ephys/"
    if not os.path.exists(maxone_wetai_kach_dir):
        os.makedirs(maxone_wetai_kach_dir)
    return maxone_wetai_kach_dir


# will download nonexistant batches into the ./ephys dir on the wetAI ephys maxwell analyzer
# Usage. Supply the batch_name for the batch you would like to download.
# If not downloaded into the local kach dir, it will be downloaded to improve loading time
def sync_s3_to_kach(batch_name):
    sync_command = f"aws --endpoint $ENDPOINT_URL s3 sync s3://braingeneers/ephys/" + batch_name + f" /home/jovyan/Projects/maxwell_analysis/ephys/" + batch_name
    os.system(sync_command)
