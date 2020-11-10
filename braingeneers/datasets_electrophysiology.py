import os
import json
import requests
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import shutil

def get_archive_path():
    """/public/groups/braingeneers/ephys  Return path to archive on the GI public server """
    return os.getenv("BRAINGENEERS_ARCHIVE_PATH", "/public/groups/braingeneers/ephys")

def get_archive_url():
    """  https://s3.nautilus.optiputer.net/braingeneers/ephys     Return URL to archive on PRP """
    return "{}/braingeneers/ephys".format(
        os.getenv("AWS_S3_ENDPOINT", "https://s3.nautilus.optiputer.net"))

def load_batch(batch_uuid):
    """
    Load the metadata for a batch of experiments and return as a dict
    Parameters
    ----------
    batch_uuid : str
        UUID of batch of experiments within the Braingeneer's archive'
        Example: 2019-02-15, or d820d4a6-f59a-4565-bcd1-6469228e8e64
    """
    full_path = "{}/{}/derived/metadata.json".format(get_archive_path(), batch_uuid)
    if os.path.exists(full_path):
        with open(full_path, "r") as f:
            return json.load(f)

    full_path_for_prp = "{}/{}/derived/metadata.json".format(get_archive_url(), batch_uuid)
    r = requests.get(full_path_for_prp)
    if r.ok:
        return r.json()
    else:
        print(full_path_for_prp)
        raise Exception('Path:', full_path_for_prp, '\nAre you sure '+ batch_uuid + ' is the right uuid?')

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
    full_path = "{}/{}/derived/{}".format(
        get_archive_path(), batch_uuid,batch["experiments"][experiment_num])

    if os.path.exists(full_path):
        with open(full_path, "r") as f:
            return json.load(f)

    # Each experiment has a metadata file with all *.rhd headers and other sample info

    full_path_for_prp = str(get_archive_url()) + '/' + batch_uuid +'/derived/' +  batch["experiments"][experiment_num]
    r = requests.get(full_path_for_prp)
    print("Full path for PRP:", full_path_for_prp)
    if r.ok:
        return r.json()
    else:
        print("Full path for PRP:", full_path_for_prp)
        raise Exception('Are you sure '+ str(experiment_num)+ ' an experiment number?')

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
    # Load all the raw files into a single matrix
    if os.path.exists("{}/{}/derived/".format(get_archive_path(), batch_uuid)):
        # Load from local archive
        filename = "{}/{}/derived/{}".format(get_archive_path(), batch_uuid, metadata["blocks"][0]["path"])
        raw_data = np.fromfile(filename, dtype='>i2', count=-1, offset=4)

    else:
        # Load from PRP S3
        url = "{}/{}/derived/{}".format(get_archive_url(), batch_uuid, metadata["blocks"][0]["path"])
        with np.DataSource(None).open(url, "rb") as f:
                raw_data = np.fromfile(f, dtype='>i2', count=-1, offset=4)


    #throw away last partial frame
    max_index = (len(raw_data)//metadata["num_channels"])*metadata["num_channels"]
    raw_data = raw_data[:max_index]

    #scale the data
    uV_data= raw_data*metadata["scaler"]

    #shape data into (frames, channels) (x, y)
    X = np.reshape(uV_data, (-1, metadata["num_channels"]),  order="C")

    #S ampling rate
    fs = metadata["sample_rate"] #Hz

    # Time
    T = (1.0/fs)*1000 #ms (given fs in Hz)
    start_t = 0
    end_t = start_t + (T * X.shape[0])
    t = np.linspace(start_t, end_t, X.shape[0], endpoint=False) #MILISECOND_TIMESCALE
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

    # Reshape interpreting as row major
    X = raw.reshape((-1, metadata["num_channels"]), order="C")
    # Convert from the raw uint16 into float "units" via "offset" and "scaler"
    X = np.multiply(metadata["scaler"], (X.astype(np.float32) - metadata["offset"]))

    # Extract sample rate for first channel and construct a time axis in ms
    fs = metadata["sample_rate"]

    start_t = (1000 / fs) * sum([s["num_frames"] for s in metadata["blocks"][0:start]])
    end_t = (1000 / fs) * sum([s["num_frames"] for s in metadata["blocks"][0:stop]])
    t = np.linspace(start_t, end_t, X.shape[0], endpoint=False)
    assert t.shape[0] == X.shape[0]

    return X, t, fs


def load_blocks(batch_uuid, experiment_num, start=0, stop=None):
    """
    Load signal blocks of data from a single experiment for Axion, Indan and Raspi
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
    metadata = load_experiment(batch_uuid, experiment_num)
    assert start >= 0 and start < len(metadata["blocks"])
    assert not stop or stop >= 0 and stop <= len(metadata["blocks"])
    assert not stop or stop > start

    if ("Raspi" in metadata["hardware"]):
        X, t, fs = load_files_raspi(metadata, batch_uuid, experiment_num, start, stop)
    elif ("Axion" in metadata["hardware"]):
        X, t, fs = load_files_axion(metadata, batch_uuid, experiment_num, start, stop)
    elif ("Intan" in metadata["hardware"]):
        X, t, fs = load_files_intan(metadata, batch_uuid, experiment_num, start, stop)
    else:
        raise Exception('hardware field in metadata.json must contain keyword Axion, Raspi, or Intan')

    return X, t, fs


def load_spikes(batch_uuid, experiment_num):
    batch = load_batch(batch_uuid)
    experiment_name_with_json = batch['experiments'][experiment_num]
    experiment_name = experiment_name_with_json[:-5].rsplit('/',1)[-1]
    path_of_firings = '/public/groups/braingeneers/ephys/' + batch_uuid + '/nico_spikes/' + experiment_name + '_spikes.npy'
    print(path_of_firings)

    try:
        firings = np.load(path_of_firings)
        spike_times= firings[1]
        return spike_times
    except:
        path_of_firings_on_prp = get_archive_url() + '/'+batch_uuid + '/nico_spikes/' + experiment_name + '_spikes.npy'
        response = requests.get(path_of_firings_on_prp, stream=True)

        with open('firings.npy', 'wb') as fin:
            shutil.copyfileobj(response.raw, fin)

        firings = np.load('firings.npy')
        spikes = firings[1]
        return spikes

def load_firings(batch_uuid, experiment_num, sorting_type): #sorting type is "ms4" or "klusta" etc
    batch = load_batch(batch_uuid)
    experiment_name_with_json = batch['experiments'][experiment_num]
    experiment_name = experiment_name_with_json[:-5].rsplit('/',1)[-1]
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
            path_of_firings_on_prp = get_archive_url() + '/'+batch_uuid + '/nico_spikes/' + experiment_name + '_spikes.npy'
        if (sorting_type == "klusta"):
            path_of_firings_on_prp = get_archive_url() + '/'+batch_uuid + '/klusta_spikes/' + experiment_name + '_spikes.npy'
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
        X, t, fs = load_blocks(batch_uuid, index, i, i+1)
        X= np.transpose(X)
        X= X[:int(experiment['num_voltage_channels'])]
        step = int(fs / 1000)
        yield np.array([[
            np.amin(X[:,j:min(j + step, X.shape[1]-1)]),
            np.amax(X[:,j:min(j + step, X.shape[1]-1)])]
          for j in range(0, X.shape[1], step)])

def create_overview(batch_uuid, experiment_num, with_spikes = True):
    #batch_uuid = '2020-02-06-kvoitiuk'

    batch = load_batch(batch_uuid)

    experiment = load_experiment(batch_uuid, experiment_num)
    index = batch['experiments'].index("{}.json".format(experiment['name']))
    plt.figure(figsize=(15,5))

    overview = np.concatenate(list(min_max_blocks(experiment, batch_uuid)))


    print('Overview Shape:',overview.shape)


    plt.title("Overview for Batch: {} Experiment: {}".format(batch_uuid, experiment["name"]))
    plt.fill_between(range(0, overview.shape[0]), overview[:,0], overview[:,1])

    blocks = load_blocks(batch_uuid, experiment_num, 0)

    if with_spikes:

        spikes = load_spikes(batch_uuid, experiment_num)

        fs = blocks[2]

        step = int(fs / 1000)

        spikes_in_correct_units = spikes/step

        for i in spikes_in_correct_units:
            plt.axvline(i, .1, .2, color = 'r', linewidth = .8, linestyle='-', alpha = .05)


    plt.show()

    #path = "archive/features/overviews/{}/{}.npy".format(batch["uuid"], experiment["name"])
    #print(path)
