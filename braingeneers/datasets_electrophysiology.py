import os
import json
import requests
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

def get_archive_path():
    """/public/groups/braingeneers/Electrophysiology  Return path to archive on the GI public server """
    return os.getenv("BRAINGENEERS_ARCHIVE_PATH", "/public/groups/braingeneers/Electrophysiology")

def get_archive_url():
    """  https://s3.nautilus.optiputer.net/braingeneers/Electrophysiology     Return URL to archive on PRP """
    return "{}/braingeneers/Electrophysiology".format(
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

    r = requests.get(full_path)
    if r.ok:
        return r.json()
    else:
        print("Unable to load {}, do you have the correct batch uuid?".format(batch_uuid))
        r.raise_for_status()
        
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
    r = requests.get("{}/derived".format(
        get_archive_url(), batch_uuid, batch["experiments"][experiment_num]))
    if r.ok:
        return r.json()
    else:
        print("Unable to load experiment {} from {}".format(experiment_num, batch_uuid))
        r.raise_for_status()
        
def load_blocks(batch_uuid, experiment_num, start=0, stop=None):
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
    metadata = load_experiment(batch_uuid, experiment_num)
    assert start >= 0 and start < len(metadata["blocks"])
    assert not stop or stop >= 0 and stop <= len(metadata["blocks"])
    assert not stop or stop > start

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
            _load_url("{}/derived/{}/{}".format(get_archive_url(), batch_uuid, s["path"]))
            for s in metadata["blocks"][start:stop]], axis=0)

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


def load_spikes(batch_uuid, experiment_num):
    batch = load_batch(batch_uuid)
    experiment_name_with_json = batch['experiments'][experiment_num]
    experiment_name = experiment_name_with_json[:-5].rsplit('/',1)[-1]
    path_of_firings = '/public/groups/braingeneers/Electrophysiology/' + batch_uuid + '/spikes/' + experiment_name + '_spikes.npy'
    print(path_of_firings)
    firings = np.load(path_of_firings)
    spike_times= firings[1]
    return spike_times
    
    
    
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
    
    spikes = load_spikes(batch_uuid, experiment_num)
    
    blocks = load_blocks(batch_uuid, experiment_num, 0)
    
    if with_spikes:
    
        fs = blocks[2]

        step = int(fs / 1000)

        spikes_in_correct_units = spikes/step 

        for i in spikes_in_correct_units:

            plt.axvline(i, .1, .2, color = 'y', linewidth = .8, linestyle='-', alpha = .05)

    plt.show()

    #path = "archive/features/overviews/{}/{}.npy".format(batch["uuid"], experiment["name"])
    #print(path)
