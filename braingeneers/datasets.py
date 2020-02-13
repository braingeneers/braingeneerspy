import os
import json
import requests
import numpy as np


def get_archive_path():
    """ Return path to archive on the GI public server """
    return os.getenv("BRAINGENEERS_ARCHIVE_PATH", "/public/groups/braingeneers/archive")


def get_archive_url():
    """ Return URL to archive on PRP """
    return "{}/braingeneers/archive".format(
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
    full_path = "{}/derived/{}/metadata.json".format(get_archive_path(), batch_uuid)
    if os.path.exists(full_path):
        with open(full_path, "r") as f:
            return json.load(f)

    r = requests.get("{}/derived/{}/metadata.json".format(get_archive_url(), batch_uuid))
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
    full_path = "{}/derived/{}/{}".format(
        get_archive_path(), batch_uuid, batch["experiments"][experiment_num])
    if os.path.exists(full_path):
        with open(full_path, "r") as f:
            return json.load(f)

    # Each experiment has a metadata file with all *.rhd headers and other sample info
    r = requests.get("{}/derived/{}/{}".format(
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
    if os.path.exists("{}/derived/{}".format(get_archive_path(), batch_uuid)):
        # Load from local archive
        raw = np.concatenate([
            _load_path("{}/derived/{}/{}".format(get_archive_path(), batch_uuid, s["path"]))
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


def list_batches_in_derived():
    """
    Lists all the batches that are in derived
    """
    full_path = "{}/derived/".format(get_archive_path())
    return os.listdir(full_path)


def list_files_in_batch(uuid):
    """
    Lists all the files in the batch (uuid)
    """
    full_path = "{}/derived/{}".format(get_archive_path(), uuid)
    return os.listdir(full_path)


def load_file(uuid, file):
    """
    Load a file from a specific batch (uuid)
    """
    full_path = "{}/derived/{}/{}".format(get_archive_path(), uuid, file)
    #print(full_path)
    if file[-3:] =='npy':
        return np.load(full_path)
    elif file[-4:]=='json':
        with open(full_path, "r") as f:
            return json.load(f)
