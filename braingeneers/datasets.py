import os
import requests
import numpy as np


def get_archive_url():
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
    return requests.get("{}/derived/{}/metadata.json".format(get_archive_url(), batch_uuid)).json()


def load_experiment(path):
    """
    Load metadata from PRP S3 for a single experiment

    Parameters
    ----------
    path : str
        Path to the experiment meta data json file in the Braingeneer's archive.
        Typically taken from a batch's metadata list of experiments.
        Example: derived/2019-02-05/OrganoidTestStimulate1.json

    Returns
    -------
    metadata : dict
        All of the metadata associated with this batch
    """

    # Each experiment has a metadata file with all *.rhd headers and other sample info
    return requests.get("{}/{}".format(get_archive_url(), path)).json()


def load_blocks(metadata, start=0, stop=None):
    """
    Load signal blocks of data from a single experiment

    Parameters
    ----------
    metadata : dict
        Metadata for an experiment returned from load_experiment

    start : int, optional
        First rhd data block to return

    stop : int, optional
        Last rhd data block to return

    step : int, optional
        Step size when selecting blocks

    Returns
    -------
    X : ndarray
        Numpy matrix with count channels by samples

    t : ndarray
        Numpy array with time in milliseconds for each sample

    fs : float
        Sample rate in Hz
    """

    # Load all the numpy files into a single matrix
    X = np.concatenate([
        np.load(np.DataSource(None).open("{}/{}"
                                         .format(get_archive_url(),
                                                 s["derived"]), "rb"))
        for s in metadata["samples"][start:stop]], axis=1)

    # Convert from the raw uint16 into float "units" via "offset" and "scaler"
    X = np.multiply(metadata["samples"][0]["scaler"],
                    (X.astype(np.float32) - metadata["samples"][0]["offset"]))

    # Extract sample rate for first channel and construct a time axis in ms
    fs = metadata["samples"][0]["frequency_parameters"]["amplifier_sample_rate"]
    t = np.linspace(0, 1000 * X.shape[1] / fs, X.shape[1])

    return X, t, fs
