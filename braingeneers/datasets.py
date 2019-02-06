import requests
import numpy as np

# Base URL to the entire Braingeneers archive on PRP S3
archive_url = "https://s3.nautilus.optiputer.net/braingeneers/archive"


def load_experiment(path, max_segments=-1):
    """
    Load signal and metadata stored in S3 via the ingest notebook.
    path = path to samples under archive, typically pulled from metadata.json
    max_segments = max number of sequential .npy files to load
    Returns:
    X = signal vector of microvolt time series readings, one for each of the channels
    t = time in milliseconds for each reading (for conveniance, all samples at fs)
    fs = sample rate
    metadata = all of the metadata associated with this experiment
    """

    # Each experiment has a metadata file with all *.rhd headers and other sample info
    metadata = requests.get("{}/{}".format(archive_url, path)).json()

    # Load all the numpy files into a single matrix
    X = np.concatenate([
        np.load(np.DataSource(None).open("{}/{}".format(archive_url, s["derived"]), "rb"))
        for s in metadata["samples"][0:max_segments]], axis=1)

    # Convert from the raw uint16 into float "units" via "offset" and "scaler"
    X = np.multiply(metadata["samples"][0]["scaler"],
                    (X.astype(np.float32) - metadata["samples"][0]["offset"]))

    # Extract sample rate for first channel and construct a time axis in seconds
    fs = metadata["samples"][0]["frequency_parameters"]["amplifier_sample_rate"]
    t = np.linspace(0, 1000 * X.shape[1] / fs, X.shape[1])

    return X, t, fs, metadata
