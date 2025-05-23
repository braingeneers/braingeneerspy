import glob
import io
import posixpath
import zipfile
from dataclasses import dataclass
from logging import getLogger
from typing import List, Tuple

import numpy as np
import pandas as pd
from deprecated import deprecated
from scipy import signal
from spikedata import SpikeData

import braingeneers.utils.smart_open_braingeneers as smart_open
from braingeneers.utils import s3wrangler
from braingeneers.utils.common_utils import get_basepath

__all__ = [
    "read_phy_files",
    "filter",
    "NeuronAttributes",
    "load_spike_data",
]

logger = getLogger("braingeneers.analysis")


@dataclass
class NeuronAttributes:
    cluster_id: int
    channel: np.ndarray
    position: Tuple[float, float]
    amplitudes: List[float]
    template: np.ndarray
    templates: np.ndarray
    label: str

    # These lists are the same length and correspond to each other
    neighbor_channels: np.ndarray
    neighbor_positions: List[Tuple[float, float]]
    neighbor_templates: List[np.ndarray]

    def __init__(self, *args, **kwargs):
        self.cluster_id = kwargs.pop("cluster_id")
        self.channel = kwargs.pop("channel")
        self.position = kwargs.pop("position")
        self.amplitudes = kwargs.pop("amplitudes")
        self.template = kwargs.pop("template")
        self.templates = kwargs.pop("templates")
        self.label = kwargs.pop("label")
        self.neighbor_channels = kwargs.pop("neighbor_channels")
        self.neighbor_positions = kwargs.pop("neighbor_positions")
        self.neighbor_templates = kwargs.pop("neighbor_templates")
        for key, value in kwargs.items():
            setattr(self, key, value)

    def add_attribute(self, key, value):
        setattr(self, key, value)

    def list_attributes(self):
        return [
            attr
            for attr in dir(self)
            if not attr.startswith("__") and not callable(getattr(self, attr))
        ]


def list_sorted_files(uuid, basepath=None):
    """
    Lists files in a directory.

    :param path: the path to the directory.
    :param pattern: the pattern to match.
    :return: a list of files.
    """
    if basepath is None:
        basepath = get_basepath()
    if "s3://" in basepath:
        return s3wrangler.list_objects(
            basepath + "ephys/" + uuid + "/derived/kilosort2/"
        )
    else:
        # return glob.glob(os.path.join(basepath, f'ephys/{uuid}/derived/kilosort2/*'))
        return glob.glob(basepath + f"ephys/{uuid}/derived/kilosort2/*")


def load_spike_data(
    uuid,
    experiment=None,
    basepath=None,
    full_path=None,
    fs=20000.0,
    groups_to_load=["good", "mua", "", np.nan, "unsorted"],
    sorter="kilosort2",
):
    """
    Loads spike data from a dataset.

    :param uuid: the UUID for a specific dataset.
    :param experiment: an optional string to specify a particular experiment in the dataset.
    :param basepath: an optional string to specify a basepath for the dataset.
    :return: SpikeData class with a list of spike time lists and a list of NeuronAttributes.
    """
    if basepath is None:
        basepath = get_basepath()

    if experiment is None:
        experiment = ""
    prefix = f"ephys/{uuid}/derived/{sorter}/{experiment}"
    logger.info("prefix: %s", prefix)
    path = posixpath.join(basepath, prefix)

    if full_path is not None:
        experiment = full_path.split("/")[-1].split(".")[0]
        logger.info("Using full path, experiment: %s", experiment)
        path = full_path
    else:
        if path.startswith("s3://"):
            logger.info("Using s3 path for experiment: %s", experiment)
            # If path is an s3 path, use wrangler
            file_list = s3wrangler.list_objects(path)

            zip_files = [file for file in file_list if file.endswith(".zip")]

            if not zip_files:
                raise ValueError("No zip files found in specified location.")
            elif len(zip_files) > 1:
                logger.warning("Multiple zip files found. Using the first one.")

            path = zip_files[0]

        else:
            logger.info("Using local path for experiment: %s", experiment)
            # If path is a local path, check locally
            file_list = glob.glob(path + "*.zip")

            zip_files = [file for file in file_list if file.endswith(".zip")]

            if not zip_files:
                raise ValueError("No zip files found in specified location.")
            elif len(zip_files) > 1:
                logger.warning("Multiple zip files found. Using the first one.")

            path = zip_files[0]

    with smart_open.open(path, "rb") as f0:
        f = io.BytesIO(f0.read())
        logger.debug("Opening zip file...")
        with zipfile.ZipFile(f, "r") as f_zip:
            assert "params.py" in f_zip.namelist(), "Wrong spike sorting output."
            logger.debug("Reading params.py...")
            with io.TextIOWrapper(f_zip.open("params.py"), encoding="utf-8") as params:
                for line in params:
                    if "sample_rate" in line:
                        fs = float(line.split()[-1])
            logger.debug("Reading spike data...")
            clusters = np.load(f_zip.open("spike_clusters.npy")).squeeze()
            templates_w = np.load(f_zip.open("templates.npy"))
            wmi = np.load(f_zip.open("whitening_mat_inv.npy"))
            channels = np.load(f_zip.open("channel_map.npy")).squeeze()
            spike_templates = np.load(f_zip.open("spike_templates.npy")).squeeze()
            spike_times = np.load(f_zip.open("spike_times.npy")).squeeze() / fs * 1e3
            positions = np.load(f_zip.open("channel_positions.npy"))
            amplitudes = np.load(f_zip.open("amplitudes.npy")).squeeze()

            # Load cluster info from the first detected of several possible filenames.
            tsv_names = {"cluster_info.tsv", "cluster_group.tsv", "cluster_KSLabel.tsv"}
            for tsv in tsv_names & set(f_zip.namelist()):
                cluster_info = pd.read_csv(f_zip.open(tsv), sep="\t")
                cluster_id = cluster_info.cluster_id.values
                # Sometimes this file has the column "KSLabel" instead of "group".
                if "KSLabel" in cluster_info:
                    cluster_info.rename(columns=dict(KSLabel="group"), inplace=True)
                labeled_clusters = cluster_id[cluster_info.group.isin(groups_to_load)]
                # Delete labeled clusters that were not assigned to any spike.
                labeled_clusters = np.intersect1d(labeled_clusters, clusters)
                break

            # If no file is detected, print a warning, but continue with filler labels.
            else:
                logger.warning(
                    "No cluster assignment TSV file found. Generating blank labels."
                )
                labeled_clusters = np.unique(clusters)
                cluster_info = pd.DataFrame(
                    {
                        "cluster_id": labeled_clusters,
                        "group": [""] * len(labeled_clusters),
                    }
                )

    assert len(labeled_clusters) > 0, "No clusters found."
    logger.debug("Reorganizing data...")
    df = pd.DataFrame(
        {"clusters": clusters, "spikeTimes": spike_times, "amplitudes": amplitudes}
    )
    cluster_agg = df.groupby("clusters").agg(
        {"spikeTimes": lambda x: list(x), "amplitudes": lambda x: list(x)}
    )
    cluster_agg = cluster_agg[cluster_agg.index.isin(labeled_clusters)]
    cls_temp = dict(zip(clusters, spike_templates))

    logger.debug("Creating neuron attributes...")
    neuron_attributes = []

    # un-whiten the templates before finding the best channel
    templates = np.dot(templates_w, wmi)

    for i in range(len(labeled_clusters)):
        c = labeled_clusters[i]
        temp = templates[cls_temp[c]].T
        amp = np.max(temp, axis=1) - np.min(temp, axis=1)
        sorted_idx = [ind for _, ind in sorted(zip(amp, np.arange(len(amp))))]
        nbgh_chan_idx = sorted_idx[::-1][:12]
        nbgh_temps = temp[nbgh_chan_idx]
        nbgh_channels = channels[nbgh_chan_idx]
        nbgh_postions = [tuple(positions[idx]) for idx in nbgh_chan_idx]
        neuron_attributes.append(
            NeuronAttributes(
                cluster_id=c,
                channel=nbgh_channels[0],
                position=nbgh_postions[0],
                amplitudes=cluster_agg["amplitudes"][c],
                template=nbgh_temps[0],
                templates=templates[cls_temp[c]].T,
                label=cluster_info["group"][cluster_info["cluster_id"] == c].values[0],
                neighbor_channels=nbgh_channels,
                neighbor_positions=nbgh_postions,
                neighbor_templates=nbgh_temps,
            )
        )

    logger.debug("Creating spike data...")

    metadata = {"experiment": experiment}
    spike_data = SpikeData(
        cluster_agg["spikeTimes"].to_list(),
        neuron_attributes=neuron_attributes,
        metadata=metadata,
    )

    logger.debug("Done.")
    return spike_data


@deprecated("Prefer load_spike_data()", version="0.1.13")
def read_phy_files(path: str, fs=20000.0):
    """
    :param path: a s3 or local path to a zip of phy files.
    :return: SpikeData class with a list of spike time lists and neuron_data.
            neuron_data = {0: neuron_dict, 1: config_dict}
            neuron_dict = {"new_cluster_id": {"channel": c, "position": (x, y),
                            "amplitudes": [a0, a1, an], "template": [t0, t1, tn],
                            "neighbor_channels": [c0, c1, cn],
                            "neighbor_positions": [(x0, y0), (x1, y1), (xn,yn)],
                            "neighbor_templates": [[t00, t01, t0n], [tn0, tn1, tnn]}}
            config_dict = {chn: pos}
    """
    assert path[-3:] == "zip", "Only zip files supported!"
    import braingeneers.utils.smart_open_braingeneers as smart_open

    with smart_open.open(path, "rb") as f0:
        f = io.BytesIO(f0.read())

        with zipfile.ZipFile(f, "r") as f_zip:
            assert "params.py" in f_zip.namelist(), "Wrong spike sorting output."
            with io.TextIOWrapper(f_zip.open("params.py"), encoding="utf-8") as params:
                for line in params:
                    if "sample_rate" in line:
                        fs = float(line.split()[-1])
            clusters = np.load(f_zip.open("spike_clusters.npy")).squeeze()
            templates = np.load(
                f_zip.open("templates.npy")
            )  # (cluster_id, samples, channel_id)
            channels = np.load(f_zip.open("channel_map.npy")).squeeze()
            templates_w = np.load(f_zip.open("templates.npy"))
            wmi = np.load(f_zip.open("whitening_mat_inv.npy"))
            spike_templates = np.load(f_zip.open("spike_templates.npy")).squeeze()
            spike_times = (
                np.load(f_zip.open("spike_times.npy")).squeeze() / fs * 1e3
            )  # in ms
            positions = np.load(f_zip.open("channel_positions.npy"))
            amplitudes = np.load(f_zip.open("amplitudes.npy")).squeeze()

            if "cluster_KSLabel.tsv" in f_zip.namelist():
                cluster_info = pd.read_csv(f_zip.open("cluster_KSLabel.tsv"), sep="\t")
                cluster_id = np.array(cluster_info["cluster_id"])
                labeled_clusters = cluster_id[
                    cluster_info["group"].isin(groups_to_load)
                ]

            elif "cluster_info.tsv" in f_zip.namelist():
                cluster_info = pd.read_csv(f_zip.open("cluster_info.tsv"), sep="\t")
                cluster_id = np.array(cluster_info["cluster_id"])
                # select clusters using curation label, remove units labeled as "noise"
                # find the best channel by amplitude
                labeled_clusters = cluster_id[cluster_info["group"] != "noise"]
            else:
                labeled_clusters = np.unique(clusters)

    df = pd.DataFrame(
        {"clusters": clusters, "spikeTimes": spike_times, "amplitudes": amplitudes}
    )
    cluster_agg = df.groupby("clusters").agg(
        {"spikeTimes": lambda x: list(x), "amplitudes": lambda x: list(x)}
    )
    cluster_agg = cluster_agg[cluster_agg.index.isin(labeled_clusters)]

    cls_temp = dict(zip(clusters, spike_templates))
    neuron_dict = dict.fromkeys(np.arange(len(labeled_clusters)), None)

    # un-whitten the templates before finding the best channel
    templates = np.dot(templates_w, wmi)

    neuron_attributes = []
    for i in range(len(labeled_clusters)):
        c = labeled_clusters[i]
        temp = templates[cls_temp[c]]
        amp = np.max(temp, axis=0) - np.min(temp, axis=0)
        sorted_idx = [ind for _, ind in sorted(zip(amp, np.arange(len(amp))))]
        nbgh_chan_idx = sorted_idx[::-1][:12]
        nbgh_temps = temp.transpose()[nbgh_chan_idx]
        best_chan_temp = nbgh_temps[0]
        nbgh_channels = channels[nbgh_chan_idx]
        nbgh_postions = [tuple(positions[idx]) for idx in nbgh_chan_idx]
        best_channel = nbgh_channels[0]
        best_position = nbgh_postions[0]
        # neighbor_templates = dict(zip(nbgh_postions, nbgh_temps))
        cls_amp = cluster_agg["amplitudes"][c]
        neuron_dict[i] = {
            "cluster_id": c,
            "channel": best_channel,
            "position": best_position,
            "amplitudes": cls_amp,
            "template": best_chan_temp,
            "neighbor_channels": nbgh_channels,
            "neighbor_positions": nbgh_postions,
            "neighbor_templates": nbgh_temps,
        }
        neuron_attributes.append(
            NeuronAttributes(
                cluster_id=c,
                channel=best_channel,
                position=best_position,
                amplitudes=cluster_agg["amplitudes"][c],
                template=best_chan_temp,
                templates=templates[cls_temp[c]].T,
                label=cluster_info["group"][cluster_info["cluster_id"] == c].values[0],
                neighbor_channels=channels[nbgh_chan_idx],
                neighbor_positions=[tuple(positions[idx]) for idx in nbgh_chan_idx],
                neighbor_templates=[templates[cls_temp[c]].T[n] for n in nbgh_chan_idx],
            )
        )

    config_dict = dict(zip(channels, positions))
    neuron_data = {0: neuron_dict}
    metadata = {0: config_dict}
    spikedata = SpikeData(
        list(cluster_agg["spikeTimes"]),
        neuron_data=neuron_data,
        metadata=metadata,
        neuron_attributes=neuron_attributes,
    )
    return spikedata


@deprecated("Prefer analysis.butter_filter()", version="0.1.14")
def filter(
    raw_data,
    fs_Hz=20000,
    filter_order=3,
    filter_lo_Hz=300,
    filter_hi_Hz=6000,
    time_step_size_s=10,
    channel_step_size=100,
    verbose=0,
    zi=None,
    return_zi=False,
):
    """
    Filter the raw data using a bandpass filter.

    :param raw_data: [channels, time] array of raw ephys data
    :param fs_Hz: sampling frequency of raw data in Hz
    :param filter_order: order of the filter
    :param filter_lo_Hz: low frequency cutoff in Hz
    :param filter_hi_Hz: high frequency cutoff in Hz
    :param filter_step_size_s: size of chunks to filter in seconds
    :param channel_step_size: number of channels to filter at once
    :param verbose: verbosity level
    :param zi: initial conditions for the filter
    :param return_zi: whether to return the final filter conditions

    :return: filtered data
    """

    time_step_size = int(time_step_size_s * fs_Hz)
    data = np.zeros_like(raw_data)

    # Get filter params
    b, a = signal.butter(
        fs=fs_Hz, btype="bandpass", N=filter_order, Wn=[filter_lo_Hz, filter_hi_Hz]
    )

    if zi is None:
        # Filter initial state
        zi = signal.lfilter_zi(b, a)
        zi = np.vstack(
            [zi * np.mean(raw_data[ch, :5]) for ch in range(raw_data.shape[0])]
        )

    # Step through the data in chunks and filter it
    for ch_start in range(0, raw_data.shape[0], channel_step_size):
        ch_end = min(ch_start + channel_step_size, raw_data.shape[0])

        logger.debug(f"Filtering channels {ch_start} to {ch_end}")

        for t_start in range(0, raw_data.shape[1], time_step_size):
            t_end = min(t_start + time_step_size, raw_data.shape[1])

            (
                data[ch_start:ch_end, t_start:t_end],
                zi[ch_start:ch_end, :],
            ) = signal.lfilter(
                b,
                a,
                raw_data[ch_start:ch_end, t_start:t_end],
                axis=1,
                zi=zi[ch_start:ch_end, :],
            )

    return data if not return_zi else (data, zi)
