import argparse
import json
import os
import pdb

from braingeneers.utils import s3wrangler as wr
from braingeneers.utils import smart_open_braingeneers as smart_open
import numpy as np
from typing import List, Union
from neuraltoolkit import ntk_filters as ntk
from braingeneers.data import datasets_electrophysiology as de
from scipy import signal as ssig
import matplotlib.pyplot as plt
import gc


def int_or_str(value):
    """
    This function is passed as type to accept the Union of two data types.
    :param value: value to consider
    :return: either int or string
    """
    try:
        return int(value)
    except ValueError:
        return value


def parse_args():
    """
    This function parses the arguments passed in via CLI
    :return: Dictionary of parsed arguments
    """
    parser = argparse.ArgumentParser(description='Convert a single neural data file using a specified filter')
    parser.add_argument('--uuid', '-u', type=str, required=True,
                        help='UUID for desired experiment batch')

    parser.add_argument('--experiment', '-e', type=int_or_str, required=True,
                        help='Experiment number. Can be passed as index (int) or experiment# (string)'
                             'e.g. 1 or \'experiment2\'')

    parser.add_argument('--outputLocation', '-o', type=str, default='local', choices=['local', 's3'],
                        help='Where to store the output. Either specify \'local\' or \'s3\', or leave blank'
                             'to have it saved locally. ')

    # parser.add_argument('--samplesize', '-s', type=int,help='Size of sample to take')
    parser.add_argument('--details', '-d', nargs='+', required=True,
                        help='CSList indicating where and how much data to '
                             'take. Usage: -a offset length channels '
                             'where offset is an int, length is an int, '
                             'and channels is a string of values separated by commas')
    parser.add_argument(
        '--apply', action='append',
        help='Filter type + arguments, --apply-filter is specified 1 or more times for each filter. Usage options:\n'
             '--apply highpass=750    (highpass filter @ 750 hz)\n'
             '--apply lowpass=8       (lowpass filter @ 8 hz)\n'
             '--apply bandpass=low,high  (bandpass values for between the low and high arguments)'
             '--apply downsample=200  (downsample to 200 hz)'
    )

    return vars(parser.parse_args())


def highpass(data: np.ndarray, hz: int, fs: int):
    data_highpass = np.vstack([
        ntk.butter_highpass(channel_data, highpass=hz, fs=fs, order=3)
        for channel_data in data
    ])
    print(
        f'Running highpass filter with parameters: hz={hz}, fs={fs}, input shape: {data.shape}, output shape: {data_highpass.shape}\n')
    return data_highpass


def lowpass(data: np.ndarray, hz: int, fs: int):
    data_lowpass = np.vstack([
        ntk.butter_lowpass(channel_data, lowpass=hz, fs=fs, order=3)
        for channel_data in data
    ])
    print(
        f'Running lowpass filter with parameters: hz={hz}, fs={fs}, input shape: {data.shape}, output shape: {data_lowpass.shape}\n')
    return data_lowpass


def bandpass(data: np.ndarray, hz_high: int, hz_low: int, fs: int):
    data_bandpass = np.vstack([
        ntk.butter_bandpass(channel_data, highpass=hz_high, lowpass=hz_low, fs=fs, order=3)
        for channel_data in data
    ])
    print(
        f'Running bandpass filter with parameters: hz_low={hz_low}, hz_high={hz_high} fs={fs}, input shape: {data.shape}, output shape: {data_bandpass.shape}\n')
    return data_bandpass


def main(uuid: str, experiment: Union[str, int], outputLocation: str, details: List[str], apply: List[str]):
    # load metadata
    offset = int(details[0])
    datalen = int(details[1])
    chans = details[2]
    metad = de.load_metadata(uuid)
    if isinstance(experiment, int):
        # change the value to a string
        e = f'experiment{experiment + 1}'
    else:
        e = experiment
    fs = metad['ephys_experiments'][e]['sample_rate']
    # then, load the data
    chans = [int(i) for i in chans.split(',')]
    dataset = de.load_data(metad, experiment=experiment, offset=offset, length=datalen, channels=chans)
    dataset = np.vstack(dataset)
    # parse out apply list
    for item in apply:
        filt, arg = item.split('=')
        if filt == 'highpass':
            filt_dataset = highpass(dataset, int(arg), fs)
            filt_dataset = np.vstack(filt_dataset)
        elif filt == 'lowpass':
            filt_dataset = lowpass(dataset, int(arg), fs)
            filt_dataset = np.vstack(filt_dataset)
        elif filt == 'bandpass':
            # 7/29/22 switched low and high since the arguments should be different
            hi_rate, low_rate = arg.split(',')
            filt_dataset = bandpass(dataset, int(hi_rate), int(low_rate), fs)
            filt_dataset = np.vstack(filt_dataset)

    # here the data should be ready to viz. todo check code works up to here
    # try putting data thru spectrogram
    # fig, axs = plt.subplots(nrows=len(chans), ncols=3, figsize=(16,8))
    fig = plt.figure(figsize=(16, 2 * len(chans) + 2))

    # create 3x1 subfigs
    subfigs = fig.subfigures(nrows=len(chans), ncols=1)
    for index in range(len(chans)):
        subfigs[index].suptitle(f'Channel {chans[index]}')

        # create 1x3 subplots per subfig
        axs = subfigs[index].subplots(nrows=1, ncols=3, subplot_kw={'anchor': 'SW'},
                                      gridspec_kw={'wspace': 0.15})
        for ax in axs:
            ax.tick_params(bottom=True, labelbottom=True, left=True, labelleft=True, right=False, labelright=False,
                           top=False, labeltop=False)

        raw_plot = dataset[index]
        filt_plot = filt_dataset[index]
        freq, times, spec = ssig.spectrogram(filt_plot, fs, window='hamming', nperseg=1000, noverlap=1000 - 1,
                                             mode='psd')

        fsemg = 1
        realtime = np.arange(np.size(raw_plot)) / fsemg
        axs[0].plot(realtime, (raw_plot - np.nanmean(raw_plot)) / np.nanstd(raw_plot))
        axs[0].set_xlim(0, datalen)
        # plot filtered data in middle
        axs[1].plot(realtime, (filt_plot - np.nanmean(filt_plot)) / np.nanstd(filt_plot))
        axs[1].set_xlim(0, datalen)
        # for spectrogram
        fmax = 64
        fmin = 1
        x_mesh, y_mesh = np.meshgrid(times, freq[(freq <= fmax) & (freq >= fmin)])
        axs[2].pcolormesh(x_mesh, y_mesh, np.log10(spec[(freq <= fmax) & (freq >= fmin)]), cmap='jet',
                          shading='auto')
    # check for output location
    # TODO: Awswrangler needs a local file for upload, so must save the image to a local place first.
    output_filename = f'Spectrogram_{uuid}_{experiment}_chan_{chans}.png'
    with open(output_filename, 'wb') as f:
        plt.savefig(f, format='png')
    # then, if it's meant to be on s3, awswrangle it up there.
    if outputLocation == 's3':
        # Check for bucket first.
        try:
            with smart_open.open(f's3://braingeneersdev/ephys/{uuid}/derived/visualization.json'):
                file_exists = True
        except OSError:
            file_exists = False
        pdb.set_trace()
        if not file_exists:
            # make metadata
            new_meta = {
                'notes': f'Raw data, filtered data, and spectrogram for {uuid} {e} ',
                'hardware': 'Maxwell'
                # ,
                # 'drug_condition': 'fill', # TODO: FIND WAY TO PARSE DRUG CONDITION FROM DATA
                # 'age_of_culture': 'age',
                # 'duration_on_chip': 'len',
                # 'cell_type': 'H9',
                # 'stimulation':'None',
                # 'external_considerations': 'microfluidics'

            }
            with open(f'{uuid}_viz_metadata.json', 'w') as f:
                json.dump(new_meta, f)
            # then, make the new bucket and put the metadata in. awswrangler for that.
            wr.upload(local_file=f'{os.getcwd()}/{uuid}_viz_metadata.json',
                      path=f's3://braingeneersdev/ephys/{uuid}/derived/metadata.json')
            wr.upload(local_file=f'{os.getcwd()}/Spectrogram_{uuid}_{experiment}_chan_{chans}.png',
                      path=f's3://braingeneersdev/ephys/{uuid}/derived/Spectrogram_{uuid}_{experiment}_chan_{chans}.png')

        # create UUID for the bucket
        # s3_url = f's3://braingeneers/ephys/{uuid}'
    plt.close()
    gc.collect()


if __name__ == '__main__':
    args = parse_args()
    # pdb.set_trace()
    main(**parse_args())
    # main('2022-05-18-e-BCCtest', 1, [6000, 10000, '0,1,2,3'], ['bandpass=500,9000'])
