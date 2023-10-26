import argparse
import json
import os
from neuraltoolkit import ntk_filters as ntk
import pdb
import braingeneers
from braingeneers.utils import s3wrangler as wr
from braingeneers.utils import smart_open_braingeneers as smart_open
import numpy as np
from typing import List, Union
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

    parser.add_argument('--details', '-d', nargs='+', required=True,
                        help='CSList indicating where and how much data to '
                             'take. Usage: -a offset length channels '
                             'where offset is an int, length is an int, '
                             'and channels is a string of values separated by slashes')
    parser.add_argument(
        '--apply', action='append', required=True,
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
    # set local endpoint for faster loading of data
    # braingeneers.set_default_endpoint(f'{os.getcwd()}')
    spectro = False
    fsemg = 1
    # TODO: Make more robust - if length is not specified, need to still make it work
    # TODO: rip out spectrogram code and make it separate file
    # load metadata
    offset = int(details[0])
    datalen = int(details[1])
    chans = details[2]
    metad = de.load_metadata(uuid)
    e = f'experiment{experiment + 1}' if isinstance(experiment, int) else experiment
    fs = metad['ephys_experiments'][e]['sample_rate']
    # then, load the data
    chans = [int(i) for i in chans.split('-')]
    dataset = de.load_data(metad, experiment=experiment, offset=offset, length=datalen, channels=chans)
    dataset = np.vstack(dataset)
    print(dataset.shape)
    # if the data is shorter than 3 min, no point in making spectrogram
    if dataset.shape[1] >= 3600000:
        spectro = True
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

    # here the data should be ready to viz.
    # try putting data thru spectrogram
    # fig, axs = plt.subplots(nrows=len(chans), ncols=3, figsize=(16,8))
    datafig = plt.figure(figsize=(16, 2 * len(chans) + 2))

    # create chan x 1 subfigs
    subfigs = datafig.subfigures(nrows=len(chans), ncols=1)
    if not spectro:
        for index in range(len(chans)):
            subfigs[index].suptitle(f'Channel {chans[index]}')

            # create 1x2 subplots per subfig
            axs = subfigs[index].subplots(nrows=1, ncols=2, subplot_kw={'anchor': 'SW'},
                                          gridspec_kw={'wspace': 0.15})
            for ax in axs:
                ax.tick_params(bottom=True, labelbottom=True, left=True, labelleft=True, right=False, labelright=False,
                               top=False, labeltop=False)

            raw_plot = dataset[index]
            filt_plot = filt_dataset[index]
            realtime = np.arange(np.size(raw_plot)) / fsemg
            axs[0].plot(realtime, (raw_plot - np.nanmean(raw_plot)) / np.nanstd(raw_plot))
            axs[0].set_xlim(0, datalen)
            # plot filtered data in middle
            axs[1].plot(realtime, (filt_plot - np.nanmean(filt_plot)) / np.nanstd(filt_plot))
            axs[1].set_xlim(0, datalen)

    # here, assume that the data is long enough to do spectrogram with
    else:
        specfig = plt.figure(figsize=(16, 2 * len(chans) + 2))
        spectro_subfigs = specfig.subfigures(nrows=len(chans), ncols=1)
        for index in range(len(chans)):
            subfigs[index].suptitle(f'Channel {chans[index]}')
            spectro_subfigs[index].suptitle(f'Channel {chans[index]}')
            # axs now needs 4 columns, but axs2 still is just 1x1
            axs = subfigs[index].subplots(nrows=1, ncols=4)
            axs2 = spectro_subfigs[index].subplots(nrows=1, ncols=1)

            for ax in axs:
                ax.tick_params(bottom=True, labelbottom=True, left=True, labelleft=True, right=False, labelright=False,
                               top=False, labeltop=False)
            axs2[0].tick_params(bottom=True, labelbottom=True, left=True, labelleft=True, right=False,
                                labelright=False,
                                top=False, labeltop=False)

            raw_plot_1 = dataset[index][:20000]  # 1 second
            raw_plot_10 = dataset[index][:200000]  # 10 seconds
            filt_plot = filt_dataset[index]
            filt_plot_1 = filt_plot[:20000]
            filt_plot_10 = filt_plot[:200000]

            realtime_1 = np.arange(np.size(raw_plot_1)) / fsemg
            realtime_10 = np.arange(np.size(raw_plot_10)) / fsemg
            axs[0].plot(realtime_1, (raw_plot_1 - np.nanmean(raw_plot_1)) / np.nanstd(raw_plot_1))
            axs[0].set_xlim(0, 20000)
            # plot filtered data in middle
            axs[1].plot(realtime_1, (filt_plot_1 - np.nanmean(filt_plot_1)) / np.nanstd(filt_plot_1))
            axs[1].set_xlim(0, 20000)

            axs[2].plot(realtime_10, (raw_plot_10 - np.nanmean(raw_plot_10)) / np.nanstd(raw_plot_10))
            axs[2].set_xlim(0, 200000)

            axs[3].plot(realtime_10, (filt_plot_10 - np.nanmean(filt_plot_10)) / np.nanstd(filt_plot_10))
            axs[3].set_xlim(0, 200000)
            # plot
            # for spectrogram
            freq, times, spec = ssig.spectrogram(filt_plot, fs, window='hamming', nperseg=1000, noverlap=1000 - 1,
                                                 mode='psd')
            fmax = 64
            fmin = 1
            x_mesh, y_mesh = np.meshgrid(times, freq[(freq <= fmax) & (freq >= fmin)])
            axs2[0].pcolormesh(x_mesh, y_mesh, np.log10(spec[(freq <= fmax) & (freq >= fmin)]), cmap='jet',
                               shading='auto')

    datapoints_filename = f'Raw_and_filtered_data_{uuid}_{experiment}_chan_{chans}.png'
    spectro_filename = f'Spectrogram_{uuid}_{experiment}_chan_{chans}.png'
    with open(datapoints_filename, 'wb') as dfig:
        plt.savefig(dfig, format='png')
    with open(spectro_filename, 'wb') as sfig:
        plt.savefig(sfig, format='png')
    # then, if it's meant to be on s3, awswrangle it up there.
    if outputLocation == 's3':
        # Check if file exists
        try:
            with smart_open.open(f's3://braingeneersdev/ephys/{uuid}/derived/{e}_visualization_metadata.json'):
                file_exists = True
        except OSError:
            file_exists = False
        # pdb.set_trace()
        if not file_exists:
            # make metadata
            new_meta = {
                'notes': f'Raw data, filtered data, and spectrogram for {uuid} {e} ',
                'hardware': 'Maxwell',
                'channels': chans
                # ,
                # 'drug_condition': 'fill', # TODO: (not immediate) FIND WAY TO PARSE DRUG CONDITION FROM DATA
                # 'age_of_culture': 'age',
                # 'duration_on_chip': 'len',
                # 'cell_type': 'H9',
                # 'stimulation':'None',
                # 'external_considerations': 'microfluidics'

            }
            with smart_open.open(f'{uuid}_viz_metadata.json', 'w') as f:
                json.dump(new_meta, f)
            # then, make the new bucket and put the metadata in. awswrangler for that.
            # pdb.set_trace()
            # print(wr.config.s3_endpoint_url)
            wr.upload(local_file=f'{os.getcwd()}/{uuid}_viz_metadata.json',
                      path=f's3://braingeneersdev/ephys/{uuid}/derived/{e}_visualization_metadata.json')

        else:
            # if the metadata exists, need to add this metadata onto the existing one
            with smart_open.open(
                    f's3://braingeneersdev/ephys/{uuid}/derived/{e}_visualization_metadata.json') as old_json:
                fixed_meta = json.load(old_json)
                old_chans = fixed_meta['channels']
                new_chans = sorted(list(set(old_chans + chans)))
                fixed_meta['channels'] = new_chans
            with smart_open.open(f's3://braingeneersdev/ephys/{uuid}/derived/{e}_visualization_metadata.json',
                                 'w') as out_metadata:
                json.dump(fixed_meta, out_metadata, indent=2)

        wr.upload(local_file=f'{os.getcwd()}/Raw_and_filtered_data_{uuid}_{experiment}_chan_{chans}.png',
                  path=f's3://braingeneersdev/ephys/{uuid}/derived/Raw_and_filtered_data_{uuid}_{experiment}_chan_{chans}.png')
        wr.upload(local_file=f'{os.getcwd()}/Spectrogram_{uuid}_{experiment}_chan_{chans}.png',
                  path=f's3://braingeneersdev/ephys/{uuid}/derived/Spectrogram_{uuid}_{experiment}_chan_{chans}.png')

    plt.close()
    gc.collect()


if __name__ == '__main__':
    # args = parse_args()
    # # pdb.set_trace()
    main(**parse_args())
    # main('2022-07-27-e-Chris_BCC_APV', 'experiment3', [6000, 10000, '0,1,2,3'], ['bandpass=500,9000'])
