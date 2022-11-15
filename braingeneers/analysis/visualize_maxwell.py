import argparse
import json
import os
import psutil
from neuraltoolkit import ntk_filters as ntk
import pdb
import braingeneers
from braingeneers.utils import s3wrangler as wr
from braingeneers.utils import smart_open_braingeneers as smart_open
import numpy as np
from typing import List, Union
from braingeneers.data import datasets_electrophysiology as de
from scipy import signal as ssig
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gc
from hanging_threads import start_monitoring
monitoring_thread = start_monitoring()

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
                        help='List indicating where and how much data to '
                             'take. Usage: -d offset length channels '
                             'where offset is an int, length is an int, '
                             'and channels is a string of values separated by slashes. '
                             'Length can be -1 or \'all\' which indicates the full length is being asked for.')
    parser.add_argument(
        '--apply', action='append', required=True,
        help='Filter type + arguments, --apply-filter is specified 1 or more times for each filter. Usage options:\n'
             '--apply highpass=750    (highpass filter @ 750 hz)\n'
             '--apply lowpass=8       (lowpass filter @ 8 hz)\n'
             '--apply bandpass=low,high  (bandpass values for between the low and high arguments)'
             '--apply downsample=200  (downsample to 200 hz)'
    )

    parser.add_argument('--spect', '-s', action='store_true', required=False,
                        help='Choice to make spectrogram or not. False by default.')

    return vars(parser.parse_args())


def highpass(data: np.ndarray, hz: int, fs: int):
    """

    :param data:
    :param hz:
    :param fs:
    :return:
    """
    data_highpass = np.vstack([
        ntk.butter_highpass(channel_data, highpass=hz, fs=fs, order=3)
        for channel_data in data
    ])
    print(
        f'Running highpass filter with parameters: hz={hz}, fs={fs}, input shape: {data.shape}, output shape: {data_highpass.shape}\n')
    return data_highpass


def lowpass(data: np.ndarray, hz: int, fs: int):
    """

    :param data:
    :param hz:
    :param fs:
    :return:
    """
    data_lowpass = np.vstack([
        ntk.butter_lowpass(channel_data, lowpass=hz, fs=fs, order=3)
        for channel_data in data
    ])
    print(
        f'Running lowpass filter with parameters: hz={hz}, fs={fs}, input shape: {data.shape}, output shape: {data_lowpass.shape}\n')
    return data_lowpass


def bandpass(data: np.ndarray, hz_high: int, hz_low: int, fs: int):
    """

    :param data:
    :param hz_high:
    :param hz_low:
    :param fs:
    :return:
    """
    data_bandpass = np.vstack([
        ntk.butter_bandpass(channel_data, highpass=hz_high, lowpass=hz_low, fs=fs, order=3)
        for channel_data in data
    ])
    print(
        f'Running bandpass filter with parameters: hz_low={hz_high}, hz_high={hz_low} fs={fs}, input shape: {data.shape}, output shape: {data_bandpass.shape}\n')
    return data_bandpass


def make_spectrogram(figure, uuid, experiment, datalen, channels, filt_dataset, fs):
    """

    :param figure:
    :param uuid:
    :param experiment:
    :param datalen:
    :param channels:
    :param filt_dataset:
    :param fs:
    :return:
    """
    spectro_subfigs = figure.subfigures(nrows=1, ncols=1)
    fmax = 64
    fmin = 1
    # make subplots for left side
    left_side_subs = spectro_subfigs.subplots(nrows=len(channels), ncols=1)
    left_side_subs = np.array([left_side_subs]) if not isinstance(left_side_subs, np.ndarray) else left_side_subs
    plt.subplots_adjust(hspace=0.3)
    for ix in range(len(left_side_subs)):
        axs2 = left_side_subs[ix]
        axs2.tick_params(bottom=True, labelbottom=True, left=True, labelleft=True, right=False,
                         labelright=False, top=False, labeltop=False)
        # set title for each subplot
        axs2.set_title(label=f'Channel {channels[ix]}')
        freq, times, spec = ssig.spectrogram(filt_dataset[ix], fs, window='hamming', nperseg=1000,
                                             noverlap=1000 - 1, mode='psd')
        x_mesh, y_mesh = np.meshgrid(times, freq[(freq <= fmax) & (freq >= fmin)])
        # log stuff for testing: norm=matplotlib.colors.SymLogNorm(linthresh=0.03)
        im1 = axs2.pcolormesh(x_mesh, y_mesh, np.log10(spec[(freq <= fmax) & (freq >= fmin)]), norm=matplotlib.colors.SymLogNorm(linthresh=0.03))
        # left_end, right_end = calc_xlim(datalen, axs2, fs)
        # axs2.set_xlim(left_end, right_end)
        axs2.set_xticks(ticks=np.arange(0, axs2.get_xlim()[1], 30))
    plt.colorbar(im1, ax=left_side_subs.ravel().tolist(), fraction=0.02)


    spectro_filename = f'Spectrogram_{uuid}_{experiment}_chan_{"-".join(map(str, channels))}.png'
    with open(spectro_filename, 'wb') as sfig:
        plt.savefig(sfig, format='png')
    print('Spectrograms produced.')


def plot_raw_and_filtered(figure, uuid, experiment, channels, raw_dataset, filt_dataset, datalen):
    """

    :param figure:
    :param uuid:
    :param experiment:
    :param channels:
    :param raw_dataset:
    :param filt_dataset:
    :param datalen:
    :return:
    """
    fsemg = 1
    subfigs = figure.subfigures(nrows=len(channels), ncols=1)
    # TODO: Need way to check if more than one subfig was made
    for index in range(len(channels)):
        try:
            subfigs[index].suptitle(f'Channel {channels[index]}')

            # create 1x2 subplots per subfig
            axs = subfigs[index].subplots(nrows=1, ncols=2, subplot_kw={'anchor': 'SW'},
                                          gridspec_kw={'wspace': 0.15})
        except TypeError:
            subfigs.suptitle(f'Channel {channels[index]}')

            # create 1x2 subplots per subfig
            axs = subfigs.subplots(nrows=1, ncols=2, subplot_kw={'anchor': 'SW'},
                                          gridspec_kw={'wspace': 0.15})
        for ax in axs:
            ax.tick_params(bottom=True, labelbottom=True, left=True, labelleft=True, right=False, labelright=False,
                           top=False, labeltop=False)

        raw_plot = raw_dataset[index]
        filt_plot = filt_dataset[index]
        realtime = np.arange(np.size(raw_plot)) / fsemg
        axs[0].plot(realtime, (raw_plot - np.nanmean(raw_plot)) / np.nanstd(raw_plot))
        axs[0].set_xlim(0, datalen)
        # plot filtered data in middle
        axs[1].plot(realtime, (filt_plot - np.nanmean(filt_plot)) / np.nanstd(filt_plot))
        axs[1].set_xlim(0, datalen)

    datapoints_filename = f'Raw_and_filtered_data_{uuid}_{experiment}_chan_{"-".join(map(str, channels))}.png'
    with open(datapoints_filename, 'wb') as dfig:
        plt.savefig(dfig, format='png')
    print('Raw data plots produced.')


def calc_xlim(datalen, axs, fs):
    """
    This function calculates how to set the xlims to show 10 minutes of data.
    :param datalen: Length of data used
    :param axs: Axes object being used
    :param fs: sample rate
    :return: values indicating proper xlimits
    """
    # gets default lims for calculations first
    ax_range = axs.get_xlim()
    leftpoint = ax_range[0]
    rightpoint = ax_range[1]
    # step is the distance default to show the data in the range, so step represents datalen
    step = rightpoint - leftpoint
    # find out how much data we're working with and how large that is compared to 10 min
    # datamin = datalen / fs / 60
    # divide datalen by sample rate to get time of data in seconds
    datasec = datalen / fs
    # now, we have a length of time that corresponds with a length of data
    scalar = 600 / datasec * step

    return leftpoint, leftpoint + scalar


def main(uuid: str, experiment: Union[str, int], outputLocation: str, details: List[str], apply: List[str],
         spect: bool):
    # braingeneers.set_default_endpoint(f'{os.getcwd()}')
    # fsemg = 1
    # fsd is rate to which the downsampling should occur
    fsd = 200
    # TODO: Make more robust - if length is not specified, need to still make it work
    # load metadata
    offset = int(details[0])
    datalen = int(details[1])
    if datalen == -1 or datalen == 'all':
        datalen = -1
    chans = details[2]
    metad = de.load_metadata(uuid)
    e = (int(experiment[-1]) - 1) if isinstance(experiment, str) else experiment
    fs = metad['ephys_experiments'][e]['sample_rate']
    # then, load the data
    if "-" in chans:
        chans = [int(i) for i in chans.split('-')]
    else:
        for_m = details[2]
        chans = np.atleast_1d( int(chans))
    dataset = de.load_data(metad, experiment=experiment, offset=offset, length=datalen, channels=chans)
    try:
        datalen = np.shape(dataset)[1]
    except IndexError:
        datalen = len(dataset)
    # turned off vstack for testing 10/25
    # dataset = np.vstack(dataset)
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
            print('Memory usage: ', psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
            gc.collect()
            filt_dataset = bandpass(dataset, int(hi_rate), int(low_rate), fs)
            print('Memory usage: ', psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
            gc.collect()
            # filt_dataset = filt_dataset[:, ::np.int64(fs/fsd)]
            filt_dataset = np.vstack(filt_dataset)

    # do raw data viz no matter what
    datafig = plt.figure(figsize=(16, 2 * len(chans) + 2))
    plot_raw_and_filtered(figure=datafig, uuid=uuid, experiment=experiment, channels=chans, raw_dataset=dataset,
                          filt_dataset=filt_dataset, datalen=datalen)

    # datapoints_filename = f'Raw_and_filtered_data_{uuid}_{experiment}_chan_{"-".join(map(str, chans))}.png'
    # # spectro_filename = f'Spectrogram_{uuid}_{experiment}_chan_{"-".join(map(str, chans))}.png'
    # with open(datapoints_filename, 'wb') as dfig:
    #     plt.savefig(dfig, format='png')

    if spect:
        specfig = plt.figure(figsize=(16, 2 * len(chans) + 2))
        # fs = fs / fsd
        filt_dataset = filt_dataset[:, ::np.int64(fs / fsd)]
        datalen = np.shape(filt_dataset)[1]
        # filt_dataset = np.hstack(filt_dataset)
        make_spectrogram(figure=specfig, uuid=uuid, channels=chans, filt_dataset=filt_dataset,
                         experiment=experiment, datalen=datalen, fs=fsd)

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
                'channels': chans.tolist() if len(chans) > 1 else for_m
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

        wr.upload(
            local_file=f'{os.getcwd()}/Raw_and_filtered_data_{uuid}_{experiment}_chan_{"-".join(map(str, chans))}.png',
            path=f's3://braingeneersdev/ephys/{uuid}/derived/visuals/Raw_and_filtered_data_{uuid}_{experiment}_chan_{"-".join(map(str, chans))}.png')
        if spect:
            wr.upload(local_file=f'{os.getcwd()}/Spectrogram_{uuid}_{experiment}_chan_{"-".join(map(str, chans))}.png',
                      path=f's3://braingeneersdev/ephys/{uuid}/derived/visuals/Spectrogram_{uuid}_{experiment}_chan_{"-".join(map(str, chans))}.png')
    print('Figures created.')
    plt.close()
    gc.collect()


if __name__ == '__main__':
    # args = parse_args()
    # # pdb.set_trace()
    main(**parse_args())
    # main('2022-07-27-e-Chris_BCC_APV', 'experiment3', [6000, 10000, '0,1,2,3'], ['bandpass=500,9000'])
