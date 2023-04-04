import argparse
import json
import os
import psutil
# from neuraltoolkit import ntk_filters as ntk
import pdb
import braingeneers
from braingeneers.utils import s3wrangler as wr
from braingeneers.utils import smart_open_braingeneers as smart_open
from braingeneers.utils import common_utils
import numpy as np
from typing import List, Union
from braingeneers.data import datasets_electrophysiology as de
from scipy import signal as ssig
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gc
# from hanging_threads import start_monitoring
# monitoring_thread = start_monitoring()

class EndpointSwitch:
  endpoint = None

  def __enter__(self):
    self.endpoint = braingeneers.get_default_endpoint()
    braingeneers.set_default_endpoint(braingeneers.utils.configure.DEFAULT_ENDPOINT)

  def __exit__(self, type, value, traceback):
    braingeneers.set_default_endpoint(self.endpoint)



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

def list_or_int(value):
    """
    Tests if something is a list or int.
    :param value:
    :return:
    """
    try:
        len(value)
        return value
    except TypeError:
        return value
def parse_args():
    """
    This function parses the arguments passed in via CLI
    :return: Dictionary of parsed arguments
    """
    # TODO: have parameters for range for raw trace
    parser = argparse.ArgumentParser(description='Create visuals for single neural recording file.')
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
    # TODO: argument for which lfp band should should show up. Maybe change main to flexibly deal with no filter specified? Need to implement in main.
    parser.add_argument(
        '--lfp', '-l', required=True, choices=['delta', 'theta', 'alpha', 'beta'],
        help='LFP band to filter for, --apply-filter is specified 1 or more times for each filter. Usage options:\n'
             '--lfp delta    (Filter for delta frequency)'
    )
    parser.add_argument(
        '--apply', action='append', required=False, default="['lowpass=100', 'bandpass=1,4']",
        help='Filter type + arguments, --apply-filter is specified 1 or more times for each filter. Usage options:\n'
             '--apply highpass=750    (highpass filter @ 750 hz)\n'
             '--apply lowpass=8       (lowpass filter @ 8 hz)\n'
             '--apply bandpass=low,high  (bandpass values for between the low and high arguments)'
    )
    parser.add_argument('--order', '-r', type=int, default=3, required=False,
                        help='Choice of order for filtering. 3 by default.')
    parser.add_argument('--spect', '-s', action='store_true', default=False, required=False,
                        help='Choice to make spectrogram or not. False by default.')
    parser.add_argument('--rawnge', '-n', nargs='+', required=False,
                        help='Range to plot raw trace, in minutes. Usage options: '
                             '--range 4 5 ( plot raw trace between 4 and 5 minutes)'
                             '--range 4 ( plot raw trace using 4 as midpoint)'
                             'If unspecified (option not used), will default to middle of raw dataset.')
    parser.add_argument('--width', '-w', type=int, default=10, required=False,
                        help='Choice of spectrogram range in minutes. 10 by default.')


    return vars(parser.parse_args())

def butter_helper(data, low, high, fs, order):
    print("Doing Sury bandpass")
    data_bandpass = np.vstack([butter_bandpass_filter(channel_data, low, high, fs, order) for channel_data in data])

    return  data_bandpass
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """

    :param data: data to filter
    :param lowcut: indicates highpass cutoff bandwidth
    :param highcut: indicates lowpass cutoff bandwidth
    :param fs: sampling rate of data
    :param order:
    :return:
    """

    band = [lowcut, highcut]
    assert len(band) == 2, "Must have lowcut and highcut!"
    Wn = [e / fs * 2 for e in band]

    filter_coeff = ssig.iirfilter(order, Wn, analog=False, btype='bandpass',
                                          ftype='butter', output='sos')
    filtered_traces = ssig.sosfiltfilt(filter_coeff, data, axis=0)
    return filtered_traces

def lfp_filter(data,lowlim,highlim,fs,order=5):
    """
    This filters the data to pull the lfp out.
    :param data: data pulled from maxwell experiment
    :param lowlim: Fl, lowest frequency data can have
    :param highlim: fH, highest frequency data can have
    :param fs: sampling rate
    :param order: order of filter used. 4 by default since Tal used that.
    :return: filtered dataset
    """
    fsd = 1000
    # first lowpass for 100 Hz
    # low_data = lowpass(data, hz=100,fs=fs,order=order)

    # try bandpassing for lowpass and see

    low_data = butter_helper(data, 0.01, 100, fs, order)
    #then, downsample to 1000 Hz
    low_data = low_data[:, ::np.int64(fs / fsd)]
    # then bandpass for frequencies desired
    low_data_b = butter_helper(low_data, lowlim, highlim, fsd, order)

    return low_data_b


# def highpass(data: np.ndarray, hz: int, fs: int, order: int):
#     """
#
#     :param data:
#     :param hz:
#     :param fs:
#     :return:
#     """
#     data_highpass = np.vstack([
#         ntk.butter_highpass(channel_data, highpass=hz, fs=fs, order=order)
#         for channel_data in data
#     ])
#     print(
#         f'Running highpass filter with parameters: hz={hz}, fs={fs}, input shape: {data.shape}, output shape: {data_highpass.shape}\n')
#     return data_highpass


# def lowpass(data: np.ndarray, hz: int, fs: int, order: int):
#     """
#
#     :param data:
#     :param hz:
#     :param fs:
#     :return:
#     """
#     data_lowpass = np.vstack([
#         ntk.butter_lowpass(channel_data, lowpass=hz, fs=fs, order=order)
#         for channel_data in data
#     ])
#     print(
#         f'Running lowpass filter with parameters: hz={hz}, fs={fs}, input shape: {data.shape}, output shape: {data_lowpass.shape}\n')
#     return data_lowpass


# def bandpass(data: np.ndarray, hz_high: int, hz_low: int, fs: int, order: int):
#     """
#
#     :param data:
#     :param hz_high:
#     :param hz_low:
#     :param fs:
#     :return:
#     """
#     data_bandpass = np.vstack([
#         ntk.butter_bandpass(channel_data, highpass=hz_high, lowpass=hz_low, fs=fs, order=order)
#         for channel_data in data
#     ])
#     print(
#         f'Running bandpass filter with parameters: highpass={hz_high}, lowpass={hz_low} fs={fs}, input shape: {data.shape}, output shape: {data_bandpass.shape}\n')
#     return data_bandpass


def make_spectrogram(figure, uuid, experiment, datalen,width, expname, channels, filt_dataset, fs):
    """

    :param expname:
    :param figure:
    :param uuid:
    :param experiment:
    :param datalen:
    :param channels:
    :param filt_dataset:
    :param fs:
    :return:
    """
    print(f'Shape of downsampled data: {np.shape(filt_dataset)}')
    # figure.suptitle(f"Spectrograms for {expname}")
    spectro_subfigs = figure.subfigures(nrows=1, ncols=1)
    spectro_subfigs.suptitle(f"Spectrograms for {expname}")
    spectro_subfigs.supxlabel("Time (min)", fontsize='x-large')
    spectro_subfigs.supylabel("Frequency (Hz)", fontsize='x-large')
    fmax = 64
    fmin = 1

    wide = 10 if width == None else width
    # TODO: test using code block below
    # left_side_subs = spectro_subfigs.subplots(nrows=len(channels), ncols=2)
    # plt.subplots_adjust(hspace=0.3)
    # axs1 = left_side_subs[0]
    # axs1.tick_params(bottom=True, labelbottom=True, left=True, labelleft=True, right=False,
    #                  labelright=False, top=False, labeltop=False)
    # chunk_data = filt_dataset[0][0:2000]
    # realtime = np.arange(np.size(chunk_data))
    # axs1.plot(realtime, (chunk_data - np.nanmean(chunk_data)) / np.nanstd(chunk_data))
    # axs2 = left_side_subs[1]
    # axs2.tick_params(bottom=True, labelbottom=True, left=True, labelleft=True, right=False,
    #                  labelright=False, top=False, labeltop=False)
    # freq, times, spec = ssig.spectrogram(filt_dataset[0], fs, window='hamming', nperseg=1000,
    #                                      noverlap=1000 - 1, mode='psd')
    # x_mesh, y_mesh = np.meshgrid(times, freq[(freq <= fmax) & (freq >= fmin)])
    # # log stuff for testing: norm=matplotlib.colors.SymLogNorm(linthresh=0.03)
    # im1 = axs2.pcolormesh(x_mesh, y_mesh, np.log10(spec[(freq <= fmax) & (freq >= fmin)]),
    #                       cmap='jet')
    # left_end, right_end = calc_xlim(datalen, axs2, fs)
    # axs2.set_xlim(left_end, right_end)


    # axs2.set_xticks(ticks=np.arange(0, axs2.get_xlim()[1], 120))
    # make subplots for left side
    # print("pre-squeeze: ",filt_dataset.shape)
    # print("post-squeeze: ",np.squeeze(filt_dataset).shape)
    print("default endpoint", braingeneers.get_default_endpoint())
    # with smart_open.open(f's3://braingeneersdev/ephys/{uuid}/derived/{experiment}_downsampled_data.npy', 'wb') as f:
    #     np.save(f, np.squeeze(filt_dataset))
    left_side_subs = spectro_subfigs.subplots(nrows=len(channels), ncols=1)
    left_side_subs = np.array([left_side_subs]) if not isinstance(left_side_subs, np.ndarray) else left_side_subs
    plt.subplots_adjust(hspace=0.3)
    for ix in range(len(left_side_subs)):
        axs2 = left_side_subs[ix]
        axs2.tick_params(bottom=True, labelbottom=True, left=True, labelleft=True, right=False,
                         labelright=False, top=False, labeltop=False)
        # set title for each subplot
        axs2.set_title(label=f'Channel {channels[ix]}')

        freq, times, spec = ssig.spectrogram(np.squeeze(filt_dataset[ix]), fs, window='hamming', nperseg=1000,
                                             noverlap=1000 - 1, mode='psd')
        x_mesh, y_mesh = np.meshgrid(times, freq[(freq <= fmax) & (freq >= fmin)])
        # log stuff for testing: norm=matplotlib.colors.SymLogNorm(linthresh=0.03)
        im1 = axs2.pcolormesh(x_mesh, y_mesh, np.log10(spec[(freq <= fmax) & (freq >= fmin)]), cmap='jet')
        # print(axs2.get_xlim())
        left_end, right_end = calc_xlim(datalen,wide, axs2, fs)
        axs2.set_xlim(left_end, right_end)
        ticklist = np.arange(axs2.get_xlim()[0], axs2.get_xlim()[1], 55.005)
        label_list = [str(round(x/55.005)) for x in ticklist]
        axs2.set_xticks(ticks=ticklist, labels=label_list)

        axs2.set_yscale('log')
        # print(axs2.get_ylim())
        # axs2.set_ylim(0, 64)
    plt.colorbar(im1, ax=left_side_subs.ravel().tolist(), fraction=0.02)
    # figure.text(0.5, 0.1, 'Time (min)', ha='center')
    # figure.text(0.1, 0.5, 'Frequency (Hz)', va='center', rotation='vertical')


    spectro_filename = f'Spectrogram_{uuid}_{experiment}_chan_{"-".join(map(str, channels))}.png'
    with open(spectro_filename, 'wb') as sfig:
        plt.savefig(sfig, format='png')
    print('Spectrograms produced.')


def plot_raw_and_filtered(figure, uuid, experiment, channels,lfp_type,rawnge, raw_dataset, filt_dataset, datalen, fs):
    """

    :param figure: Figure object passed in to use
    :param uuid: UUID of dataset as a string
    :param experiment: experiment number as a string
    :param channels: list of channels used
    :param lfp_type:
    :param range: from where the raw data should be taken from, as a list. if only one value is given, treats it as midpoint.
                    If no values are given, default is middle of the dataset.
    :param raw_dataset:
    :param filt_dataset:
    :param datalen:
    :param fs:
    :return:
    """

    #Tal's paper says to downsample to 1Hz
    # fsd = 1000
    fsemg = 1
    # fdata = filt_dataset[:, ::np.int64(fs / fsd)]
    # rdata = raw_dataset[:, ::np.int64(fs / fsd)]
    flen = filt_dataset.shape[1]
    rlen = raw_dataset.shape[1]

    # check what range looks like. If range is None (unspecified)
    try:
        # change int value to reflect frames
        start = rawnge[0] * fs * 60
        spread = "middle" if len(rawnge)==1 else "forward"
    except TypeError:
        # this means rawnge was None, and should be treated to the default case.
        start = int(rlen / 2)
        spread = "middle"
    subfigs = figure.subfigures(nrows=len(channels), ncols=1)
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
            ax.set_xlabel("Time (sec)")
        # TODO: testing 20 seconds of data plotting
        if spread == "middle":
            raw_plot = raw_dataset[index][start - (5 * fs):start + (5 * fs)]
            # uses 1000 since it was downsampled
            filt_plot = filt_dataset[index][int(flen / 2) - (5 * 1000):int(flen / 2) + (5 * 1000)]
        else:
            raw_plot = raw_dataset[index][start :start + (10 * fs)]
            # uses 1000 since it was downsampled
            filt_plot = filt_dataset[index][int(flen / 2):int(flen / 2) + (10 * 1000)]
        realtime = np.arange(np.size(raw_plot)) / fsemg
        filt_time = np.arange(np.size(filt_plot)) / fsemg
        axs[0].plot(realtime, (raw_plot - np.nanmean(raw_plot)) / np.nanstd(raw_plot))
        axs[0].set_xlim(0, len(raw_plot))
        axs[0].set_ylim(-4, 4)
        ticklist_r = np.arange(0, axs[0].get_xlim()[1], fs)
        label_list_r = [str(round(x/fs)) for x in ticklist_r]
        axs[0].set_xticks(ticks=ticklist_r, labels=label_list_r)
        axs[0].legend(labels=['raw data'])

        # plot filtered data in middle
        axs[1].plot(filt_time, (filt_plot - np.nanmean(filt_plot)) / np.nanstd(filt_plot))
        axs[1].set_xlim(0, len(filt_plot))
        axs[1].set_ylim(-4, 4)
        # Do i need to make this flexible???
        ticklist_f = np.arange(0, axs[1].get_xlim()[1], 1000)
        label_list_f = [str(round(x/1000)) for x in ticklist_f]
        axs[1].set_xticks(ticks=ticklist_f, labels=label_list_f)
        axs[1].legend(labels=[f'{lfp_type}'])

    datapoints_filename = f'Raw_and_filtered_data_{uuid}_{experiment}_chan_{"-".join(map(str, channels))}.png'
    with open(datapoints_filename, 'wb') as dfig:
        plt.savefig(dfig, format='png')
    print('Raw data plots produced.')


def calc_xlim(datalen,width, axs, fs):
    """
    This function calculates how to set the xlims to show 30 minutes of data.
    :param datalen: Length of data used
    :param axs: Axes object being used
    :param fs: sample rate
    :return: values indicating proper xlimits
    """
    # gets default lims for calculations first
    # TODO: Somehow need to get the width for 1 min of data to flexibly make the spectrograms. Try the one in raw data plot.
    wsec = width * 60
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
    scalar = wsec / datasec * step

    return leftpoint, leftpoint + scalar


def main(uuid: str, experiment: Union[str, int], outputLocation: str, details: List[str], lfp: str, apply: List[str], order: int,
         spect: bool, rawnge: List[int] = None, width: int = None):
    # braingeneers.set_default_endpoint(f'{os.getcwd()}')
    # fsemg = 1
    # fsd is rate to which the downsampling should occur
    print('Visualization starting')
    fsd = 200
    try:
        rawnge = [int(x) for x in rawnge]
    except TypeError:
        pass
    # load metadata
    offset = int(details[0])
    datalen = details[1]
    if datalen == -1 or datalen == 'all':
        datalen = -1
    else:
        datalen = int(datalen)

    chans = details[2]
    # print(braingeneers.get_default_endpoint())
    # print(de.get_basepath())
    metad = de.load_metadata(uuid)
    # if experiment is a string, need to keep it for uploading but have an int.
    # If it's an int, need to make it a string but still use it.
    # e will be the int version, exp will be the string.
    # TODO: Go into metadata and use experiment number to get the name of the Trace
    e = (int(experiment[-1]) - 1) if isinstance(experiment, str) else experiment
    exp = f'experiment{experiment + 1}' if isinstance(experiment, int) else experiment
    # now that it's a dictionary under ephys experiments, must use .keys() to access properly
    # for x in range(len(metad['ephys_experiments'].keys()) - 1):
    #     # exp_key = next(iter(metad['ephys_experiments'].keys()))
    #
    # exp_key = metad['ephys_experiments'].keys()[e]
    mlist = list(metad['ephys_experiments'])
    keyname  = mlist[e]
    hardware = metad['ephys_experiments'][keyname]['hardware']
    expname = keyname if hardware=="Maxwell" else exp
    fs = metad['ephys_experiments'][keyname]['sample_rate']
    # then, load the data
    if "-" in chans:
        split_up = chans.split('-')
        chans = [int(i) for i in split_up]
        for_m = ','.join(split_up)
    else:
        for_m = details[2]
        chans = np.atleast_1d( int(chans))
    dataset = de.load_data(metad, experiment=keyname, offset=offset, length=datalen, channels=chans, dtype=np.float32)
    print("shape of entire dataset:", np.shape(dataset))
    # print(dataset.dtype)
    try:
        datalen = np.shape(dataset)[1]
    except IndexError:
        datalen = len(dataset)
    # turned off vstack for testing 10/25
    # dataset = np.vstack(dataset)

    #TESTING NEW FILTER FOR LFP
    # TODO: get params from lfp - do this later
    if lfp == "alpha":
        lowlim = 8
        highlim = 13
    elif lfp == "beta":
        lowlim = 13
        highlim = 30
    elif lfp == "delta":
        lowlim = 0.5
        highlim = 4

    elif lfp == "theta":
        lowlim = 4
        highlim = 8
    # for param in apply:
    #     filt, args = param.split('=')
    #     # if there is a comma in args, it contains bandpass limits
    #     vals = args.split(',') if "," in args else args
    #     filt_data = lfp_filter(dataset, )

    filt_dataset = lfp_filter(dataset, lowlim=lowlim, highlim=highlim, fs=fs, order=order)


    # parse out apply list
    # for item in apply:
    #     filt, arg = item.split('=')
    #     if filt == 'highpass':
    #         filt_dataset = highpass(dataset, int(arg), fs, order)
    #         filt_dataset = np.vstack(filt_dataset)
    #     elif filt == 'lowpass':
    #         filt_dataset = lowpass(dataset, int(arg), fs, order)
    #         filt_dataset = np.vstack(filt_dataset)
    #     elif filt == 'bandpass':
    #         # 7/29/22 switched low and high since the arguments should be different
    #         hi_rate, low_rate = arg.split(',')
    #         print('Memory usage: ', psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
    #         gc.collect()
    #         filt_dataset = bandpass(dataset, int(hi_rate), int(low_rate), fs, order)
    #         filt_data_2 = butter_helper(dataset, int(hi_rate), int(low_rate), fs, order)
    #         print('Memory usage: ', psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
    #         gc.collect()
    #         # filt_dataset = filt_dataset[:, ::np.int64(fs/fsd)]
    #         filt_data_2 = np.vstack(filt_data_2)
    #         filt_dataset = np.vstack(filt_dataset)

    # do raw data viz no matter what
    datafig = plt.figure(figsize=(16, 2 * len(chans) + 4))
    plot_raw_and_filtered(figure=datafig, uuid=uuid, experiment=keyname, lfp_type=lfp, rawnge=rawnge, channels=chans, raw_dataset=dataset,
                          filt_dataset=filt_dataset, datalen=datalen, fs=fs)

    # datapoints_filename = f'Raw_and_filtered_data_{uuid}_{experiment}_chan_{"-".join(map(str, chans))}.png'
    # # spectro_filename = f'Spectrogram_{uuid}_{experiment}_chan_{"-".join(map(str, chans))}.png'
    # with open(datapoints_filename, 'wb') as dfig:
    #     plt.savefig(dfig, format='png')

    if spect:
        specfig = plt.figure(figsize=(16, 2 * len(chans) + 4))
        # fs = fs / fsd
        downsampled_dataset = dataset[:, ::np.int64(fs / fsd)]
        dsp_datalen = np.shape(downsampled_dataset)[1]
        # filt_dataset = np.hstack(filt_dataset)
        make_spectrogram(figure=specfig, uuid=uuid, channels=chans, filt_dataset=downsampled_dataset,
                         experiment=keyname,expname=expname, datalen=dsp_datalen, fs=fsd, width=width)

    # then, if it's meant to be on s3, awswrangle it up there.
    print("before switch", wr.config.s3_endpoint_url)
    with EndpointSwitch():
        if outputLocation == 's3':
            # Check if file exists
            # endpoint = braingeneers.get_default_endpoint()
            # braingeneers.set_default_endpoint(braingeneers.utils.configure.DEFAULT_ENDPOINT)
            print("after switch", wr.config.s3_endpoint_url)

            file_exists = common_utils.file_exists(
                f's3://braingeneersdev/ephys/{uuid}/derived/{exp}_visualization_metadata.json')
            # if braingeneers.get_default_endpoint().startswith("http"):
            #
            #     file_exists = common_utils.file_exists(f's3://braingeneersdev/ephys/{uuid}/derived/{exp}_visualization_metadata.json')
            # else:
            #     file_exists = common_utils.file_exists(f'{os.getcwd()')
            # try:
            #     with smart_open.open(f's3://braingeneersdev/ephys/{uuid}/derived/{exp}_visualization_metadata.json'):
            #         file_exists = True
            #         print('File exists')
            # except OSError:
            #     file_exists = False
            #     print('File does not exist')
            # # pdb.set_trace()
            # except Exception as e:
            #     print(f'I got {e} instead')
            if not file_exists:
                # make metadata
                new_meta = {
                    'notes': f'Raw data, filtered data, and spectrogram for {uuid} {exp} ',
                    'hardware': hardware,
                    'channels': for_m
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
                print("before wrangling", wr.config.s3_endpoint_url)
                wr.upload(local_file=f'{os.getcwd()}/{uuid}_viz_metadata.json',
                          path=f's3://braingeneersdev/ephys/{uuid}/derived/{exp}_visualization_metadata.json')

            else:
                # if the metadata exists, need to add this metadata onto the existing one
                # TODO: Need to find good way to add channels together. Consider single channel, more than one, etc.
                # TODO: will it pull a list from the metadata or just a string? Check with maxwell data too

                newchans = set()
                with smart_open.open(
                        f's3://braingeneersdev/ephys/{uuid}/derived/{exp}_visualization_metadata.json') as old_json:
                    fixed_meta = json.load(old_json)
                    cur_chans = fixed_meta['channels']
                    # if there's more than one value, split it and put all into a set
                    if ',' in cur_chans:
                        for chan in cur_chans.split(','):
                            newchans.add(chan)
                    else:
                        newchans.add(cur_chans)
                    # TODO: Add all channels to list. Since chans is a list of int (normally), could maybe just join into string and set as new value? -> won't deal with repeats
                    if len(chans) > 1:
                        for chan in chans:
                            newchans.add(str(chan))
                    # for single channels
                    elif len(chans) == 1:
                        newchans.add(str(chans[0]))

                    # for chan in chans:
                    #     newchans.add(str(chan))
                    #new_chans = sorted(list(set(old_chans + chans.astype(str))))
                    fixed_meta['channels'] = ','.join(sorted(list(newchans)))
                with smart_open.open(f's3://braingeneersdev/ephys/{uuid}/derived/{exp}_visualization_metadata.json',
                                     'w') as out_metadata:
                    json.dump(fixed_meta, out_metadata, indent=2)

                # braingeneers.set_default_endpoint(endpoint)


                # llllll = Spectrogram_{uuid}_{experiment}_chan_{"-".join(map(str, channels))}.png'

            wr.upload(
                local_file=f'{os.getcwd()}/Raw_and_filtered_data_{uuid}_{keyname}_chan_{"-".join(map(str, chans))}.png',
                path=f's3://braingeneersdev/ephys/{uuid}/derived/visuals/{exp}/raw_and_filtered_data/{uuid}_{expname}_frame_{offset}-{offset + datalen}_chan_{"-".join(map(str, chans))}.png')
            print("Raw data figure uploaded.")
            if spect:
                wr.upload(local_file=f'{os.getcwd()}/Spectrogram_{uuid}_{keyname}_chan_{"-".join(map(str, chans))}.png',
                          path=f's3://braingeneersdev/ephys/{uuid}/derived/visuals/{exp}/spectrogram/{uuid}_{expname}_frame_{offset}-{offset + datalen}_chan_{"-".join(map(str, chans))}.png')
                print("Spectrogram uploaded.")
    print('Figures created.')
    plt.close()
    gc.collect()


if __name__ == '__main__':

    main(**parse_args())
    # main('2022-07-27-e-Chris_BCC_APV', 'experiment3', [6000, 10000, '0,1,2,3'], ['bandpass=500,9000'])
