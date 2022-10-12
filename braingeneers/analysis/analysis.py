import heapq
import numpy as np
from scipy import sparse, stats, signal
import itertools
from collections import namedtuple
import powerlaw
import os
import contextlib
from braingeneers.utils.data_access_objects import SpikeData, ThresholdedSpikeData


DCCResult = namedtuple('DCCResult', 'dcc p_size p_duration')


def filter(raw_data, fs_Hz=20000, filter_order=3,
                filter_lo_Hz=300, filter_hi_Hz=6000,
                time_step_size_s=10, channel_step_size = 100,
                verbose = 0, zi = None, return_zi = False):
    '''
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
    '''
    

    time_step_size = int(time_step_size_s * fs_Hz)
    data = np.zeros_like(raw_data)
    

    # Get filter params
    b, a = signal.butter(fs=fs_Hz, btype='bandpass', #output='sos',
                        N=filter_order, Wn=[filter_lo_Hz, filter_hi_Hz])

    if zi is None:
        # Filter initial state
        zi = signal.lfilter_zi(b, a)
        zi = np.vstack([zi*raw_data[ch,0] for ch in range(raw_data.shape[0])])
    

    # Step through the data in chunks and filter it
    for ch_start in range(0, raw_data.shape[0], channel_step_size):
        ch_end = min(ch_start + channel_step_size, raw_data.shape[0])
        
        if verbose:
            print(f'Filtering channels {ch_start} to {ch_end}')

        for t_start in range(0, raw_data.shape[1], time_step_size):
            t_end = min(t_start + time_step_size, raw_data.shape[1])
            
            data[ch_start:ch_end, t_start:t_end], zi[ch_start:ch_end,:] = signal.lfilter(
                    b, a, raw_data[ch_start:ch_end, t_start:t_end], 
                    axis=1, zi=zi[ch_start:ch_end,:])
    
    return data if not return_zi else (data, zi)


        # # Step through the data in chunks, filtering each one.
        # for f_step in np.arange(filter_step_size, raw_data.shape[1],
        #                         filter_step_size):
        #     # Filter one chunk of data.
            
        #     print('filtering')
        #     y, zi = signal.lfilter(b, a,
        #                 raw_data[:, f_step-filter_step_size:f_step], zi=zi)
        #     print('Did filter step', f_step/fs_Hz,'seconds')
        #     # Save the filtered chunk.
        #     data[:, f_step-filter_step_size:f_step] = y

        #     # If its the last chunk of data
        #     if f_step + filter_step_size > raw_data.shape[1]:
        #         # Filter the last chunk of data.
        #         y, zi = signal.lfilter(b, a,
        #                     raw_data[:, f_step:], zi=zi)
        #         # Save the filtered chunk.
        #         data[:, f_step:] = y
            

def _resampled_isi(spikes, times):
    '''
    Calculate the firing rate of a spike train at specific times, based on
    the reciprocal inter-spike interval. It is assumed to have been sampled
    halfway between any two given spikes, and then linearly interpolated.
    '''
    if len(spikes) == 0:
        return np.zeros_like(times)
    elif len(spikes) == 1:
        return np.ones_like(times) / spikes[0]
    else:
        x = 0.5*(spikes[:-1] + spikes[1:])
        y = 1/np.diff(spikes)
        return np.interp(times, x, y)


def _p_and_alpha(data, N_surrogate=1000, pval_truncated=0.0):
    '''
    Perform a power-law fit to some data, and return a p-value for the
    hypothesis that this fit is poor, together with just the exponent
    of the fit.

    A positive value of `pval_truncated` means to allow the hypothesis
    of a truncated power law, which must be better than the plain
    power law with the given significance under powerlaw's default
    nested hypothesis comparison test.

    The returned significance value is computed by sampling N surrogate
    datasets and counting what fraction are further from the fitted
    distribution according to the one-sample Kolmogorov-Smirnoff test.
    '''
    # Perform the fits and compare the distributions with IO
    # silenced because there's no option to disable printing
    # in this library...
    with open(os.devnull, 'w') as f, \
            contextlib.redirect_stdout(f), \
            contextlib.redirect_stderr(f):
        fit = powerlaw.Fit(data)
        stat, p = fit.distribution_compare('power_law',
                                           'truncated_power_law',
                                           nested=True)

    # If the truncated power law is a significantly better
    # explanation of the data, use it.
    if stat < 0 and p < pval_truncated:
        dist = fit.truncated_power_law
    else:
        dist = fit.power_law

    # The p-value of the fit is the fraction of surrogate
    # datasets which it fits worse than the input dataset.
    ks = stats.ks_1samp(data, dist.cdf)
    p = np.mean([stats.ks_1samp(dist.generate_random(len(data)),
                                dist.cdf) > ks
                 for _ in range(N_surrogate)])
    return p, dist.alpha


def _train_from_i_t_list(idces, times, N):
    '''
    Given lists of spike times and indices, produce a list whose
    ith entry is a list of the spike times of the ith unit.
    '''
    idces, times = np.asarray(idces), np.asarray(times)
    if N is None:
        N = idces.max() + 1

    ret = []
    for i in range(N):
        ret.append(times[idces == i])
    return ret


def fano_factors(raster):
    '''
    Given arrays of spike times and the corresponding units which
    produced them, computes the Fano factor of the corresponding spike
    raster.

    If a unit doesn't fire, a Fano factor of 1 is returned because in
    the limit of events happening at a rate ε->0, either as
    a Bernoulli process or in the many-bins limit of a single event,
    the Fano factor converges to 1.
    '''
    if sparse.issparse(raster):
        mean = np.array(raster.mean(1)).ravel()
        moment = np.array(raster.multiply(raster).mean(1)).ravel()

        # Silly numbers to make the next line return f=1 for a unit
        # that never spikes.
        moment[mean == 0] = 2
        mean[mean == 0] = 1

        # This is the variance/mean ratio computed in a sparse-friendly
        # way. This algorithm is numerically unstable in general, but
        # should only be a problem if your bin size is way too big.
        return moment/mean - mean

    else:
        mean = np.asarray(raster).mean(1)
        var = np.asarray(raster).var(1)
        mean[mean == 0] = var[mean == 0] = 1.0
        return var / mean


def _sttc_ta(tA, delt, tmax):
    '''
    Helper function for spike time tiling coefficients: calculate the
    total amount of time within a range delt of spikes within the
    given sorted list of spike times tA.
    '''
    if len(tA) == 0:
        return 0

    base = min(delt, tA[0]) + min(delt, tmax - tA[-1])
    return base + np.minimum(np.diff(tA), 2*delt).sum()


def _sttc_na(tA, tB, delt):
    '''
    Helper function for spike time tiling coefficients: given two
    sorted lists of spike times, calculate the number of spikes in
    spike train A within delt of any spike in spike train B.
    '''
    if len(tB) == 0:
        return 0
    tA, tB = np.asarray(tA), np.asarray(tB)

    # Find the closest spike in B after spikes in A.
    iB = np.searchsorted(tB, tA)

    # Clip to ensure legal indexing, then check the spike at that
    # index and its predecessor to see which is closer.
    np.clip(iB, 1, len(tB)-1, out=iB)
    dt_left = np.abs(tB[iB] - tA)
    dt_right = np.abs(tB[iB-1] - tA)

    # Return how many of those spikes are actually within delt.
    return (np.minimum(dt_left, dt_right) <= delt).sum()


def pearson(spikes):
    '''
    Compute a Pearson correlation coefficient matrix for a spike
    raster. Includes a sparse-friendly method for very large spike
    rasters, but falls back on np.corrcoef otherwise because this
    method can be numerically unstable.
    '''
    if not sparse.issparse(spikes):
        return np.corrcoef(spikes)

    Exy = (spikes @ spikes.T) / spikes.shape[1]
    Ex = np.array(spikes.mean(axis=1))

    # Calculating std is convoluted
    spikes2 = spikes.copy()
    spikes2.data **= 2
    Ex2 = np.array(spikes2.mean(axis=1))
    σx = np.sqrt(Ex2 - Ex**2)

    # Some cells won't fire in the whole observation window.
    # These should be treated as uncorrelated with everything
    # else, rather than generating infinite Pearson coefficients.
    σx[σx == 0] = np.inf

    # This is by the formula, but there's also a hack to deal with the
    # numerical issues that break the invariant that every variable
    # should have a Pearson autocorrelation of 1.
    corr = np.array(Exy - Ex*Ex.T) / (σx*σx.T)
    np.fill_diagonal(corr, 1)
    return corr


def cumulative_moving_average(hist):
    'The culmulative moving average for a histogram. Return a list of CMA.'
    ret = []
    for h in hist:
        cma = 0
        cma_list = []
        for i in range(len(h)):
            cma = (cma * i + h[i]) / (i+1)
            cma_list.append(cma)
        ret.append(cma_list)
    return ret


def burst_detection(spike_times, burst_threshold, spike_num_thr=3):
    '''
    Detect burst from spike times with a interspike interval
    threshold (burst_threshold) and a spike number threshold (spike_num_thr).
    Returns:
        spike_num_list -- a list of burst features
          [index of burst start point, number of spikes in this burst]
        burst_set -- a list of spike times of all the bursts.
    '''
    spike_num_burst = 1
    spike_num_list = []
    for i in range(len(spike_times)-1):
        if spike_times[i+1] - spike_times[i] <= burst_threshold:
            spike_num_burst += 1
        else:
            if spike_num_burst >= spike_num_thr:
                spike_num_list.append([i-spike_num_burst+1, spike_num_burst])
                spike_num_burst = 1
            else:
                spike_num_burst = 1
    burst_set = []
    for loc in spike_num_list:
        for i in range(loc[1]):
            burst_set.append(spike_times[loc[0]+i])
    return spike_num_list, burst_set
