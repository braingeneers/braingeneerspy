import numpy as np
import scipy.stats

def temporal_binning(spike_times, bin_size=40):
    """
    Given a sorted list of spike times (with no channel or neuron 
    number information), quantizes time into intervals of bin_size 
    and counts the number of events in each bin.
    """
    # Create enough bins to hold all the spikes.
    n_bins = (spike_times[-1] + bin_size - 1) // bin_size
    counts = np.zeros(n_bins, np.int)
        
    # Put each spike in the appropriate bin.
    for time in spike_times:
        counts[time // bin_size] += 1
    return counts


def find_avalanches(counts, thresh):
    """
    Given a list of spikes per bucket and a threshold number of spike
    events above which a bucket is considered "active", return a list
    of avalanches, represented as (size, duration) pairs.
    """
    avalanches = []
    size, duration = 0, 0
    for count in counts:
        if count > thresh:
            size += count
            duration += 1
        elif duration != 0:
            avalanches.append((size, duration))
            size = duration = 0
    return avalanches


def vuong(data, A, B, deltaK=None): 
    """ 
    Perform's Vuong's closeness test to compare the relative goodness
    of fit between two (non-nested) models A and B following a
    scipy.stats API.  Returns the statistic and its quantile on the
    standard normal distribution. If deltaK, the difference in
    parameter count of A vs. B, is not passed, it will be assumed
    equal to the difference in the length of the distributions'
    argument lists.

    If the statistic is above the (1-alpha) quantile, model A is
    preferred with significance level alpha; likewise, if the
    statistic is below the alpha quantile, model B is preferred at
    the same significance level.  
    """
    # Log likelihood of each individual data point.
    L1 = A.logpdf(data)
    L2 = B.logpdf(data)

    # Count the parameters...
    if deltaK is None:
        deltaK = len(A.args) - len(B.args)

    # Log likelihood ratio, variance estimate, and the stat itself.
    LR = L1.sum() - L2.sum() - 0.5*deltaK*np.log(len(data))
    omega = np.std(L1 - L2)
    stat = LR / omega / np.sqrt(len(data))

    # Return the statistic and its quantile on the standard normal.
    return stat, scipy.stats.norm.cdf(stat)
