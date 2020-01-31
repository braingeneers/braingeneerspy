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


def get_avalanches(counts, thresh):
    """
    Given a list of spikes per bucket and a threshold number of spike
    events above which a bucket is considered "active", return a list
    of avalanches, represented as lists of counts per bucket.
    """
    avalanches = []
    this_one = []
    for count in counts:
        if count > thresh:
            this_one.append(count)
        elif len(this_one) != 0:
            avalanches.append(this_one)
            this_one = []
    return avalanches


def find_avalanches(counts, thresh):
    """
    Given a list of spikes per bucket and a threshold number of spike
    events above which a bucket is considered "active", return the
    sizes and durations of all the avalanches that would be returned
    by get_avalanches with the same arguments.
    """
    avalanches = get_avalanches(counts, thresh)
    sizes = [sum(av) for av in avalanches]
    durations = [len(av) for av in avalanches]
    return sizes, durations


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

def small_world(XY, plocal, beta):
    """
    This model is inspired by the Watts-Strogatz model, but describes
    the relationships between objects with physical positions rather
    than on an abstract cycle. The positions XY describe the
    physical locations of the nodes---a distance less than 1 is
    considered close, so normalize before passing this parameter. The
    local connection probability plocal describes the connection
    probability for nodes near to each other; after these connections
    have been formed, a random fraction beta of them are reassigned to
    arbitrary nodes irrespective of distance.
    """
    N = XY.shape[1]

    # Symmetric array of distances between nodes i and j.
    dij = np.sqrt(np.sum((XY[:,:,None] - XY[:,None,:])**2, axis=0))

    # With probability plocal, connect nodes with distance less than 1.
    edges = (dij < 1) & (np.random.rand(N,N) < plocal)

    # Relocate edges with probability beta, keeping the same origin
    # but changing the target.
    to_change = edges & (np.random.rand(N,N) < beta)
    edges ^= to_change
    new_edges = to_change.sum(0)
    for j,n in enumerate(new_edges):
        targets = np.random.choice(N, size=n, replace=False)
        edges[targets,j] = True

    # Ensure no self-edges and return.
    np.fill_diagonal(edges, False)
    return edges
