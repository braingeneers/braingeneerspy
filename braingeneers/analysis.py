import numpy as np
from scipy import stats, sparse


def sparse_raster(times, idces, cells=None, bin_size=20):
    '''
    Given a list of spike times and the corresponding cells which
    produced them, bin the spike times and create a sparse matrix
    where entry (i,j) is the number of times cell i fired in bin j.

    You can specify which cells should and should not be included in
    the resulting sparse matrix by providing an interable of cell
    indices in the parameter cells. Alternately, you can provide a
    number of cells, in which case it is replaced by a range.
    '''
    # These need to be numpy arrays.
    times, idces = np.asarray(times), np.asarray(idces)

    # We're going to need to iterate over the cells more than once, so
    # convert them to a list; if they're not a list yet, assume
    # they're just a single integer, and replace them with a range.
    try:
        cells = list(cells)
    except TypeError:
        cells = np.arange(cells if cells is not None else idces.max()+1)

    indices = np.hstack([times[idces == i] // bin_size for i in cells])
    indptr = np.cumsum([0] + [(idces == i).sum() for i in cells])
    return sparse.csr_matrix((np.ones_like(indices), indices, indptr))


def pearson(sparse_raster):
    '''
    Compute a Pearson correlation coefficient matrix for a sparse
    spike raster in the format produced by sparse_raster(). Don't use
    this method unless the spike raster is sparse: it's numerically
    worse than the standard method used by scipy for dense matrices.
    '''
    spikes = sparse_raster

    Exy = (spikes @ spikes.T) / spikes.shape[1]
    Ex = np.array(spikes.mean(axis=1))

    # Calculating std is convoluted
    spikes2 = spikes.copy()
    spikes2.data **= 2
    Ex2 = np.array(spikes2.mean(axis=1))
    σx = np.sqrt(Ex2 - Ex**2)

    # Some cells won't fire in the whole observation window.
    # These should be treated as uncorrelated with everything
    # else, rather than generating infinice Pearson coefficients.
    σx[σx == 0] = np.inf

    # This is by the formula, but there's also a hack to deal with the
    # numerical issues that break the invariant that every variable
    # should have a Pearson autocorrelation of 1.
    corr = np.array(Exy - Ex*Ex.T) / (σx*σx.T)
    np.fill_diagonal(corr, 1)
    return corr


def temporal_binning(spike_times, bin_size=40):
    """
    Given an in-order iterable of spike times (with no channel or
    neuron number information), quantizes time into intervals of
    bin_size and counts the number of events in each bin.
    """
    bin, count = 1, 0
    for time in spike_times:
        while time >= bin*bin_size:
            yield count
            bin, count = bin+1, 0
        count += 1
    yield count


def get_avalanches(counts, thresh):
    """
    Given a list of spikes per bucket and a threshold number of spike
    events above which a bucket is considered "active", generate the
    spike counts in each bucket.
    """
    this_av = []
    for count in counts:
        if count > thresh:
            this_av.append(count)
        elif this_av:
            yield this_av
            this_av = []
    if this_av:
        yield this_av


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
    return stat, stats.norm.cdf(stat)


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
