import heapq
import numpy as np
from scipy import stats, sparse, optimize
import itertools


class SpikeData():
    '''
    Generic representation for spiking data from spike sorters and
    simulations.
    '''

    def __init__(self, arg1, arg2=None):
        '''
        Parses different argument list possibilities into the desired
        format: a list indexed by unit ID, where each element is
        a list of spike times. The three possibilities accepted are:
        (1) a pair of lists corresponding to unit indices and times,
        (2) a list of lists of spike times, and (3) a list of Neuron
        objects whose parameter spike_time is a list of spike times.

        Spike times should be in units of milliseconds, unless a list
        of Neurons is given; these have spike times in units of
        samples, which are converted to milliseconds using the sample
        rate saved in the Neuron object.
        '''
        if arg2 is not None:
            self.train = _train_from_i_t_list(arg1, arg2)
        else:
            try:
                self.train = [np.asarray(n.spike_time)/n.fs*1e3
                              for n in arg1]
            except AttributeError:
                if all([len(arg)==2 for arg in arg1]):
                    idces = [i for i,_ in arg1]
                    times = [t for i,t in arg1]
                    self.train = _train_from_i_t_list(idces, times)
                else:
                    self.train = arg1

        # Make sure each individual spike train is sorted, because
        # none of the formats guarantee this but all the algorithms
        # expect it.
        self.train = [np.sort(times) for times in self.train]

    @property
    def times(self):
        'Iterate spike times for all units in time order.'
        return heapq.merge(*self.train)

    @property
    def index_time(self):
        'Iterate (index,time) pairs for all units in time order.'
        return heapq.merge(*[zip(itertools.repeat(i), t)
                             for (i,t) in enumerate(self.train)],
                           key=lambda x: x[1])

    def frames(self, length, overlap=0):
        '''
        Iterate new SpikeData objects corresponding to subwindows of
        a given `length` with a fixed `overlap`.
        '''
        interval = length - overlap
        window, events = 1, []
        for (index,time) in self.index_time:
            while time >= window*interval:
                yield SpikeData(events)
                window, events = window+1, []
            events.append((index,time))
        yield SpikeData(events)

    def binned(self, bin_size=40, unit=None):
        '''
        Quantizes time into intervals of bin_size and counts the
        number of events in each bin.
        '''
        bin, count = 1, 0
        for time in self.times:
            while time >= bin*bin_size:
                yield count
                bin, count = bin+1, 0
            count += 1
        yield count

    def subset(self, units):
        'Return a new SpikeData with spike times for only some units.'
        train = [ts for i,ts in enumerate(self.train) if i in units]
        return self.__class__(train)

    def subtime(self, start, end):
        'Return a new SpikeData with only spikes in a time range.'
        train = [t[(t >= start) & (t < end)] for t in self.train]
        return self.__class__(train)

    def sparse_raster(self, bin_size=20):
        '''
        Bin all spike times and create a sparse matrix where entry
        (i,j) is the number of times cell i fired in bin j.
        '''
        indices = np.hstack([ts // bin_size for ts in self.train])
        units = np.hstack([0] + [len(ts) for ts in self.train])
        indptr = np.cumsum(units)
        values = np.ones_like(indices)
        return sparse.csr_matrix((values, indices, indptr))

    def raster(self, bin_size=20):
        '''
        Bin all spike times and create a dense matrix where entry
        (i,j) is the number of times cell i fired in bin j.
        '''
        return self.sparse_raster(bin_size).toarray()

    def interspike_intervals(self):
        'Produce a list of arrays of interspike intervals per unit.'
        return [ts[1:] - ts[:-1] for ts in self.train]

    def burstiness_index(self, bin_size=40):
        '''
        Compute the burstiness index [1], a number from 0 to 1 which
        quantifies synchronization of activity in neural cultures.

        Spikes are binned, and the fraction of spikes accounted for by
        the top 15% will be 0.15 if activity is fully asynchronous, and
        1.0 if activity is fully synchronized into just a few bins. This
        is linearly rescaled to the range 0--1 for clearer interpretation.

        [1] Wagenaar, Madhavan, Pine & Potter. Controlling bursting
            in cortical cultures with closed-loop multi-electrode
            stimulation. J Neurosci 25:3, 680–688 (2005).
        '''
        binned = np.array(list(self.binned(bin_size)))
        binned.sort()
        N85 = int(np.round(len(binned) * 0.85))

        if N85 == len(binned):
            return 1.0
        else:
            f15 = binned[N85:].sum() / binned.sum()
            return (f15 - 0.15) / 0.85

    def spike_time_tiling(self, i, j, delt=20, tmax=None):
        '''
        Given the indices of two units of interest, compute the spike
        time tiling coefficient [1], a metric for causal relationships
        between spike trains with some improved intuitive properties
        compared to the Pearson correlation coefficient.

        [1] Cutts & Eglen. Detecting pairwise correlations in spike
            trains: An objective comparison of methods and application
            to the study of retinal waves. J Neurosci 34:43,
            14288–14303 (2014).
        '''
        tA, tB = self.train[i], self.train[j]

        if tmax is None:
            tmax = max(max(tA), max(tB))

        TA = _sttc_ta(tA, delt, tmax) / tmax
        TB = _sttc_ta(tB, delt, tmax) / tmax

        PA = _sttc_na(tA, tB, delt) / len(tA)
        PB = _sttc_na(tB, tA, delt) / len(tB)

        aa = (PA-TB)/(1-PA*TB) if PA*TB != 1 else 0
        bb = (PB-TA)/(1-PB*TA) if PB*TA != 1 else 0
        return (aa + bb) / 2

    def avalanches(self, thresh, bin_size=40):
        """
        Given a list of spikes per bucket and a threshold number of spike
        events above which a bucket is considered "active", generate the
        spike counts in each bucket.
        """
        this_av = []
        for count in self.binned(bin_size):
            if count > thresh:
                this_av.append(count)
            elif this_av:
                yield this_av
                this_av = []
        if this_av:
            yield this_av


def _train_from_i_t_list(idces, times):
    '''
    Given lists of spike times and indices, produce a list whose
    ith entry is a list of the spike times of the ith unit.
    '''
    idces, times = np.asarray(idces), np.asarray(times)
    ret = []
    for i in range(idces.max()+1):
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
    sorted lists of spike times, calculate the fraction of spikes in
    spike train A within delt of any spike in spike train B. This
    algorithm is my own, modified to use the closed interval instead
    of open after reading the original code.
    '''
    count = 0
    ia, ib = 0, 0

    # Microoptimizations because Python won't do it for me: don't
    # recompute len during each iteration, and subtract once.
    LA, LB = len(tA), len(tB)

    while ia < LA and ib < LB:
        cg = tB[ib] - tA[ia]
        if cg < -delt:
            ib += 1
        elif cg > delt:
            ia += 1
        else:
            ia += 1
            count += 1
    return count


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
    # else, rather than generating infinice Pearson coefficients.
    σx[σx == 0] = np.inf

    # This is by the formula, but there's also a hack to deal with the
    # numerical issues that break the invariant that every variable
    # should have a Pearson autocorrelation of 1.
    corr = np.array(Exy - Ex*Ex.T) / (σx*σx.T)
    np.fill_diagonal(corr, 1)
    return corr


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


def criticality_metric(times):
    """
    Given lists of spike times and indices, separate them into
    avalanches and calculate a metric of criticality.

    The method uses the exponents of the power-law distributions to
    calculate the "Deviation from Criticality Coefficient" (DCC)
    described by Ma (2019). Additionally, quantify the badness of fit
    of the two power laws by comparing the Kolmogorov-Smirnoff
    statistic to the lognormal fits. If either lognormal fit is better
    than the corresponding power-law fit, add the difference in KS
    statistic to the DCC in order to get a metric that takes into
    account the possibility that your culture may not be demonstrating
    power-law-style criticality in the first place.
    """
    times = np.array(times)

    # First, decide on a threshold (per bin) from firing rate quantiles.
    bin_size = 20
    bin_sec = bin_size / 1e3
    rates = np.array(list(temporal_binning(times, bin_size=bin_size)))
    thresh = stats.mstats.mquantiles(rates, [0.3])[0]

    # Now find the avalanches using methods above.
    avalanches = list(get_avalanches(rates, thresh))
    sizes = np.array([sum(av) for av in avalanches])
    durations = np.array([len(av)*bin_sec for av in avalanches])

    # Fit the distributions of avalanche size and duration.
    pl, ln = stats.pareto, stats.lognorm
    sizes_pl = pl(*pl.fit(sizes, floc=0, fscale=thresh))
    sizes_ln = ln(*ln.fit(sizes, floc=0))
    durations_pl = pl(*pl.fit(durations, floc=0, fscale=bin_sec))
    durations_ln = ln(*ln.fit(durations, floc=0))

    # Measure the badness of fit of power-law vs lognormal.
    def badness(points, dist):
        return stats.kstest(points, dist.cdf).statistic
    durations_pl_badness = badness(durations, durations_pl)
    durations_ln_badness = badness(durations, durations_ln)
    sizes_pl_badness = badness(sizes, sizes_pl)
    sizes_ln_badness = badness(sizes, sizes_ln)
    durations_worseness = \
        max(durations_pl_badness - durations_ln_badness, 0)
    sizes_worseness = max(sizes_pl_badness - sizes_ln_badness, 0)
    total_worseness = durations_worseness + sizes_worseness

    # Finally, measure the DCC.
    scale_fit, m_fit = optimize.curve_fit(lambda x, a,b: a*x**b,
                                          durations, sizes)[0]
    m_pred = durations_pl.args[0] / sizes_pl.args[0]
    DCC = abs(m_pred - m_fit)

    # And return the sum of the DCC with the badness of fit!
    return DCC + total_worseness
