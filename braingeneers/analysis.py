import heapq
import numpy as np
from scipy import sparse, special, optimize
import itertools


class SpikeData():
    '''
    Generic representation for spiking data from spike sorters and
    simulations.
    '''

    def __init__(self, arg1, arg2=None, *, N=None, length=None):
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

        # The length of the spike train defaults to the last spike
        # time it contains.
        if length is None:
            length = max((t[-1] for t in self.train if len(t) > 0))
        self.length = length

        # If a number of units was provided, make the list of spike
        # trains consistent with that number.
        if N is not None and len(self.train) < N:
            self.train += [[]]*(N - len(self.train))
        self.N = len(self.train)

    @property
    def times(self):
        'Iterate spike times for all units in time order.'
        return heapq.merge(*self.train)

    @property
    def events(self):
        'Iterate (index,time) pairs for all units in time order.'
        return heapq.merge(*[zip(itertools.repeat(i), t)
                             for (i,t) in enumerate(self.train)],
                           key=lambda x: x[1])

    def idces_times(self):
        '''
        Return separate lists of times and indices, e.g. for raster
        plots. This is not a property unlike `times` and `events`
        because the lists must actually be constructed in memory.
        '''
        idces, times = [], []
        for i,t in self.events:
            idces.append(i)
            times.append(t)
        return np.array(idces), np.array(times)

    def frames(self, length, overlap=0):
        '''
        Iterate new SpikeData objects corresponding to subwindows of
        a given `length` with a fixed `overlap`.
        '''
        interval = length - overlap
        window, events = 1, []
        for (index,time) in self.events:
            while time >= window*interval:
                yield SpikeData(events, length=interval)
                window, events = window+1, []
            events.append((index, time - (window-1)*interval))
        yield SpikeData(events, length=interval)

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
        return self.__class__(train, length=self.length)

    def subtime(self, start, end):
        'Return a new SpikeData with only spikes in a time range.'
        train = [t[(t >= start) & (t < end)] - start
                 for t in self.train]
        return self.__class__(train, length=end-start)

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

    def spike_time_tiling(self, i, j, delt=20):
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

        TA = _sttc_ta(tA, delt, self.length) / self.length
        TB = _sttc_ta(tB, delt, self.length) / self.length

        PA = _sttc_na(tA, tB, delt) / len(tA) if len(tA) > 0 else 1.0
        PB = _sttc_na(tB, tA, delt) / len(tB) if len(tB) > 0 else 1.0

        aa = (PA-TB)/(1-PA*TB) if PA*TB != 1 else 0
        bb = (PB-TA)/(1-PB*TA) if PB*TA != 1 else 0
        return (aa + bb) / 2

    def avalanches(self, thresh, bin_size=40):
        '''
        Bin the spikes in this data, and group the result into lists
        corresponding to avalanches, defined as deviations above
        a given threshold spike count.
        '''
        this_av = []
        for count in self.binned(bin_size):
            if count > thresh:
                this_av.append(count)
            elif this_av:
                yield this_av
                this_av = []
        if this_av:
            yield this_av

    def duration_size(self, thresh, bin_size=40):
        '''
        Collect the avalanches in this data and regroup them into
        a pair of lists: durations and sizes.
        '''
        durations, sizes = [], []
        for avalanche in self.avalanches(thresh, bin_size):
            durations.append(len(avalanche))
            sizes.append(sum(avalanche))
        return np.array(durations), np.array(sizes)

    def deviation_from_criticality(self, thresh, bin_size=40,
                                   significance=True, p_value=0.05):
        '''
        Calculates the deviation from criticality according to the
        method of Ma et al. (2019), who used the relationship of the
        dynamical critical exponent to the exponents of the separate
        power laws corresponding to the avalanche size and duration
        distributions as a metric for suboptimal cortical function
        following monocular deprivation.

        [1] Ma, Z., Turrigiano, G. G., Wessel, R. & Hengen, K. B.
            Cortical circuit dynamics are homeostatically tuned to
            criticality in vivo. Neuron 104, 655-664.e4 (2019).
        '''
        # Gather durations and sizes, and fit power laws.
        durations, sizes = self.duration_size(thresh, bin_size)
        fit_duration = power_law_fit(durations)
        fit_size = power_law_fit(sizes)

        # If either of them wasn't a power law, obviously we can't
        # have criticality here either, so just return failure.
        if fit_size[-1] == 1.0 or fit_duration[-1] == 1.0:
            return np.inf
        p_duration = power_law_significance(durations, *fit_duration)
        if p_duration < p_value:
            return np.inf
        p_size = power_law_significance(sizes, *fit_size)
        if p_size < p_value:
            return np.inf

        # Fit and predict the dynamical critical exponent, and return
        # the absolute difference between them.
        τ_fit, a = np.polyfit(np.log(durations), np.log(sizes), 1)
        τ_pred = (fit_duration[0] - 1) / (fit_size[0] - 1)
        return abs(τ_pred - τ_fit)


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
    # else, rather than generating infinice Pearson coefficients.
    σx[σx == 0] = np.inf

    # This is by the formula, but there's also a hack to deal with the
    # numerical issues that break the invariant that every variable
    # should have a Pearson autocorrelation of 1.
    corr = np.array(Exy - Ex*Ex.T) / (σx*σx.T)
    np.fill_diagonal(corr, 1)
    return corr

def _power_law_pmf(X, τ, x0, xM):
    '''
    Evaluate the probability mass function for a discrete power law at
    point(s) X, with exponent τ, minimum value x0, and maximum xM.
    '''
    X = np.atleast_1d(X)
    ret = np.zeros(len(X))
    mask = (X >= x0) & (X <= xM)
    ret[mask] = X[mask]**-τ
    sum_from_xM = special.zeta(τ, xM+1)
    sum_from_x0 = special.zeta(τ, x0)
    return ret / (sum_from_x0 - sum_from_xM)

def _power_law_one_sample(τ, x0, xM):
    '''
    Sample from a discrete power law using the method of inverses by
    directly searching for a numerical solution [1], plus rejection
    sampling to implement the maximum. Shockingly, this is
    significantly faster than scipy.stats.zipfian.

    [1] Clauset, A., Shalizi, C. R. & Newman, M. E. J. Power-law
        distributions in empirical data. SIAM Rev. 51, 661–703 (2009).
    '''
    z0 = special.zeta(τ, x0)
    while True:
        r = np.random.rand()
        x1, x2 = x0, 2*x0

        def P(x):
            return special.zeta(τ, x) / z0

        while P(x2) >= 1-r:
            x1 = x2
            x2 = 2*x1
        while x2 - x1 > 1:
            xmid = int((x1 + x2)//2)
            if P(xmid) < 1-r:
                x2 = xmid
            else:
                x1 = xmid
        ret = int((x1 + x2)//2)
        if ret > xM:
            continue
        else:
            return ret

def _power_law_sample(M, τ, x0, xM):
    '''
    Sample random variables from a discrete power law using the
    inverse method of _powser_law_one_sample().
    '''
    return np.array([_power_law_one_sample(τ, x0, xM)
                     for _ in range(M)])

def _power_law_log_likelihood(X, τ, x0, xM):
    '''
    Calculate log likelihood for a dataset X of avalanche sizes with
    power law exponent τ and minimum and maximum values x0 and xM.
    '''
    sum_from_xM = special.zeta(τ, xM+1)
    sum_from_x0 = special.zeta(τ, x0)
    return -τ*np.log(X).mean() - np.log(sum_from_x0 - sum_from_xM)

def _power_law_ks_1samp(X, τ, x0, xM):
    '''
    Calculate the one-sample Kolmogorov-Smirnoff test statistic for
    a given dataset X under a provided discrete power law fit with
    exponent τ and minimum and maximum values x0 and xM.
    '''
    pmf = _power_law_pmf(np.arange(0, max(X)+1), τ, x0, xM)

    emp_pmf = np.zeros(max(X)+1)
    vals, cnts = np.unique(X, return_counts=True)
    emp_pmf[vals] = cnts / len(X)

    return np.abs(pmf.cumsum() - emp_pmf.cumsum()).max()

def power_law_fit(X, approximate=False):
    '''
    Fit a truncated discrete power law to a collection of integers `X`
    as is common in neuroscience literature [1]. Finds the largest
    possible maximum avalanche size for which the best fit satisfies
    a Kolmogorov-Smirnoff statistic condition for some reasonable
    value of the minimum. Returns the exponent, minimum and maximum
    vaules for the fit, and the calculated Kolmogorov-Smirnoff
    statistic. Optionally, use an approximate closed form of the
    maximum likelihood calculation [2], which may be faster sometimes.

    [1] Shew, W. L. et al. Adaptation to sensory input tunes visual
        cortex to criticality. Nature Physics 11, 659–663 (2015).
    [2] Clauset, A., Shalizi, C. R. & Newman, M. E. J. Power-law
        distributions in empirical data. SIAM Rev. 51, 661–703 (2009).
    '''
    Xfull = X
    Xvals = np.unique(Xfull)
    best = 2.0, X.min(), X.max(), 1.0
    for xM in Xvals[::-1]:
        Xbelow = Xfull[Xfull <= xM]
        for x0 in Xvals:
            X = Xbelow[Xbelow >= x0]
            if len(X) < 0.9*len(Xfull):
                break

            if approximate:
                τ = 1 + len(X)/np.log(X / (x0 - 0.5)).sum()
            else:
                τ = optimize.minimize_scalar(
                    lambda τ: -_power_law_log_likelihood(X,τ,x0,xM),
                    bounds=(1,4), method='bounded').x

            # Choose the first fit with sufficiently low KS statistic,
            # but if none attains the mark, return the best.
            ks = _power_law_ks_1samp(X, τ, x0, xM)
            if ks < best[-1]:
                best = τ, x0, xM, ks
            if ks < np.sqrt(1/len(X)):
                return best

    return best

def power_law_significance(M, τ, x0, xM, ks=None, N=1000):
    '''
    Estimate the significance level of the hypothesis that the
    provided dataset `X` is drawn from a different distribution than
    a power law with the given parameters, by sampling `N` surrogate
    datasets from that distribution with the same size as `X` and
    returning what fraction of them have worse Kolmogorov-Smirnoff
    statistic.
    '''
    try:
        X = M[(M >= x0) & (M <= xM)]
        M = len(X)
        ks = _power_law_ks_1samp(X, τ, x0, xM) if ks is None else ks
    except TypeError:
        if ks is None:
            raise ValueError('Must provide ks statistic or full sample.')

    return np.mean([
        power_law_fit(_power_law_sample(M, τ, x0, xM))[-1] > ks
        for _ in range(N)
    ])
