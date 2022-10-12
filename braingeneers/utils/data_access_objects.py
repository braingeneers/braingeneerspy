import heapq
import numpy as np
from scipy import sparse, stats, signal
import itertools
from collections import namedtuple
import powerlaw
import os
import contextlib


class SpikeData:
    '''
    Generic representation for spiking data from spike sorters and
    simulations.
    '''

    def __init__(self, arg1, arg2=None, *, N=None, length=None,
                 neuron_data={}, metadata={},
                 raw_data=None, raw_time=None):
        '''
        Parses different argument list possibilities into the desired
        format: a list indexed by unit ID, where each element is
        a list of spike times. The five possibilities accepted are:
        (1) a pair of lists corresponding to unit indices and times,
        (2) a NEST spike recorder plus the collection of nodes to
        record from, (3) a list of lists of spike times, (4) a list of
        channel-time pairs, (5) a list of Neuron objects whose
        parameter spike_time is a list of spike times.

        Metadata can also be passed in to the constructor, on a global
        basis in a dict called `metadata` or on a per-neuron basis in
        a dict of lists `neuron_data`.

        Raw timeseries data can be passed in as `raw_data`. If this is
        used, `raw_time` is also obligatory. This can be a series of
        sample times or just a sample rate in kHz. In this case, it is
        assumed that the start of the raw data corresponds with t=0,
        and a raw time array is generated.

        Spike times should be in units of milliseconds, unless a list
        of Neurons is given; these have spike times in units of
        samples, which are converted to milliseconds using the sample
        rate saved in the Neuron object.
        '''
        # Install the metadata and neuron_data.
        self.metadata = metadata.copy()
        self.neuron_data = neuron_data.copy()

        # If two arguments are provided, they're either a NEST spike
        # detector plus NodeCollection, or just a list of indices and
        # times.
        if arg2 is not None:

            # First, try parsing spikes from a NEST spike detector. Accept
            # either a number of cells or a NodeCollection as arg2.
            try:
                times = arg1.events['times']
                idces = arg1.events['senders']
                try:
                    N = arg2
                    cells = np.arange(N) + 1
                except ValueError:
                    cells = np.array(arg2)
                    N = cells.max()
                cellrev = np.zeros(N + 1, int)
                cellrev[cells] = np.arange(len(cells))

                # Store the underlying NEST cell IDs in the neuron_data.
                self.neuron_data['nest_id'] = cells

                self.train = [[] for _ in cells]
                for i, t in zip(idces, times):
                    if i <= N:
                        self.train[cellrev[i]].append(t)

            # If that fails, we must have lists of indices and times.
            except AttributeError:
                self.train = _train_from_i_t_list(arg1, arg2, N)

        else:
            # The input could be a list [musclebeachtools.Neuron]
            try:
                self.train = [np.asarray(n.spike_time) / n.fs * 1e3
                              for n in arg1]

            # Now it could be either (channel, time) pairs or
            # a complete prebuilt spike train.
            except AttributeError:

                # If all the elements are length 2, it must be pairs.
                if all([len(arg) == 2 for arg in arg1]):
                    idces = [i for i, _ in arg1]
                    times = [t for i, t in arg1]
                    self.train = _train_from_i_t_list(idces, times, N)
                # Otherwise, it's just a plain spike train.
                else:
                    self.train = arg1

        # Make sure each individual spike train is sorted, because
        # none of the formats guarantee this but all the algorithms
        # expect it. This also copies each array to avoid aliasing.
        self.train = [np.sort(times) for times in self.train]

        # The length of the spike train defaults to the last spike
        # time it contains.
        if length is None:
            length = max((t[-1] for t in self.train if len(t) > 0))
        self.length = length

        # If a number of units was provided, make the list of spike
        # trains consistent with that number.
        if N is not None and len(self.train) < N:
            self.train += [np.array([], float) for _ in
                           range(N - len(self.train))]
        self.N = len(self.train)

        # Add the raw data if present, including generating raw time.
        if (raw_data is None) != (raw_time is None):
            raise ValueError('Must provide both or neither of '
                             '`raw_data` and `raw_time`.')
        if raw_data is not None:
            self.raw_data = np.asarray(raw_data)
            self.raw_time = np.asarray(raw_time)
            if self.raw_time.shape == ():
                self.raw_time = np.arange(self.raw_data.shape[-1]) / raw_time
            elif self.raw_data.shape[-1:] != self.raw_time.shape:
                raise ValueError('Length of `raw_data` and '
                                 '`raw_time` must match.')
        else:
            self.raw_data = np.zeros((0, 0))
            self.raw_time = np.zeros((0,))

        # Double-check that the neuron_data has the right number of values.
        for k, values in self.neuron_data.items():
            if len(values) != self.N:
                raise ValueError('Malformed metadata: '
                                 f'neuron_data[{k}] should have '
                                 f'{self.N} items.')

    @property
    def times(self):
        'Iterate spike times for all units in time order.'
        return heapq.merge(*self.train)

    @property
    def events(self):
        'Iterate (index,time) pairs for all units in time order.'
        return heapq.merge(*[zip(itertools.repeat(i), t)
                             for (i, t) in enumerate(self.train)],
                           key=lambda x: x[1])

    def idces_times(self):
        '''
        Return separate lists of times and indices, e.g. for raster
        plots. This is not a property unlike `times` and `events`
        because the lists must actually be constructed in memory.
        '''
        idces, times = [], []
        for i, t in self.events:
            idces.append(i)
            times.append(t)
        return np.array(idces), np.array(times)

    def frames(self, length, overlap=0):
        '''
        Iterate new SpikeData objects corresponding to subwindows of
        a given `length` with a fixed `overlap`.
        '''
        for start in np.arange(0, self.length, length - overlap):
            yield self.subtime(start, start + length)

    def binned(self, bin_size=40):
        '''
        Quantizes time into intervals of bin_size and counts the
        number of events in each bin, considered as a lower half-open
        interval of times, with the exception that events at time
        precisely zero will be included in the first bin.
        '''
        return self.raster(bin_size).sum(0)

    def rates(self, unit='kHz'):
        '''
        Calculate the firing rate of each neuron as an average number
        of events per time over the duration of the data. The unit may
        be either `Hz` or `kHz` (default).
        '''
        rates = np.array([len(t) for t in self.train]) / self.length

        if unit == 'Hz':
            return 1e3 * rates
        elif unit == 'kHz':
            return rates
        else:
            raise ValueError(f'Unknown unit {unit} (try Hz or kHz)')

    def resampled_isi(self, times):
        '''
        Calculate firing rate at the given times by interpolating the
        inverse ISI, considered valid in between any two spikes. If any
        neuron has only one spike, the rate is assumed to be zero.
        '''
        return np.array([_resampled_isi(t, times) for t in self.train])

    def subset(self, units, by=None):
        '''
        Return a new SpikeData with spike times for only some units,
        selected either byy their indices or by an ID stored under a given
        key in the neuron_data. If IDs are not unique, every neuron which
        matches is included in the output.

        Metadata and raw data are propagated exactly, while neuron
        data is subsetted in the same way as the spike trains.
        '''
        # The inclusion condition depends on whether we're selecting by ID
        # or by index.
        if by is None:
            def cond(i):
                return i in units
        else:
            def cond(i):
                return self.neuron_data[by][i] in units

        train = [ts for i, ts in enumerate(self.train) if cond(i)]
        neuron_data = {k: [v for i, v in enumerate(vs) if cond(i)]
                       for k, vs in self.neuron_data.items()}
        return SpikeData(train, length=self.length, N=len(train),
                         neuron_data=neuron_data,
                         metadata=self.metadata,
                         raw_time=self.raw_time,
                         raw_data=self.raw_data)

    def subtime(self, start, end):
        '''
        Return a new SpikeData with only spikes in a time range,
        closed on top but open on the bottom unless the lower bound is
        zero, consistent with the binning methods. This is to ensure
        no overlap between adjacent slices.

        Start and end can be negative, in which case they are counted
        backwards from the end. They can also be None or Ellipsis,
        which results in only paying attention to the other bound.

        All metadata and neuron data are propagated, while raw data is
        sliced to the same range of times, but overlap is okay, so we
        include all samples within the closed interval.
        '''
        if start is None or start is Ellipsis:
            start = 0
        elif start < 0:
            start += self.length

        if end is None or end is Ellipsis:
            end = self.length
        elif end < 0:
            end += self.length

        # Special case out the start=0 case by nopping the comparison.
        lower = start if start > 0 else -np.inf

        # Subset the spike train by time.
        train = [t[(t > lower) & (t <= end)] - start
                 for t in self.train]

        # Subset and propagate the raw data.
        rawmask = (self.raw_time >= lower) & (self.raw_time <= end)
        return SpikeData(train, length=end - start, N=self.N,
                         neuron_data=self.neuron_data,
                         metadata=self.metadata,
                         raw_time=self.raw_time[rawmask] - start,
                         raw_data=self.raw_data[:, rawmask])

    def __getitem__(self, key):
        '''
        Overloads the [] operator to allow for slicing of the spikeData object.
        Uses the subtime method to slice the spikeData object.
        '''
        # print(start, stop, fs)

        if isinstance(key, slice):
            return self.subtime(key.start, key.stop)
        # print(start, stop, fs)
        if start is None:
            start = 0
        if stop is None:
            stop = self.length

        return self.subtime(start, stop)

    def sparse_raster(self, bin_size=20):
        '''
        Bin all spike times and create a sparse matrix where entry
        (i,j) is the number of times cell i fired in bin j. Bins are
        left-open and right-closed intervals except the first, which
        will capture any spikes occurring exactly at t=0.
        '''
        indices = np.hstack([np.ceil(ts / bin_size) - 1
                             for ts in self.train]).astype(int)
        units = np.hstack([0] + [len(ts) for ts in self.train])
        indptr = np.cumsum(units)
        values = np.ones_like(indices)
        length = int(np.ceil(self.length / bin_size))
        np.clip(indices, 0, length - 1, out=indices)
        ret = sparse.csr_matrix((values, indices, indptr),
                                shape=(self.N, length))
        return ret

    def raster(self, bin_size=20):
        '''
        Bin all spike times and create a dense matrix where entry
        (i,j) is the number of times cell i fired in bin j.
        '''
        return self.sparse_raster(bin_size).toarray()

    def interspike_intervals(self):
        'Produce a list of arrays of interspike intervals per unit.'
        return [np.diff(ts) for ts in self.train]

    def isi_skewness(self):
        'Skewness of interspike interval distribution.'
        intervals = self.interspike_intervals()
        return [stats.skew(intl) for intl in intervals]

    def isi_log_histogram(self, bin_num=300):
        '''
        Logarithmic (log base 10) interspike interval histogram.
        Return histogram and bins in log10 scale.
        '''
        intervals = self.interspike_intervals()
        ret = []
        ret_logbins = []
        for ts in intervals:
            log_bins = np.geomspace(min(ts), max(ts), bin_num + 1)
            hist, _ = np.histogram(ts, log_bins)
            ret.append(hist)
            ret_logbins.append(log_bins)
        return ret, ret_logbins

    def isi_threshold_cma(self, hist, bins, coef=1):
        '''
        Calculate interspike interval threshold from cumulative moving
        average [1]. The threshold is the bin that has the max CMA on
        the interspike interval histogram. Histogram and bins are
        logarithmic by default. `coef` is an input variable for
        threshold.

        [1] Kapucu, et al. Frontiers in computational neuroscience 6 (2012): 38
        '''
        isi_thr = []
        for n in range(len(hist)):
            h = hist[n]
            max_idx = 0
            cma = 0
            cma_list = []
            for i in range(len(h)):
                cma = (cma * i + h[i]) / (i + 1)
                cma_list.append(cma)
            max_idx = np.argmax(cma_list)
            thr = (bins[n][max_idx + 1]) * coef
            isi_thr.append(thr)
        return isi_thr

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
        binned = self.binned(bin_size)
        binned.sort()
        N85 = int(np.round(len(binned) * 0.85))

        if N85 == len(binned):
            return 1.0
        else:
            f15 = binned[N85:].sum() / binned.sum()
            return (f15 - 0.15) / 0.85

    def spike_time_tilings(self, delt=20):
        '''
        Compute the full spike time tiling coefficient matrix.
        '''
        ret = np.diag(np.ones(self.N))
        for i in range(self.N):
            for j in range(i + 1, self.N):
                ret[i, j] = ret[j, i] = self.spike_time_tiling(i, j, delt)
        return ret

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

        if len(tA) == 0 or len(tB) == 0:
            return 0.0

        TA = _sttc_ta(tA, delt, self.length) / self.length
        TB = _sttc_ta(tB, delt, self.length) / self.length

        PA = _sttc_na(tA, tB, delt) / len(tA)
        PB = _sttc_na(tB, tA, delt) / len(tB)

        aa = (PA - TB) / (1 - PA * TB) if PA * TB != 1 else 0
        bb = (PB - TA) / (1 - PB * TA) if PB * TA != 1 else 0
        return (aa + bb) / 2

    def avalanches(self, thresh, bin_size=40):
        '''
        Bin the spikes in this data, and group the result into lists
        corresponding to avalanches, defined as deviations above
        a given threshold spike count.
        '''
        counts = self.binned(bin_size)
        active = counts > thresh
        toggles = np.where(np.diff(active))[0]

        # If we start inactive, the first toggle begins the first
        # avalanche. Otherwise, we have to ignore it because we don't
        # know how long the system was active before.
        if active[0]:
            ups = toggles[1::2]
            downs = toggles[2::2]
        else:
            ups = toggles[::2]
            downs = toggles[1::2]

        # Now batch up the transitions and create a list of spike
        # counts in between them.
        return [counts[up + 1:down + 1] for up, down in zip(ups, downs)]

    def avalanche_duration_size(self, thresh, bin_size=40):
        '''
        Collect the avalanches in this data and regroup them into
        a pair of lists: durations and sizes.
        '''
        durations, sizes = [], []
        for avalanche in self.avalanches(thresh, bin_size):
            durations.append(len(avalanche))
            sizes.append(sum(avalanche))
        return np.array(durations), np.array(sizes)

    def deviation_from_criticality(self, quantile=0.35, bin_size=40,
                                   N=1000, pval_truncated=0.05):
        '''
        Calculates the deviation from criticality according to the
        method of Ma et al. (2019), who used the relationship of the
        dynamical critical exponent to the exponents of the separate
        power laws corresponding to the avalanche size and duration
        distributions as a metric for suboptimal cortical function
        following monocular deprivation.

        The returned DCCResult struct contains not only the DCC metric
        itself but also the significance of the hypothesis that the
        size and duration distributions of the extracted avalanches
        are poorly fit by power laws.

        [1] Ma, Z., Turrigiano, G. G., Wessel, R. & Hengen, K. B.
            Cortical circuit dynamics are homeostatically tuned to
            criticality in vivo. Neuron 104, 655-664.e4 (2019).
        '''
        # Calculate the spike count threshold corresponding to
        # the given quantile.
        thresh = np.quantile(self.binned(bin_size), quantile)

        # Gather durations and sizes. If there are no avalanches, we
        # very much can't say the system is critical.
        durations, sizes = self.avalanche_duration_size(thresh, bin_size)
        if len(durations) == 0:
            return DCCResult(dcc=np.inf, p_size=1.0, p_duration=1.0)

        # Call out to all the actual statistics.
        p_size, alpha_size = _p_and_alpha(sizes, N, pval_truncated)
        p_dur, alpha_dur = _p_and_alpha(durations, N, pval_truncated)

        # Fit and predict the dynamical critical exponent.
        τ_fit = np.polyfit(np.log(durations), np.log(sizes), 1)[0]
        τ_pred = (alpha_dur - 1) / (alpha_size - 1)
        dcc = abs(τ_pred - τ_fit)

        # Return the DCC value and significance.
        return DCCResult(dcc=dcc, p_size=p_size, p_duration=p_dur)

    def latencies(self, times, window_ms=100):
        '''
        Given a sorted list of times, compute the latencies from that time to
        each spike in the train within a window

        :param times: list of times
        :param window_ms: window in ms

        :return: 2d list, each row is a list of latencies
                        from a time to each spike in the train
        '''
        latencies = []
        for train in self.train:
            cur_latencies = np.hstack([train[(train >= time) & (train <= time + window_ms)] - time
                                       for time in times])
            latencies.append(cur_latencies)

        return latencies

    def latencies_to_index(self, i, window_ms=100):
        '''
        Given an index, compute latencies using self.latencies()

        :param i: index of the unit
        :param window_ms: window in ms

        :return: 2d list, each row is a list of latencies per neuron
        '''

        return self.latencies(self.train[i], window_ms)


class ThresholdedSpikeData(SpikeData):
    '''
    SpikeData generated by applying filtering and thresholding to raw ephys
    data in [channels, time] format.
    '''

    def __init__(self, raw_data, fs_Hz=20000, threshold_sigma=5,
                 filter_order=3, filter_lo_Hz=300, filter_hi_Hz=6000,
                 time_step_size_s=10, do_filter=True, hysteresis=True,
                 direction='both'):
        '''
        :param raw_data: [channels, time] array of raw ephys data
        :param fs_Hz: sampling frequency of raw data in Hz
        :param threshold_sigma: threshold for spike detection in units of
               standard deviation
        :param filter_spec: dictionary of filter parameters
        :param filter_step_size_s: size of chunks to filter in seconds
        '''
        # Filter the data.
        if do_filter:
            data = filter(raw_data, fs_Hz, filter_order, filter_lo_Hz,
                          filter_hi_Hz, time_step_size_s)
        else:
            # This is bad form
            data = raw_data

        threshold = threshold_sigma * np.std(data, axis=1, keepdims=True)

        if direction == 'both':
            raster = (data > threshold) | (data < -threshold)
        elif direction == 'up':
            raster = data > threshold
        elif direction == 'down':
            raster = data < -threshold

        if hysteresis:
            raster = np.diff(np.array(raster, dtype=int), axis=1) == 1

        self.idces, t_idces = np.nonzero(raster)

        self.times_ms = t_idces / fs_Hz * 1000

        self.N = data.shape[0]
        fs_ms = fs_Hz / 1000
        self.length = data.shape[1] / fs_ms

        # If no spikes were found, we can't do anything else.
        if len(self.idces) == 0:
            self.has_spikes = False
        else:
            self.has_spikes = True

        # change this to be an instance of the parent class instead
        # super().__init__(idces, times_ms, **kwargs)

    def to_spikeData(self, N=None, length=None):
        if self.has_spikes:
            if N is None:
                N = self.N
            if length is None:
                length = self.length
            return SpikeData(self.idces, self.times_ms, N=N, length=length)
        else:
            return None
