import unittest
from scipy import stats, sparse
import numpy as np
import braingeneers.analysis as ba
from collections import namedtuple

DerpNeuron = namedtuple('Neuron', 'spike_time fs')

class DerpSpikeRecorder:
    'Weird mockup of a NEST spike recorder.'
    def __init__(self, idces, times):
        self.events = dict(senders=idces, times=times)
    def __getattr__(self, attr):
        return self.__dict__[attr]
    def __iter__(self):
        yield self

class AnalysisTest(unittest.TestCase):

    def assertSpikeDataEqual(self, sda, sdb, msg=None):
        'Assert that two SpikeData objects contain the same data.'
        for a,b in zip(sda.train, sdb.train):
            self.assertTrue(len(a) == len(b) and np.allclose(a, b),
                            msg=msg)

    def assertAll(self, bools, msg=None):
        'Assert that two arrays are equal elementwise.'
        self.assertTrue(np.all(bools), msg=msg)

    def assertClose(self, a, b, msg=None, **kw):
        'Assert that two arrays are equal within tolerance.'
        self.assertTrue(np.allclose(a, b, **kw), msg=msg)

    def test_spike_data(self):
        # Generate a bunch of random spike times and indices.
        times = np.random.rand(100) * 100
        idces = np.random.randint(5, size=100)

        # Test two-argument constructor and spike time list.
        sd = ba.SpikeData(idces, times, length=100.0)
        self.assertAll(np.sort(times) == list(sd.times))

        # Test event-list constructor.
        sd1 = ba.SpikeData(list(zip(idces, times)))
        self.assertSpikeDataEqual(sd, sd1)

        # Test 'list of lists' constructor.
        sd2 = ba.SpikeData(sd.train)
        self.assertSpikeDataEqual(sd, sd2)

        # Test 'list of Neuron()s' constructor.
        fs = 10
        ns = [DerpNeuron(spike_time=ts*fs, fs=fs*1e3) for ts in sd.train]
        sd3 = ba.SpikeData(ns)
        self.assertSpikeDataEqual(sd, sd3)

        # Test events.
        sd4 = ba.SpikeData(list(sd.events))
        self.assertSpikeDataEqual(sd, sd4)

        # Test idces_times().
        sd5 = ba.SpikeData(*sd.idces_times())
        self.assertSpikeDataEqual(sd, sd5)

        # Test 'NEST SpikeRecorder' constructor.
        recorder = DerpSpikeRecorder(idces, times)
        sd6 = ba.SpikeData(recorder)
        self.assertSpikeDataEqual(sd, sd6)

        # Test subset() constructor.
        idces = [1, 2, 3]
        sdsub = sd.subset(idces)
        for i,j in enumerate(idces):
            self.assertAll(sdsub.train[i] == sd.train[j])

        # Test subtime() constructor idempotence.
        sdtimefull = sd.subtime(0, 100)
        self.assertSpikeDataEqual(sd, sdtimefull)

        # Test subtime() constructor actually grabs subsets.
        sdtime = sd.subtime(20, 50)
        for i in range(len(sd.train)):
            self.assertAll(sdtime.train[i] > 0)
            self.assertAll(sdtime.train[i] <= 30)
            n_in_range = np.sum((sd.train[i] > 20) & (sd.train[i] <= 50))
            self.assertTrue(len(sdtime.train[i]) == n_in_range)

        # Test subtime() with negative arguments.
        sdtime = sd.subtime(-80, -50)
        for i in range(len(sd.train)):
            self.assertAll(sdtime.train[i] > 0)
            self.assertAll(sdtime.train[i] <= 30)
            n_in_range = np.sum((sd.train[i] > 20) & (sd.train[i] <= 50))
            print(sd.length)
            print(sdtime.length)
            print(n_in_range, len(sdtime.train[i]))
            print(sdtime.train[i])
            print(sd.train[i])
            self.assertTrue(len(sdtime.train[i]) == n_in_range)

        # Check subtime() with ... first argument.
        sdtime = sd.subtime(..., 50)
        for i in range(len(sd.train)):
            self.assertAll(sdtime.train[i] > 0)
            self.assertAll(sdtime.train[i] <= 50)
            n_in_range = np.sum(sd.train[i] <= 50)
            self.assertTrue(len(sdtime.train[i]) == n_in_range)

        # Check subtime() with ... second argument.
        sdtime = sd.subtime(20, ...)
        for i in range(len(sd.train)):
            self.assertAll(sdtime.train[i] > 0)
            self.assertAll(sdtime.train[i] <= 80)
            n_in_range = np.sum(sd.train[i] > 20)
            self.assertTrue(len(sdtime.train[i]) == n_in_range)

        # Test consistency between subtime() and frames().
        for i,frame in enumerate(sd.frames(20)):
            self.assertSpikeDataEqual(frame,
                                      sd.subtime(i*20, (i+1)*20))

        # Regression test for overlap parameter of frames().
        for i,frame in enumerate(sd.frames(20, overlap=10)):
            self.assertSpikeDataEqual(frame,
                                      sd.subtime(i*10, i*10+20))

    def test_raster(self):
        # Generate Poisson spike trains and make sure no spikes are
        # lost in translation.
        N = 10000
        times = np.random.rand(N) * 1e4
        idces = np.random.randint(10, size=N)
        sd = ba.SpikeData(idces, times)

        # Try both a sparse and a dense raster.
        self.assertEqual(sd.raster().sum(), N)
        self.assertAll(sd.sparse_raster() == sd.raster())

        # Make sure the length of the raster is going to be consistent
        # no matter how many spikes there are.
        N = 10
        length = 1e4
        sdA = ba.SpikeData(np.zeros(N, int),
                           np.random.rand(N) * length,
                           length=length)
        sdB = ba.SpikeData(np.zeros(N, int),
                           np.random.rand(N) * length,
                           length=length)
        self.assertEqual(sdA.raster().shape, sdB.raster().shape)

        # Corner cases of raster binning rules: spikes exactly at
        # 0 end up in the first bin, but other bins should be
        # lower-open and upper-closed.
        ground_truth = [[1, 1, 0, 1]]
        sd = ba.SpikeData([0,0,0], [0,20,40])
        self.assertEqual(sd.length, 40)
        self.assertAll(sd.raster(10) == ground_truth)

        # Also verify that binning rules are consistent between the
        # raster and other binning methods.
        binned = np.array([list(sd.binned(10))])
        self.assertAll(sd.raster(10) == binned)


    def test_pearson(self):
        # These four cells are constructed so that A is perfectly
        # correlated with B, perfectly anticorrelated with C, and
        # uncorrelated with D.
        cellA = [1, 0, 1, 1, 0, 0]
        cellB = cellA
        cellC = [0, 1, 0, 0, 1, 1]
        cellD = [1, 1, 1, 0, 0, 1]

        # Construct the true raster, use it to produce times and
        # indices using numpy methods, and ensure that the sparse
        # matrix generated is correct.
        ground_truth = np.stack((cellA, cellB, cellC, cellD))
        times, idces = np.where(ground_truth.T)
        raster = ba.SpikeData(idces, times + 0.5).sparse_raster(bin_size=1)
        print(raster.todense())
        print(ground_truth)
        self.assertAll(raster == ground_truth)

        # Finally, check the calculated Pearson coefficients to ensure
        # they're numerically close enough to the intended values.
        true_pearson = [
            [1, 1, -1, 0],
            [1, 1, -1, 0],
            [-1, -1, 1, 0],
            [0, 0, 0, 1]]
        sparse_pearson = ba.pearson(raster)
        self.assertClose(sparse_pearson, true_pearson)

        # Test on dense matrices (fallback to np.pearson).
        dense_pearson = ba.pearson(raster.todense())
        np_pearson = np.corrcoef(raster.todense())
        self.assertClose(dense_pearson, np_pearson)

        # Also check the calculations.
        self.assertEqual(dense_pearson.shape, sparse_pearson.shape)
        self.assertClose(dense_pearson, sparse_pearson)

    def test_burstiness_index(self):
        # Something completely uniform should have zero burstiness,
        # but ensure there's no spike at time zero.
        uniform = ba.SpikeData([0.5+np.arange(1000)])
        self.assertEqual(uniform.burstiness_index(10), 0)

        # All spikes at the same time is technically super bursty,
        # just make sure that they happen late enough that there are
        # actually several bins to count.
        atonce = ba.SpikeData([[1]]*1000)
        self.assertEqual(atonce.burstiness_index(0.01), 1)

        # Added code to deal with a corner case so it's really ALWAYS
        # in the zero to one range. I think this only happens with
        # very small values.
        self.assertEqual(ba.SpikeData([[1]]).burstiness_index(), 1)

    def test_interspike_intervals(self):
        # Uniform spike train: uniform ISIs. Also make sure it returns
        # a list of just the one array.
        N = 10000
        ar = np.arange(N)
        ii = ba.SpikeData(np.zeros(N,int), ar).interspike_intervals()
        self.assertTrue((ii[0]==1).all())
        self.assertEqual(len(ii[0]), N-1)
        self.assertEqual(len(ii), 1)

        # Also make sure multiple spike trains do the same thing.
        ii = ba.SpikeData(ar % 10, ar).interspike_intervals()
        self.assertEqual(len(ii), 10)
        for i in ii:
            self.assertTrue((i==10).all())
            self.assertEqual(len(i), N/10 - 1)

        # Finally, check with random ISIs.
        truth = np.random.rand(N)
        spikes = ba.SpikeData(np.zeros(N,int), truth.cumsum())
        ii = spikes.interspike_intervals()
        self.assertClose(ii[0], truth[1:])

    def test_fano_factors(self):
        N = 10000

        def spikes(units):
            times = np.random.rand(N)*N
            idces = np.random.randint(units, size=N)
            return ba.SpikeData(idces, times)

        # If there's no variance, Fano factors should be zero, for
        # both sparse and dense implementations. Also use todense()
        # next to  toarray() to show that both np.matrix and np.array
        # spike rasters are acceptable. Note that the numerical issues
        # in the sparse version mean that it's not precisely zero, so
        # we use assertAlmostEqual() in this case.
        ones = sparse.csr_matrix(np.ones(N))
        self.assertAlmostEqual(ba.fano_factors(ones)[0], 0)
        self.assertEqual(ba.fano_factors(ones.todense())[0], 0)
        self.assertEqual(ba.fano_factors(ones.toarray())[0], 0)

        # Poisson spike trains should have Fano factors about 1.
        # This is only rough because random, but the sparse and dense
        # versions should both be equal to each other.
        foo = spikes(1).sparse_raster(1)
        f_sparse = ba.fano_factors(foo)[0]
        f_dense = ba.fano_factors(foo.toarray())[0]
        self.assertAlmostEqual(f_sparse, 1, 1)
        self.assertAlmostEqual(f_dense, 1, 1)
        self.assertAlmostEqual(f_sparse, f_dense)

        # Make sure the sparse and dense are equal when there are
        # multiple spike trains as well.
        foo = spikes(10).sparse_raster(10)
        f_sparse = ba.fano_factors(foo)
        f_dense = ba.fano_factors(foo.toarray())
        self.assertClose(f_sparse, f_dense)

    def test_spike_time_tiling_ta(self):
        # Trivial base cases.
        self.assertEqual(ba._sttc_ta([42], 1, 100), 2)
        self.assertEqual(ba._sttc_ta([], 1, 100), 0)

        # When spikes don't overlap, you should get exactly 2ndt.
        self.assertEqual(ba._sttc_ta(np.arange(42)+1, 0.5, 100), 42.0)

        # When spikes overlap fully, you should get exactly
        # (tmax-tmin) + 2dt
        self.assertEqual(ba._sttc_ta(np.arange(42)+100, 100, 300), 241)

    def test_spike_time_tiling_na(self):
        # Trivial base cases.
        self.assertEqual(ba._sttc_na([1,2,3], [], 1), 0)
        self.assertEqual(ba._sttc_na([], [1,2,3], 1), 0)

        self.assertEqual(ba._sttc_na([1], [2], 0.5), 0)
        self.assertEqual(ba._sttc_na([1], [2], 1), 1)

        # Make sure closed intervals are being used.
        na = ba._sttc_na(np.arange(10), np.arange(10)+0.5, 0.5)
        self.assertEqual(na, 10)

        # Skipping multiple spikes in spike train B.
        self.assertEqual(ba._sttc_na([4], [1, 2, 3, 4.5], 0.1), 0)
        self.assertEqual(ba._sttc_na([4], [1, 2, 3, 4.5], 0.5), 1)

        # Many spikes in train B covering a single one in A.
        self.assertEqual(ba._sttc_na([2], [1, 2, 3], 0.1), 1)
        self.assertEqual(ba._sttc_na([2], [1, 2, 3], 1), 1)

        # Many spikes in train A are covered by one in B.
        self.assertEqual(ba._sttc_na([1, 2, 3], [2], 0.1), 1)
        self.assertEqual(ba._sttc_na([1, 2, 3], [2], 1), 3)

    def test_spike_time_tiling_coefficient(self):
        # Examples to use in different cases.
        N = 10000

        def spikes():
            return np.random.rand(N) * N

        # Any spike train should be exactly equal to itself, and the
        # result shouldn't depend on which train is A and which is B.
        foo = ba.SpikeData([spikes(), spikes()])
        self.assertEqual(foo.spike_time_tiling(0, 0, 1), 1.0)
        self.assertEqual(foo.spike_time_tiling(1, 1, 1), 1.0)
        self.assertEqual(foo.spike_time_tiling(0, 1, 1),
                         foo.spike_time_tiling(1, 0, 1))

        # Default arguments, inferred value of tmax.
        tmax = max(foo.train[0].ptp(), foo.train[1].ptp())
        self.assertEqual(foo.spike_time_tiling(0, 1),
                         foo.spike_time_tiling(0, 1, tmax))

        # The uncorrelated spike trains above should stay near zero.
        # I'm not sure how many significant figures to expect with the
        # randomness, though, so it's really easy to pass.
        self.assertAlmostEqual(foo.spike_time_tiling(0, 1, 1), 0, 1)

        # Two spike trains that are in complete disagreement. This
        # should be exactly -0.8, but there's systematic error
        # proportional to 1/N, even in their original implementation.
        bar = ba.SpikeData([np.arange(N)+0.0, np.arange(N)+0.5])
        self.assertAlmostEqual(bar.spike_time_tiling(0, 1, 0.4),
                               -0.8, int(np.log10(N)))

        # As you vary dt, that alternating spike train actually gets
        # the STTC to go continuously from 0 to approach a limit of
        # lim(dt to 0.5) STTC(dt) = -1, but STTC(dt >= 0.5) = 0.
        self.assertEqual(bar.spike_time_tiling(0, 1, 0.5), 0)

        # Make sure it stays within range. Technically it goes a tiny
        # bit out of range on the negative extremes due to numerical
        # issues, but it should be fine in general.
        for _ in range(100):
            baz = ba.SpikeData([spikes(), spikes()])
            sttc = baz.spike_time_tiling(0, 1, np.random.lognormal())
            self.assertLessEqual(sttc, 1)
            self.assertGreaterEqual(sttc, -1)

    def test_binning_doesnt_lose_spikes(self):
        # Generate the times of a Poisson spike train, and ensure that
        # no spikes are lost in translation.
        N = 1000
        times = stats.expon.rvs(size=N).cumsum()
        spikes = ba.SpikeData([times])
        self.assertEqual(sum(spikes.binned(5)), N)

    def test_binning(self):
        # Here's a totally arbitrary list of spike times to bin.
        spikes = ba.SpikeData([[1, 2, 5, 15, 16, 20, 22, 25]])
        self.assertListEqual(list(spikes.binned(4)),
                             [2, 1, 0, 2, 1, 1, 1])

    def test_avalanches(self):
        def sd_from_counts(counts):
            'Generate a SpikeData whose raster matches given counts.'
            times = np.hstack([i*np.ones(c) for i,c in enumerate(counts)])
            return ba.SpikeData([times + 0.5])

        # Double-check that this helper method works...
        counts = np.random.randint(10, size=1000)
        sd = sd_from_counts(counts)
        self.assertAll(sd.binned(1) == counts)

        # The simple case where there are avalanches in the middle.
        sd = sd_from_counts([1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0])
        self.assertListEqual(
            [len(av) for av in sd.avalanches(1, bin_size=1)],
            [5, 3])
        self.assertListEqual(
            [sum(av) for av in sd.avalanches(1, bin_size=1)],
            [2+3+4+3+2, 2+3+2])

        # Also the duration-size lists of the same data.
        durations, sizes = sd.duration_size(1, bin_size=1)
        self.assertListEqual(list(durations), [5,3])
        self.assertListEqual(list(sizes), [2+3+4+3+2, 2+3+2])

        # Ensure that avalanches coinciding with the start and end of
        # recording don't get counted because there's no way to know
        # how long they are.
        sd = sd_from_counts([2,5,3, 0,1,0,0, 2,2, 0,0,0,0, 4,3,4, 0, 42])
        self.assertListEqual(
            [len(av) for av in sd.avalanches(1, bin_size=1)],
            [2, 3])

        # Corner cases where there are no avalanches: no transitions
        # because threshold too low, because threshold too high,
        # because only crosses once downwards, and because only
        # crosses once upwards.
        sd = sd_from_counts([1, 2, 3, 4, 5])
        self.assertListEqual(sd.avalanches(0, bin_size=1), [])
        self.assertListEqual(sd.avalanches(10, bin_size=1), [])
        self.assertListEqual(sd.avalanches(3, bin_size=1), [])
        sd = sd_from_counts([5, 4, 3, 2, 1])
        self.assertListEqual(sd.avalanches(3, bin_size=1), [])

    def test_metadata(self):
        # Make sure there's an error if the metadata is gibberish.
        self.assertRaises(ValueError,
                          lambda: ba.SpikeData([], N=5, length=100,
                                               neuron_data=dict(trash=[47])))

        # Overall propagation testing...
        foo = ba.SpikeData([], N=5, length=1000,
                           metadata=dict(name='Marvin'),
                           neuron_data=dict(size=np.random.rand(5)))

        # Make sure subset propagates all metadata and correctly
        # subsets the neuron_data.
        subset = [1, 3]
        truth = foo.neuron_data['size'][subset]
        bar = foo.subset(subset)
        self.assertDictEqual(foo.metadata, bar.metadata)
        self.assertAll(bar.neuron_data['size'] == truth)

        # Change the metadata of foo and see that it's copied, so the
        # change doesn't propagate.
        foo.metadata['name'] = 'Ford'
        baz = bar.subtime(500, 1000)
        self.assertDictEqual(bar.metadata, baz.metadata)
        self.assertIsNot(bar.metadata, baz.metadata)
        self.assertNotEqual(foo.metadata['name'], bar.metadata['name'])
        self.assertDictEqual(bar.neuron_data, baz.neuron_data)

    def test_raw_data(self):
        # Make sure there's an error if only one of raw_data and
        # raw_time is provided to the constructor.
        self.assertRaises(ValueError,
                          lambda: ba.SpikeData([], N=5, length=100,
                                               raw_data=[]))
        self.assertRaises(ValueError,
                          lambda: ba.SpikeData([], N=5, length=100,
                                               raw_time=42))

        # Make sure inconsistent lengths throw an error as well.
        self.assertRaises(ValueError,
                          lambda: ba.SpikeData([], N=5, length=100,
                                               raw_data=np.zeros((5,100)),
                                               raw_time=np.arange(42)))

        # Check automatic generation of the time array.
        sd = ba.SpikeData([], N=5, length=100,
                          raw_data=np.random.rand(5,100),
                          raw_time=1.0)
        self.assertAll(sd.raw_time == np.arange(100))

        # Make sure the raw data is sliced properly with time.
        sd2 = sd.subtime(20, 30)
        self.assertAll(sd2.raw_time == np.arange(11))
        self.assertAll(sd2.raw_data == sd.raw_data[:,20:31])
