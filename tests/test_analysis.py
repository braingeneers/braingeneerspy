import unittest
from scipy import stats, sparse
import numpy as np
import braingeneers.analysis as ba
from collections import namedtuple

DerpNeuron = namedtuple('Neuron', 'spike_time fs')

class AnalysisTest(unittest.TestCase):

    def test_spike_data(self):
        # Generate a bunch of random spike times and indices.
        times = np.random.rand(100) * 100
        idces = np.random.randint(5, size=100)

        # Test two-argument constructor and spike time list.
        sd = ba.SpikeData(idces, times)
        self.assertTrue(np.all(np.sort(times) == list(sd.times)))

        # Test event-list constructor.
        sd1 = ba.SpikeData(list(zip(idces, times)))
        for ta,tb in zip(sd.train, sd1.train):
            self.assertTrue(np.all(ta == tb))

        # Test 'list of lists' constructor.
        sd2 = ba.SpikeData(sd.train)
        for ta,tb in zip(sd.train, sd2.train):
            self.assertTrue(np.all(ta == tb))

        # Test 'list of Neuron()s' constructor.
        fs = 10
        ns = [DerpNeuron(spike_time=ts*fs, fs=fs*1e3) for ts in sd.train]
        sd3 = ba.SpikeData(ns)
        for ta,tb in zip(sd.train, sd3.train):
            self.assertTrue(np.isclose(ta, tb).all())

        # Test subset() constructor.
        idces = [1, 2, 3]
        sdsub = sd.subset(idces)
        for i,j in enumerate(idces):
            self.assertTrue(np.all(sdsub.train[i] == sd.train[j]))

    def test_sparse_raster(self):
        # Generate Poisson spike trains and make sure no spikes are
        # lost in translation.
        N = 10000
        times = np.random.rand(N) * 1e4
        idces = np.random.randint(10, size=N)
        raster = ba.SpikeData(idces, times).sparse_raster()
        self.assertEqual(raster.sum(), N)

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
        raster = ba.SpikeData(idces, times).sparse_raster(bin_size=1)
        self.assertTrue(np.all(raster == ground_truth))

        # Finally, check the calculated Pearson coefficients to ensure
        # they're numerically close enough to the intended values.
        true_pearson = [
            [1, 1, -1, 0],
            [1, 1, -1, 0],
            [-1, -1, 1, 0],
            [0, 0, 0, 1]]
        sparse_pearson = ba.pearson(raster)
        self.assertTrue(np.isclose(sparse_pearson, true_pearson).all())

        # Test on dense matrices (fallback to np.pearson).
        dense_pearson = ba.pearson(raster.todense())
        np_pearson = np.corrcoef(raster.todense())
        self.assertTrue(np.isclose(dense_pearson, np_pearson).all())

        # Also check the calculations.
        self.assertEqual(dense_pearson.shape, sparse_pearson.shape)
        self.assertTrue(np.isclose(dense_pearson, sparse_pearson).all())

    def test_burstiness_index(self):
        # Something completely uniform should have zero burstiness.
        uniform = ba.SpikeData([np.arange(1000)])
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
        ii = ba.SpikeData(ar%10, ar).interspike_intervals()
        self.assertEqual(len(ii), 10)
        for i in ii:
            self.assertTrue((i==10).all())
            self.assertEqual(len(i), N/10 - 1)

        # Finally, check with random ISIs.
        truth = np.random.rand(N)
        spikes = ba.SpikeData(np.zeros(N,int), truth.cumsum())
        ii = spikes.interspike_intervals()
        self.assertTrue(np.isclose(ii[0], truth[1:]).all())

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
        self.assertTrue(np.isclose(f_sparse, f_dense).all())

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


class AvalancheTest(unittest.TestCase):
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
                             [2, 1, 0, 1, 1, 2, 1])

    def test_avalanches(self):
        # Here's a potential list of binned spike counts; ensure that
        # the final avalanche doesn't get dropped like it used to.
        counts = [2,5,3, 0,1,0,0, 2,2, 0, 42]
        times = np.hstack([i*np.ones(c) for i,c in enumerate(counts)])
        spikes = ba.SpikeData([times])
        self.assertListEqual(list(spikes.binned(1)), counts)
        self.assertListEqual(
            [len(av) for av in spikes.avalanches(1, bin_size=1)],
            [3, 2, 1])
