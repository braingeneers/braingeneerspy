import unittest
from braingeneers.analysis import *

class AnalysisTest(unittest.TestCase):
    def test_sparse_raster(self):
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
        raster = sparse_raster(times, idces, bin_size=1)

        self.assertTrue(np.all(raster == ground_truth), 
                'Incorrect construction of spike raster.')

        # Finally, check the calculated Pearson coefficients to ensure
        # they're numerically close enough to the intended values.
        true_pearson = [
                [1, 1, -1, 0],
                [1, 1, -1, 0],
                [-1, -1, 1, 0],
                [0, 0, 0, 1]]
        close = np.isclose(pearson(raster), true_pearson)
        self.assertTrue(close.all(), 
                'Wrong Pearson correlation matrix')


class AvalancheTest(unittest.TestCase):
    def test_binning_doesnt_lose_spikes(self):
        # Generate the times of a Poisson spike train, and ensure that
        # no spikes are lost in translation.
        times = stats.expon.rvs(size=1000).cumsum()
        self.assertEqual(sum(temporal_binning(times, 5)), 1000,
                'Temporal binning lost some spikes!')

    def test_binning(self):
        # Here's a totally arbitrary list of spike times to bin.
        times = [1, 2, 5, 15, 16, 20, 22, 25]
        self.assertListEqual(
                list(temporal_binning(times, 4)),
                [2, 1, 0, 1, 1, 2, 1],
                'Mistake in temporal binning!')

    def test_avalanches(self):
        # Here's a potential list of binned spike counts; ensure that
        # the final avalanche doesn't get dropped like it used to.
        counts = [2,5,3, 0,1,0,0, 2,2, 0, 42]
        self.assertListEqual(
                [len(av) for av in get_avalanches(counts, 1)], 
                [3, 2, 1],
                'Counted avalanches wrong!')
