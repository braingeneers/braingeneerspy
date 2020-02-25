import unittest
from braingeneers.drylab import *

class WrapperBackendTest(unittest.TestCase):
    def test_consistency(self):
        np.random.seed(42)
        nn = OrganoidWrapper(N=1000, backend=backend_numpy)
        out_nn = nn.total_firings(input=1, interval=10)

        np.random.seed(42)
        tt = OrganoidWrapper(N=1000, backend=backend_torch)
        out_tt = tt.total_firings(input=1, interval=10)

        self.assertListEqual(list(out_nn), list(out_tt),
                'Discrepancy between numpy and torch results')


