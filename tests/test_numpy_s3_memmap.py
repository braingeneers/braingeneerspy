import unittest

import numpy as np

from braingeneers.utils.configure import skip_unittest_if_offline
from braingeneers.utils.numpy_s3_memmap import NumpyS3Memmap


class TestNumpyS3Memmap(unittest.TestCase):
    @skip_unittest_if_offline
    def test_numpy32memmap_online(self):
        """Note: this is an online test requiring access to the PRP/S3 braingeneersdev bucket."""
        x = NumpyS3Memmap("s3://braingeneersdev/dfparks/test/test.npy")

        # Online test data at s3://braingeneersdev/dfparks/test/test.npy
        # array([[1., 2., 3.],
        #        [4., 5., 6.]], dtype=float32)

        e = np.arange(1, 7, dtype=np.float32).reshape(2, 3)

        self.assertTrue(np.all(x[0] == e[0]))
        self.assertTrue(np.all(x[:, 0:2] == e[:, 0:2]))
        self.assertTrue(np.all(x[:, [0, 1]] == e[:, [0, 1]]))

    @skip_unittest_if_offline
    def test_online_in_the_wild_file(self):
        """
        This test assumes online access.
        Specifically this test case found a bug in numpy arrays for fortran order.
        """
        x = NumpyS3Memmap(
            "s3://braingeneersdev/ephys/2020-07-06-e-MGK-76-2614-Drug/numpy/"
            "well_A1_chan_group_idx_1_time_000.npy"
        )
        self.assertEqual(x.shape, (3750000, 4))

        all_data = x[:]
        self.assertEqual(all_data.shape, (3750000, 4))


if __name__ == "__main__":
    unittest.main()
