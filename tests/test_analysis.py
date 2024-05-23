import unittest

import numpy as np

import braingeneers.analysis as ba
from braingeneers import skip_unittest_if_offline


class TestSpikeDataLoaders(unittest.TestCase):
    def assertAll(self, bools, msg=None):
        "Assert that two arrays are equal elementwise."
        self.assertTrue(np.all(bools), msg=msg)

    @skip_unittest_if_offline
    def testSpikeAttributes(self):
        uuid = "2023-04-17-e-causal_v1"
        sd = ba.load_spike_data(uuid)
        self.assertTrue(isinstance(sd, ba.SpikeData))
        r = sd.raster(1)
        rr = sd.randomized(1).raster(1)
        self.assertAll(r.sum(1) == rr.sum(1))
        self.assertAll(r.sum(0) == rr.sum(0))

    @skip_unittest_if_offline
    def testSpikeAttributesDiffSorter(self):
        uuid = "2023-04-17-e-causal_v1"
        exp = "data_phy.zip"
        sorter = "kilosort3"
        sd = ba.load_spike_data(uuid, exp, sorter=sorter)
        self.assertTrue(isinstance(sd, ba.SpikeData))
