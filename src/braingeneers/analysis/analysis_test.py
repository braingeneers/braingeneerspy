import unittest

import braingeneers.analysis as ba
from braingeneers import skip_unittest_if_offline


class TestSpikeDataLoaders(unittest.TestCase):
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
    def testReadPhyFiles(self):
        from posixpath import join

        from braingeneers.utils.common_utils import get_basepath

        uuid = "2023-04-17-e-connectoid16235_CCH"
        sorter = "kilosort2"
        file = "Trace_20230418_15_10_08_chip16235_curated_s1.zip"
        path = join(get_basepath(), "ephys", uuid, "derived", sorter, file)
        sd = ba.read_phy_files(path)
        self.assertTrue(isinstance(sd, ba.SpikeData))

    @skip_unittest_if_offline
    def testSpikeAttributesDiffSorter(self):
        uuid = "2023-04-17-e-causal_v1"
        exp = "data_phy.zip"
        sorter = "kilosort3"
        sd = ba.load_spike_data(uuid, exp, sorter=sorter)
        self.assertTrue(isinstance(sd, ba.SpikeData))
