import json
import shutil
import tempfile
import threading
import unittest
import sys
from unittest.mock import patch

import diskcache
import numpy as np
import pytest

import braingeneers.data.datasets_electrophysiology as ephys
import braingeneers.utils.smart_open_braingeneers as smart_open
from braingeneers import skip_unittest_if_offline
from braingeneers.data.datasets_electrophysiology import cached_load_data


@pytest.mark.filterwarnings("ignore::UserWarning")
class MaxwellReaderTests(unittest.TestCase):
    @skip_unittest_if_offline
    def test_online_maxwell_stitched_uuid(self):
        uuid = "2023-04-17-e-causal_v1"
        metadata = ephys.load_metadata(uuid)
        data = ephys.load_data(
            metadata=metadata, experiment=0, offset=0, length=4, channels=[0, 1]
        )
        self.assertEqual(data.shape, (2, 4))  # trivial check that we read data

    @skip_unittest_if_offline
    def test_online_maxwell_load_data(self):
        uuid = "2022-05-18-e-connectoid"
        metadata = ephys.load_metadata(uuid)
        data = ephys.load_data(
            metadata=metadata,
            experiment="experiment1",
            offset=0,
            length=4,
            channels=[0],
        )
        self.assertEqual(data.shape, (1, 4))  # trivial check that we read data

    @skip_unittest_if_offline
    def test_load_data_maxwell_per_channel(self):
        """Reads a single channel from a maxwell data file without any parallelism"""
        filepath = "s3://braingeneersdev/dfparks/omfg_stim.repack4-1.raw.h5"  # a repacked V1 HDF5 file
        data = ephys._load_data_maxwell_per_channel(filepath, 42, 5, 10)
        self.assertEqual(data.shape, (10,))
        self.assertListEqual(
            data.tolist(), [497, 497, 497, 495, 496, 497, 497, 496, 497, 497]
        )  # manually confirmed result

    @skip_unittest_if_offline
    def test_read_maxwell_parallel_maxwell_v1_format(self):
        """V1 maxwell HDF5 data format"""
        uuid = "2021-10-05-e-org1_real"
        metadata = ephys.load_metadata(uuid)
        data = ephys.load_data_maxwell_parallel(
            metadata=metadata,
            batch_uuid=uuid,
            experiment="experiment1",
            channels=[42, 43],
            offset=5,
            length=10,
        )
        self.assertEqual(data.shape, (2, 10))
        self.assertListEqual(
            data.tolist(),
            [
                [527, 527, 527, 527, 526, 526, 526, 527, 526, 527],
                [511, 511, 511, 511, 512, 511, 510, 511, 512, 511],
            ],
        )

    @skip_unittest_if_offline
    def test_read_data_maxwell_v1_format(self):
        """V1 maxwell HDF5 data format"""
        uuid = "2021-10-05-e-org1_real"
        metadata = ephys.load_metadata(uuid)
        data = ephys.load_data(
            metadata=metadata,
            experiment=0,
            channels=[42, 43],
            offset=5,
            length=10,
        )
        self.assertEqual(data.shape, (2, 10))
        self.assertListEqual(
            data.tolist(),
            [
                [527, 527, 527, 527, 526, 526, 526, 527, 526, 527],
                [511, 511, 511, 511, 512, 511, 510, 511, 512, 511],
            ],
        )

    @unittest.skipIf(sys.platform.startswith("win"), "TODO: Test is broken on Windows.")
    @skip_unittest_if_offline
    def test_read_data_maxwell_v2_format(self):
        """V2 maxwell HDF5 data format"""
        uuid = "2023-02-08-e-mouse_updates"
        metadata = ephys.load_metadata(uuid)
        data = ephys.load_data(
            metadata=metadata,
            experiment=0,
            channels=[73, 42],
            offset=5,
            length=10,
        )
        self.assertEqual(data.shape, (2, 10))
        self.assertListEqual(
            data.tolist(),
            [
                [507, 508, 509, 509, 509, 508, 507, 507, 509, 509],
                [497, 497, 497, 498, 498, 498, 497, 497, 498, 498],
            ],
        )

    @skip_unittest_if_offline
    def test_non_int_offset_length(self):
        """Bug found while reading Maxwell V2 file"""
        with self.assertRaises(AssertionError):
            uuid = "2023-04-17-e-connectoid16235_CCH"
            metadata = ephys.load_metadata(uuid)
            fs = 20000
            time_from = 14.75
            time_to = 16
            offset = 14.75 * fs
            length = int((time_to - time_from) * fs)
            ephys.load_data(
                metadata=metadata, experiment=0, offset=offset, length=length
            )

    def test_modify_maxwell_metadata(self):
        """Update an older Maxwell metadata json with new metadata and NWB file paths, if they exist."""
        with open("tests/test_data/maxwell-metadata.old.json", "r") as f:
            metadata = json.load(f)
            # use mock to ensure that new NWB files ALWAYS exist
            with patch(
                "braingeneers.utils.s3wrangler.does_object_exist"
            ) as mock_does_object_exist:
                mock_does_object_exist.return_value = True
                modified_metadata = ephys.modify_metadata_maxwell_raw_to_nwb(metadata)
        assert isinstance(modified_metadata["timestamp"], str)
        assert len(modified_metadata["timestamp"]) == len("2023-10-05T18:10:02")
        modified_metadata["timestamp"] = ""

        with open("tests/test_data/maxwell-metadata.expected.json", "r") as f:
            expected_metadata = json.load(f)
            expected_metadata["timestamp"] = ""

        assert modified_metadata == expected_metadata

    @skip_unittest_if_offline
    def test_load_gpio_maxwell(self):
        """Read gpio event for Maxwell V1 file"""
        data_1 = (
            "s3://braingeneers/ephys/"
            "2023-04-02-hc328_rec/original/data/"
            "2023_04_02_hc328_0.raw.h5"
        )
        data_2 = (
            "s3://braingeneers/ephys/"
            "2023-04-04-e-hc328_hckcr1-2_040423_recs/original/data/"
            "hc3.28_hckcr1_chip8787_plated4.4_rec4.4.raw.h5"
        )
        data_3 = (
            "s3://braingeneers/ephys/"
            "2023-04-04-e-hc328_hckcr1-2_040423_recs/original/data/"
            "2023_04_04_hc328_hckcr1-2_3.raw.h5"
        )
        gpio_1 = ephys.load_gpio_maxwell(data_1)
        gpio_2 = ephys.load_gpio_maxwell(data_2)
        gpio_3 = ephys.load_gpio_maxwell(data_3)
        self.assertEqual(gpio_1.shape, (1, 2))
        self.assertEqual(gpio_2.shape, (0,))
        self.assertEqual(gpio_3.shape, (29,))


class MEArecReaderTests(unittest.TestCase):
    """The fake reader test."""

    batch_uuid = "2023-08-29-e-mearec-6cells-tetrode"

    @skip_unittest_if_offline
    def test_online_mearec_generate_metadata(self):
        """
         Metadata json output should be this with different timestamps:

        {"uuid": "2023-08-29-e-mearec-6cells-tetrode",
         "timestamp": "2023-09-20T14:59:37",
         "notes": {"comments": "This data is a simulated recording generated by MEArec."},
         "ephys_experiments": {
             "experiment0": {
                 "name": "experiment0",
                 "hardware": "MEArec Simulated Recording",
                 "notes": "This data is a simulated recording generated by MEArec.",
                 "timestamp": "2023-09-20T14:59:37",
                 "sample_rate": 32000,
                 "num_channels": 4,
                 "num_current_input_channels": 0,
                 "num_voltage_channels": 4,
                 "channels": [0, 1, 2, 3],
                 "offset": 0,
                 "voltage_scaling_factor": 1,
                 "units": "\u00b5V",
                 "version": "0.0.0",
                 "blocks": [{"num_frames": 960000,
                             "path": "s3://braingeneers/ephys/2023-08-29-e-mearec-6cells-tetrode/original/data/recordings_6cells_tetrode_30.0_10.0uV.h5",
                             "timestamp": "2023-09-20T14:59:37",
                             "data_order": "rowmajor"}]}}}
        """
        metadata = ephys.generate_metadata_mearec(self.batch_uuid)
        experiment0 = metadata["ephys_experiments"]["experiment0"]

        self.assertTrue(isinstance(metadata.get("notes").get("comments"), str))
        self.assertTrue("timestamp" in metadata)
        self.assertEqual(metadata["uuid"], self.batch_uuid)
        self.assertEqual(experiment0["hardware"], "MEArec Simulated Recording")
        self.assertEqual(experiment0["name"], "experiment0")
        self.assertTrue(isinstance(experiment0.get("notes"), str))
        self.assertEqual(experiment0["num_channels"], 4)
        self.assertEqual(experiment0["num_current_input_channels"], 0)
        self.assertEqual(experiment0["num_voltage_channels"], 4)
        self.assertEqual(experiment0["offset"], 0)
        self.assertEqual(experiment0["sample_rate"], 32000)
        self.assertTrue(isinstance(experiment0["sample_rate"], int))
        self.assertEqual(experiment0["units"], "\u00b5V")
        # validate json serializability
        json.dumps(metadata)

    @skip_unittest_if_offline
    def test_online_mearec_generate_data(self):
        """Ensure that MEArec data loads correctly."""
        data = ephys.load_data_mearec(
            ephys.load_metadata(self.batch_uuid),
            self.batch_uuid,
            channels=[1, 2],
            length=4,
        )
        assert data.tolist() == [
            [
                24.815574645996094,
                9.68782901763916,
                -5.6944580078125,
                13.871763229370117,
            ],
            [
                -7.700503349304199,
                0.8792770504951477,
                -15.32259750366211,
                -6.081937789916992,
            ],
        ]
        data = ephys.load_data_mearec(
            ephys.load_metadata(self.batch_uuid),
            self.batch_uuid,
            channels=[1],
            length=2,
        )
        assert data.tolist() == [[24.815574645996094, 9.68782901763916]]


class AxionReaderTests(unittest.TestCase):
    """
    Online test cases require access to braingeneers/S3 including ~/.aws/credentials file
    """

    filename = (
        "s3://braingeneers/ephys/2020-07-06-e-MGK-76-2614-Wash/original/data/"
        "H28126_WK27_010320_Cohort_202000706_Wash(214).raw"
    )

    batch_uuid = "2020-07-06-e-MGK-76-2614-Wash"

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    # TypeError: unhashable type: 'dict' (in load_data_axion L381)
    @unittest.expectedFailure
    @unittest.skip("Large (many GB) data transfer")
    def test_online_multiple_files(self):
        metadata = ephys.load_metadata("2021-09-23-e-MR-89-0526-drug-3hr")
        data = ephys.load_data(metadata, "A3", 0, 45000000, None)
        self.assertTrue(data.shape[1] == 45000000)

    @skip_unittest_if_offline
    def test_online_read_beyond_eof(self):
        metadata = ephys.load_metadata(self.batch_uuid)
        dataset_size = sum(
            [
                block["num_frames"]
                for block in metadata["ephys_experiments"]["A1"]["blocks"]
            ]
        )
        with self.assertRaises(IndexError):
            ephys.load_data(metadata, "A1", offset=dataset_size - 10, length=20)

    @skip_unittest_if_offline
    def test_online_axion_generate_metadata(self):
        metadata = ephys.generate_metadata_axion(self.batch_uuid)
        experiment0 = list(metadata["ephys_experiments"].values())[0]

        self.assertEqual(len(metadata["ephys_experiments"]), 6)
        self.assertEqual(metadata["issue"], "")
        self.assertEqual(metadata["notes"], "")
        self.assertTrue("timestamp" in metadata)
        self.assertEqual(metadata["uuid"], self.batch_uuid)
        self.assertEqual(len(metadata), 6)

        self.assertEqual(experiment0["hardware"], "Axion BioSystems")
        self.assertEqual(experiment0["name"], "A1")
        self.assertEqual(experiment0["notes"], "")
        self.assertEqual(
            experiment0["num_channels"], 384
        )  # 6 well, 64 channel per well
        self.assertEqual(experiment0["num_current_input_channels"], 0)
        self.assertEqual(experiment0["num_voltage_channels"], 384)
        self.assertEqual(experiment0["offset"], 0)
        self.assertEqual(experiment0["sample_rate"], 12500)
        self.assertEqual(experiment0["axion_channel_offset"], 0)
        self.assertTrue(isinstance(experiment0["sample_rate"], int))
        self.assertAlmostEqual(
            experiment0["voltage_scaling_factor"], -5.484861781483107e-08
        )
        self.assertTrue(isinstance(experiment0["voltage_scaling_factor"], float))
        self.assertTrue("T" in experiment0["timestamp"])
        self.assertEqual(experiment0["units"], "\u00b5V")
        self.assertEqual(experiment0["version"], "1.0.0")

        self.assertEqual(len(experiment0["blocks"]), 267)
        self.assertEqual(experiment0["blocks"][0]["num_frames"], 3750000)
        self.assertEqual(
            experiment0["blocks"][0]["path"],
            "H28126_WK27_010320_Cohort_202000706_Wash(000).raw",
        )
        self.assertTrue("T" in experiment0["blocks"][0]["timestamp"])

        self.assertEqual(
            list(metadata["ephys_experiments"].values())[1]["axion_channel_offset"], 64
        )

        # validate json serializability
        json.dumps(metadata)

        # save metadata files - used in development, kept here for quick reference
        # with smart_open.open(f's3://braingeneers/ephys/{self.batch_uuid}/metadata.json', 'w') as f:
        #     json.dump(metadata, f, indent=2)

    @skip_unittest_if_offline
    def test_online_load_data_axion(self):
        file_214_offset = 802446875
        metadata = ephys.load_metadata(self.batch_uuid)
        data = ephys.load_data(
            metadata=metadata,
            experiment=1,
            offset=file_214_offset,
            length=4,
            channels=[0],
        )

        voltage_scaling_factor = -5.484861781483107e-08

        # Test a few manually selected values are returned correctly
        self.assertAlmostEqual(data[0][0], -9 * voltage_scaling_factor)
        self.assertAlmostEqual(data[0][1], -18 * voltage_scaling_factor)
        self.assertAlmostEqual(data[0][2], 10 * voltage_scaling_factor)
        self.assertAlmostEqual(data[0][3], 30 * voltage_scaling_factor)

    @skip_unittest_if_offline
    def test_online_axion_generate_metadata_24well(self):
        uuid_24well_data = "2021-09-23-e-MR-89-0526-spontaneous"
        metadata_json = ephys.generate_metadata_axion(uuid_24well_data)
        self.assertTrue(len(metadata_json) > 0)  # Trivial validation
        self.assertEqual(len(metadata_json["ephys_experiments"]), 24)

        # save metadata files - used in development, kept here for quick reference
        with smart_open.open(
            f"s3://braingeneers/ephys/{uuid_24well_data}/metadata.json", "w"
        ) as f:
            json.dump(metadata_json, f, indent=2)

    @skip_unittest_if_offline
    def test_online_axion_load_data_24well(self):
        uuid_24well_data = "2021-09-23-e-MR-89-0526-spontaneous"
        metadata_json = ephys.load_metadata(uuid_24well_data)
        data = ephys.load_data(
            metadata=metadata_json, experiment="B1", offset=0, length=10, channels=0
        )
        self.assertEqual(
            data.shape, (1, 10)
        )  # trivial validation, needs to be improved

    @skip_unittest_if_offline
    def test_online_axion_load_data_24well_int_index(self):
        uuid_24well_data = "2021-09-23-e-MR-89-0526-spontaneous"
        metadata_json = ephys.load_metadata(uuid_24well_data)
        data = ephys.load_data(
            metadata=metadata_json, experiment=1, offset=0, length=10, channels=0
        )
        self.assertEqual(
            data.shape, (1, 10)
        )  # trivial validation, needs to be improved

    @skip_unittest_if_offline
    def test_online_load_metadata(self):
        metadata = ephys.load_metadata(self.batch_uuid)
        self.assertTrue("uuid" in metadata)  # sanity check only
        self.assertTrue(len(metadata["ephys_experiments"]) == 6)  # sanity check only
        self.assertTrue(
            "voltage_scaling_factor" in metadata["ephys_experiments"]["A1"]
        )  # sanity check only

    @skip_unittest_if_offline
    def test_online_axion_load_data_none_for_all_channels(self):
        """axion should accept None for "all" channels"""
        file_214_offset = 802446875
        metadata = ephys.load_metadata(self.batch_uuid)
        data = ephys.load_data(
            metadata=metadata,
            experiment=1,
            offset=file_214_offset,
            length=4,
            channels=None,
        )

        voltage_scaling_factor = -5.484861781483107e-08

        # Test a few manually selected values are returned correctly
        self.assertAlmostEqual(data[0][0], -9 * voltage_scaling_factor)
        self.assertAlmostEqual(data[0][1], -18 * voltage_scaling_factor)
        self.assertAlmostEqual(data[0][2], 10 * voltage_scaling_factor)
        self.assertAlmostEqual(data[0][3], 30 * voltage_scaling_factor)

    @skip_unittest_if_offline
    def test_bug_read_length_neg_one(self):
        """
        Tests a bug reported by Matt getting the error: ValueError: read length must be non-negative or -1
        :return:
        """
        metadata = ephys.load_metadata("2021-09-23-e-MR-89-0526-drug-3hr")
        ephys.load_data(
            metadata=metadata,
            experiment="D2",
            offset=0,
            length=450000,
            channels=[0, 2, 6, 7],
        )
        self.assertTrue("No exception, no problem.")


class HengenlabReaderTests(unittest.TestCase):
    batch_uuid = "2020-04-12-e-hengenlab-caf26"

    @skip_unittest_if_offline
    def test_online_load_data_hengenlab_across_data_files(self):
        metadata = ephys.load_metadata(batch_uuid=self.batch_uuid)

        # Read across 2 data files
        data = ephys.load_data(
            metadata=metadata,
            experiment="experiment1",
            offset=7500000 - 2,
            length=4,
            dtype="int16",
        )

        self.assertEqual((192, 4), data.shape)
        self.assertEqual(
            [-1072, -1128, -1112, -1108], data[1, :].tolist()
        )  # manually checked values using ntk without applying gain
        self.assertEqual(np.int16, data.dtype)

    @skip_unittest_if_offline
    def test_online_load_data_hengenlab_select_channels(self):
        metadata = ephys.load_metadata(batch_uuid=self.batch_uuid)

        # Read across 2 data files
        data = ephys.load_data(
            metadata=metadata,
            experiment="experiment1",
            offset=7500000 - 2,
            length=4,
            channels=[0, 1],
            dtype="int16",
        )

        self.assertEqual(
            [-1072, -1128, -1112, -1108], data[1, :].tolist()
        )  # manually checked values using ntk without applying gain
        self.assertEqual((2, 4), data.shape)
        self.assertEqual(np.int16, data.dtype)

    @skip_unittest_if_offline
    def test_online_load_data_hengenlab_float32(self):
        metadata = ephys.load_metadata(batch_uuid=self.batch_uuid)

        # Read across 2 data files
        data = ephys.load_data(
            metadata=metadata,
            experiment="experiment1",
            offset=7500000 - 2,
            length=4,
            dtype="float32",
        )

        gain = np.float64(0.19073486328125)
        expected_raw = [-1072, -1128, -1112, -1108]
        expected_float32 = np.array(expected_raw, dtype=np.int16) * gain

        # this can't be checked with ntk easily because ntk also applies an odd int16 scaling of the data
        self.assertTrue(np.all(expected_float32 == data[1, :]))
        self.assertEqual((192, 4), data.shape)
        self.assertEqual(np.float32, data.dtype)


class TestCachedLoadData(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for the cache
        self.cache_dir = tempfile.mkdtemp(prefix="test_cache_")

    def tearDown(self):
        # Remove the temporary directory after the test
        shutil.rmtree(self.cache_dir)

    @unittest.skipIf(sys.platform.startswith("win"), "TODO: Test is broken on Windows.")
    @patch("braingeneers.data.datasets_electrophysiology.load_data")
    def test_caching_mechanism(self, mock_load_data):
        """
        Test that data is properly cached and retrieved on subsequent calls with the same parameters.
        """
        mock_load_data.return_value = "mock_data"
        metadata = {"uuid": "test_uuid"}

        # First call should invoke load_data
        first_call_data = cached_load_data(
            self.cache_dir, metadata=metadata, experiment=0
        )
        mock_load_data.assert_called_once()

        # Second call should retrieve data from cache and not invoke load_data again
        second_call_data = cached_load_data(
            self.cache_dir, metadata=metadata, experiment=0
        )
        self.assertEqual(first_call_data, second_call_data)
        mock_load_data.assert_called_once()  # Still called only once

    @unittest.skipIf(sys.platform.startswith("win"), "TODO: Test is broken on Windows.")
    @patch("braingeneers.data.datasets_electrophysiology.load_data")
    def test_cache_eviction_when_full(self, mock_load_data):
        """
        Test that the oldest items are evicted from the cache when it exceeds its size limit.
        """
        mock_load_data.side_effect = lambda **kwargs: f"data_{kwargs['experiment']}"
        max_size_gb = 0.000001  # Set a very small cache size to test eviction

        # Populate the cache with enough data to exceed its size limit
        for i in range(10):
            cached_load_data(
                self.cache_dir,
                max_size_gb=max_size_gb,
                metadata={"uuid": "test_uuid"},
                experiment=i,
            )

        cache = diskcache.Cache(self.cache_dir)
        self.assertLess(len(cache), 10)  # Ensure some items were evicted

    @unittest.skipIf(sys.platform.startswith("win"), "TODO: Test is broken on Windows.")
    @patch("braingeneers.data.datasets_electrophysiology.load_data")
    def test_arguments_passed_to_load_data(self, mock_load_data):
        """
        Test that all arguments after cache_path are correctly passed to the underlying load_data function.
        """
        # Mock load_data to return a serializable object, e.g., a numpy array
        mock_load_data.return_value = np.array([1, 2, 3])

        kwargs = {
            "metadata": {"uuid": "test_uuid"},
            "experiment": 0,
            "offset": 0,
            "length": 1000,
        }
        cached_load_data(self.cache_dir, **kwargs)
        mock_load_data.assert_called_with(**kwargs)

    @unittest.skipIf(sys.platform.startswith("win"), "TODO: Test is broken on Windows.")
    @patch("braingeneers.data.datasets_electrophysiology.load_data")
    def test_multiprocessing_thread_safety(self, mock_load_data):
        """
        Test that the caching mechanism is multiprocessing/thread-safe.
        """
        # Mock load_data to return a serializable object, e.g., a numpy array
        mock_load_data.return_value = np.array([1, 2, 3])

        def thread_function(cache_path, metadata, experiment):
            # This function uses the mocked load_data indirectly via cached_load_data
            cached_load_data(cache_path, metadata=metadata, experiment=experiment)

        metadata = {"uuid": "test_uuid"}
        threads = []
        for i in range(10):
            t = threading.Thread(
                target=thread_function, args=(self.cache_dir, metadata, i)
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # If the cache is thread-safe, this operation should complete without error
        # This assertion is basic and assumes the test's success implies thread safety
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
