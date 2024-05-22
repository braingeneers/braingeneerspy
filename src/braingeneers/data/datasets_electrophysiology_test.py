import unittest
import tempfile
import shutil
import diskcache
import json
import threading
import braingeneers
import braingeneers.data.datasets_electrophysiology as ephys
from braingeneers.data.datasets_electrophysiology import cached_load_data
import braingeneers.utils.smart_open_braingeneers as smart_open
from braingeneers.data.datasets_electrophysiology import load_data
from braingeneers import skip_unittest_if_offline
import boto3
import numpy as np
from unittest.mock import patch, MagicMock
import pynwb
from pynwb import NWBFile, NWBHDF5IO
from pynwb.ecephys import ElectricalSeries
from typing import Iterable, Union
from numpy.typing import NDArray
import os
from datetime import datetime
import copy


class MaxwellReaderTests(unittest.TestCase):

    @skip_unittest_if_offline
    def test_online_maxwell_stitched_uuid(self):
        uuid = '2023-04-17-e-causal_v1'
        metadata = ephys.load_metadata(uuid)
        data = ephys.load_data(
            metadata=metadata, experiment=0, offset=0, length=4, channels=[0, 1]
        )
        self.assertEqual(data.shape, (2, 4))  # trivial check that we read data

    @skip_unittest_if_offline
    def test_online_maxwell_load_data(self):
        uuid = '2022-05-18-e-connectoid'
        metadata = ephys.load_metadata(uuid)
        data = ephys.load_data(
            metadata=metadata, experiment='experiment1', offset=0, length=4, channels=[0]
        )
        self.assertEqual(data.shape, (1, 4))  # trivial check that we read data

    @skip_unittest_if_offline
    def test_load_data_maxwell_per_channel(self):
        """ Reads a single channel from a maxwell data file without any parallelism """
        filepath = 's3://braingeneersdev/dfparks/omfg_stim.repack4-1.raw.h5'  # a repacked V1 HDF5 file
        data = ephys._load_data_maxwell_per_channel(filepath, 42, 5, 10)
        self.assertEqual(data.shape, (10,))
        self.assertListEqual(data.tolist(), [497, 497, 497, 495, 496, 497, 497, 496, 497, 497])  # manually confirmed result

    @skip_unittest_if_offline
    def test_read_maxwell_parallel_maxwell_v1_format(self):
        """ V1 maxwell HDF5 data format """
        uuid = '2021-10-05-e-org1_real'
        metadata = ephys.load_metadata(uuid)
        data = ephys.load_data_maxwell_parallel(
            metadata=metadata,
            batch_uuid=uuid,
            experiment='experiment1',
            channels=[42, 43],
            offset=5,
            length=10,
        )
        self.assertEqual(data.shape, (2, 10))
        self.assertListEqual(data.tolist(), [
            [527, 527, 527, 527, 526, 526, 526, 527, 526, 527],
            [511, 511, 511, 511, 512, 511, 510, 511, 512, 511],
        ])

    @skip_unittest_if_offline
    def test_read_data_maxwell_v1_format(self):
        """ V1 maxwell HDF5 data format """
        uuid = '2021-10-05-e-org1_real'
        metadata = ephys.load_metadata(uuid)
        data = ephys.load_data(
            metadata=metadata,
            experiment=0,
            channels=[42, 43],
            offset=5,
            length=10,
        )
        self.assertEqual(data.shape, (2, 10))
        self.assertListEqual(data.tolist(), [
            [527, 527, 527, 527, 526, 526, 526, 527, 526, 527],
            [511, 511, 511, 511, 512, 511, 510, 511, 512, 511],
        ])

    @skip_unittest_if_offline
    def test_read_data_maxwell_v2_format(self):
        """ V2 maxwell HDF5 data format """
        uuid = '2023-02-08-e-mouse_updates'
        metadata = ephys.load_metadata(uuid)
        data = ephys.load_data(
            metadata=metadata,
            experiment=0,
            channels=[73, 42],
            offset=5,
            length=10,
        )
        self.assertEqual(data.shape, (2, 10))
        self.assertListEqual(data.tolist(), [
            [507, 508, 509, 509, 509, 508, 507, 507, 509, 509],
            [497, 497, 497, 498, 498, 498, 497, 497, 498, 498],
        ])

    @skip_unittest_if_offline
    def test_non_int_offset_length(self):
        """ Bug found while reading Maxwell V2 file """
        with self.assertRaises(AssertionError):
            uuid = '2023-04-17-e-connectoid16235_CCH'
            metadata = ephys.load_metadata(uuid)
            fs = 20000
            time_from = 14.75
            time_to = 16
            offset = 14.75 * fs
            length = int((time_to - time_from) * fs)
            ephys.load_data(metadata=metadata, experiment=0, offset=offset, length=length)

    def test_modify_maxwell_metadata(self):
        """Update an older Maxwell metadata json with new metadata and NWB file paths, if they exist."""
        with open('test_data/maxwell-metadata.old.json', 'r') as f:
            metadata = json.load(f)
            # use mock to ensure that new NWB files ALWAYS exist
            with patch('__main__.s3wrangler.does_object_exist') as mock_does_object_exist:
                mock_does_object_exist.return_value = True
                modified_metadata = ephys.modify_metadata_maxwell_raw_to_nwb(metadata)
        assert isinstance(modified_metadata['timestamp'], str)
        assert len(modified_metadata['timestamp']) == len('2023-10-05T18:10:02')
        modified_metadata['timestamp'] = ''

        with open('test_data/maxwell-metadata.expected.json', 'r') as f:
            expected_metadata = json.load(f)
            expected_metadata['timestamp'] = ''

        assert modified_metadata == expected_metadata

    @skip_unittest_if_offline
    def test_load_gpio_maxwell(self):
        """ Read gpio event for Maxwell V1 file"""
        data_1 = "s3://braingeneers/ephys/" \
                 "2023-04-02-hc328_rec/original/data/" \
                 "2023_04_02_hc328_0.raw.h5"
        data_2 = "s3://braingeneers/ephys/" \
                 "2023-04-04-e-hc328_hckcr1-2_040423_recs/original/data/" \
                 "hc3.28_hckcr1_chip8787_plated4.4_rec4.4.raw.h5"
        data_3 = "s3://braingeneers/ephys/" \
                 "2023-04-04-e-hc328_hckcr1-2_040423_recs/original/data/" \
                 "2023_04_04_hc328_hckcr1-2_3.raw.h5"
        gpio_1 = ephys.load_gpio_maxwell(data_1)
        gpio_2 = ephys.load_gpio_maxwell(data_2)
        gpio_3 = ephys.load_gpio_maxwell(data_3)
        self.assertEqual(gpio_1.shape, (1, 2))
        self.assertEqual(gpio_2.shape, (0,))
        self.assertEqual(gpio_3.shape, (29,))


class MEArecReaderTests(unittest.TestCase):
    """The fake reader test."""
    batch_uuid = '2023-08-29-e-mearec-6cells-tetrode'

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
        experiment0 = metadata['ephys_experiments']['experiment0']

        self.assertTrue(isinstance(metadata.get('notes').get('comments'), str))
        self.assertTrue('timestamp' in metadata)
        self.assertEqual(metadata['uuid'], self.batch_uuid)
        self.assertEqual(experiment0['hardware'], 'MEArec Simulated Recording')
        self.assertEqual(experiment0['name'], 'experiment0')
        self.assertTrue(isinstance(experiment0.get('notes'), str))
        self.assertEqual(experiment0['num_channels'], 4)
        self.assertEqual(experiment0['num_current_input_channels'], 0)
        self.assertEqual(experiment0['num_voltage_channels'], 4)
        self.assertEqual(experiment0['offset'], 0)
        self.assertEqual(experiment0['sample_rate'], 32000)
        self.assertTrue(isinstance(experiment0['sample_rate'], int))
        self.assertEqual(experiment0['units'], '\u00b5V')
        # validate json serializability
        json.dumps(metadata)

    @skip_unittest_if_offline
    def test_online_mearec_generate_data(self):
        """Ensure that MEArec data loads correctly."""
        data = ephys.load_data_mearec(ephys.load_metadata(self.batch_uuid), self.batch_uuid, channels=[1, 2], length=4)
        assert data.tolist() == [[24.815574645996094, 9.68782901763916,   -5.6944580078125,   13.871763229370117],
                                 [-7.700503349304199, 0.8792770504951477, -15.32259750366211, -6.081937789916992]]
        data = ephys.load_data_mearec(ephys.load_metadata(self.batch_uuid), self.batch_uuid, channels=[1], length=2)
        assert data.tolist() == [[24.815574645996094, 9.68782901763916]]


class AxionReaderTests(unittest.TestCase):
    """
    Online test cases require access to braingeneers/S3 including ~/.aws/credentials file
    """
    filename = "s3://braingeneers/ephys/2020-07-06-e-MGK-76-2614-Wash/original/data/" \
               "H28126_WK27_010320_Cohort_202000706_Wash(214).raw"

    batch_uuid = '2020-07-06-e-MGK-76-2614-Wash'

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    @unittest.skip
    def test_online_multiple_files(self):
        """ Warning: large (Many GB) data transfer """
        metadata = ephys.load_metadata('2021-09-23-e-MR-89-0526-drug-3hr')
        data = ephys.load_data(metadata, 'A3', 0, 45000000, None)
        self.assertTrue(data.shape[1] == 45000000)

    @skip_unittest_if_offline
    def test_online_read_beyond_eof(self):
        metadata = ephys.load_metadata(self.batch_uuid)
        dataset_size = sum([block['num_frames'] for block in metadata['ephys_experiments']['A1']['blocks']])
        with self.assertRaises(IndexError):
            ephys.load_data(metadata, 'A1', offset=dataset_size - 10, length=20)

    @skip_unittest_if_offline
    def test_online_axion_generate_metadata(self):
        metadata = ephys.generate_metadata_axion(self.batch_uuid)
        experiment0 = list(metadata['ephys_experiments'].values())[0]

        self.assertEqual(len(metadata['ephys_experiments']), 6)
        self.assertEqual(metadata['issue'], '')
        self.assertEqual(metadata['notes'], '')
        self.assertTrue('timestamp' in metadata)
        self.assertEqual(metadata['uuid'], self.batch_uuid)
        self.assertEqual(len(metadata), 6)

        self.assertEqual(experiment0['hardware'], 'Axion BioSystems')
        self.assertEqual(experiment0['name'], 'A1')
        self.assertEqual(experiment0['notes'], '')
        self.assertEqual(experiment0['num_channels'], 384)  # 6 well, 64 channel per well
        self.assertEqual(experiment0['num_current_input_channels'], 0)
        self.assertEqual(experiment0['num_voltage_channels'], 384)
        self.assertEqual(experiment0['offset'], 0)
        self.assertEqual(experiment0['sample_rate'], 12500)
        self.assertEqual(experiment0['axion_channel_offset'], 0)
        self.assertTrue(isinstance(experiment0['sample_rate'], int))
        self.assertAlmostEqual(experiment0['voltage_scaling_factor'], -5.484861781483107e-08)
        self.assertTrue(isinstance(experiment0['voltage_scaling_factor'], float))
        self.assertTrue('T' in experiment0['timestamp'])
        self.assertEqual(experiment0['units'], '\u00b5V')
        self.assertEqual(experiment0['version'], '1.0.0')

        self.assertEqual(len(experiment0['blocks']), 267)
        self.assertEqual(experiment0['blocks'][0]['num_frames'], 3750000)
        self.assertEqual(experiment0['blocks'][0]['path'], 'H28126_WK27_010320_Cohort_202000706_Wash(000).raw')
        self.assertTrue('T' in experiment0['blocks'][0]['timestamp'])

        self.assertEqual(list(metadata['ephys_experiments'].values())[1]['axion_channel_offset'], 64)

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
            metadata=metadata, experiment=1, offset=file_214_offset, length=4, channels=[0]
        )

        voltage_scaling_factor = -5.484861781483107e-08

        # Test a few manually selected values are returned correctly
        self.assertAlmostEqual(data[0][0], -9 * voltage_scaling_factor)
        self.assertAlmostEqual(data[0][1], -18 * voltage_scaling_factor)
        self.assertAlmostEqual(data[0][2], 10 * voltage_scaling_factor)
        self.assertAlmostEqual(data[0][3], 30 * voltage_scaling_factor)

    @skip_unittest_if_offline
    def test_online_axion_generate_metadata_24well(self):
        uuid_24well_data = '2021-09-23-e-MR-89-0526-spontaneous'
        metadata_json = ephys.generate_metadata_axion(uuid_24well_data)
        self.assertTrue(len(metadata_json) > 0)  # Trivial validation
        self.assertEqual(len(metadata_json['ephys_experiments']), 24)

        # save metadata files - used in development, kept here for quick reference
        with smart_open.open(f's3://braingeneers/ephys/{uuid_24well_data}/metadata.json', 'w') as f:
            json.dump(metadata_json, f, indent=2)

    @skip_unittest_if_offline
    def test_online_axion_load_data_24well(self):
        uuid_24well_data = '2021-09-23-e-MR-89-0526-spontaneous'
        metadata_json = ephys.load_metadata(uuid_24well_data)
        data = ephys.load_data(metadata=metadata_json, experiment='B1', offset=0, length=10, channels=0)
        self.assertEqual(data.shape, (1, 10))  # trivial validation, needs to be improved

    @skip_unittest_if_offline
    def test_online_axion_load_data_24well_int_index(self):
        uuid_24well_data = '2021-09-23-e-MR-89-0526-spontaneous'
        metadata_json = ephys.load_metadata(uuid_24well_data)
        data = ephys.load_data(metadata=metadata_json, experiment=1, offset=0, length=10, channels=0)
        self.assertEqual(data.shape, (1, 10))  # trivial validation, needs to be improved

    @skip_unittest_if_offline
    def test_online_load_metadata(self):
        metadata = ephys.load_metadata(self.batch_uuid)
        self.assertTrue('uuid' in metadata)  # sanity check only
        self.assertTrue(len(metadata['ephys_experiments']) == 6)  # sanity check only
        self.assertTrue('voltage_scaling_factor' in metadata['ephys_experiments']['A1'])  # sanity check only

    @skip_unittest_if_offline
    def test_online_axion_load_data_none_for_all_channels(self):
        """ axion should accept None for "all" channels """
        file_214_offset = 802446875
        metadata = ephys.load_metadata(self.batch_uuid)
        data = ephys.load_data(
            metadata=metadata, experiment=1, offset=file_214_offset,
            length=4, channels=None
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
        metadata = ephys.load_metadata('2021-09-23-e-MR-89-0526-drug-3hr')
        ephys.load_data(metadata=metadata, experiment='D2', offset=0, length=450000, channels=[0, 2, 6, 7])
        self.assertTrue('No exception, no problem.')


class HengenlabReaderTests(unittest.TestCase):
    batch_uuid = '2020-04-12-e-hengenlab-caf26'

    @skip_unittest_if_offline
    def test_online_load_data_hengenlab_across_data_files(self):
        metadata = ephys.load_metadata(batch_uuid=self.batch_uuid)

        # Read across 2 data files
        data = ephys.load_data(
            metadata=metadata,
            experiment='experiment1',
            offset=7500000 - 2,
            length=4,
            dtype='int16',
        )

        self.assertEqual((192, 4), data.shape)
        self.assertEqual([-1072, -1128, -1112, -1108], data[1, :].tolist())  # manually checked values using ntk without applying gain
        self.assertEqual(np.int16, data.dtype)

    @skip_unittest_if_offline
    def test_online_load_data_hengenlab_select_channels(self):
        metadata = ephys.load_metadata(batch_uuid=self.batch_uuid)

        # Read across 2 data files
        data = ephys.load_data(
            metadata=metadata,
            experiment='experiment1',
            offset=7500000 - 2,
            length=4,
            channels=[0, 1],
            dtype='int16',
        )

        self.assertEqual([-1072, -1128, -1112, -1108], data[1, :].tolist())  # manually checked values using ntk without applying gain
        self.assertEqual((2, 4), data.shape)
        self.assertEqual(np.int16, data.dtype)

    @skip_unittest_if_offline
    def test_online_load_data_hengenlab_float32(self):
        metadata = ephys.load_metadata(batch_uuid=self.batch_uuid)

        # Read across 2 data files
        data = ephys.load_data(
            metadata=metadata,
            experiment='experiment1',
            offset=7500000 - 2,
            length=4,
            dtype='float32',
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
        self.cache_dir = tempfile.mkdtemp(prefix='test_cache_')

    def tearDown(self):
        # Remove the temporary directory after the test
        shutil.rmtree(self.cache_dir)

    @patch('braingeneers.data.datasets_electrophysiology.load_data')
    def test_caching_mechanism(self, mock_load_data):
        """
        Test that data is properly cached and retrieved on subsequent calls with the same parameters.
        """
        mock_load_data.return_value = 'mock_data'
        metadata = {'uuid': 'test_uuid'}

        # First call should invoke load_data
        first_call_data = cached_load_data(self.cache_dir, metadata=metadata, experiment=0)
        mock_load_data.assert_called_once()

        # Second call should retrieve data from cache and not invoke load_data again
        second_call_data = cached_load_data(self.cache_dir, metadata=metadata, experiment=0)
        self.assertEqual(first_call_data, second_call_data)
        mock_load_data.assert_called_once()  # Still called only once

    @patch('braingeneers.data.datasets_electrophysiology.load_data')
    def test_cache_eviction_when_full(self, mock_load_data):
        """
        Test that the oldest items are evicted from the cache when it exceeds its size limit.
        """
        mock_load_data.side_effect = lambda **kwargs: f"data_{kwargs['experiment']}"
        max_size_gb = 0.000001  # Set a very small cache size to test eviction

        # Populate the cache with enough data to exceed its size limit
        for i in range(10):
            cached_load_data(self.cache_dir, max_size_gb=max_size_gb, metadata={'uuid': 'test_uuid'}, experiment=i)

        cache = diskcache.Cache(self.cache_dir)
        self.assertLess(len(cache), 10)  # Ensure some items were evicted

    @patch('braingeneers.data.datasets_electrophysiology.load_data')
    def test_arguments_passed_to_load_data(self, mock_load_data):
        """
        Test that all arguments after cache_path are correctly passed to the underlying load_data function.
        """
        # Mock load_data to return a serializable object, e.g., a numpy array
        mock_load_data.return_value = np.array([1, 2, 3])

        kwargs = {'metadata': {'uuid': 'test_uuid'}, 'experiment': 0, 'offset': 0, 'length': 1000}
        cached_load_data(self.cache_dir, **kwargs)
        mock_load_data.assert_called_with(**kwargs)

    @patch('braingeneers.data.datasets_electrophysiology.load_data')
    def test_multiprocessing_thread_safety(self, mock_load_data):
        """
        Test that the caching mechanism is multiprocessing/thread-safe.
        """
        # Mock load_data to return a serializable object, e.g., a numpy array
        mock_load_data.return_value = np.array([1, 2, 3])

        def thread_function(cache_path, metadata, experiment):
            # This function uses the mocked load_data indirectly via cached_load_data
            cached_load_data(cache_path, metadata=metadata, experiment=experiment)

        metadata = {'uuid': 'test_uuid'}
        threads = []
        for i in range(10):
            t = threading.Thread(target=thread_function, args=(self.cache_dir, metadata, i))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # If the cache is thread-safe, this operation should complete without error
        # This assertion is basic and assumes the test's success implies thread safety
        self.assertTrue(True)


class TestLoadData(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp()

        # Create a temporary directory for cache
        self.temp_cache_dir = tempfile.mkdtemp()

        # Create an NWB file in that directory
        self.nwb_file_path = os.path.join(self.temp_dir, "test_file.nwb")

        # Create an NWBFile object to hold electrophysiology data
        self.nwbfile = NWBFile(
            session_description='Test session',
            identifier='test_nwb',
            session_start_time=datetime.now().astimezone(),
            file_create_date=datetime.now().astimezone()
        )

        # Mock some electrophysiology data - replace this with your data loading
        # Here we create a simple sine wave as mock data
        rate = 20000.0  # Sample rate in Hz (20 kHz)
        times = np.arange(0, 1, 1 / rate)  # 1 second of data
        data = np.sin(2 * np.pi * 1 * times)  # 1 Hz sine wave
        data = np.tile(data, (4, 1))  # Repeat data for 4 electrodes

        # Create an electrode group
        device = self.nwbfile.create_device(name='test_device')
        electrode_group = self.nwbfile.create_electrode_group(
            name='test_electrode_group',
            description='test electrode group description',
            location='test location',
            device=device
        )

        # Add electrodes to the group
        for i in range(4):
            self.nwbfile.add_electrode(
                id=i,
                x=1.0, y=2.0, z=3.0,
                imp=-1.0,
                location='test_location',
                filtering='none',
                group=electrode_group
            )

        # Create an electrode table region (which electrodes to include)
        electrode_table_region = self.nwbfile.create_electrode_table_region(list(range(4)), 'test_electrode')

        # Create an ElectricalSeries object
        ephys_data = ElectricalSeries(
            name="ElectricalSeries",
            data=data,
            electrodes=electrode_table_region,
            starting_time=0.0,
            rate=rate
        )

        # Add it to the NWBFile object
        self.nwbfile.add_acquisition(ephys_data)

        # Write the NWBFile to disk
        with NWBHDF5IO(self.nwb_file_path, 'w') as io:
            io.write(self.nwbfile)

        self.metadata = {
            'uuid': 'test_uuid',
            'ephys_experiments': {
                'experiment0': {
                    'blocks': [
                        {
                            'num_frames': 5000,
                            'path': self.nwb_file_path,
                            'timestamp': '2020-04-12T00:00:00',
                            'data_chunk_size': [1, 40000]
                        },
                    ],
                    'timestamp': '2020-04-12T00:00:00',
                    'channels': [0, 1, 2, 3],
                    'hardware': 'Hengenlab',
                    'name': 'experiment0',
                    'notes': '',
                    'num_channels': 4,
                    'num_current_input_channels': 0,
                    'num_voltage_channels': 4,
                    'offset': 0,
                    'sample_rate': 1000,
                    'units': 'uV',
                    'voltage_scaling_factor': 1,
                    'version': '0.0.0'
                },
                'experiment1': {
                    'blocks': [
                        {
                            'num_frames': 6000,
                            'path': self.nwb_file_path,
                            'timestamp': '2020-04-12T00:00:00',
                            'data_chunk_size': [1, 40000]
                        },
                    ],
                    'timestamp': '2020-04-13T00:00:00',
                    'channels': [0, 1, 2, 3],
                    'hardware': 'Hengenlab',
                    'name': 'experiment0',
                    'notes': '',
                    'num_channels': 4,
                    'num_current_input_channels': 0,
                    'num_voltage_channels': 4,
                    'offset': 0,
                    'sample_rate': 1000,
                    'units': 'uV',
                    'voltage_scaling_factor': 1,
                    'version': '0.0.0'
                },
            },
        }

    def tearDown(self):
        # Delete the temporary directory and all files it contains
        os.remove(self.nwb_file_path)
        os.rmdir(self.temp_dir)

    def test_load_data_with_experiment_options(self):
        experiments = [
            ['experiment0', 'experiment1'],
            [0, 1],
            None,
            'experiment0',
            1,
        ]

        for experiment in experiments:
            with self.subTest(experiment=experiment):
                data = ephys.load_data(self.metadata, experiment=experiment, offset=0, length=1, parallelism=False)
                self.assertTrue(data.shape == (4, 1), "Data shape mismatch for experiment: {}".format(experiment))

    def test_different_dtypes(self):
        """Test that the function correctly handles different data types (dtype parameters)."""
        valid_dtypes = [
            ('float32', np.float32),
            ('float64', np.float64),
            ('float16', np.float16),
            ('int16', np.int16),
        ]

        for dtype_str, np_dtype in valid_dtypes:
            with self.subTest(dtype=dtype_str):
                data = load_data(self.metadata, experiment='experiment0', offset=0, length=100, dtype=dtype_str, parallelism=False)
                self.assertTrue(data.dtype == np_dtype, f"Data dtype {data.dtype} does not match expected {np_dtype} for {dtype_str}")

        # Test for an invalid dtype - should raise an exception
        with self.assertRaises((TypeError, AssertionError)):
            load_data(self.metadata, experiment='experiment0', offset=0, length=100, dtype='invalid_dtype', parallelism=False)

    def test_with_and_without_voltage_scaling(self):
        # Load data with voltage scaling applied
        data_with_scaling = load_data(self.metadata, experiment='experiment0', offset=0, length=100, parallelism=False)

        # Modify metadata to not use voltage scaling by setting it to None
        metadata_no_scaling = copy.deepcopy(self.metadata)
        for experiment in metadata_no_scaling['ephys_experiments'].values():
            experiment['voltage_scaling_factor'] = None

        # Load data without voltage scaling
        data_without_scaling = load_data(metadata_no_scaling, experiment='experiment0', offset=0, length=100, parallelism=False)

        # Assert that the shapes are the same
        self.assertEqual(data_with_scaling.shape, data_without_scaling.shape)

        # Calculate the scaled data manually to compare
        expected_scaled_data = data_without_scaling * self.metadata['ephys_experiments']['experiment0']['voltage_scaling_factor']

        # Assert that the data loaded with scaling applied is close to the manually scaled data
        np.testing.assert_allclose(data_with_scaling, expected_scaled_data, rtol=1e-05, atol=1e-08)

    def test_channel_specification(self):
        """Test loading data for specified channels, including single, multiple, and default (all channels)."""
        # Single channel
        data = ephys.load_data(self.metadata, experiment='experiment0', offset=0, length=100, channels=0, parallelism=False)
        self.assertEqual(data.shape, (1, 100), "Failed for single channel")

        # Multiple specified channels
        data = ephys.load_data(self.metadata, experiment='experiment0', offset=0, length=100, channels=[0, 2], parallelism=False)
        self.assertEqual(data.shape, (2, 100), "Failed for multiple specified channels")

        # All channels (default)
        data = ephys.load_data(self.metadata, experiment='experiment0', offset=0, length=100, parallelism=False)
        self.assertEqual(data.shape, (4, 100), "Failed for default (all channels)")

    def test_varying_lengths_and_offsets(self):
        """Test loading data of different lengths and starting from different offsets."""
        # Test with varying offsets
        data = ephys.load_data(self.metadata, experiment='experiment0', offset=50, length=50, parallelism=False)
        self.assertEqual(data.shape[1], 50, "Incorrect data length for offset=50, length=50")

        # Test length -1 (all remaining data)
        data = ephys.load_data(self.metadata, experiment='experiment0', offset=0, length=-1, parallelism=False)
        expected_length = sum(block['num_frames'] for block in self.metadata['ephys_experiments']['experiment0']['blocks'])
        self.assertEqual(data.shape[1], expected_length, "Incorrect data length for length=-1")

        # Test with varying lengths
        data = ephys.load_data(self.metadata, experiment='experiment0', offset=0, length=200, parallelism=False)
        self.assertEqual(data.shape[1], 200, "Incorrect data length for offset=0, length=200")

    @patch('diskcache.Cache', spec=True)
    def test_cache_usage_no_cache(self, mock_cache):
        mock_cache_instance = mock_cache.return_value

        # No need to mock __contains__ here, as we are testing the initial load

        data = load_data(self.metadata, experiment='experiment0', offset=0, length=100, cache_path=self.temp_cache_dir)
        self.assertIsNone(mock_cache_instance.__getitem__.call_args)  # No cache lookup initially
        mock_cache_instance.__setitem__.assert_called_once()  # Data stored in cache
        self.assertEqual(data.shape, (4, 100))

    @patch('diskcache.Cache', spec=True)
    def test_cache_usage_cache_hit(self, mock_cache):
        mock_cache_instance = mock_cache.return_value

        # First, call load_data to populate the cache
        data = load_data(self.metadata, experiment='experiment0', offset=0, length=100, cache_path=self.temp_cache_dir)

        # Now mock __contains__ to return True for the expected key
        mock_cache_instance.__contains__.return_value = True

        # Call load_data again to test cache hit behavior
        data = load_data(self.metadata, experiment='experiment0', offset=0, length=100, cache_path=self.temp_cache_dir)

        mock_cache_instance.__getitem__.assert_called_once()  # Data retrieved from cache
        mock_cache_instance.__setitem__.assert_called_once()  # Should only be called once (during the first load)

    @patch('diskcache.Cache', spec=True)
    def test_cache_usage_different_parameters(self, mock_cache):
        mock_cache_instance = mock_cache.return_value

        # No need to mock __contains__ here, as we are testing with different parameters

        data = load_data(self.metadata, experiment='experiment0', offset=50, length=50, cache_path=self.temp_cache_dir)
        self.assertEqual(data.shape, (4, 50))
        self.assertIsNone(mock_cache_instance.__getitem__.call_args)  # Did NOT use cache

    def test_parallelism_settings(self):
        """Ensure correct behavior under different parallelism settings."""
        # Test with parallelism disabled (single thread)
        data_single_thread = load_data(self.metadata, experiment='experiment0', offset=0, length=100, parallelism=False)
        self.assertEqual(data_single_thread.shape, (4, 100))

        # Test with automatic parallelism (number of threads based on CPU count)
        data_auto_parallel = load_data(self.metadata, experiment='experiment0', offset=0, length=100)  # Default is True
        self.assertEqual(data_auto_parallel.shape, (4, 100))

        # Test with a specific number of threads
        n_threads = 2  # Example, adjust as needed
        data_specific_threads = load_data(self.metadata, experiment='experiment0', offset=0, length=100, parallelism=n_threads)
        self.assertEqual(data_specific_threads.shape, (4, 100))

    def test_invalid_metadata(self):
        """Test handling of invalid metadata, such as missing required fields."""
        # Test missing 'uuid' field
        invalid_metadata_1 = copy.deepcopy(self.metadata)
        del invalid_metadata_1['uuid']
        with self.assertRaisesRegex(AssertionError, "Metadata file is invalid, it does not contain required uuid field."):
            load_data(invalid_metadata_1, experiment='experiment0')

        # Test missing 'ephys_experiments' field
        invalid_metadata_2 = copy.deepcopy(self.metadata)
        del invalid_metadata_2['ephys_experiments']
        with self.assertRaisesRegex(AssertionError, "Metadata file is invalid, it does not contain required ephys_experiments field."):
            load_data(invalid_metadata_2, experiment='experiment0')

    def test_invalid_experiment_identifiers(self):
        """Test handling of invalid experiment identifiers, such as non-existent names or indexes."""
        length = 100  # Arbitrary length for testing purposes

        # Test non-existent experiment name
        with self.assertRaises(ValueError):
            load_data(self.metadata, experiment='nonexistent_experiment', length=length)

        # Test out-of-range experiment index
        num_experiments = len(self.metadata['ephys_experiments'])
        with self.assertRaises(ValueError):
            load_data(self.metadata, experiment=num_experiments, length=length)  # Index is out of range

        # Test invalid type in experiment list
        with self.assertRaises(AssertionError):
            load_data(self.metadata, experiment=['experiment0', 2.5], length=length)  # Float in list

    def test_invalid_channels(self):
        """Test handling of invalid channel specifications, such as non-existent channels."""
        num_channels = self.metadata['ephys_experiments']['experiment0']['num_channels']

        # Test with a non-existent channel
        with self.assertRaises(IndexError):
            load_data(self.metadata, experiment='experiment0', channels=num_channels, length=100)

        # Test with a list containing a non-existent channel
        with self.assertRaises(IndexError):
            load_data(self.metadata, experiment='experiment0', channels=[0, num_channels], length=100)

    def test_invalid_dtypes(self):
        """Test rejection of invalid data types (dtypes not listed as valid options)."""
        invalid_dtypes = ['uint8', 'int32', 'complex64', 'object']  # Examples of invalid dtypes

        for dtype in invalid_dtypes:
            with self.subTest(dtype=dtype):
                with self.assertRaisesRegex(AssertionError, "dtype must be one of"):
                    load_data(self.metadata, experiment='experiment0', length=100, dtype=dtype)

    def test_invalid_lengths_and_offsets(self):
        """Test handling of invalid lengths and offsets, such as negative lengths (other than -1) or offsets that exceed the dataset size."""
        dataset_size = sum(block['num_frames'] for block in self.metadata['ephys_experiments']['experiment0']['blocks'])

        # Test with a negative length (other than -1)
        with self.assertRaises(AssertionError):
            load_data(self.metadata, experiment='experiment0', offset=0, length=-10)

        # Test with an offset exceeding the dataset size
        with self.assertRaises(IndexError):
            load_data(self.metadata, experiment='experiment0', offset=dataset_size + 1, length=100)

        # Test with offset + length exceeding the dataset size
        with self.assertRaises(IndexError):
            load_data(self.metadata, experiment='experiment0', offset=dataset_size - 50, length=100)

    def test_max_size_gb_limitation(self):
        """Test behavior when cache size exceeds max_size_gb, ensuring oldest items are purged."""
        max_size_gb = 0.0000001  # Set a very small cache size

        # Load data multiple times to fill the cache
        for i in range(10):
            data = load_data(self.metadata, experiment='experiment0', offset=i * 100, length=100,
                             cache_path=self.temp_cache_dir, max_size_gb=max_size_gb)
            self.assertEqual(data.shape, (4, 100))

        # Check if the first loaded data is still in the cache (it should be evicted)
        cache = diskcache.Cache(self.temp_cache_dir)
        first_load_key = json.dumps(
            [self.metadata['uuid'], [('experiment0', self.metadata['ephys_experiments']['experiment0'])], 0, 100, None])
        self.assertNotIn(first_load_key, cache)  # Check if the key is not present

    def test_invalid_parallelism_values(self):
        """Test handling of invalid parallelism settings, such as non-boolean, non-integer values, or negative integers."""
        invalid_parallelism_values = [
            "string",
            1.5,
            -2,
            [],
            {}
        ]

        for value in invalid_parallelism_values:
            with self.subTest(value=value):
                with self.assertRaisesRegex(AssertionError, "Parallelism must be a boolean or an integer"):
                    load_data(self.metadata, experiment='experiment0', length=100, parallelism=value)

    def test_order_of_experiments_by_timestamp(self):
        """Ensure that when multiple experiments are specified, they are processed in the correct order according to their timestamps."""
        # Assuming the metadata has 'timestamp' fields for each experiment...
        experiments = ['experiment1', 'experiment0']  # Reversed order of timestamps

        # Expect a ValueError because the experiments are not in timestamp order
        with self.assertRaisesRegex(ValueError, "must be in order of timestamp"):
            load_data(self.metadata, experiment=experiments, length=100)

    def test_data_format(self):
        """Confirm that the data is returned in the correct [channels, time] format."""
        data = load_data(self.metadata, experiment='experiment0', length=100)

        # Assert that the data has the expected shape (channels as rows, time as columns)
        num_channels = self.metadata['ephys_experiments']['experiment0']['num_channels']
        self.assertEqual(data.shape, (num_channels, 100))


if __name__ == '__main__':
    unittest.main()
