import unittest
import braingeneers
import braingeneers.data.datasets_electrophysiology as ephys
import json
from braingeneers import skip_unittest_if_offline
# import braingeneers.utils.smart_open_braingeneers as smart_open
import smart_open
import boto3
import numpy as np


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

    @skip_unittest_if_offline
    def test_online_generate_metadata(self):
        metadata = ephys.generate_metadata_hengenlab(
            batch_uuid=self.batch_uuid,
            dataset_name='CAF26',
            save=False,
        )

        # top level items
        self.assertEqual(metadata['uuid'], '2020-04-12-e-hengenlab-caf26')
        self.assertEqual(metadata['timestamp'], '2020-08-07T14:00:15')
        self.assertEqual(metadata['issue'], '')
        self.assertEqual(metadata['headstage_types'], ['EAB50chmap_00', 'APT_PCB', 'APT_PCB'])

        # notes
        self.assertEqual(metadata['notes']['biology']['sample_type'], 'mouse')
        self.assertEqual(metadata['notes']['biology']['dataset_name'], 'CAF26')
        self.assertEqual(metadata['notes']['biology']['birthday'], '2020-02-20T07:30:00')
        self.assertEqual(metadata['notes']['biology']['genotype'], 'wt')

        # ephys_experiments
        self.assertEqual(len(metadata['ephys_experiments']), 1)
        self.assertTrue(isinstance(metadata['ephys_experiments'], list))

        experiment = metadata['ephys_experiments'][0]
        self.assertEqual(experiment['name'], 'experiment1')
        self.assertEqual(experiment['hardware'], 'Hengenlab')
        self.assertEqual(experiment['num_channels'], 192)
        self.assertEqual(experiment['sample_rate'], 25000)
        self.assertEqual(experiment['voltage_scaling_factor'], 0.19073486328125)
        self.assertEqual(experiment['timestamp'], '2020-08-07T14:00:15')
        self.assertEqual(experiment['units'], '\u00b5V')
        self.assertEqual(experiment['version'], '1.0.0')
        self.assertEqual(len(experiment['blocks']), 324)

        block1 = metadata['ephys_experiments'][0]['blocks'][1]
        self.assertEqual(block1['num_frames'], 7500000)
        self.assertEqual(block1['path'], 'original/experiment1/Headstages_192_Channels_int16_2020-08-07_14-05-16.bin')
        self.assertEqual(block1['timestamp'], '2020-08-07T14:05:16')
        self.assertEqual(block1['ecube_time'], 301061600050)


if __name__ == '__main__':
    unittest.main()
