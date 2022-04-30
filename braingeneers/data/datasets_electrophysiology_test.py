import unittest
import braingeneers.data.datasets_electrophysiology as ephys
import json
from braingeneers import skip_unittest_if_offline
import braingeneers.utils.smart_open_braingeneers as smart_open


class MaxwellReaderTests(unittest.TestCase):
    @skip_unittest_if_offline
    def test_online_maxwell_load_data(self):
        self.fail()


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
        experiment0 = metadata['ephys_experiments'][0]

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

        self.assertEqual(metadata['ephys_experiments'][1]['axion_channel_offset'], 64)

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
        self.assertEqual(data[0][0], -9 * voltage_scaling_factor)
        self.assertEqual(data[0][1], -18 * voltage_scaling_factor)
        self.assertEqual(data[0][2], 10 * voltage_scaling_factor)
        self.assertEqual(data[0][3], 30 * voltage_scaling_factor)


class HengenlabReaderTests(unittest.TestCase):
    batch_uuid = '2020-04-12-e-hengenlab-caf26'

    @skip_unittest_if_offline
    def test_online_generate_metadata(self):
        metadata = ephys.generate_metadata_hengenlab(
            batch_uuid=self.batch_uuid,
            dataset_name='CAF26',
        )

        self.assertEqual(metadata['issue'], '')
        self.assertEqual(metadata['notes'], '')
        self.assertEqual(metadata['timestamp'], '2020-08-07T14:00:15')
        self.assertEqual(metadata['uuid'], '2020-04-12-e-hengenlab-caf26')
        self.assertEqual(len(metadata['ephys_experiments']), 1)

        experiment = metadata['ephys_experiments'][0]
        self.assertEqual(experiment['name'], 'experiment1')
        self.assertEqual(experiment['hardware'], 'Hengenlab eCube')
        self.assertEqual(experiment['notes'], '')
        self.assertEqual(experiment['num_channels'], 192)
        self.assertEqual(experiment['sample_rate'], 25000)
        self.assertEqual(experiment['voltage_scaling_factor'], 'TODO')
        self.assertEqual(experiment['timestamp'], '2020-08-07T14:00:15')
        self.assertEqual(experiment['units'], '\u00b5V')
        self.assertEqual(experiment['version'], '1.0.0')
        self.assertEqual(len(experiment['blocks']), 324)

        block1 = metadata['ephys_experiments'][0]['blocks'][1]
        self.assertEqual(block1['num_frames'], 7500000)
        self.assertEqual(block1['path'], 'original/experiment1/Headstages_192_Channels_int16_2020-08-07_14-05-16.bin')
        self.assertEqual(block1['timestamp'], '2020-08-07T14:05:16')
        self.assertEqual(block1['ecube_time'], None)  # todo

        self.fail()


if __name__ == '__main__':
    unittest.main()
