import unittest
import datasets_electrophysiology as ephys
import json
import braingeneers.utils.smart_open_braingeneers as smart_open


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

    def test_online_axion_get_data(self):
        """
        This test case assumes a local axion raw file.
        It should be skipped for automated tests.
        :return:
        """
        data_start, data_length, num_channels, corrected_map, sampling_frequency, voltage_scale = \
            ephys._axion_generate_per_block_metadata(self.filename)

        voltage_data = ephys._axion_get_data(
            self.filename, data_start, 0, 20, num_channels, corrected_map
        )

        for i in range(10):
            for j in range(0, 31):
                print(voltage_data[j][i], end="\t")
            print()
        print()

        # A2 = (1,2) 64-127
        for i in range(10):
            for j in range(64, 95):
                print(voltage_data[j][i], end="\t")
            print()
        print()

        # A3 = (1,3) 128-191
        for i in range(10):
            for j in range(128, 159):
                print(voltage_data[j][i], end="\t")
            print()
        print()

        # B1 = (2,1) 192-255
        for i in range(10):
            for j in range(192, 223):
                print(voltage_data[j][i], end="\t")
            print()
        print()

        # B2 = (2,2) 256-319
        for i in range(10):
            for j in range(256, 287):
                print(voltage_data[j][i], end="\t")
            print()
        print()

        # B3 = (2,3) 320-383
        for i in range(10):
            for j in range(320, 351):
                print(voltage_data[j][i], end="\t")
            print()
        print()

        self.assertTrue(True)  # simple success if not exception before this occurred

    def test_online_axion_generate_metadata(self):
        metadata = ephys.generate_metadata_axion(self.batch_uuid)
        experiment0 = metadata['ephys-experiments'][0]

        self.assertEqual(len(metadata['ephys-experiments']), 6)
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

        # validate json serializability
        json.dumps(metadata)

        # save metadata files - used in development, kept here for quick reference
        # with smart_open.open(f's3://braingeneers/ephys/{self.batch_uuid}/metadata.json', 'w') as f:
        #     json.dump(metadata, f, indent=2)

    def test_online_load_data_axion(self):
        file_214_offset = 802446875
        metadata = ephys.load_metadata(self.batch_uuid)
        data = ephys.load_data(
            metadata=metadata, experiment=1, offset=file_214_offset, length=4, channels=[0]
        )

        # Test a few manually selected values are returned correctly
        self.assertEqual(data[0][0], -9)
        self.assertEqual(data[0][1], -18)
        self.assertEqual(data[0][2], 10)
        self.assertEqual(data[0][3], 30)

    def test_online_axion_generate_metadata_24well(self):
        uuid_24well_data = '2021-09-23-e-MR-89-0526-spontaneous'
        metadata_json = ephys.generate_metadata_axion(uuid_24well_data)
        self.assertTrue(len(metadata_json) > 0)  # Trivial validation
        self.assertEquals(len(metadata_json['ephys-experiments']), 24)

        # save metadata files - used in development, kept here for quick reference
        with smart_open.open(f's3://braingeneers/ephys/{uuid_24well_data}/metadata.json', 'w') as f:
            json.dump(metadata_json, f, indent=2)

    def test_online_axion_load_data_24well(self):
        uuid_24well_data = '2021-09-23-e-MR-89-0526-spontaneous'
        metadata_json = ephys.load_metadata(uuid_24well_data)
        data = ephys.load_data(metadata=metadata_json, experiment='B1', offset=0, length=10, channels=0)
        self.assertEqual(data.shape, (1, 10))  # trivial validation, needs to be improved

    def test_online_axion_load_data_24well_int_index(self):
        uuid_24well_data = '2021-09-23-e-MR-89-0526-spontaneous'
        metadata_json = ephys.load_metadata(uuid_24well_data)
        data = ephys.load_data(metadata=metadata_json, experiment=1, offset=0, length=10, channels=0)
        self.assertEqual(data.shape, (1, 10))  # trivial validation, needs to be improved

    def test_online_load_metadata(self):
        metadata = ephys.load_metadata(self.batch_uuid)
        self.assertTrue('uuid' in metadata)  # sanity check only
        self.assertTrue(len(metadata['ephys-experiments']) == 6)  # sanity check only
        self.assertTrue('voltage_scaling_factor' in metadata['ephys-experiments'])  # sanity check only


if __name__ == '__main__':
    unittest.main()
