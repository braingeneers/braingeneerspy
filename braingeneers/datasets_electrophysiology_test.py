import unittest
import datasets_electrophysiology as ephys
import json


class DatasetsElectrophysiologyTestCase(unittest.TestCase):
    """
    Online test cases require access to braingeneers/S3 including ~/.aws/credentials file
    """
    filename = "s3://braingeneers/ephys/2020-07-06-e-MGK-76-2614-Wash/original/data/" \
               "H28126_WK27_010320_Cohort_202000706_Wash(214).raw"

    batch_uuid = '2020-07-06-e-MGK-76-2614-Wash'

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
        """

        :return:
        """
        metadata_json, experiment1_json = ephys.axion_generate_metadata(self.batch_uuid)

        self.assertCountEqual(metadata_json['experiments'], ['experiment1.json'])
        self.assertEqual(metadata_json['issue'], '')
        self.assertEqual(metadata_json['notes'], '')
        self.assertTrue('timestamp' in metadata_json)
        self.assertEqual(metadata_json['uuid'], self.batch_uuid)
        self.assertEqual(len(metadata_json), 5)

        self.assertEqual(experiment1_json['hardware'], 'Axion BioSystems')
        self.assertEqual(experiment1_json['name'], 'MGK-76-2614-Wash')
        self.assertEqual(experiment1_json['notes'], '')
        self.assertEqual(experiment1_json['num_channels'], 384)  # 6 well, 64 channel per well
        self.assertEqual(experiment1_json['num_current_input_channels'], 0)
        self.assertEqual(experiment1_json['num_voltage_channels'], 384)
        self.assertEqual(experiment1_json['offset'], 0)
        self.assertEqual(experiment1_json['sample_rate'], 12500)
        self.assertTrue(isinstance(experiment1_json['sample_rate'], int))
        self.assertAlmostEqual(experiment1_json['voltage_scaling_factor'], -5.484861781483107e-08)
        self.assertTrue(isinstance(experiment1_json['voltage_scaling_factor'], float))
        self.assertTrue('T' in experiment1_json['timestamp'])
        self.assertEqual(experiment1_json['units'], '\u00b5V')
        self.assertEqual(experiment1_json['version'], '1.0.0')

        self.assertEqual(len(experiment1_json['blocks']), 267)
        self.assertEqual(experiment1_json['blocks'][0]['num_frames'], 3750000)
        self.assertEqual(experiment1_json['blocks'][0]['path'], 'H28126_WK27_010320_Cohort_202000706_Wash(000).raw')
        self.assertTrue('T' in experiment1_json['blocks'][0]['timestamp'])

        # validate json serializability
        json.dumps(metadata_json)
        json.dumps(experiment1_json)

        # save metadata files - used in development, kept here for quick reference
        # with open('../tmp/metadata.json', 'w') as f_metadata:
        #     json.dump(metadata_json, f_metadata, indent=2)
        # with open('../tmp/experiment1.json', 'w') as f_experiment1:
        #     json.dump(experiment1_json, f_experiment1, indent=2)

    def test_online_load_data_axion(self):
        file_214_offset = 802446875
        data = ephys.load_data(
            batch_uuid=self.batch_uuid, experiment_num=0, offset=file_214_offset, length=4, channels=[64]
        )

        # Test a few manually selected values are returned correctly
        self.assertEqual(data[0][0], -9)
        self.assertEqual(data[0][1], -18)
        self.assertEqual(data[0][2], 10)
        self.assertEqual(data[0][3], 30)


if __name__ == '__main__':
    unittest.main()
