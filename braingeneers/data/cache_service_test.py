from datetime import datetime, timedelta
import unittest
import tempfile
import braingeneers.data.cache_service as cache_service
from braingeneers.data.cache_service import CacheConfig
import os


class CacheConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.td = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.td.cleanup()

    def test_save_json_and_to_json(self):
        cache_config = CacheConfig()
        save_filepath = os.path.join(self.td.name, 'cache_config.json')
        cache_config.save_json(save_filepath)

        with open(save_filepath, 'r') as f:
            expected = f.read()
            actual = cache_config.to_json()
            self.assertEqual(expected, actual)

    def test_load_json_and_from_json(self):
        cache_config_before_load = CacheConfig()
        save_filepath = os.path.join(self.td.name, 'cache_config.json')
        cache_config_before_load.save_json(save_filepath)

        cache_config_after_load = CacheConfig.load_json(save_filepath)

        for key, expected in vars(cache_config_before_load).items():
            actual = getattr(cache_config_after_load, key)
            self.assertEqual(expected, actual)

    def test_set_uuid(self):
        uuid = '0000-00-00-e-test'
        cache_config = CacheConfig().set_uuid(uuid)
        self.assertEqual(uuid, cache_config.uuid)

    def test_set_data_range_default(self):
        cache_config = CacheConfig().set_data_range()
        self.assertEqual(cache_config.data_ranges, [(0, -1)])

    def test_set_data_range_specified_value(self):
        cache_config = CacheConfig().set_data_range(offset=100, size=500)
        self.assertEqual(cache_config.data_ranges, [(100, 500)])

        cache_config.set_data_range(offset=1000, size=5000)
        self.assertEqual(cache_config.data_ranges, [(100, 500), (1000, 5000)])

    def test_set_data_range_invalid_value(self):
        with self.assertRaises(ValueError):
            CacheConfig().set_data_range(offset=-2)

        with self.assertRaises(ValueError):
            CacheConfig().set_data_range(size=0)

        with self.assertRaises(ValueError):
            CacheConfig().set_data_range(size=0.0)

    def test_set_channels_individual(self):
        cache_config = CacheConfig().set_channels_individual(42)
        self.assertEqual(cache_config.channels, [(42,)])
        cache_config.set_channels_individual([0, 1, 2])
        self.assertEqual(cache_config.channels, [(0,), (1,), (2,), (42,)])

    def test_set_channels_individual_invalid_input(self):
        with self.assertRaises(ValueError):
            CacheConfig().set_channels_individual(42.0)
        with self.assertRaises(ValueError):
            CacheConfig().set_channels_individual(object())

    def test_set_channels_grouped_range(self):
        cache_config = CacheConfig().set_channels_grouped(range(64, 128))
        expected = [tuple(range(64, 128))]
        self.assertEqual(cache_config.channels, expected)

    def test_set_channels_grouped_set(self):
        cache_config = CacheConfig().set_channels_grouped([0, 1, 2])
        self.assertEqual(cache_config.channels, [(0, 1, 2)])

    def test_set_cache_until(self):
        test_strings = [
            ('1d', 1),
            ('1w', 7),
            ('1m', 31),
            ('1m3d', 34),
            ('3d 1w', 10),
        ]
        for test_string, days in test_strings:
            cache_config = CacheConfig().set_cache_until(test_string)
            dt_epoch = datetime.utcfromtimestamp(0)
            dt_a = (datetime.fromisoformat(cache_config.cache_until_iso) - dt_epoch).total_seconds()
            dt_b = (datetime.now() + timedelta(days=days) - dt_epoch).total_seconds()
            self.assertTrue(abs(dt_a - dt_b) < 1)  # less than 1s difference between now and actual timestamp

    def test_set_local_cache(self):
        cache_config = CacheConfig().set_local_cache(self.td.name)
        self.assertEqual(self.td.name, cache_config.local_cache)

    def test_prepare_cache(self):
        cache_config = CacheConfig()
        cache_service._prepare_cache(cache_config)
        self.fail()
