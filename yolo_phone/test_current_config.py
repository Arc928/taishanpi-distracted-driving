"""Checks that config/current.yaml points at real, consistent artifacts."""
import tempfile
import unittest
from pathlib import Path

import yaml

import current_config
from annotation_io import NAMES


class CurrentConfigTests(unittest.TestCase):
    def test_weights_exist_and_hash_matches(self):
        path = current_config.weights_path(verify=True)
        self.assertTrue(path.is_file())

    def test_dataset_classes_match_fixed_ids(self):
        data = yaml.safe_load(current_config.dataset_path().read_text(encoding='utf-8'))
        self.assertEqual(data['names'], NAMES)

    def test_values_in_allowed_range(self):
        self.assertEqual(current_config.imgsz(), 640)
        self.assertTrue(0.1 <= current_config.confidence() <= 0.95)

    def test_rejects_non_640_imgsz(self):
        bad = dict(current_config.load(), imgsz=960)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'current.yaml'
            path.write_text(yaml.safe_dump(bad, allow_unicode=True), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '640'):
                current_config.load(path)


if __name__ == '__main__':
    unittest.main()
