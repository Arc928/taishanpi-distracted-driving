"""config/datasets.yaml must match what is on disk."""
import unittest
from pathlib import Path

from core import config as current_config
from core import registry
from core.io import digest
from core.paths import BASE, artifact


class DatasetRegistryTests(unittest.TestCase):
    def test_every_registered_path_exists_and_manifest_unchanged(self):
        for dataset_id, entry in registry.load().items():
            with self.subTest(dataset=dataset_id):
                root = BASE / entry['path']
                self.assertTrue(root.is_dir(), root)
                if 'manifest_sha256' in entry:
                    self.assertEqual(digest(root / 'manifest.csv'), entry['manifest_sha256'])
                else:
                    self.assertFalse((root / 'manifest.csv').exists(), '有 manifest 却未登记哈希')

    def test_every_dataset_directory_on_disk_is_registered(self):
        registered = {(BASE / e['path']).resolve() for e in registry.load().values()}
        candidates = {p.parent.resolve() for p in BASE.glob('*/manifest.csv')}
        candidates |= {p.parent.resolve() for p in (BASE / 'data').rglob('manifest.csv')}
        candidates |= {p.resolve() for p in (BASE / 'archive' / 'data').iterdir() if p.is_dir()}
        # a manifest nested inside a registered dataset (e.g. its eval_statefarm copy) belongs to it
        unregistered = [c for c in candidates if not ({c, *c.parents} & registered)]
        self.assertEqual(sorted(map(str, unregistered)), [])

    def test_data_yaml_path_points_at_its_own_directory(self):
        # Ultralytics resolves a relative path against its own datasets dir, so path stays absolute
        import yaml
        for yaml_file in BASE.glob('data/*/data.yaml'):
            with self.subTest(data=yaml_file.parent.name):
                path = yaml.safe_load(yaml_file.read_text(encoding='utf-8'))['path']
                self.assertEqual(Path(path).resolve(), yaml_file.parent.resolve())

    def test_lineage_has_no_cycles(self):
        data = registry.load()

        def ancestors(key, seen=()):
            self.assertNotIn(key, seen, f'循环来源: {seen}')
            for parent in data[key].get('parents', []):
                ancestors(parent, seen + (key,))
        for key in data:
            ancestors(key)

    def test_current_dataset_is_an_active_training_set(self):
        dataset_id = current_config.load()['dataset']
        self.assertEqual(registry.entry(dataset_id)['status'], 'active')
        self.assertTrue(current_config.dataset_path().is_file())

    def test_unknown_names_raise_instead_of_falling_back(self):
        with self.assertRaises(KeyError):
            registry.path('no_such_dataset')
        with self.assertRaises(KeyError):
            artifact('no_such_directory')
        self.assertEqual(artifact('selftrain_23train_round2'), BASE / 'archive/data/selftrain_23train_round2')


if __name__ == '__main__':
    unittest.main()
