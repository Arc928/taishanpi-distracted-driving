"""Plan validation for train.py: hyperparameters only from the plan, no silent overrides."""
import tempfile
import unittest
from pathlib import Path

import yaml

from core.io import digest
from train import load_plan

TEMPLATE = Path(__file__).resolve().parents[1] / 'experiments' / 'plan_template.yaml'


class TrainPlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.model = root / 'start.pt'
        self.model.write_bytes(b'not a real checkpoint')
        self.data = root / 'data.yaml'
        self.data.write_text('train: images/train\nval: images/val\nnames:\n  0: texting\n  1: calling\n')
        self.plan = {'name': 'unit_test_run', 'data': str(self.data), 'model': str(self.model),
                     'model_sha256': digest(self.model), 'train': {'epochs': 1, 'batch': 2}}

    def tearDown(self):
        self.tmp.cleanup()

    def load(self, **changes):
        path = Path(self.tmp.name) / 'plan.yaml'
        path.write_text(yaml.safe_dump({**self.plan, **changes}, allow_unicode=True))
        return load_plan(path)

    def test_valid_plan_passes_train_block_verbatim_and_fixes_imgsz(self):
        _, resolved = self.load()
        kwargs = resolved['train_kwargs']
        self.assertEqual((kwargs['epochs'], kwargs['batch'], kwargs['imgsz']), (1, 2, 640))
        self.assertEqual(kwargs['name'], 'unit_test_run')

    def test_template_placeholder_name_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'name'):
            load_plan(TEMPLATE)

    def test_reserved_keys_cannot_be_overridden(self):
        for key in ('imgsz', 'data', 'resume'):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                self.load(train={'epochs': 1, key: 960})

    def test_start_weights_hash_must_match(self):
        with self.assertRaisesRegex(ValueError, '哈希'):
            self.load(model_sha256='0'*64)

    def test_dataset_class_order_must_match_fixed_ids(self):
        self.data.write_text('names:\n  0: calling\n  1: texting\n')
        with self.assertRaisesRegex(ValueError, '类别'):
            self.load()


if __name__ == '__main__':
    unittest.main()
