"""Raw-data location config, path localization and symlink rewriting."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from core import sources
from relink_sources import apply, plan_internal, plan_source


class SourceConfigTests(unittest.TestCase):
    def test_configured_roots_exist_on_this_machine(self):
        for name in sources.load():
            with self.subTest(source=name):
                self.assertTrue(sources.root(name).is_dir(), sources.root(name))

    def test_localize_is_identity_while_data_has_not_moved(self):
        recorded = sources.load()['mdad']['recorded'] + '/Day-RGB1/RGB1/S1/AC1/x.jpg'
        self.assertEqual(Path(sources.localize(recorded)).resolve(), Path(recorded).resolve())

    def test_environment_variable_relocates_recorded_paths(self):
        entry = sources.load()['mdad']
        with mock.patch.dict(os.environ, {entry['env']: '/Volumes/disk/3MDAD'}):
            self.assertEqual(sources.localize(entry['recorded'] + '/Day-RGB2/a.jpg'), '/Volumes/disk/3MDAD/Day-RGB2/a.jpg')
            # a sibling directory sharing the prefix text is not rewritten
            self.assertEqual(sources.localize(entry['recorded'] + '_old/a.jpg'), entry['recorded'] + '_old/a.jpg')


class RelinkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.top = Path(self.tmp.name).resolve()
        for d in ('old/imgs', 'new/imgs', 'project/data', 'project/archive'):
            (self.top / d).mkdir(parents=True)
        for d in ('old', 'new'):
            (self.top / d / 'imgs/a.jpg').write_bytes(b'x')
        (self.top / 'project/archive/b.json').write_text('{}')
        os.symlink(self.top / 'old/imgs/a.jpg', self.top / 'project/data/a.jpg')
        os.symlink(self.top / 'project/archive/b.json', self.top / 'project/data/b.json')

    def tearDown(self):
        self.tmp.cleanup()

    def test_source_relink_rewrites_only_matching_prefix(self):
        changes = plan_source(str(self.top / 'old'), self.top / 'new', top=self.top / 'project')
        self.assertEqual([c[0].name for c in changes], ['a.jpg'])
        apply(changes)
        self.assertEqual(os.readlink(self.top / 'project/data/a.jpg'), str(self.top / 'new/imgs/a.jpg'))

    def test_internal_links_become_relative_and_still_resolve(self):
        apply(plan_internal(top=self.top / 'project'))
        link = self.top / 'project/data/b.json'
        self.assertEqual(os.readlink(link), '../archive/b.json')
        self.assertTrue(link.exists())
        self.assertEqual(os.readlink(self.top / 'project/data/a.jpg'), str(self.top / 'old/imgs/a.jpg'))


if __name__ == '__main__':
    unittest.main()
