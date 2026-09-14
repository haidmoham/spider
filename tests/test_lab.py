"""Protect migrated evidence and importable fixtures without running lessons."""
import hashlib
import importlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LabMigrationTests(unittest.TestCase):
    def test_preserved_history_matches_source_hashes(self):
        manifest = json.loads((ROOT / 'lab/history/migration.json').read_text())
        sources = [entry['source'] for entry in manifest['files']]
        self.assertEqual(len(sources), len(set(sources)))
        for entry in manifest['files']:
            destination = entry['destination']
            if destination is None:
                self.assertIn('retired', entry['disposition'])
                continue
            path = ROOT / destination
            with self.subTest(source=entry['source']):
                self.assertTrue(path.is_file(), destination)
                if destination.startswith('lab/history/'):
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), entry['sha256'])

    def test_fixtures_import_without_opening_a_viewer_or_stepping_physics(self):
        from unittest.mock import patch
        with patch('mujoco.mj_step', side_effect=AssertionError('import stepped physics')), \
             patch('mujoco.viewer.launch_passive', side_effect=AssertionError('import opened viewer')):
            for name in ('pendulum', 'two_link_coupling', 'model_based_control',
                         'jacobians_task_space', 'two_joint_planar_leg',
                         'tripod_static_support', 'static_support_boundary', 'c1n_leg_workspace'):
                with self.subTest(fixture=name):
                    importlib.import_module('lab.' + name)


if __name__ == '__main__':
    unittest.main()
