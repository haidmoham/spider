"""Exercise notebook plumbing with synthetic transitions, never train the robot.

These tests intentionally do not judge the learner's estimator or objective.
"""
import contextlib
import copy
from dataclasses import replace
import io
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nbformat
import numpy as np
from spider.learning import LearningSimulation, PolicyRecording
from spider.simulation import measured_state, neutral_targets

LEARNING_ENV = all(importlib.util.find_spec(name) for name in ('torch', 'pandas'))
if LEARNING_ENV:
    import pandas as pd
    import torch
    from lab.reinforce_inspection import collect

ROOT = Path(__file__).resolve().parents[1]


class SyntheticSimulation(LearningSimulation):
    """Use the model schema but advance fabricated measurements, not physics."""

    def step(self, offsets, *, physics_steps):
        self.data.time += physics_steps * self.model.opt.timestep
        self.data.ctrl[:] = np.clip(
            np.asarray(neutral_targets()) + offsets,
            self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])
        state = measured_state(self.model, self.data)
        return replace(state, torso_position=(self.data.time * .03, 0., .45),
                       torso_velocity=(.03, 0., 0.))


@unittest.skipUnless(LEARNING_ENV, 'Optional notebook PyTorch/pandas environment')
class ReinforceWorkbenchTests(unittest.TestCase):
    def test_training_evaluation_and_artifacts_without_physics(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        notebook = nbformat.read(ROOT / 'lab/notebooks/04_mdp_contract.ipynb', as_version=4)
        cells = {cell.id: cell.source for cell in notebook.cells}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / 'lab/notebooks/04_mdp_contract.ipynb'
            target.parent.mkdir(parents=True)
            nbformat.write(notebook, target)
            scope = dict(REPO_ROOT=root, sim=SyntheticSimulation(), np=np, pd=pd,
                         geometry_gait=lambda state: np.zeros(18))
            with (contextlib.redirect_stdout(io.StringIO()),
                  patch.object(PolicyRecording, 'watch', return_value=None) as solo_viewer,
                  patch('spider.recording.launch_replay_grid', return_value=None) as grid_viewer):
                exec(compile(cells['rf4-initialize'], str(target), 'exec'), scope)
                exec(compile(cells['draft-reward-workbench'], str(target), 'exec'), scope)
                scope['SETTINGS'].update(horizon=3, batch_episodes=2, eval_seeds=[101])
                scope['collect'] = lambda *a, **kw: collect(*a, **kw, env_factory=SyntheticSimulation)
                exec(compile(cells['rf4-train'].replace('UPDATES_THIS_BLOCK = 2', 'UPDATES_THIS_BLOCK = 1'),
                             str(target), 'exec'), scope)
                before_evaluation = copy.deepcopy(scope['actor'].state_dict())
                exec(compile(cells['rf4-evaluate'], str(target), 'exec'), scope)
                solo_viewer.assert_not_called()
                grid_viewer.assert_called_once()
                self.assertEqual(len(grid_viewer.call_args.args[0]), 4)
            self.assertEqual(scope['updates'], 1)
            self.assertEqual(len(scope['last_batch']), 2)
            for key, value in scope['actor'].state_dict().items():
                torch.testing.assert_close(value, before_evaluation[key])
            for episode in scope['last_batch']:
                rows = episode['rows']
                self.assertEqual(len(rows), 3)
                self.assertTrue(rows[-1]['truncated'])
                self.assertFalse(rows[-1]['terminated'])
                for before, after in zip(rows, rows[1:]):
                    torch.testing.assert_close(before['next_observation'], after['observation'])
                self.assertTrue(np.isfinite(episode['log'].reward).all())
                self.assertFalse(rows[0]['log_prob'].requires_grad)
            saved = list(scope['run_directory'].rglob('transitions.npz'))
            self.assertEqual(len(saved), 8)
            with np.load(saved[0], allow_pickle=False) as data:
                self.assertEqual(data['observations'].shape, (3, 47))
                self.assertEqual(data['latent_actions'].shape, (3, 18))
                self.assertTrue(np.isfinite(data['log_probs']).all())
            self.assertEqual(len(list(scope['run_directory'].rglob('checkpoint-*.pt'))), 1)
            plt.close('all')


if __name__ == '__main__':
    unittest.main()
