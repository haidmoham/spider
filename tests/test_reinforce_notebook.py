"""Check notebook wiring with synthetic states; never run its robot experiment."""

import ast
import copy
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Optional notebook PyTorch environment')
class ReinforceNotebookTests(unittest.TestCase):
    def setUp(self):
        import torch
        from spider.learning import PolicyRecording

        self.torch = torch
        path = Path(__file__).resolve().parents[1] / 'lab/notebooks/03_coordinated_baseline.ipynb'
        notebook = json.loads(path.read_text(encoding='utf-8'))
        self.cells = {c['id']: ''.join(c['source']) for c in notebook['cells']}

        def measured(time):
            return SimpleNamespace(time=time, joint_positions=(0.1,) * 18,
                                   torso_position=(2 * time, 0.0, 0.45))

        class SyntheticSimulation:
            def __init__(self):
                self.model = SimpleNamespace(opt=SimpleNamespace(timestep=0.002))
                self.data = SimpleNamespace(time=0.0, ctrl=np.zeros(18))

            def reset(self):
                return measured(0.0)

            def step(self, offsets, *, physics_steps):
                self.data.time += physics_steps * 0.002
                self.data.ctrl = np.asarray(offsets).copy()
                return measured(self.data.time)

        class SyntheticReplay:
            def __init__(self, model, label):
                self.model, self.label = model, label
                self.states, self.times = [], []

            def capture(self, data):
                self.states.append(np.array([data.time]))
                self.times.append(data.time)

        self.namespace = dict(
            torch=torch, np=np, Normal=torch.distributions.Normal,
            LearningSimulation=SyntheticSimulation, TreatmentReplay=SyntheticReplay,
            PolicyRecording=PolicyRecording, RF_ACTION_COUNT=3, RF_PHYSICS_STEPS=10,
            RF_BOUND_RAD=0.05, RF_STD=0.20,
            observation_from=lambda state: torch.tensor(state.joint_positions),
            neutral_targets=lambda: np.zeros(18),
        )
        # Load definitions only. Top-level notebook setup, rollouts and updates do not run.
        definitions = []
        for identity in ('reinforce-setup', 'reinforce-collector'):
            definitions.extend(node for node in ast.parse(self.cells[identity]).body
                               if isinstance(node, ast.FunctionDef))
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), 'exec'), self.namespace)

    def test_returns_use_only_future_rewards_and_detach(self):
        rewards = self.torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        returns = self.namespace['rf_reward_to_go'](rewards)
        self.torch.testing.assert_close(returns, self.torch.tensor([6.0, 5.0, 3.0]))
        self.assertFalse(returns.requires_grad)

    def test_synthetic_transitions_align_rewards_and_retain_gradients(self):
        torch = self.torch
        network = torch.nn.Linear(18, 18)
        rng_before = torch.get_rng_state().clone()
        rows, recording = self.namespace['rf_rollout'](network, seed=5, label='synthetic', track_grad=True)
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
        self.assertEqual(len(rows), 3)
        self.assertEqual(len(recording.measurements), 4)
        for index, row in enumerate(rows):
            self.assertAlmostEqual(row['time'], index * 0.02)
            self.assertAlmostEqual(row['next_state'].time, (index + 1) * 0.02)
            self.assertAlmostEqual(row['reward'].item(), 2 * (index + 1) * 0.02 + 0.045, places=6)
            self.assertEqual(tuple(row['observation'].shape), (18,))
            self.assertEqual(tuple(row['sampled_action'].shape), (18,))
            self.assertFalse(row['sampled_action'].requires_grad)
            self.assertTrue(row['log_prob'].requires_grad)
            self.assertLessEqual(np.max(np.abs(row['offsets_rad'])), 0.05)
        rewards = torch.stack([row['reward'] for row in rows])
        log_probs = torch.stack([row['log_prob'] for row in rows])
        returns = self.namespace['rf_reward_to_go'](rewards)
        before = copy.deepcopy(network.state_dict())
        self.namespace.update(
            rf_policy=network, rf_optimizer=torch.optim.Adam(network.parameters(), lr=3e-4),
            rf_log_probs=log_probs, rf_returns=returns, rf_updates=0,
            rf_rollout_update=0, rf_rollout_consumed=False, copy=copy,
            pd=SimpleNamespace(DataFrame=lambda rows: rows), display=lambda value: None,
        )
        # Exercise the actual update cell on synthetic transitions, not robot data.
        with redirect_stdout(io.StringIO()):
            exec(self.cells['reinforce-update'], self.namespace)
        for parameter in network.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.norm().item(), 0)
        self.assertTrue(any(not torch.equal(before[k], v) for k, v in network.state_dict().items()))
        self.assertEqual(self.namespace['rf_updates'], 1)
        with self.assertRaisesRegex(RuntimeError, 'fresh rollout'):
            exec(self.cells['reinforce-update'], self.namespace)

    def test_control_and_evaluation_have_no_training_graph(self):
        rows, recording = self.namespace['rf_rollout'](None, seed=5, label='synthetic control')
        np.testing.assert_array_equal(recording.offsets, np.zeros((3, 18)))
        self.assertTrue(all(row['log_prob'] is None for row in rows))
        network = self.torch.nn.Linear(18, 18)
        rows, _ = self.namespace['rf_rollout'](network, seed=5, label='synthetic eval')
        self.assertTrue(all(not row['log_prob'].requires_grad for row in rows))


if __name__ == '__main__':
    unittest.main()
