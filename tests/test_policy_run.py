"""Verify platform wiring using a fake stepper, without any robot rollout."""

from contextlib import redirect_stdout
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from spider import policy_run
from spider.__main__ import main


class PolicyRunTests(unittest.TestCase):
    def test_saved_control_cadence_and_episode_cutoff(self):
        model = SimpleNamespace(opt=SimpleNamespace(timestep=.002))
        data = SimpleNamespace(time=0.)
        commands = []
        decisions = []

        class FakePolicy:
            settings = dict(physics_steps=10, horizon=250, fall_height_m=.25)
            updates = 100
            checkpoint_sha256 = "test"
            description = "test policy"
            config = {}

            def __init__(self, *args, **kwargs):
                pass

            def targets(self, observed):
                decisions.append(observed.time)
                return (float(len(decisions)),) * 18

        class FakeReplay:
            def __init__(self, model, label):
                self.states, self.times = [], []

            def capture(self, data):
                self.states.append(np.array([data.time]))
                self.times.append(data.time)

        def observed(*args):
            return SimpleNamespace(time=data.time, torso_position=(data.time, 0., .45),
                                   foot_contacts=(1, 2, 3), support_margin=.02)

        def step(model, data, targets):
            commands.append(targets)
            data.time += .002

        with TemporaryDirectory() as temporary, \
                patch.dict("sys.modules", {"spider.policy": SimpleNamespace(PPOPolicy=FakePolicy)}), \
                patch.object(policy_run.simulation, "load_model", return_value=model), \
                patch.object(policy_run.simulation, "reset"), \
                patch.object(policy_run.simulation, "measured_state", side_effect=observed), \
                patch.object(policy_run.simulation, "step", side_effect=step), \
                patch.object(policy_run.mujoco, "MjData", return_value=data), \
                patch.object(policy_run.mujoco, "mj_saveModel"), \
                patch.object(policy_run, "version", return_value="test"), \
                patch.object(policy_run, "TreatmentReplay", FakeReplay):
            result = policy_run.run_policy(seconds=.04, directory=Path(temporary) / "recording")
        self.assertEqual(len(decisions), 2)
        self.assertEqual(commands[:10], [(1.,) * 18] * 10)
        self.assertEqual(commands[10:], [(2.,) * 18] * 10)
        self.assertAlmostEqual(result["seconds"], .04)
        self.assertFalse(result["terminated"])
        self.assertTrue(result["truncated"])

    def test_compare_dispatch_is_explicit_and_headless_does_not_watch(self):
        with patch("sys.argv", ["spider", "policy", "--compare", "--headless"]), \
                patch.object(policy_run, "compare_policies", return_value=Path("out")) as compare, \
                patch.object(policy_run, "run_policy") as single, \
                redirect_stdout(io.StringIO()):
            main()
        compare.assert_called_once()
        self.assertFalse(compare.call_args.kwargs["watch"])
        self.assertFalse(compare.call_args.kwargs["sampled"])
        single.assert_not_called()

    def test_single_dispatch_preserves_named_treatment(self):
        with patch("sys.argv", ["spider", "policy", "--headless", "--treatment", "lower"]), \
                patch.object(policy_run, "run_policy", return_value={}) as single, \
                patch.object(policy_run, "compare_policies") as compare, \
                redirect_stdout(io.StringIO()):
            main()
        self.assertEqual(single.call_args.kwargs["treatment"], "lower")
        compare.assert_not_called()
