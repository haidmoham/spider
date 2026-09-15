"""A bounded training round must preserve a failed empirical gate."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from spider.stride_round import run_stride_round


class StrideRoundTests(unittest.TestCase):
    def test_fixed_budget_and_failed_gate_are_saved(self):
        trainer = Mock()
        trainer.return_value.train.return_value = Path("checkpoint.pt")
        assess = Mock(return_value={"numerical_pass": False})
        sample = dict(speed_m_s=0., forward_m=0., lateral_m=0., terminated=True,
                      minimum_height_m=.2, steady_height_mean_m=.3,
                      steady_height_std_m=0., action_delta_rms_rad=0.,
                      stance_foot_speed_rms_m_s=0., vertical_velocity_rms_m_s=0.)
        def evaluate(*args, sampled=False, **kwargs):
            return dict(sample, mode="sampled" if sampled else "mean")
        with TemporaryDirectory() as directory, patch.dict("sys.modules", {
                "torch": SimpleNamespace(set_num_threads=Mock()),
                "spider.stride_training": SimpleNamespace(FreshTrainingSession=trainer),
                "spider.policy_acceptance": SimpleNamespace(assess_round=assess)}), \
                patch("spider.stride_round.run_policy", side_effect=evaluate) as run, \
                redirect_stdout(io.StringIO()):
            output = run_stride_round(Path(directory) / "round")
            receipt = json.loads((output / "receipt.json").read_text())
        self.assertEqual(trainer.call_count, 3)
        self.assertEqual([call.args[1] for call in trainer.call_args_list], [11, 22, 33])
        self.assertEqual(trainer.return_value.train.call_count, 3)
        for call in trainer.return_value.train.call_args_list:
            self.assertEqual(call.kwargs, {"updates": 50})
        self.assertEqual(run.call_count, 52)
        assess.assert_called_once_with(output)
        self.assertEqual(receipt["status"], "failed-acceptance")
        self.assertFalse(receipt["numerical_pass"])
