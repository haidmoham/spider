"""A completed evaluation must not return success for a failed policy gate."""

from contextlib import redirect_stdout
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from spider.__main__ import main


class PolicyAcceptanceCliTests(unittest.TestCase):
    def test_failed_recorded_gate_exits_nonzero_without_training(self):
        assess = Mock(return_value={"numerical_pass": False, "policies": {}})
        with patch.dict("sys.modules", {"spider.policy_acceptance": SimpleNamespace(assess_round=assess)}), \
                patch("sys.argv", ["spider", "check-policy", "--round", "recorded"]), \
                redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as failure:
            main()
        self.assertEqual(failure.exception.code, 1)
        assess.assert_called_once_with(Path("recorded"))

    def test_passing_numerical_gate_returns_normally(self):
        assess = Mock(return_value={"numerical_pass": True, "accepted": False})
        with patch.dict("sys.modules", {"spider.policy_acceptance": SimpleNamespace(assess_round=assess)}), \
                patch("sys.argv", ["spider", "check-policy", "--round", "recorded"]), \
                redirect_stdout(io.StringIO()):
            main()
