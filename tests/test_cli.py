"""The public command must dispatch headless runs and return usable measurements."""

import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CommandLineTests(unittest.TestCase):
    def test_headless_experiments_return_the_requested_state(self):
        for experiment in ("none", "stand", "shuffle"):
            with self.subTest(experiment=experiment):
                process = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "spider",
                        "run",
                        "--headless",
                        "--experiment",
                        experiment,
                        "--seconds",
                        "0.02",
                    ],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    timeout=20,
                    check=True,
                )
                state = json.loads(process.stdout)
                self.assertEqual(state["experiment"], experiment)
                self.assertAlmostEqual(state["time"], 0.02)
                self.assertEqual(len(state["controls"]), 18)

    def test_help_does_not_start_a_viewer(self):
        process = subprocess.run(
            [sys.executable, "-m", "spider", "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        for command in ("run", "command", "replay", "render"):
            self.assertIn(command, process.stdout)


if __name__ == "__main__":
    unittest.main()
