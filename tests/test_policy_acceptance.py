from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import mujoco

from spider.policy_acceptance import assess_round
from spider.simulation import MODEL_PATH, load_model, reset


CHECKPOINT = Path(__file__).resolve().parent.parent / "artifacts/ppo-crude-baseline-20260915/checkpoint-00100.pt"
MODEL_HASH = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()


class AcceptanceTests(unittest.TestCase):
    def fixture(self, root: Path, *, low_speed=False, fall=False, missing_seed=None,
                duplicate=False, nan=False, missing_frames=False, invalid_model=False,
                hidden_fall=False, metadata_mismatch=False):
        entries = []
        model = load_model()
        data = mujoco.MjData(model)
        reset(model, data)
        state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        base = np.empty(mujoco.mj_stateSize(model, state_spec))
        mujoco.mj_getState(model, data, base, state_spec)
        cases = [("mean", 201)] + [("sampled", seed) for seed in range(201, 213)]
        if missing_seed is not None:
            cases.remove(("sampled", missing_seed))
        if duplicate:
            cases.append(("sampled", 201))
        for mode, seed in cases:
            recording = root / "control" / "evaluation" / f"{mode}-{seed}-{len(entries)}"
            recording.mkdir(parents=True)
            forward = 1.0 if low_speed else (2.0 if mode == "mean" else 1.5)
            rows = []
            times = np.arange(1, 251) * 0.02
            heights = np.full(250, 0.45)
            if hidden_fall:
                heights[-1] = 0.20
            for index, time in enumerate(times):
                row = {"time_s": time,
                       "forward_m": float("nan") if nan and index == 249 else forward * time / 5,
                       "height_m": heights[index], "terminated": fall and index == 249}
                row.update({f"target_{joint}_rad": 0.0 for joint in range(18)})
                rows.append(row)
            with (recording / "trace.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            states = np.repeat(base[None, :], 251, axis=0)
            state_times = np.r_[0.0, times]
            states[:, 0] = state_times
            states[1:, 1] = np.asarray([row["forward_m"] for row in rows], dtype=float)
            states[1:, 3] = heights
            if missing_frames:
                states, state_times = states[:-1], state_times[:-1]
            np.savez(recording / "states.npz", states=states, times=state_times)
            if invalid_model:
                (recording / "model.mjb").write_bytes(b"fixture")
            else:
                mujoco.mj_saveModel(model, str(recording / "model.mjb"))
            metadata = {"status": "complete", "model_sha256": MODEL_HASH,
                        "checkpoint": str(CHECKPOINT),
                        "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
                        "mode": "wrong" if metadata_mismatch else mode, "seed": seed,
                        "seconds": 5.0, "control_interval_s": 0.02,
                        "terminated": fall, "truncated": not fall}
            (recording / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            entries.append({"policy": "control", "recording": str(recording), "mode": mode,
                            "seed": seed, "checkpoint": str(CHECKPOINT),
                            "checkpoint_sha256": metadata["checkpoint_sha256"]})
        (root / "evaluations.json").write_text(json.dumps(entries), encoding="utf-8")

    def assess(self, **kwargs):
        temporary = tempfile.TemporaryDirectory(); root = Path(temporary.name)
        self.fixture(root, **kwargs); report = assess_round(root)
        self.addCleanup(temporary.cleanup)
        return report

    def test_valid_control_passes_numerically_but_requires_visual_acceptance(self):
        report = self.assess()
        self.assertTrue(report["numerical_pass"])
        self.assertFalse(report["accepted"])
        self.assertTrue(report["visual_review_required"])

    def test_rejects_fall_missing_duplicate_nan_and_low_speed(self):
        cases = ({"fall": True}, {"missing_seed": 212}, {"duplicate": True},
                 {"nan": True}, {"low_speed": True})
        for case in cases:
            with self.subTest(case=case):
                self.assertFalse(self.assess(**case)["numerical_pass"])

    def test_rejects_missing_frames_hidden_fall_invalid_model_and_metadata_mismatch(self):
        for case in ({"missing_frames": True}, {"hidden_fall": True},
                     {"invalid_model": True}, {"metadata_mismatch": True}):
            with self.subTest(case=case):
                self.assertFalse(self.assess(**case)["numerical_pass"])

    def test_rejects_policy_group_missing_from_evaluations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            (root / "checkpoints.json").write_text(json.dumps({
                "control": str(CHECKPOINT), "missing_candidate": str(CHECKPOINT)
            }), encoding="utf-8")
            report = assess_round(root)
        self.assertFalse(report["numerical_pass"])
        self.assertIn("missing_candidate", report["policies"])
        self.assertFalse(report["policies"]["missing_candidate"]["numerical_pass"])


if __name__ == "__main__": unittest.main()
