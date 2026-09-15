"""Evaluate saved residual checkpoints with Notebook 4's seeds and open replays.

No training takes place here. The notebook and both accepted controls stay frozen.
Each window shows PPO-100 and approved tripod controls above sampled and mean
seed-201 recordings from one candidate checkpoint. Failed evaluations remain visible.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import torch

from . import simulation
from .gait_preview import _accepted_baseline
from .reference_evaluation import evaluate


def compare(training: Path, output: Path, checkpoints=(50, 100), *, view=True,
            control: Path | None = None) -> dict:
    training, output = training.resolve(), output.resolve()
    paths = [training / f"checkpoint-{step:05d}.pt" for step in checkpoints]
    payloads = [torch.load(path, map_location="cpu", weights_only=True) for path in paths]
    for step, payload in zip(checkpoints, payloads):
        if payload["updates"] != step:
            raise ValueError("checkpoint update count does not match filename")
        for key in ("settings", "ppo", "reward", "model_xml"):
            if payload[key] != payloads[0][key]:
                raise ValueError(f"checkpoint comparison changes {key}")
    baseline = _accepted_baseline()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__), output / "reference_comparison.py")
    if control is None:
        approved = output / "approved_tripod"
        with zipfile.ZipFile(simulation.ROOT / "artifacts/tripod-openloop-baseline-20260915/replay.zip") as archive:
            archive.extractall(approved)
    else:
        approved = control.resolve()
        for name in ("model.mjb", "states.npz", "metadata.json"):
            if not (approved / name).is_file():
                raise FileNotFoundError(approved / name)
    report = dict(training=str(training), checkpoints=list(checkpoints),
                  protocol="Notebook 4: seeds 201..212, sampled and mean, five seconds",
                  comparison_scope="Candidate versus accepted PPO-100 and selected measured control",
                  selected_control=str(approved),
                  accepted=False, evaluations=[], windows=[])
    rows = []
    for step, checkpoint in zip(checkpoints, paths):
        destination = output / f"n{step:05d}"
        result = evaluate(checkpoint, destination, notebook_seeds=True)
        report["evaluations"].append(dict(update=step, checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                                           **{k: v for k, v in result.items() if k != "records"}))
        rows.extend(dict(update=step, mode=r["mode"], seed=r["seed"], seconds=r["seconds"],
                         speed_m_s=r["speed_m_s"], terminated=r["terminated"],
                         max_joint_limit_violation_rad=r["max_joint_limit_violation_rad"])
                    for r in result["records"])
        recordings = [baseline, approved, destination / "sampled-201", destination / "mean-201"]
        command = [sys.executable, "-m", "spider", "replay-grid", *map(str, recordings),
                   "--speed", "1", "--presentation", "stalk",
                   "--ready", str(destination / "viewer.ready"),
                   "--screenshot", str(destination / "preview.png")]
        window = dict(update=step, recordings=list(map(str, recordings)), command=command)
        if view:
            with (destination / "viewer.stdout.log").open("w") as stdout, (destination / "viewer.stderr.log").open("w") as stderr:
                process = subprocess.Popen(command, cwd=simulation.ROOT, stdout=stdout, stderr=stderr,
                                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            window["pid"] = process.pid
        report["windows"].append(window)
        (output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        with (output / "comparison.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        print(f"n={step}: {len(result['records'])} recorded evaluations; numerical pass={result['goal_numerical_pass']}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[50, 100])
    parser.add_argument("--no-view", action="store_true")
    parser.add_argument("--control", type=Path, help="Measured top-right control instead of the approved tripod")
    args = parser.parse_args()
    compare(args.training, args.output, args.checkpoints, view=not args.no_view, control=args.control)


if __name__ == "__main__":
    main()
