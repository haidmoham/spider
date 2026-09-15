"""Review the candidate chassis using recorded dynamics and labelled target poses."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import sys
import zipfile

import mujoco

from . import simulation
from .gait_preview import _accepted_baseline, _record_reference
from .gait_reference import GaitReference
from .viewing.film import render_comparison


def run(output: Path, measured: Path, *, view: bool = True,
        refresh_appearance: bool = False) -> Path:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    measured = measured.resolve()
    metadata = json.loads((measured / "metadata.json").read_text())
    if not metadata["summary"]["completed_full_duration"]:
        raise ValueError("candidate must have a complete measured five-second recording")
    if refresh_appearance:
        from .chassis_candidate import candidate_xml, load_candidate
        from .policy_acceptance import _same_physics
        displayed_model = load_candidate()
        measured_model = mujoco.MjModel.from_binary_path(str(measured / "model.mjb"))
        if not _same_physics(displayed_model, measured_model):
            raise ValueError("appearance refresh changes physics; a new measured run is required")
        displayed = output / "candidate_measured_display"
        displayed.mkdir()
        for filename in ("states.npz", "trace.csv"):
            shutil.copy2(measured / filename, displayed / filename)
        mujoco.mj_saveModel(displayed_model, str(displayed / "model.mjb"))
        xml = candidate_xml()
        (displayed / "display_model.xml").write_text(xml, encoding="utf-8")
        metadata = dict(metadata, recorded_source=str(measured),
                        display_xml_sha256=hashlib.sha256(xml.encode()).hexdigest(),
                        display_note="Appearance refresh; physical arrays verified identical; states unchanged")
        (displayed / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        measured = displayed
    approved = output / "approved_tripod"
    with zipfile.ZipFile(simulation.ROOT / "artifacts/tripod-openloop-baseline-20260915/replay.zip") as archive:
        archive.extractall(approved)
    original = GaitReference(simulation.load_model(), "tripod")
    target = output / "candidate_target"
    _record_reference(measured / "model.mjb", "tripod", target,
                      {"measured_source": str(measured)}, foot_centers=original.foot_centers)
    recordings = [_accepted_baseline(), approved, target, measured]
    labels = ["PPO-100 | LEARNED CONTROL",
              "APPROVED TRIPOD | ORIGINAL CHASSIS | UNTRAINED",
              "CANDIDATE TARGET | KINEMATIC ONLY",
              "CANDIDATE TRIPOD | MEASURED | UNTRAINED"]
    film = render_comparison(recordings, labels, output / "comparison.mp4", fps=25)
    (output / "review.json").write_text(json.dumps({
        "recordings": list(map(str, recordings)), "labels": labels,
        "purpose": "chassis and gait direction review, not learned policy acceptance",
        "candidate_summary": metadata["summary"],
        "leg_count": 6, "playback_speed": 1,
    }, indent=2) + "\n")
    if view:
        with (output / "viewer.stdout.log").open("w") as stdout, (output / "viewer.stderr.log").open("w") as stderr:
            subprocess.Popen([sys.executable, "-m", "spider", "replay-grid",
                              *map(str, recordings), "--speed", "1", "--presentation", "stalk",
                              "--ready", str(output / "review.ready")], cwd=simulation.ROOT,
                             stdout=stdout, stderr=stderr,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return film


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--measured", required=True, type=Path,
                        help="Candidate tripod recording from gait_feasibility")
    parser.add_argument("--no-view", action="store_true")
    parser.add_argument("--refresh-appearance", action="store_true",
                        help="Use current appearance only after exact physical-model equality check")
    args = parser.parse_args()
    print(run(args.output, args.measured, view=not args.no_view,
              refresh_appearance=args.refresh_appearance))


if __name__ == "__main__":
    main()
