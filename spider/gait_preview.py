"""Record and view untrained kinematic gait references beside the accepted PPO baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import mujoco
import numpy as np

from .gait_reference import GaitReference
from .policy_run import DEFAULT_CHECKPOINT
from .recording import STATE
from .simulation import ROOT
from .viewing.film import _matching_indices, render_comparison


FPS = 25
DURATION_S = 5.0
FRAME_TIMES = np.linspace(0.0, DURATION_S, round(DURATION_S * FPS) + 1)
BASELINE_ROUND = ROOT / "telemetry/tuning/20260915-forward-exploration-round-01"
GAIT_NAMES = ("wave", "ripple", "tripod")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _accepted_baseline(round_directory: Path = BASELINE_ROUND) -> Path:
    """Find the recorded accepted mean-policy run without relying on a seed path."""
    candidates: list[Path] = []
    for metadata_path in sorted((round_directory / "accepted_baseline").rglob("metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("treatment") == "baseline" and metadata.get("mode") == "mean":
            recording = metadata_path.parent
            if all((recording / name).is_file() for name in ("model.mjb", "states.npz")):
                candidates.append(recording)
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one accepted mean-policy recording under {round_directory}; "
            f"found {len(candidates)}"
        )
    return candidates[0]


def _save_recording(
    directory: Path,
    model: mujoco.MjModel,
    states: np.ndarray,
    times: np.ndarray,
    label: str,
    metadata: dict,
) -> None:
    directory.mkdir(parents=True)
    mujoco.mj_saveModel(model, str(directory / "model.mjb"))
    np.savez_compressed(directory / "states.npz", states=states, times=times, label=label)
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _record_reference(
    source_model: Path, name: str, destination: Path, provenance: dict[str, str], *, foot_centers=None
) -> None:
    model = mujoco.MjModel.from_binary_path(str(source_model))
    reference = GaitReference(model, name, foot_centers=foot_centers)
    data = mujoco.MjData(model)
    states = np.empty((len(FRAME_TIMES), mujoco.mj_stateSize(model, STATE)))
    for index, time_s in enumerate(FRAME_TIMES):
        mujoco.mj_resetData(model, data)
        data.time = float(time_s)
        data.qpos[0] = reference.speed_mps * time_s
        data.qpos[2] = reference.height_m
        data.qpos[3] = 1.0
        data.qpos[7:] = reference.pose(float(time_s))
        mujoco.mj_forward(model, data)
        mujoco.mj_getState(model, data, states[index], STATE)
    label = (
        f"{name.upper()} / KINEMATIC / UNTRAINED / "
        f"prescribed {reference.speed_mps:.2f} m/s, z={reference.height_m:.2f} m"
    )
    _save_recording(
        destination,
        model,
        states,
        FRAME_TIMES,
        label,
        {
            "schema_version": 1,
            "name": name,
            "status": "kinematic_untrained_reference",
            "warning": "Prescribed pose preview only. No dynamics, stability, or locomotion claim.",
            "fps": FPS,
            "duration_s": DURATION_S,
            "frame_count_including_end": len(FRAME_TIMES),
            "speed_mps": reference.speed_mps,
            "height_m": reference.height_m,
            "config": reference.config,
            "provenance": provenance,
        },
    )


def prepare_recordings(output: str | Path) -> list[Path]:
    """Create the baseline-plus-reference recordings consumed by both viewers."""
    destination = Path(output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    baseline = _accepted_baseline()
    source_metadata_path = baseline / "metadata.json"
    source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    checkpoint_hash = _sha256(DEFAULT_CHECKPOINT)
    if source_metadata.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("Accepted baseline checkpoint hash does not match DEFAULT_CHECKPOINT")
    provenance = {
        "checkpoint": str(DEFAULT_CHECKPOINT.relative_to(ROOT)),
        "checkpoint_sha256": checkpoint_hash,
        "source_recording": str(baseline.relative_to(ROOT)),
        "source_metadata_sha256": _sha256(source_metadata_path),
        "source_model_sha256": _sha256(baseline / "model.mjb"),
        "source_states_sha256": _sha256(baseline / "states.npz"),
    }
    baseline_output = destination / "accepted_baseline"
    baseline_output.mkdir()
    shutil.copy2(baseline / "model.mjb", baseline_output / "model.mjb")
    shutil.copy2(source_metadata_path, baseline_output / "source-metadata.json")
    with np.load(baseline / "states.npz", allow_pickle=False) as archive:
        source_times = archive["times"]
        indices = _matching_indices(source_times, FRAME_TIMES)
        baseline_states = archive["states"][indices]
        baseline_times = source_times[indices]
    baseline_label = "RECORDED ACCEPTED PPO-100 / MEASURED / comparator"
    np.savez_compressed(
        baseline_output / "states.npz",
        states=baseline_states,
        times=baseline_times,
        label=baseline_label,
    )
    (baseline_output / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "accepted_recorded_baseline_comparator",
                "source": str(baseline.relative_to(ROOT)),
                "fps": FPS,
                "duration_s": DURATION_S,
                "frame_count_including_end": len(FRAME_TIMES),
                "provenance": provenance,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    recordings = [baseline_output]
    for name in GAIT_NAMES:
        gait_output = destination / name
        _record_reference(baseline / "model.mjb", name, gait_output, provenance)
        recordings.append(gait_output)
    return recordings


def run(output: str | Path, *, launch_preview: bool = True) -> tuple[Path, subprocess.Popen | None]:
    """Save endpoint-inclusive panes, render a five-second film, and open the replay grid."""
    destination = Path(output).resolve()
    recordings = prepare_recordings(destination)
    reference_metadata = [
        json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        for path in recordings[1:]
    ]
    labels = [
        "ACCEPTED PPO-100 | MEASURED",
        *[
            f"{name.upper()} | KINEMATIC / UNTRAINED | "
            f"{metadata['speed_mps']:.2f}m/s z={metadata['height_m']:.2f}m"
            for name, metadata in zip(GAIT_NAMES, reference_metadata, strict=True)
        ],
    ]
    # Keep the standard five-second, 125-frame film. The saved recordings retain
    # the 5.00 s endpoint for replay and later endpoint-inclusive inspection.
    film = render_comparison(
        recordings,
        labels,
        destination / "comparison.mp4",
        fps=FPS,
        duration_s=DURATION_S,
    )
    process = None
    if launch_preview:
        command = [
            sys.executable,
            "-m",
            "spider",
            "replay-grid",
            *(str(path) for path in recordings),
            "--speed",
            "1",
            "--presentation",
            "stalk",
            "--ready",
            str(destination / "review.ready"),
        ]
        with (destination / "viewer.stdout.log").open("w", encoding="utf-8") as stdout, (
            destination / "viewer.stderr.log"
        ).open("w", encoding="utf-8") as stderr:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=stdout,
                stderr=stderr,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
    return film, process


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-view", action="store_true", help="render files without opening GLFW")
    args = parser.parse_args(argv)
    film, _ = run(args.output, launch_preview=not args.no_view)
    print(f"Saved kinematic reference comparison: {film}")


if __name__ == "__main__":
    main()
