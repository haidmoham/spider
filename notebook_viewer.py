"""Replay recorded notebook states in a separate native MuJoCo window."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import mujoco
import numpy as np


STATE = mujoco.mjtState.mjSTATE_INTEGRATION


class TreatmentReplay:
    """Capture states without stepping or changing the experiment."""

    def __init__(self, model: mujoco.MjModel, label: str):
        self.model = model
        self.label = label
        self.states: list[np.ndarray] = []
        self.times: list[float] = []

    def capture(self, data: mujoco.MjData) -> None:
        state = np.empty(mujoco.mj_stateSize(self.model, STATE))
        mujoco.mj_getState(self.model, data, state, STATE)
        self.states.append(state)
        self.times.append(float(data.time))

    def launch(self, directory: Path, speed: float = 0.1) -> subprocess.Popen:
        """Open a looping replay. Slow motion changes display timing only."""
        if not self.states or not np.isfinite(speed) or speed <= 0:
            raise ValueError("Capture states and choose a finite positive playback speed.")
        directory.mkdir(parents=True, exist_ok=True)
        run = Path(tempfile.mkdtemp(prefix="treatment-", dir=directory))
        mujoco.mj_saveModel(self.model, str(run / "model.mjb"))
        np.savez(run / "states.npz", states=np.asarray(self.states),
                 times=np.asarray(self.times), label=self.label)
        with (run / "viewer.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), str(run), "--speed", str(speed)],
                stdout=log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if (run / "ready").exists():
                print(f"Viewer opened (PID {process.pid}). Recorded states: {run}")
                return process
            if process.poll() is not None:
                raise RuntimeError((run / "viewer.log").read_text(encoding="utf-8"))
            time.sleep(0.05)
        raise TimeoutError(f"Viewer startup is still pending (PID {process.pid}); see {run / 'viewer.log'}")


def replay(run: Path, speed: float) -> None:
    import mujoco.viewer

    model = mujoco.MjModel.from_binary_path(str(run / "model.mjb"))
    data = mujoco.MjData(model)
    with np.load(run / "states.npz", allow_pickle=False) as archive:
        states, times = archive["states"], archive["times"]
        label = str(archive["label"])
    duration = float(times[-1] - times[0]) / speed
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = (0, 0, 0.25)
        viewer.cam.distance = 1.55
        viewer.cam.azimuth = 225
        viewer.cam.elevation = -25
        started = time.monotonic()
        ready = False
        while viewer.is_running():
            frame_start = time.monotonic()
            elapsed = (frame_start - started) % (duration + 1.0)
            shown_time = times[0] + min(elapsed, duration) * speed
            index = min(len(times) - 1, max(0, int(np.searchsorted(times, shown_time, side="right")) - 1))
            with viewer.lock():
                mujoco.mj_setState(model, data, states[index], STATE)
                # Rebuild display transforms; never integrate physics in the viewer.
                mujoco.mj_forward(model, data)
            viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_100,
                               mujoco.mjtGridPos.mjGRID_TOPLEFT,
                               f"RECORDED TREATMENT / {speed:g}x / loops\n{label}",
                               f"t = {times[index]:.3f} s\nClose this window to stop")])
            viewer.sync(state_only=True)
            if not ready:
                (run / "ready").touch()
                ready = True
            time.sleep(max(0, 1 / 60 - (time.monotonic() - frame_start)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--speed", type=float, default=0.1)
    args = parser.parse_args()
    replay(args.run, args.speed)
