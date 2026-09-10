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

    def __init__(self, model: mujoco.MjModel, label: str, actuator_name: str = ""):
        self.model = model
        self.label = label
        self.actuator_name = actuator_name
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
                 times=np.asarray(self.times), label=self.label, actuator_name=self.actuator_name)
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


def _inspection(model, states, times, actuator_name):
    """Build a static pose reference and an angle trace from recorded states."""
    actuator = model.actuator(actuator_name).id
    joint = int(model.actuator_trnid[actuator, 0])
    address = int(model.jnt_qposadr[joint])
    reference = mujoco.MjData(model)
    mujoco.mj_setState(model, reference, states[0], STATE)
    mujoco.mj_forward(model, reference)
    body = int(model.jnt_bodyid[joint])
    bodies = {body}
    for candidate in range(body + 1, model.nbody):
        if int(model.body_parentid[candidate]) in bodies:
            bodies.add(candidate)
    geoms = [i for i in range(model.ngeom) if int(model.geom_bodyid[i]) in bodies]
    recorded = mujoco.MjData(model)
    angles = []
    for state in states:
        mujoco.mj_setState(model, recorded, state, STATE)
        angles.append(float(recorded.qpos[address]))
    changes = np.rad2deg(np.asarray(angles) - angles[0])
    figure = mujoco.MjvFigure()
    figure.title = "Recorded joint change (degrees)"
    figure.xlabel = "Simulation time (s)"
    figure.flg_extend = 0
    figure.flg_legend = 1
    figure.linename[0], figure.linename[1] = "measured", "shown frame"
    figure.linergb[0], figure.linergb[1] = (1, 0.7, 0.3), (0.6, 0.85, 1)
    figure.figurergba[:] = (0.09, 0.09, 0.14, 0.96)
    figure.panergba[:] = (0.12, 0.12, 0.18, 1)
    figure.textrgb[:] = (0.91, 0.88, 0.84)
    figure.gridrgb[:] = (0.27, 0.26, 0.34)
    figure.linewidth = 2
    indices = np.linspace(0, len(times) - 1, min(len(times), mujoco.mjMAXLINEPNT), dtype=int)
    figure.linepnt[0] = len(indices)
    figure.linedata[0, :2 * len(indices)] = np.column_stack((times[indices], changes[indices])).ravel()
    low, high = min(0, float(changes.min())), max(0, float(changes.max()))
    padding = max((high - low) * 0.15, 0.01)
    figure.range[:] = ((times[0], max(times[-1], times[0] + 0.001)), (low - padding, high + padding))
    return reference, geoms, changes, figure


def replay(run: Path, speed: float, actuator_name: str = "") -> None:
    import mujoco.viewer

    model = mujoco.MjModel.from_binary_path(str(run / "model.mjb"))
    data = mujoco.MjData(model)
    with np.load(run / "states.npz", allow_pickle=False) as archive:
        states, times = archive["states"], archive["times"]
        label = str(archive["label"])
        actuator_name = actuator_name or (str(archive["actuator_name"]) if "actuator_name" in archive else "")
    inspection = _inspection(model, states, times, actuator_name) if actuator_name else None
    duration = float(times[-1] - times[0]) / speed
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = (0, 0, 0.25)
        viewer.cam.distance = 1.55
        viewer.cam.azimuth = 225
        viewer.cam.elevation = -25
        if inspection:
            reference, geoms, changes, figure = inspection
            viewer.cam.lookat[:] = np.mean(reference.geom_xpos[geoms], axis=0)
            viewer.cam.distance = 0.95
            viewer.cam.azimuth = 225
            viewer.cam.elevation = -20
            for index, geom_id in enumerate(geoms):
                ghost = viewer.user_scn.geoms[index]
                mujoco.mjv_initGeom(ghost, int(model.geom_type[geom_id]), model.geom_size[geom_id],
                                   reference.geom_xpos[geom_id], reference.geom_xmat[geom_id],
                                   np.array((0.6, 0.85, 1, 0.28), dtype=np.float32))
                ghost.emission = 0.4
            viewer.user_scn.ngeom = len(geoms)
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
                               f"t = {times[index]:.3f} s\nClose this window to stop"),
                              (mujoco.mjtFontScale.mjFONTSCALE_150,
                               mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
                               (f"Joint change: {changes[index]:+.3f} deg\nCyan ghost: starting pose, not a control run"
                                if inspection else ""), "")])
            if inspection:
                figure.linepnt[1] = 2
                figure.linedata[1, :4] = (times[index], figure.range[1, 0], times[index], figure.range[1, 1])
                viewport = viewer.viewport
                viewer.set_figures([(mujoco.MjrRect(viewport.left + 10, viewport.bottom + 10,
                                                   min(500, viewport.width // 2), min(240, viewport.height // 3)), figure)])
            viewer.sync(state_only=True)
            if not ready:
                (run / "ready").touch()
                ready = True
            time.sleep(max(0, 1 / 60 - (time.monotonic() - frame_start)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--speed", type=float, default=0.1)
    parser.add_argument("--actuator", default="")
    args = parser.parse_args()
    replay(args.run, args.speed, args.actuator)
