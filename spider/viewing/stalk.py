"""A restrained, presentation-only look for recorded C-1N motion."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import tempfile
import zipfile

import mujoco
import numpy as np
from PIL import Image

from ..recording import STATE


def apply_stalk_presentation(model: mujoco.MjModel) -> None:
    """Apply dark-stage colors and steady key/rim lights to a render model.

    This function changes MuJoCo visual fields only. It does not touch bodies,
    joints, actuators, contacts, solver settings, or ``MjData``.
    """
    ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    if ground_id >= 0:
        model.geom_rgba[ground_id] = (0.135, 0.145, 0.165, 1.0)

    model.vis.rgba.fog[:] = (0.018, 0.021, 0.028, 1.0)
    model.vis.rgba.haze[:] = (0.035, 0.041, 0.052, 1.0)
    model.vis.headlight.active = 1
    model.vis.headlight.ambient[:] = (0.125, 0.130, 0.145)
    model.vis.headlight.diffuse[:] = (0.38, 0.37, 0.36)
    model.vis.headlight.specular[:] = (0.09, 0.09, 0.10)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, 1152)
    model.vis.global_.offheight = max(model.vis.global_.offheight, 648)

    # A neutral overhead key keeps the porcelain body readable. The cool,
    # restrained rear light separates the leg tips from the matte stage.
    if model.nlight > 0:
        model.light_pos[0] = (1.15, -1.35, 2.35)
        model.light_dir[0] = (-0.37, 0.43, -0.82)
        model.light_diffuse[0] = (0.82, 0.75, 0.64)
        model.light_specular[0] = (0.30, 0.27, 0.23)
        model.light_ambient[0] = (0.045, 0.043, 0.040)
        model.light_castshadow[0] = 1
    if model.nlight > 1:
        model.light_pos[1] = (-1.25, 1.10, 1.35)
        model.light_dir[1] = (0.52, -0.46, -0.72)
        model.light_diffuse[1] = (0.28, 0.36, 0.50)
        model.light_specular[1] = (0.10, 0.14, 0.21)
        model.light_ambient[1] = (0.008, 0.011, 0.018)
        model.light_castshadow[1] = 1


def stalk_camera(model: mujoco.MjModel, data: mujoco.MjData) -> mujoco.MjvCamera:
    """Return a low three-quarter camera that follows the recorded torso."""
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    target = data.xpos[torso_id] if torso_id >= 0 else data.qpos[:3]
    camera.lookat[:] = (float(target[0]) + 0.10, float(target[1]), 0.24)
    camera.distance = 1.36
    camera.azimuth = 222.0
    camera.elevation = -13.0
    return camera


@contextmanager
def _recording_directory(source: Path):
    if source.is_dir():
        yield source
        return
    if source.suffix.lower() != ".zip":
        raise ValueError(f"Expected a replay directory or zip file: {source}")
    with tempfile.TemporaryDirectory(prefix="c1n-stalk-") as temporary:
        with zipfile.ZipFile(source) as archive:
            archive.extractall(temporary)
        yield Path(temporary)


def render_recorded_preview(
    directory: str | Path,
    output: str | Path,
    *,
    pane: int = 3,
    frame: int | None = None,
) -> Path:
    """Render one saved replay frame without integrating the simulation."""
    source, destination = Path(directory), Path(output)
    with _recording_directory(source) as root:
        replay = root / str(pane) if (root / str(pane)).is_dir() else root
        model = mujoco.MjModel.from_binary_path(str(replay / "model.mjb"))
        data = mujoco.MjData(model)
        with np.load(replay / "states.npz", allow_pickle=False) as archive:
            states = archive["states"]
        index = len(states) // 2 if frame is None else frame
        if not -len(states) <= index < len(states):
            raise IndexError(f"Frame {index} is outside replay length {len(states)}")

        apply_stalk_presentation(model)
        mujoco.mj_setState(model, data, states[index], STATE)
        mujoco.mj_forward(model, data)
        with mujoco.Renderer(model, height=648, width=1152) as renderer:
            renderer.update_scene(data, camera=stalk_camera(model, data))
            image = Image.fromarray(renderer.render())

    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)
    return destination.resolve()
