"""Reusable C-1N front-left leg workspace fixture for issue #31.

The fixture owns model loading, the support target, and torso-frame foot
measurement. The companion notebook uses these functions for visual and
numerical analysis. This module does not run an experiment at import time.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np


MODEL_PATH = Path(__file__).resolve().parent / "models" / "c1n_two_joint.xml"
TARGET = np.array((0.30, 0.32, -0.30))
REACH_TOLERANCE_M = 0.01
FRONT_LEFT_JOINTS = ("front_left_hip", "front_left_knee")


def load_fixed_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load the historical two-joint C-1N model with gravity disabled for kinematics."""
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    model.opt.gravity[:] = 0.0
    return model, mujoco.MjData(model)


def torso_frame_foot_position(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_names: tuple[str, ...],
    angles: np.ndarray,
    site_name: str = "front_left_contact_site",
) -> np.ndarray:
    """Return one foot point in torso coordinates at declared joint angles."""
    data.qpos[:] = 0.0
    data.qpos[3] = 1.0
    for joint_name, angle in zip(joint_names, angles):
        joint = model.joint(joint_name)
        data.qpos[joint.qposadr[0]] = angle
    mujoco.mj_forward(model, data)
    torso_id = model.body("torso").id
    torso_rotation = data.xmat[torso_id].reshape(3, 3)
    site_id = model.site(site_name).id
    return torso_rotation.T @ (data.site_xpos[site_id] - data.xpos[torso_id])
