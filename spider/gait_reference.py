"""Inspectable foot trajectories for C-1N; no training or physics integration.

These are motion references, not learned policies. A stance foot travels backward
relative to a translating torso, so it remains planted in world coordinates.
Swing uses a Hermite path with matching endpoint velocity and a smooth lift.
"""

from __future__ import annotations

import math
import mujoco
import numpy as np

from . import simulation


GAITS = {
    # Order: front left/right, middle left/right, rear left/right.
    "wave": dict(frequency_hz=0.60, stance_fraction=5 / 6,
                 offsets=(2 / 6, 5 / 6, 1 / 6, 4 / 6, 0, 3 / 6)),
    "ripple": dict(frequency_hz=0.85, stance_fraction=2 / 3,
                   offsets=(0, 1 / 2, 1 / 3, 5 / 6, 2 / 3, 1 / 6)),
    "tripod": dict(frequency_hz=1.10, stance_fraction=0.65,
                   offsets=(0, 1 / 2, 1 / 2, 0, 0, 1 / 2)),
}


class GaitReference:
    """Pure periodic target generator using the canonical leg geometry.

    ``pose`` solves each requested foot location exactly or raises. It never
    clips an unreachable target into a superficially plausible animation.
    """

    def __init__(self, model: mujoco.MjModel, name: str, *, foot_centers=None):
        if name not in GAITS:
            raise ValueError(f"unknown gait: {name}")
        self.name = name
        self.config = dict(GAITS[name], height_m=0.37, stride_length_m=0.18,
                           clearance_m=0.04, radial_spread_m=0.17,
                           evidence_kind="untrained_kinematic_reference")
        self.height_m = self.config["height_m"]
        self.speed_mps = (self.config["stride_length_m"] * self.config["frequency_hz"]
                          / self.config["stance_fraction"])
        self._bases, self._rotations, self._lengths, self._limits = [], [], [], []
        # Do not use the simulation's id-keyed neutral-position cache here:
        # temporary original/candidate models can reuse a released object's id.
        # Reference construction is infrequent; compute from this exact model.
        neutral_data = mujoco.MjData(model)
        simulation.reset(model, neutral_data)
        neutral = {name: neutral_data.geom_xpos[model.geom(name + "_foot").id].copy()
                   for name in simulation.FOOT_NAMES}
        self._centers = []
        for name in simulation.FOOT_NAMES:
            body = model.body(name)
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, body.quat)
            self._rotations.append(rotation.reshape(3, 3).copy())
            self._bases.append(body.pos.copy())
            shin = model.body(name + "_shin")
            foot = model.geom(name + "_foot")
            self._lengths.append((float(np.linalg.norm(shin.pos)),
                                  float(np.linalg.norm(foot.pos))))
            center = np.array(neutral[name])
            radial = center[:2] - body.pos[:2]
            center[:2] += radial / np.linalg.norm(radial) * self.config["radial_spread_m"]
            center[2] = float(foot.size[0])
            self._centers.append(center)
            limits = []
            for suffix in ("coxa", "hip", "knee"):
                joint = model.joint(name + "_" + suffix)
                actuator_ids = np.flatnonzero(model.actuator_trnid[:, 0] == joint.id)
                low, high = joint.range
                for actuator in actuator_ids:
                    if model.actuator_ctrllimited[actuator]:
                        low = max(low, model.actuator_ctrlrange[actuator, 0])
                        high = min(high, model.actuator_ctrlrange[actuator, 1])
                limits.append((low, high))
            self._limits.append(limits)
        self._centers = np.asarray(self._centers)
        if foot_centers is not None:
            centers = np.asarray(foot_centers, dtype=float)
            if centers.shape != (6, 3) or not np.isfinite(centers).all():
                raise ValueError("foot_centers must contain six finite XYZ positions")
            self._centers = centers.copy()
        self._limits = np.asarray(self._limits)

    @property
    def foot_centers(self) -> np.ndarray:
        """Copy of the centred stance anchors for controlled chassis comparisons."""
        return self._centers.copy()

    def feet(self, time_s: float) -> np.ndarray:
        """Desired foot centres in a frame translating with the level torso.

        Z is ground referenced. The body origin is (0, 0, height_m).
        """
        if not math.isfinite(time_s):
            raise ValueError("time must be finite")
        cfg = self.config
        phase = (time_s * cfg["frequency_hz"] + np.asarray(cfg["offsets"])) % 1
        duty, length = cfg["stance_fraction"], cfg["stride_length_m"]
        result = self._centers.copy()
        for leg, p in enumerate(phase):
            if p < duty:
                x = length * (0.5 - p / duty)
            else:
                u = (p - duty) / (1 - duty)
                # Both endpoint tangents match the backward stance velocity.
                tangent = -length * (1 - duty) / duty
                x = ((2*u**3 - 3*u**2 + 1) * (-length/2)
                     + (u**3 - 2*u**2 + u) * tangent
                     + (-2*u**3 + 3*u**2) * (length/2)
                     + (u**3 - u**2) * tangent)
                result[leg, 2] += cfg["clearance_m"] * math.sin(math.pi * u)**2
            result[leg, 0] += x
        return result

    def pose(self, time_s: float) -> np.ndarray:
        joints = []
        for leg, foot in enumerate(self.feet(time_s)):
            relative = foot - np.asarray((0, 0, self.height_m)) - self._bases[leg]
            x, y, z = self._rotations[leg].T @ relative
            radial = math.hypot(x, y)
            upper, lower = self._lengths[leg]
            cosine = (radial**2 + z**2 - upper**2 - lower**2) / (2*upper*lower)
            if abs(cosine) > 1 + 1e-10:
                raise ValueError(f"unreachable {simulation.FOOT_NAMES[leg]} at {time_s:g}s")
            knee = math.acos(np.clip(cosine, -1, 1))
            hip = math.atan2(-z, radial) - math.atan2(lower*math.sin(knee),
                                                                  upper+lower*math.cos(knee))
            angles = np.asarray((math.atan2(y, x), hip, knee))
            if np.any(angles < self._limits[leg, :, 0] - 1e-9) or np.any(
                    angles > self._limits[leg, :, 1] + 1e-9):
                raise ValueError(f"joint limit: {simulation.FOOT_NAMES[leg]} at {time_s:g}s: {angles}")
            joints.extend(angles)
        return np.asarray(joints)
