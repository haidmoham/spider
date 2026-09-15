"""Isolated wide-range chassis candidate derived from the canonical C-1N XML."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import mujoco

from . import simulation


LEG_NAMES = simulation.FOOT_NAMES
LEG_YAWS_DEG = {
    "front_left": 38,
    "front_right": -38,
    "middle_left": 90,
    "middle_right": -90,
    "rear_left": 142,
    "rear_right": -142,
}
MOUNT_PITCH_REMOVED_DEG = 55.0

PARAMETER_MANIFEST = {
    "source": str(simulation.MODEL_PATH),
    "isolation": "generated in memory; canonical model/spider.xml is never written",
    "physical_edits": {
        "leg_root_euler_deg": {
            name: [yaw, 0, 0] for name, yaw in LEG_YAWS_DEG.items()
        },
        "hip_joint_range_deg": [-45, 100],
        "knee_joint_range_deg": [-80, 140],
        "hip_actuator_ctrlrange_rad": [-0.7854, 1.7453],
        "knee_actuator_ctrlrange_rad": [-1.3963, 2.4435],
        "reset_hip_offset_deg": MOUNT_PITCH_REMOVED_DEG,
    },
    "preserved": [
        "masses and density",
        "friction and contacts",
        "link lengths and physical geom sizes",
        "actuator gains",
    ],
    "visual_edits": (
        "render-only graphite abdomen, dorsal armor, spikes, and leg shrouds; "
        "ruby accents; canonical googly eyes retained"
    ),
}


def _site(parent: ET.Element, name: str, site_type: str, **attributes: str) -> None:
    ET.SubElement(parent, "site", name=name, type=site_type, group="2", **attributes)


def _add_shell(torso: ET.Element) -> None:
    _site(torso, "candidate_abdomen_visual", "ellipsoid", pos="-0.105 0 0.075",
          size="0.245 0.145 0.075", rgba="0.23 0.25 0.30 1")
    _site(torso, "candidate_rear_taper_visual", "ellipsoid", pos="-0.285 0 0.072",
          size="0.135 0.095 0.052", rgba="0.15 0.17 0.22 1")
    for index, x in enumerate((0.10, 0.025, -0.055, -0.135, -0.215)):
        _site(torso, f"candidate_dorsal_plate_{index}", "box",
              pos=f"{x} 0 0.142", size="0.050 0.126 0.010",
              euler=f"0 {12 - index * 5} 0", rgba="0.28 0.30 0.35 1")
    for index, (x, z, length) in enumerate(((-0.02, 0.18, 0.075),
                                             (-0.12, 0.19, 0.090),
                                             (-0.22, 0.17, 0.070))):
        _site(torso, f"candidate_dorsal_spike_{index}", "capsule",
              fromto=f"{x} 0 {z - length} {x - 0.035} 0 {z}", size="0.010",
              rgba="0.035 0.038 0.055 1")
    for side, y in (("left", 0.137), ("right", -0.137)):
        _site(torso, f"candidate_{side}_ruby_visual", "ellipsoid",
              pos=f"-0.10 {y} 0.085", size="0.016 0.008 0.010",
              material="candidate_life_glow", rgba="0.95 0.015 0.025 1")
        _site(torso, f"candidate_{side}_breathing_seam", "capsule",
              fromto=f"-0.23 {y*0.80} 0.123 0.09 {y*0.80} 0.123", size="0.0035",
              material="candidate_life_glow", rgba="0.95 0.015 0.025 1")
    _site(torso, "candidate_heartbeat_core", "ellipsoid", pos="-0.12 0 0.159",
          size="0.027 0.017 0.008", material="candidate_life_glow",
          rgba="0.95 0.015 0.025 1")
    ET.SubElement(torso, "light", name="candidate_underlight", pos="0 0 -0.075",
                  dir="0 0 -1", diffuse="0.10 0.0015 0.0025", specular="0 0 0",
                  ambient="0 0 0", castshadow="false", cutoff="85")
    ET.SubElement(torso, "site", name="googly_enabled", pos="0 0 0", size="0.001",
                  rgba="0 0 0 0", group="5")


def _add_leg_shrouds(leg: ET.Element, name: str) -> None:
    _site(leg, f"{name}_candidate_thigh_shroud", "capsule",
          fromto="0.015 0 0.018 0.245 0 0.018", size="0.046",
          rgba="0.20 0.22 0.27 1")
    _site(leg, f"{name}_candidate_thigh_point", "ellipsoid", pos="0.255 0 0.018",
          size="0.070 0.040 0.025", rgba="0.16 0.18 0.23 1")
    shin = leg.find(f"body[@name='{name}_shin']")
    if shin is None:
        raise ValueError(f"canonical model is missing {name}_shin")
    _site(shin, f"{name}_candidate_shin_shroud", "capsule",
          fromto="0.012 0 0.018 0.222 0 0.018", size="0.039",
          rgba="0.13 0.15 0.20 1")
    _site(shin, f"{name}_candidate_distal_point", "ellipsoid", pos="0.225 0 0.018",
          size="0.062 0.033 0.022", rgba="0.025 0.028 0.045 1")


def candidate_xml() -> str:
    """Return a standalone candidate XML source derived from the canonical file."""
    root = ET.parse(simulation.MODEL_PATH).getroot()
    torso = root.find("./worldbody/body[@name='torso']")
    actuator = root.find("actuator")
    if torso is None or actuator is None:
        raise ValueError("canonical model is missing torso or actuator section")

    torso_geom = torso.find("geom[@name='torso_geom']")
    cowl = torso.find("site[@name='cowl_visual']")
    if torso_geom is None or cowl is None:
        raise ValueError("canonical torso visual anchors are missing")
    torso_geom.set("rgba", "0.76 0.70 0.61 0")
    cowl.set("rgba", "0.095 0.10 0.17 0")
    _add_shell(torso)

    # Zero-mass, non-contact mesh fins affect silhouette only. The original
    # collision chassis and automatically computed physical masses stay intact.
    asset = root.find("asset")
    if asset is None:
        raise ValueError("canonical asset section is missing")
    ET.SubElement(asset, "material", name="candidate_life_glow",
                  rgba="0.95 0.015 0.025 1", emission="0.7", specular="0.15")
    ET.SubElement(asset, "material", name="candidate_metal",
                  rgba="0.18 0.19 0.21 1", specular="0.75", shininess="0.7")
    ET.SubElement(asset, "mesh", name="candidate_armor_fin", scale="0.11 0.035 0.07",
                  vertex="0 -1 0 0 1 0 -1 0 0 -0.8 0 1.3",
                  face="0 2 1 0 1 3 1 2 3 2 0 3")
    for index, x in enumerate((0.06, -0.04, -0.14, -0.24)):
        ET.SubElement(torso, "geom", name=f"candidate_fin_{index}", type="mesh",
                      mesh="candidate_armor_fin", pos=f"{x} 0 0.14",
                      mass="0", contype="0", conaffinity="0", group="2",
                      rgba="0.24 0.27 0.32 1")

    for name in LEG_NAMES:
        leg = torso.find(f"body[@name='{name}']")
        if leg is None:
            raise ValueError(f"canonical model is missing {name}")
        leg.set("euler", f"{LEG_YAWS_DEG[name]} 0 0")
        hip = leg.find(f"joint[@name='{name}_hip']")
        shin = leg.find(f"body[@name='{name}_shin']")
        knee = None if shin is None else shin.find(f"joint[@name='{name}_knee']")
        if hip is None or knee is None:
            raise ValueError(f"canonical model is missing joints for {name}")
        hip.set("range", "-45 100")
        knee.set("range", "-80 140")
        hip_motor = actuator.find(f"position[@name='{name}_hip_motor']")
        knee_motor = actuator.find(f"position[@name='{name}_knee_motor']")
        if hip_motor is None or knee_motor is None:
            raise ValueError(f"canonical model is missing actuators for {name}")
        hip_motor.set("ctrlrange", "-0.7854 1.7453")
        knee_motor.set("ctrlrange", "-1.3963 2.4435")
        _add_leg_shrouds(leg, name)
        for geom in leg.iter("geom"):
            geom.set("rgba", "0.11 0.12 0.14 1")
            geom.set("material", "candidate_metal")
    for site in torso.iter("site"):
        name = site.get("name", "")
        if "candidate_" in name and site.get("material") != "candidate_life_glow":
            if name == "googly_enabled":
                continue
            site.set("material", "candidate_metal")
            if "shroud" in name or "abdomen" in name:
                site.set("rgba", "0.18 0.19 0.21 1")
        if "hub_visual" in name:
            site.set("material", "candidate_metal")
            site.set("rgba", "0.23 0.24 0.26 1")
        if name in ("spine_visual", "rear_badge_visual"):
            site.set("material", "candidate_life_glow")
            site.set("rgba", "0.95 0.015 0.025 1")
    return ET.tostring(root, encoding="unicode")


def load_candidate() -> mujoco.MjModel:
    """Compile and return a new candidate model without touching canonical XML."""
    model = mujoco.MjModel.from_xml_string(candidate_xml())
    feet = [model.geom(i).name for i in range(model.ngeom)
            if model.geom(i).name.endswith("_foot")]
    expected = {name + "_foot" for name in LEG_NAMES}
    if len(feet) != 6 or set(feet) != expected or model.nu != 18:
        raise ValueError("C-1N must retain exactly six legs with three actuators per leg")
    return model


def reset_candidate(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[float, ...]:
    """Reset with the hip offset that replaces the removed 55-degree mount pitch."""
    targets = list(simulation.reset(model, data))
    offset = math.radians(MOUNT_PITCH_REMOVED_DEG)
    for name in LEG_NAMES:
        joint = model.joint(f"{name}_hip")
        actuator = model.actuator(f"{name}_hip_motor")
        data.qpos[int(model.jnt_qposadr[joint.id])] += offset
        data.ctrl[actuator.id] += offset
        targets[actuator.id] += offset
    mujoco.mj_forward(model, data)
    return tuple(targets)
