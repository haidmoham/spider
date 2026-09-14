"""Make a two-joint leg's pose-dependent velocity Jacobian observable."""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np


XML = """
<mujoco model="two_joint_planar_leg_jacobian">
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <light pos="0 0 4"/>
    <geom type="plane" size="3 3 0.1" rgba="0.91 0.925 0.945 1"/>
    <body name="hip" pos="0 0 1.5">
      <joint name="hip" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.75" size="0.05" mass="1" rgba="0.01 0.01 0.015 1"/>
      <body name="knee" pos="0 0 -0.75">
        <joint name="knee" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.65" size="0.045" mass="0.7" rgba="0.01 0.01 0.015 1"/>
        <site name="foot" pos="0 0 -0.65" size="0.06" rgba="0.71 0.263 0.247 1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

POSES = {
    "bent": np.array([0.45, -0.95]),
    "open": np.array([1.15, -0.35]),
}
QDOT = np.array([0.70, -0.35])
ARROW_SCALE = 0.40
VIEWER_SWEEP_SECONDS = 4.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Sweep smoothly between the two evaluated poses in MuJoCo.",
    )
    return parser.parse_args()


def evaluate_pose(model, data, foot_site_id, name, q):
    """Evaluate the instantaneous planar velocity map at one configuration."""
    data.qpos[:] = q
    data.qvel[:] = QDOT
    mujoco.mj_forward(model, data)

    jacobian_position = np.zeros((3, model.nv))
    jacobian_rotation = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacobian_position, jacobian_rotation, foot_site_id)
    jacobian_xz = jacobian_position[[0, 2], :2]
    predicted_xdot = jacobian_xz @ QDOT

    spatial_velocity = np.zeros(6)
    mujoco.mj_objectVelocity(
        model, data, mujoco.mjtObj.mjOBJ_SITE, foot_site_id, spatial_velocity, 0
    )
    simulated_xdot = spatial_velocity[3:][[0, 2]]

    return {
        "name": name,
        "q": q.copy(),
        "x": data.site_xpos[foot_site_id, [0, 2]].copy(),
        "jacobian": jacobian_xz,
        "predicted_xdot": predicted_xdot,
        "simulated_xdot": simulated_xdot,
        "error": predicted_xdot - simulated_xdot,
    }


def print_result(result):
    print(f"\n{result['name']} pose")
    print(f"q       = {np.array2string(result['q'], precision=3)} rad")
    print(f"qdot    = {np.array2string(QDOT, precision=3)} rad/s")
    print(f"x(q)    = {np.array2string(result['x'], precision=3)} m  [world X, world Z]")
    print("J(q)    =")
    print(np.array2string(result["jacobian"], precision=3))
    print(
        "J(q) qdot = "
        f"{np.array2string(result['predicted_xdot'], precision=6)} m/s"
    )
    print(
        "MuJoCo xdot = "
        f"{np.array2string(result['simulated_xdot'], precision=6)} m/s"
    )
    print(f"difference = {np.linalg.norm(result['error']):.2e} m/s")


def add_velocity_arrow(scene, start, velocity_xz, color, label):
    """Draw one X-Z velocity vector in the visible motion plane."""
    end = start + ARROW_SCALE * np.array([velocity_xz[0], 0.0, velocity_xz[1]])
    arrow = scene.geoms[scene.ngeom]
    mujoco.mjv_connector(
        arrow,
        mujoco.mjtGeom.mjGEOM_ARROW,
        0.018,
        start,
        end,
    )
    arrow.rgba = color
    arrow.label = label
    scene.ngeom += 1


def draw_jacobian_vectors(scene, model, data, foot_site_id):
    """Show the two local Jacobian columns and their qdot-weighted sum."""
    jacobian_position = np.zeros((3, model.nv))
    jacobian_rotation = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacobian_position, jacobian_rotation, foot_site_id)
    start = data.site_xpos[foot_site_id].copy()
    add_velocity_arrow(
        scene, start, jacobian_position[[0, 2], 0], np.array([0.18, 0.41, 0.68, 1.0]), "J hip"
    )
    add_velocity_arrow(
        scene, start, jacobian_position[[0, 2], 1], np.array([0.95, 0.55, 0.10, 1.0]), "J knee"
    )
    add_velocity_arrow(
        scene, start, jacobian_position[[0, 2], :2] @ QDOT, np.array([0.25, 0.80, 0.40, 1.0]), "J qdot"
    )


def run_viewer(model, data, foot_site_id):
    """Sweep through the pose change so the local map is continuously visible."""
    bent_q = POSES["bent"]
    open_q = POSES["open"]
    start = time.perf_counter()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = (0.0, 0.0, 0.8)
        viewer.cam.distance = 2.8
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = 0.0
        while viewer.is_running():
            phase = (time.perf_counter() - start) / VIEWER_SWEEP_SECONDS
            blend = 0.5 - 0.5 * np.cos(2.0 * np.pi * phase)
            data.qpos[:] = (1.0 - blend) * bent_q + blend * open_q
            data.qvel[:] = QDOT
            mujoco.mj_forward(model, data)
            viewer.user_scn.ngeom = 0
            draw_jacobian_vectors(viewer.user_scn, model, data, foot_site_id)
            viewer.sync()
            time.sleep(1.0 / 120.0)


def main():
    args = parse_args()
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    foot_site_id = model.site("foot").id
    results = [
        evaluate_pose(model, data, foot_site_id, name, q)
        for name, q in POSES.items()
    ]
    for result in results:
        print_result(result)

    print("\nSame qdot; different pose -> different J(q) -> different foot xdot.")
    if args.viewer:
        run_viewer(model, data, foot_site_id)


if __name__ == "__main__":
    main()
