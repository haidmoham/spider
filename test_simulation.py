"""Regression checks for C-1N's deterministic simulation boundary."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np

from interact import build_simulation, execute, run_headless, shove_cases
from simulate import run
from simulation import FOOT_NAMES, JOINTS_PER_LEG, load_model, measured_state, neutral_foot_positions, neutral_targets, reset, set_targets, step
from standing import SupportAwareStanceController
from walk import GaitCoordinator, apply_gait_control


class SimulationCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = load_model()
        self.data = mujoco.MjData(self.model)

    def test_reset_is_deterministic_and_has_six_foot_positions(self) -> None:
        reset(self.model, self.data)
        first = measured_state(self.model, self.data)
        step(self.model, self.data)
        reset(self.model, self.data)
        second = measured_state(self.model, self.data)

        self.assertEqual(first, second)
        self.assertEqual(len(first.foot_positions), len(FOOT_NAMES))
        self.assertEqual(first.torso_position, (0.0, 0.0, 0.45))
        self.assertEqual(first.torso_orientation, (1.0, 0.0, 0.0, 0.0))
        self.assertEqual(first.torso_angular_velocity, (0.0, 0.0, 0.0))
        self.assertEqual(first.joint_positions, (0.0, -0.2, 1.1) * 6)
        self.assertEqual(first.joint_velocities, (0.0,) * 18)
        self.assertEqual(first.actuator_forces, (0.0,) * 18)
        self.assertEqual(first.foot_position_residual, (0.0,) * 18)
        self.assertEqual(first.foot_position_residual_norm, 0.0)
        self.assertEqual(first.joint_space_update_direction, (0.0,) * 18)
        self.assertEqual(first.joint_space_update_direction_norm, 0.0)
        self.assertEqual(first.foot_positions, neutral_foot_positions(self.model))

    def test_targets_are_commands_not_measured_state(self) -> None:
        reset(self.model, self.data)
        before = measured_state(self.model, self.data)
        set_targets(self.data, (0.0, -0.1, 1.0) * 6)
        self.assertEqual(measured_state(self.model, self.data), before)
        step(self.model, self.data)
        self.assertEqual(tuple(self.data.ctrl), (0.0, -0.1, 1.0) * 6)
        self.assertGreater(measured_state(self.model, self.data).time, 0.0)

    def test_static_and_legacy_shuffle_failure_signatures_are_reproducible(self) -> None:
        """Protect current physics evidence without treating it as a capability claim."""
        reset(self.model, self.data)
        for _ in range(round(1.0 / self.model.opt.timestep)):
            step(self.model, self.data, neutral_targets())
        self.assertAlmostEqual(self.data.qpos[2], 0.45237846181248814, places=10)
        self.assertAlmostEqual(self.data.qvel[2], -6.674825905960546e-05, places=10)

        reset(self.model, self.data)
        shuffle = GaitCoordinator(self.model, self.data)
        for _ in range(round(2.0 / self.model.opt.timestep)):
            targets, _, _ = apply_gait_control(shuffle, self.data)
            step(self.model, self.data, targets)
        self.assertAlmostEqual(self.data.qpos[0], -1.1185716166355097e-14, places=10)
        self.assertAlmostEqual(self.data.qpos[1], -0.16214037173688392, places=10)
        self.assertAlmostEqual(self.data.qpos[2], 0.5765503932250885, places=10)

    def test_static_and_live_surfaces_share_the_neutral_reset(self) -> None:
        static_model, _ = run(0.002)
        live_model, live_data, power, coordinator, controller, perturbation = build_simulation("none")
        static_initial = mujoco.MjData(static_model)
        reset(static_model, static_initial)
        self.assertEqual(measured_state(static_model, static_initial), measured_state(live_model, live_data))

        response = execute({"command": "reset"}, live_model, live_data, power, "none", coordinator, controller, perturbation)
        self.assertEqual(response["torso_position"], [0.0, 0.0, 0.45])
        self.assertEqual(response["torso_orientation"], [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(response["torso_angular_velocity"], [0.0, 0.0, 0.0])
        self.assertEqual(response["joint_velocities"], [0.0] * 18)
        self.assertEqual(response["actuator_forces"], [0.0] * 18)
        self.assertEqual(response["controls"], [0.0, -0.2, 1.1] * 6)

    def test_interact_state_reports_live_measured_telemetry(self) -> None:
        model, data, power, coordinator, controller, perturbation = build_simulation("none")

        initial = execute({"command": "state"}, model, data, power, "none", coordinator, controller, perturbation)
        observed_initial = measured_state(model, data)
        self.assertEqual(initial["joint_positions"], list(observed_initial.joint_positions))
        self.assertEqual(initial["joint_velocities"], list(observed_initial.joint_velocities))
        self.assertEqual(initial["actuator_forces"], list(observed_initial.actuator_forces))
        self.assertEqual(initial["foot_contacts"], list(observed_initial.foot_contacts))
        self.assertEqual(initial["controls"], data.ctrl.tolist())
        self.assertEqual(set(initial["legs"]), set(FOOT_NAMES))

        stepped = execute({"command": "step", "n": 2}, model, data, power, "none", coordinator, controller, perturbation)
        observed_stepped = measured_state(model, data)
        self.assertEqual(stepped["time"], observed_stepped.time)
        self.assertEqual(stepped["torso_position"], list(observed_stepped.torso_position))
        self.assertEqual(stepped["torso_orientation"], list(observed_stepped.torso_orientation))
        self.assertEqual(stepped["torso_velocity"], list(observed_stepped.torso_velocity))
        self.assertEqual(stepped["torso_angular_velocity"], list(observed_stepped.torso_angular_velocity))
        self.assertEqual(stepped["joint_positions"], list(observed_stepped.joint_positions))
        self.assertEqual(stepped["joint_velocities"], list(observed_stepped.joint_velocities))
        self.assertEqual(stepped["actuator_forces"], list(observed_stepped.actuator_forces))
        self.assertEqual(stepped["foot_positions"], {name: list(position) for name, position in observed_stepped.foot_positions.items()})
        self.assertEqual(stepped["foot_contacts"], list(observed_stepped.foot_contacts))
        self.assertEqual(stepped["whole_stance_kinematics"]["foot_position_residual"], list(observed_stepped.foot_position_residual))
        self.assertEqual(stepped["whole_stance_kinematics"]["foot_position_residual_norm_m"], observed_stepped.foot_position_residual_norm)
        self.assertEqual(stepped["whole_stance_kinematics"]["joint_space_update_direction"], list(observed_stepped.joint_space_update_direction))
        self.assertEqual(stepped["whole_stance_kinematics"]["joint_space_update_direction_norm"], observed_stepped.joint_space_update_direction_norm)
        self.assertEqual(stepped["controls"], data.ctrl.tolist())
        self.assertTrue(any(value != 0.0 for value in stepped["joint_velocities"]))
        self.assertTrue(any(value != 0.0 for value in stepped["actuator_forces"]))
        for index, name in enumerate(FOOT_NAMES):
            leg = stepped["legs"][name]
            self.assertEqual(leg["foot_position"], list(observed_stepped.foot_positions[name]))
            self.assertEqual(leg["in_ground_contact"], name in observed_stepped.foot_contacts)
            for offset, joint in enumerate(("coxa", "hip", "knee")):
                joint_state = leg["joints"][joint]
                actuator = index * JOINTS_PER_LEG + offset
                self.assertEqual(joint_state["target"], data.ctrl[actuator])
                self.assertEqual(joint_state["position"], observed_stepped.joint_positions[actuator])
                self.assertEqual(joint_state["velocity"], observed_stepped.joint_velocities[actuator])
                self.assertEqual(joint_state["actuator_force"], observed_stepped.actuator_forces[actuator])

    def test_live_surface_schedules_and_clears_a_torso_force_pulse(self) -> None:
        model, data, power, coordinator, controller, perturbation = build_simulation("none")
        scheduled = execute(
            {"command": "perturb", "force_n": [1.0, 0.0, 0.0], "seconds": 0.01},
            model,
            data,
            power,
            "none",
            coordinator,
            controller,
            perturbation,
        )
        self.assertEqual(scheduled["perturbation"]["force_n"], [1.0, 0.0, 0.0])
        self.assertEqual(scheduled["perturbation"]["remaining_steps"], 5)

        reset_state = execute({"command": "reset"}, model, data, power, "none", coordinator, controller, perturbation)
        self.assertEqual(reset_state["perturbation"]["force_n"], [0.0, 0.0, 0.0])
        self.assertEqual(reset_state["perturbation"]["remaining_steps"], 0)

    def test_standing_controller_yields_to_control_panel_and_resumes_after_clear(self) -> None:
        reset(self.model, self.data)
        controller = SupportAwareStanceController()
        automatic = controller.targets(self.model, self.data)
        set_targets(self.data, automatic)

        manual = list(automatic)
        manual[0] = 0.2
        set_targets(self.data, manual)
        self.assertEqual(controller.targets(self.model, self.data), tuple(manual))
        self.assertTrue(controller.last_telemetry.manual_override)

        set_targets(self.data, (0.0,) * len(self.data.ctrl))
        resumed = controller.targets(self.model, self.data)
        self.assertFalse(controller.last_telemetry.manual_override)
        self.assertNotEqual(resumed, (0.0,) * len(self.data.ctrl))

    def test_headless_rollout_trace_contains_compact_telemetry_v1_samples(self) -> None:
        with TemporaryDirectory() as directory:
            trace_path = Path(directory) / "stand.npz"
            final_state = run_headless(0.004, "stand", trace_path)
            with np.load(trace_path) as trace:
                metadata = json.loads(str(trace["metadata_json"]))
                samples = trace["time_s"]
                force = trace["force_along_shove_n"]
                margin = trace["support_margin_m"]

        self.assertEqual(metadata["telemetry_standard"], "Telemetry v1")
        self.assertEqual(metadata["c1n_iteration"], "v0.11")
        self.assertEqual(metadata["experiment"], "stand")
        self.assertEqual(samples[0], 0.0)
        self.assertEqual(len(samples), 1)
        self.assertEqual(force[0], 0.0)
        self.assertTrue(np.isfinite(margin[0]))
        self.assertEqual(final_state["time"], 0.004)

    def test_shove_grid_has_one_baseline_then_eight_directions_per_force(self) -> None:
        cases = shove_cases(self.model)

        self.assertEqual(len(cases), 33)
        self.assertEqual({metadata["direction_label"] for _, _, metadata in cases}, {f"{angle:03d}deg" for angle in range(0, 360, 45)})
        self.assertEqual({metadata["label"] for _, _, metadata in cases}, {"0mg", "0.25mg", "0.5mg", "0.75mg", "1mg"})
        self.assertEqual(sum(metadata["case_role"] == "control" for _, _, metadata in cases), 1)
        self.assertEqual(sum(metadata["case_role"] == "treatment" for _, _, metadata in cases), 32)
        self.assertEqual([metadata["label"] for _, _, metadata in cases], ["0mg"] + [f"{multiple:g}mg" for multiple in (0.25, 0.5, 0.75, 1.0) for _ in range(8)])


if __name__ == "__main__":
    unittest.main()
