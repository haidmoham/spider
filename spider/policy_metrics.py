"""Motion-quality measurements shared by platform evaluation and tuning."""

import numpy as np

from .simulation import FOOT_NAMES


def transition_metrics(before, after, targets, previous_targets, dt):
    """Measure commands and stance-foot displacement over one control interval.

    Slip is an endpoint estimate for feet contacting at both ends. It cannot
    resolve sliding within the interval or brief touchdown/liftoff events.
    """
    target_change = np.asarray(targets) - np.asarray(previous_targets)
    supported = set(before.foot_contacts) & set(after.foot_contacts)
    squared_speeds = [float(np.square(
        (np.asarray(after.foot_positions[name])[:2] - np.asarray(before.foot_positions[name])[:2]) / dt
    ).sum()) for name in supported]
    return dict(
        action_delta_mean_sq_rad2=float(np.square(target_change).mean()),
        stance_speed_sq_sum=float(sum(squared_speeds)),
        stance_samples=len(squared_speeds),
        vertical_velocity_m_s=float(after.torso_velocity[2]),
        **{f"contact_{name}": int(name in after.foot_contacts) for name in FOOT_NAMES},
        **{f"foot_height_{name}_m": float(after.foot_positions[name][2]) for name in FOOT_NAMES},
    )


def episode_metrics(rows):
    """Summarize one recording; never infer gait quality from these alone."""
    steady = [row for row in rows if row["time_s"] >= 2.] or rows
    stance_count = sum(row["stance_samples"] for row in rows)
    result = dict(
        speed_m_s=float(rows[-1]["forward_m"] / rows[-1]["time_s"]),
        action_delta_rms_rad=float(np.sqrt(np.mean([row["action_delta_mean_sq_rad2"] for row in rows]))),
        stance_foot_speed_rms_m_s=float(np.sqrt(sum(row["stance_speed_sq_sum"] for row in rows) / stance_count))
            if stance_count else None,
        steady_height_mean_m=float(np.mean([row["height_m"] for row in steady])),
        steady_height_std_m=float(np.std([row["height_m"] for row in steady])),
        vertical_velocity_rms_m_s=float(np.sqrt(np.mean([row["vertical_velocity_m_s"] ** 2 for row in rows]))),
        contact_duty={name: float(np.mean([row[f"contact_{name}"] for row in rows])) for name in FOOT_NAMES},
    )
    # Heights refer to sphere centres. Subtract the canonical 45 mm foot radius
    # when reporting visible ground clearance. This is measured, not a reward.
    result["maximum_foot_clearance_m"] = {
        name: max(0., max(row[f"foot_height_{name}_m"] for row in rows) - .045)
        for name in FOOT_NAMES
    }
    result["liftoff_count"] = {
        name: sum(bool(before[f"contact_{name}"]) and not after[f"contact_{name}"]
                  for before, after in zip(rows, rows[1:]))
        for name in FOOT_NAMES
    }
    # Count only fully observed swings, so episode truncation cannot inflate
    # the estimated duration. Contact sampling can still miss brief events.
    durations = {name: [] for name in FOOT_NAMES}
    for name in FOOT_NAMES:
        started = None
        for before, after in zip(rows, rows[1:]):
            if before[f"contact_{name}"] and not after[f"contact_{name}"]:
                started = after["time_s"]
            elif not before[f"contact_{name}"] and after[f"contact_{name}"] and started is not None:
                durations[name].append(after["time_s"] - started)
                started = None
    result["mean_complete_swing_s"] = {
        name: float(np.mean(values)) if values else None for name, values in durations.items()
    }
    if all(f"joint_{index}_rad" in row for row in rows for index in range(18)):
        result["joint_range_of_motion_rad"] = [
            max(row[f"joint_{index}_rad"] for row in rows)
            - min(row[f"joint_{index}_rad"] for row in rows) for index in range(18)
        ]
    return result
