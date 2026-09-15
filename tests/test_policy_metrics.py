"""Distinguish observed foot swings from contact flicker and incomplete swings."""

import unittest

from spider.policy_metrics import episode_metrics
from spider.simulation import FOOT_NAMES


class PolicyMetricsTests(unittest.TestCase):
    def test_clearance_and_complete_swings_use_observed_contacts(self):
        rows = []
        for index, contact in enumerate((1, 0, 0, 1, 0)):
            rows.append(dict(
                time_s=(index + 1) * .04, forward_m=0., height_m=.43,
                stance_samples=0, stance_speed_sq_sum=0.,
                action_delta_mean_sq_rad2=0., vertical_velocity_m_s=0.,
                **{f"contact_{name}": contact for name in FOOT_NAMES},
                **{f"foot_height_{name}_m": .045 if contact else .105 for name in FOOT_NAMES},
                **{f"joint_{joint}_rad": index * .1 for joint in range(18)},
            ))
        result = episode_metrics(rows)
        for name in FOOT_NAMES:
            self.assertEqual(result["liftoff_count"][name], 2)
            self.assertAlmostEqual(result["mean_complete_swing_s"][name], .08)
            self.assertAlmostEqual(result["maximum_foot_clearance_m"][name], .06)
        self.assertIsNone(result["stance_foot_speed_rms_m_s"])
        self.assertAlmostEqual(result["joint_range_of_motion_rad"][0], .4)
