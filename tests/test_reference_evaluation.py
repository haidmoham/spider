import copy
import unittest

try:
    from spider.reference_evaluation import summarize
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    raise unittest.SkipTest("PyTorch is not installed")


class ReferenceEvaluationTests(unittest.TestCase):
    def records(self):
        return [dict(mode=mode, seed=seed, seconds=5., speed_m_s=.4,
                     terminated=False, max_joint_limit_violation_rad=0.)
                for mode, seed in [("mean", 201), *[("sampled", seed) for seed in range(201, 213)]]]

    def test_numeric_success_still_requires_visual_review(self):
        result = summarize(self.records(), .37882745)
        self.assertTrue(result["goal_numerical_pass"])
        self.assertFalse(result["accepted"])
        self.assertTrue(result["visual_review_required"])

    def test_notebook_protocol_requires_all_mean_seeds(self):
        rows = self.records()
        self.assertFalse(summarize(rows, .37882745, notebook_seeds=True)["goal_numerical_pass"])
        rows += [dict(rows[0], seed=seed) for seed in range(202, 213)]
        self.assertTrue(summarize(rows, .37882745, notebook_seeds=True)["goal_numerical_pass"])
        rows[-1]["terminated"] = True
        self.assertFalse(summarize(rows, .37882745, notebook_seeds=True)["goal_numerical_pass"])

    def test_rejects_missing_duplicate_slow_fallen_invalid_records(self):
        original = self.records()
        candidates = [original[:-1]]
        for field, value in [("seed", 201), ("speed_m_s", 0.), ("terminated", True),
                             ("seconds", 1.), ("max_joint_limit_violation_rad", .02),
                             ("seconds", float("nan"))]:
            rows = copy.deepcopy(original)
            rows[-1][field] = value
            candidates.append(rows)
        for rows in candidates:
            with self.subTest(last=rows[-1]):
                self.assertFalse(summarize(rows, .37882745)["goal_numerical_pass"])


if __name__ == "__main__":
    unittest.main()
