"""Shared replay timing and layout, without a window or simulation."""
import unittest
import numpy as np

from spider.viewing.replay_grid import frame_index, quadrants


class ReplayGridTests(unittest.TestCase):
    def test_shared_time_and_short_episode_hold(self):
        long = np.array([10., 10.02, 10.04, 10.06])
        short = np.array([0., .02])
        self.assertEqual(frame_index(long, .03), 1)
        self.assertEqual(frame_index(short, .03), 1)
        self.assertEqual(frame_index(long, .06), 3)
        self.assertEqual(frame_index(short, 10.), 1)
        self.assertEqual(frame_index(np.array([2.]), 1.), 0)

    def test_quadrants_stay_separate_on_odd_dimensions(self):
        rectangles = quadrants(1441, 961)
        self.assertGreater(rectangles[0][1], rectangles[2][1])
        self.assertLess(rectangles[0][0], rectangles[1][0])
        for i, (x, y, w, h) in enumerate(rectangles):
            self.assertGreater(w, 0)
            self.assertGreater(h, 0)
            self.assertLessEqual(x+w, 1441)
            self.assertLessEqual(y+h, 961)
            for a, b, c, d in rectangles[i+1:]:
                self.assertTrue(x+w <= a or a+c <= x or y+h <= b or b+d <= y)


if __name__ == '__main__':
    unittest.main()
