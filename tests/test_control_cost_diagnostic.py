"""Check command meanings independently of any reward coefficient or trainer."""
import importlib.util
import unittest
import numpy as np

HAS_PANDAS = importlib.util.find_spec('pandas') is not None
if HAS_PANDAS:
    from lab.control_cost_diagnostic import command_quantities


@unittest.skipUnless(HAS_PANDAS, 'Optional learning environment')
class ControlQuantityTests(unittest.TestCase):
    def test_constant_offset_has_size_but_no_later_change(self):
        offsets = np.array([[.1, -.2], [.1, -.2]])
        values = command_quantities(offsets, offsets + [0, 1])
        np.testing.assert_allclose(values['offset_sq_rad2'], [.05, .05])
        np.testing.assert_allclose(values['action_change_sq_rad2'], [.05, 0])
        np.testing.assert_allclose(values['absolute_target_sq_rad2'], [.65, .65])

    def test_neutral_target_is_not_zero_absolute_target(self):
        offsets = np.zeros((2, 18))
        targets = np.tile([0, -.2, 1.1], (2, 6))
        values = command_quantities(offsets, targets)
        np.testing.assert_allclose(values['offset_sq_rad2'], 0)
        np.testing.assert_allclose(values['action_change_sq_rad2'], 0)
        np.testing.assert_allclose(values['absolute_target_sq_rad2'], 7.5)
