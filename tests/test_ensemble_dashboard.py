import unittest

import numpy as np

from evaluation.ensemble_dashboard import _average_simplex_actions, select_ensemble_members


class EnsembleDashboardTest(unittest.TestCase):
    def test_average_simplex_actions_returns_normalized_action(self):
        actions = [
            np.array([0.2, 0.8, 0.0]),
            np.array([0.4, 0.1, 0.5]),
        ]

        averaged = _average_simplex_actions(actions)

        np.testing.assert_allclose(averaged, np.array([0.3, 0.45, 0.25]), atol=1e-8)
        self.assertAlmostEqual(float(averaged.sum()), 1.0)

    def test_average_simplex_actions_uses_member_weights(self):
        actions = [
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]

        averaged = _average_simplex_actions(actions, weights=[0.75, 0.25])

        np.testing.assert_allclose(averaged, np.array([0.0, 0.75, 0.25]), atol=1e-8)

    def test_select_ensemble_members_supports_equal_weighting(self):
        members = [
            {"validation_score": 10.0, "path": "a"},
            {"validation_score": 2.0, "path": "b"},
            {"validation_score": 1.0, "path": "c"},
        ]

        selected = select_ensemble_members(members, top_k=2, weighting="equal")

        self.assertEqual(["a", "b"], [member["path"] for member in selected])
        np.testing.assert_allclose([member["ensemble_weight"] for member in selected], [0.5, 0.5])

    def test_select_ensemble_members_rejects_unknown_weighting(self):
        members = [{"validation_score": 1.0, "path": "a"}]

        with self.assertRaises(ValueError):
            select_ensemble_members(members, weighting="unsupported")


if __name__ == "__main__":
    unittest.main()
