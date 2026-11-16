"""Regression tests for fast fitting model selection."""
import unittest
from unittest.mock import Mock, patch

import numpy as np

from SparseSC.fit_fast import fit_fast, _fit_fast_inner, _fit_fast_match
from SparseSC.utils.match_space import Fixed_V_factory
from SparseSC.utils.misc import _get_fit_units
from SparseSC.utils.penalty_utils import RidgeCVSolution


class TestFitFastModelTypes(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1., 2.], [2., 1.], [3., 4.], [4., 3.]])
        self.Y = np.array([[2., 3.], [3., 2.], [5., 7.], [20., 30.]])
        self.V = np.ones(2)
        self.controls = [0, 1, 2]
        self.treated = [3]

    def test_fit_units_for_each_model_type(self):
        for model_type, expected in [
            ("retrospective", self.controls),
            ("prospective", list(range(4))),
            ("prospective-restricted", self.treated),
            ("full", list(range(4))),
        ]:
            with self.subTest(model_type=model_type):
                self.assertEqual(
                    list(_get_fit_units(model_type, self.controls, self.treated, 4)),
                    expected,
                )

    def test_default_matches_explicit_retrospective(self):
        maker = Mock(wraps=Fixed_V_factory(self.V))
        kwargs = dict(treated_units=self.treated, w_pens=[1.],
                      w_pen_inner=False, match_space_maker=maker)
        default = fit_fast(self.X, self.Y, **kwargs)
        np.testing.assert_array_equal(maker.call_args[0][0], self.X[self.controls])
        np.testing.assert_array_equal(maker.call_args[0][1], self.Y[self.controls])
        explicit = fit_fast(self.X, self.Y, model_type="retrospective", **kwargs)
        self.assertEqual(default.model_type, "retrospective")
        self.assertEqual(default.score, explicit.score)
        self.assertEqual(default.fitted_w_pen, explicit.fitted_w_pen)
        np.testing.assert_allclose(default.sc_weights, explicit.sc_weights)

    def test_helper_defaults_match_retrospective(self):
        for fn, extra in [(_fit_fast_inner, dict(w_pens=[1.], w_pen_inner=False)),
                          (_fit_fast_match, dict(best_w_pen=1.))]:
            with self.subTest(function=fn.__name__):
                kwargs = dict(treated_units=self.treated, **extra)
                default = fn(self.X, self.X, self.Y, self.V, **kwargs)
                explicit = fn(self.X, self.X, self.Y, self.V,
                              model_type="retrospective", **kwargs)
                self.assertEqual(default.model_type, "retrospective")
                self.assertEqual(default.score, explicit.score)
                np.testing.assert_allclose(default.sc_weights, explicit.sc_weights)

    def test_restricted_model_fits_and_scores_only_treated_units(self):
        maker = Mock(wraps=Fixed_V_factory(self.V))
        with patch("SparseSC.fit_fast.RidgeCVSolution", wraps=RidgeCVSolution) as ridge:
            result = fit_fast(
                self.X, self.Y, model_type="prospective-restricted",
                treated_units=self.treated, match_space_maker=maker, w_pens=[0.1, 1.],
            )
        np.testing.assert_array_equal(maker.call_args[0][0], self.X[self.treated])
        np.testing.assert_array_equal(maker.call_args[0][1], self.Y[self.treated])
        self.assertFalse(ridge.call_args[0][2])
        self.assertEqual(ridge.call_args[0][3], self.treated)
        residuals = self.Y - result.predict()
        self.assertAlmostEqual(result.score, np.square(residuals[self.treated]).sum())


if __name__ == "__main__":
    unittest.main()
