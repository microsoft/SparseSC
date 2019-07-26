"""
Tests for model fitness
"""

import unittest
import random
import numpy as np
import pandas as pd

#import warnings
#warnings.simplefilter("error")

try:
    import SparseSC as SC
except ImportError:
    raise RuntimeError("SparseSC is not installed. use 'pip install -e .' from repo root to install in dev mode")
from os.path import join, abspath, dirname
from dgp.factor_model import factor_dgp

# import matplotlib.pyplot as plt

import sys
sys.path.insert(0, join(dirname(abspath(__file__)), "..", "examples"))
from example_graphs import *


# pylint: disable=no-self-use, missing-docstring

# here for lexical scoping
command_line_options = {}

class TestEstimationForErrors(unittest.TestCase):
    def setUp(self):

        random.seed(12345)
        np.random.seed(101101001)
        control_units = 50
        treated_units = 2
        N = control_units + treated_units
        T = 15
        K_X = 2

        self.Y = np.random.rand(N, T)
        self.X = np.random.rand(N, K_X)
        self.treated_units = np.arange(treated_units)
        self.unit_treatment_periods = np.full((N), np.nan)
        self.unit_treatment_periods[0] = 7
        self.unit_treatment_periods[1] = 8
        #self.
        #self.unit_treatment_periods[treated_name] = treatment_date_ms

    @classmethod
    def run_test(cls, obj, model_type="retrospective", frame_type="ndarray"): #"NDFrame", "pandas_timeindex", NDFrame
        X = obj.X
        Y = obj.Y
        unit_treatment_periods = obj.unit_treatment_periods
        if frame_type=="NDFrame" or frame_type=="timeindex":
            X = pd.DataFrame(X)
            Y = pd.DataFrame(Y)
            if frame_type=="timeindex":
                t_index = pd.Index(np.datetime64('2000-01-01','D') + range(Y.shape[1]))
                unit_treatment_periods = pd.Series(np.datetime64('NaT'), index=Y.index)
                unit_treatment_periods[0] = t_index[7]
                unit_treatment_periods[1] = t_index[8]
                Y.columns = t_index

        SC.estimate_effects(X=X, Y=Y, model_type=model_type, unit_treatment_periods=unit_treatment_periods)

    def test_all(self): #RidgeCV returns: RuntimeWarning: invalid value encountered in true_divide \n return (c / G_diag) ** 2, c
        for model_type in ["retrospective", "prospective", "prospective-restricted"]:
            for frame_type in ["ndarray", "NDFrame", "timeindex"]:
                TestEstimationForErrors.run_test(self, model_type, frame_type)

class TestDGPs(unittest.TestCase):
    """
    testing fixture
    """
    @staticmethod
    def simple_summ(fit, Y):
        #print("V_pen=%s, W_pen=%s" % (fit.fitted_v_pen, fit.fitted_w_pen))
        if fit.match_space_desc is not None:
            print(fit.match_space_desc)
        else:
            print("V=%s" % np.diag(fit.V))
        print("Treated weights: sim=%s, uns=%s, sum=%s" % ( fit.sc_weights[0, 49], fit.sc_weights[0, 99], sum(fit.sc_weights[0, :]),))
        print("Sim Con weights: sim=%s, uns=%s, sum=%s" % ( fit.sc_weights[1, 49], fit.sc_weights[1, 99], sum(fit.sc_weights[1, :]),))
        print("Uns Con weights: sim=%s, uns=%s, sum=%s" % ( fit.sc_weights[51, 49], fit.sc_weights[51, 99], sum(fit.sc_weights[51, :]),))
        Y_sc = fit.predict(Y)
        print("Treated diff: %s" % (Y - Y_sc)[0, :])


    def testSimpleTrendDGP(self):
        """
        No X, just Y; half the donors are great, other half are bad
        """
        N1, N0_sim, N0_not = 1, 50, 50
        N0 = N0_sim + N0_not
        N = N1 + N0
        treated_units, control_units  = range(N1), range(N1, N)
        T0, T1 = 5, 2
        T = T0 + T1 # unused
        proto_sim = np.array([1, 2, 3, 4, 5] + [6,7], ndmin=2)
        proto_not = np.array([0, 2, 4, 6, 8] + [10, 12], ndmin=2)
        te = 2
        proto_tr = proto_sim + np.hstack((np.zeros((1, T0)), np.full((1, T1), te)))
        Y1 = np.matmul(np.ones((N1, 1)), proto_tr)
        Y0_sim = np.matmul(np.ones((N0_sim, 1)), proto_sim)
        Y0_sim = Y0_sim + np.random.normal(0,0.1,Y0_sim.shape)
        #Y0_sim = Y0_sim + np.hstack((np.zeros((N0_sim,1)), 
        #                             np.random.normal(0,0.1,(N0_sim,1)),
        #                             np.zeros((N0_sim,T-2))))
        Y0_not = np.matmul(np.ones((N0_not, 1)), proto_not)
        Y0_not = Y0_not + np.random.normal(0,0.1,Y0_not.shape)
        Y = np.vstack((Y1, Y0_sim, Y0_not))

        unit_treatment_periods = np.full((N), -1)
        unit_treatment_periods[0] = T0

        # Y += np.random.normal(0, 0.01, Y.shape)

        # OPTIMIZE OVER THE V_PEN'S
        # for v_pen, w_pen in [(1,1), (1,1e-10), (1e-10,1e-10), (1e-10,1), (None, None)]: #
        # print("\nv_pen=%s, w_pen=%s" % (v_pen, w_pen))
        ret = SC.estimate_effects(
            Y,
            unit_treatment_periods,
            ret_CI=True,
            max_n_pl=200,
            fast = True,
            #stopping_rule=4,
            **command_line_options,
        )
        TestDGPs.simple_summ(ret.fits[T0], Y)
        V_penalty = ret.fits[T0].fitted_v_pen

        Y_sc = ret.fits[T0].predict(Y)# [control_units, :]
        te_vec_est = (Y - Y_sc)[0:T0:]
        # weight_sums = np.sum(ret.fit.sc_weights, axis=1)

        # print(ret.fit.scores)
        p_value = ret.p_value
        #print("p-value: %s" % p_value)
        #print( ret.CI)
        #print(np.diag(ret.fit.V))
        #import pdb; pdb.set_trace()
        # print(ret)
        assert te in ret.CI, "Confidence interval does not include the true effect"
        assert p_value is not None
        assert p_value < 0.1, "P-value is larger than expected"

        # [sc_raw, sc_diff] = ind_sc_plots(Y[0, :], Y_sc[0, :], T0, ind_ci=ret.ind_CI)
        # plt.figure("sc_raw")
        # plt.title("Unit 0")
        # ### SHOW() blocks!!!!
        # # plt.show()
        # plt.figure("sc_diff")
        # plt.title("Unit 0")
        # # plt.show()
        # [te] = te_plot(ret)
        # plt.figure("te")
        # plt.title("Average Treatment Effect")
        # # plt.show()

    def testFactorDGP(self):
        """
        factor dbp based test
        """
        N1, N0 = 2, 100
        treated_units = [0, 1]
        T0, T1 = 20, 10
        K, R, F = 5, 5, 5
        (
            Cov_control,
            Cov_treated,
            Out_pre_control,
            Out_pre_treated,
            Out_post_control,
            Out_post_treated,
            _,
            _,
        ) = factor_dgp(N0, N1, T0, T1, K, R, F)

        Cov = np.vstack((Cov_treated, Cov_control))
        Out_pre = np.vstack((Out_pre_treated, Out_pre_control))
        Out_post = np.vstack((Out_post_treated, Out_post_control))

        SC.estimate_effects(
            Out_pre,
            Out_post,
            treated_units,
            Cov,
            # constrain="simplex", -- handled by argparse now..
            **command_line_options,
        )

        # print(fit_res)
        # est_res = SC.estimate_effects(
        #   Cov, Out_pre, Out_post, treated_units, V_penalty=0, W_penalty=0.001
        # )
        # print(est_res)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog="PROG", allow_abbrev=False)
    parser.add_argument(
        "--constrain", choices=["orthant", "simplex"], default="orthant"
    )
    args = parser.parse_args()
    command_line_options.update(vars(args))

    random.seed(12345)
    np.random.seed(10101)

    t = TestEstimationForErrors()
    t.setUp()
    t.test_all()
    # unittest.main()
