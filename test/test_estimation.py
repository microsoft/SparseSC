"""
Tests for model fitness
"""

import unittest
import numpy as np
import random
import SparseSC as SC
from os.path import join, abspath, dirname
from dgp.factor_model import factor_dgp

# import matplotlib.pyplot as plt

exec(
    open(join(dirname(abspath(__file__)), "..", "examples", "example_graphs.py")).read()
)  # if we don't want an __init__.py


class TestDGPs(unittest.TestCase):
    def testSimpleTrendDGP(self):
        """
        No X, just Y; half the donors are great, other half are bad
        """
        N1, N0_sim, N0_not = 2, 50, 50
        N0 = N0_sim + N0_not
        N = N1 + N0
        treated_units = range(N1)
        control_units = range(N1, N)
        T0, T1 = 5, 5
        T = T0 + T1
        proto_sim = np.array(range(0, T, 1), ndmin=2)
        proto_not = np.array(range(0, 2 * T, 2), ndmin=2)
        te = np.hstack((np.zeros((1, T0)), np.full((1, T0), 2)))
        proto_tr = proto_sim + te
        Y1 = np.matmul(np.ones((N1, 1)), proto_tr)
        Y0_sim = np.matmul(np.ones((N0_sim, 1)), proto_sim)
        Y0_not = np.matmul(np.ones((N0_not, 1)), proto_not)
        Y = np.vstack((Y1, Y0_sim, Y0_not))
        ret_full = SC.estimate_effects(
            Y[:, :T0], Y[:, T0:], treated_units, ret_CI=True, max_n_pl=200
        )  # just getting V_pen
        V_penalty = ret_full.fit.V_penalty

        ret = SC.estimate_effects(
            Y[:, :T0],
            Y[:, T0:],
            treated_units,
            v_pen=[V_penalty],
            w_pen=0.00000000001,
            ret_CI=True,
        )
        Y_sc = ret.fit.predict(Y[control_units, :])
        te = (Y - Y_sc)[0:T0:]
        # weight_sums = np.sum(ret.fit.sc_weights, axis=1)
        # print(weight_sums[0])
        # print(np.mean(weight_sums[0]))

        # print(ret)
        assert 2 in ret.CI, "Confidence interval does not include the true effect"
        p_value = ret.p_value
        assert p_value is not None
        assert p_value < 0.001, "P-value is larger than expected"

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
        ) = factor_dgp(N0, N1, T0, T1, K, R, F)

        Cov = np.vstack((Cov_treated, Cov_control))
        Out_pre = np.vstack((Out_pre_treated, Out_pre_control))
        Out_post = np.vstack((Out_post_treated, Out_post_control))

        SC.estimate_effects(Out_pre, Out_post, treated_units, Cov)
        # print(fit_res)
        # est_res = SC.estimate_effects(
        #   Cov, Out_pre, Out_post, treated_units, V_penalty=0, W_penalty=0.001
        # )
        # print(est_res)

    # Simulations
    # 1) As T0 and N0 increases do
    ##a) SC match actuals in terms of the factor loadings
    ##b) our estimates look consistent and have good coverage
    ##c) Can we match a longer set of factor loadings
    # Other Counterfactual prediction:
    ## a) Compare to SC (big N0, small T0, then SC; or many factors; should do bad) to basic time-series model


if __name__ == "__main__":
    random.seed(12345)
    np.random.seed(10101)

    t = TestDGPs()
    t.testSimpleTrendDGP()
    # unittest.main()
