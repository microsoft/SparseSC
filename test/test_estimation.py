"""
Tests for model fitness
"""

import unittest
import numpy as np
try:
    import SparseSC as SC
except ImportError:
    raise RuntimeError("SparseSC is not installed. use 'pip install -e .' to install")
from os.path import join, abspath, dirname
from dgp.factor_model import factor_dgp

# import matplotlib.pyplot as plt

exec(# pylint: disable=exec-used
    open(join(dirname(abspath(__file__)), "..", "examples", "example_graphs.py")).read()
)  # if we don't want an __init__.py


# pylint: disable=no-self-use


class TestDGPs(unittest.TestCase):
    """
    testing fixture
    """

    def testSimpleTrendDGP(self):
        """
        No X, just Y; half the donors are great, other half are bad
        """
        N1, N0_sim, N0_not = 1, 50, 50
        N0 = N0_sim + N0_not
        N = N1 + N0
        treated_units = range(N1)
        control_units = range(N1, N)
        T0, T1 = 2, 1
        T = T0 + T1
        proto_sim = np.array([1,0]+[0], ndmin=2)
        proto_not = np.array([0,1]+[1], ndmin=2)
        #proto_not[0,2] += 1
        te = 2
        te_vec = np.hstack((np.zeros((1, T0)), np.full((1, T1), te)))
        proto_tr = proto_sim + te_vec
        Y1 = np.matmul(np.ones((N1, 1)), proto_tr)
        Y0_sim = np.matmul(np.ones((N0_sim, 1)), proto_sim)
        Y0_not = np.matmul(np.ones((N0_not, 1)), proto_not)
        Y = np.vstack((Y1, Y0_sim, Y0_not))

        def simple_summ(fit, Y):
            print("V_pen=%s, W_pen=%s" % (fit.fitted_v_pen, fit.fitted_w_pen))
            print("V=%s" % np.diag(fit.V))
            print("Treated weights: sim=%s, uns=%s, sum=%s" % (fit.sc_weights[0,49], fit.sc_weights[0,99], sum(fit.sc_weights[0,:])))
            print("Sim Con weights: sim=%s, uns=%s, sum=%s" % (fit.sc_weights[1,49], fit.sc_weights[1,99], sum(fit.sc_weights[1,:])))
            print("Uns Con weights: sim=%s, uns=%s, sum=%s" % (fit.sc_weights[51,49], fit.sc_weights[51,99], sum(fit.sc_weights[51,:])))
            Y_sc = fit.predict(Y[fit.control_units, :])
            print("Treated diff: %s" % (Y - Y_sc)[0,:])

        #Y += np.random.normal(0, 0.01, Y.shape)

        # OPTIMIZE OVER THE V_PEN'S
        #for v_pen, w_pen in [(1,1), (1,1e-10), (1e-10,1e-10), (1e-10,1), (None, None)]: #
        #print("\nv_pen=%s, w_pen=%s" % (v_pen, w_pen))
        ret = SC.estimate_effects(
            Y[:, :T0],
            Y[:, T0:],
            treated_units,
            ret_CI=True,
            max_n_pl=200,
            progress = False,
            stopping_rule=4,
            constrain="simplex",
            #v_pen = v_pen, w_pen=w_pen
        ) 
        simple_summ(ret.fit, Y)
        V_penalty = ret.fit.fitted_v_pen

        Y_sc = ret.fit.predict(Y[control_units, :])
        te_vec_est = (Y - Y_sc)[0:T0:]
        # weight_sums = np.sum(ret.fit.sc_weights, axis=1)

        #print(ret.fit.scores)
        p_value = ret.p_value
        #print("p-value: %s" % p_value)
        #print( ret.CI)
        #print(np.diag(ret.fit.V))
        #import pdb; pdb.set_trace()
        # print(ret)
        assert te in ret.CI, "Confidence interval does not include the true effect"
        assert p_value is not None
        import pdb; pdb.set_trace()
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
        ) = factor_dgp(N0, N1, T0, T1, K, R, F)

        Cov = np.vstack((Cov_treated, Cov_control))
        Out_pre = np.vstack((Out_pre_treated, Out_pre_control))
        Out_post = np.vstack((Out_post_treated, Out_post_control))

        SC.estimate_effects(Out_pre, Out_post, treated_units, Cov,constrain="simplex" )
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
    import random

    random.seed(12345)
    np.random.seed(10101)

    t = TestDGPs()
    t.testSimpleTrendDGP()
    # unittest.main()
