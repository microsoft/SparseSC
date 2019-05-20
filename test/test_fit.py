# --------------------------------------------------------------------------------
# Programmer: Jason Thorpe
# Date    1/25/2019 3:34:02 PM
# Language:  Python (.py) Version 2.7 or 3.5
# Usage:
#
# Test all model types
#
#     \SpasrseSC > python -m unittest test/test_fit.py
#
# Test a specific model type (e.g. "prospective-restricted"):
#
#     \SpasrseSC > python -m unittest test.test_fit.TestFit.test_retrospective
#
# --------------------------------------------------------------------------------

from __future__ import print_function  # for compatibility with python 2.7
import sys, os, random
import unittest
import warnings
from scipy.optimize.linesearch import LineSearchWarning
import numpy as np
import traceback
import pdb 

try:
    import SparseSC as SC
except ImportError:
    raise RuntimeError("SparseSC is not installed. use 'pip install -e .' from repo root to install in dev mode")
from SparseSC.fit import fit


class TestFit(unittest.TestCase):
    def setUp(self):

        random.seed(12345)
        np.random.seed(101101001)
        control_units = 50
        treated_units = 20
        features = 10
        targets = 5

        self.X = np.random.rand(control_units + treated_units, features)
        self.Y = np.random.rand(control_units + treated_units, targets)
        self.treated_units = np.arange(treated_units)

    @classmethod
    def run_test(cls, obj, model_type, verbose=False):
        """
        main test runner
        """
        if verbose:
            print("Calling fit with `model_type  = '%s'`..." % (model_type,), end="")
        sys.stdout.flush()

        # Catch the LineSearchWarning silently, but allow others
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",category=PendingDeprecationWarning)
            warnings.filterwarnings("ignore",category=LineSearchWarning)
            try:
                fit(
                    X=obj.X,
                    Y=obj.Y,
                    model_type=model_type,
                    treated_units=obj.treated_units
                    if model_type
                    in ("retrospective", "prospective", "prospective-restricted")
                    else None,
                    # KWARGS:
                    print_path=False,
                    stopping_rule=1,
                    progress=verbose,
                    grid_length=5,
                    min_iter=-1,
                    tol=1,
                    verbose=0,
                )
                if verbose:
                    print("DONE")
            except LineSearchWarning:
                pass
            except PendingDeprecationWarning: 
                pass
            except Exception as exc: # pytlint: disable=broad-exception
                print("Failed with {}({})/n{}\n==================================".format(exc.__class__.__name__, getattr(exc,"message",""),traceback.format_exc()))
                raise exc

    def test_retrospective(self):
        TestFit.run_test(self, "retrospective")

    def test_prospective(self):
        TestFit.run_test(self, "prospective")

    def test_prospective_restrictive(self):
        TestFit.run_test(self, "prospective-restricted")

    def test_full(self):
        TestFit.run_test(self, "full")
        


class TestFitToy(unittest.TestCase):
    @staticmethod
    def simple_summ(fit, Y):
        #print("V_pen=%s, W_pen=%s" % (fit.fitted_v_pen, fit.fitted_w_pen))
        print("V=%s" % np.diag(fit.V))
        print("Treated weights: sim=%s, uns=%s, sum=%s" % ( fit.sc_weights[0, 49], fit.sc_weights[0, 99], sum(fit.sc_weights[0, :]),))
        print("Sim Con weights: sim=%s, uns=%s, sum=%s" % ( fit.sc_weights[1, 49], fit.sc_weights[1, 99], sum(fit.sc_weights[1, :]),))
        print("Uns Con weights: sim=%s, uns=%s, sum=%s" % ( fit.sc_weights[51, 49], fit.sc_weights[51, 99], sum(fit.sc_weights[51, :]),))
        Y_sc = fit.predict(Y)#[fit.control_units, :]
        print("Treated diff: %s" % (Y - Y_sc)[0, :])

    def test0s(self):
        N1, N0_sim, N0_not = 1, 50, 50
        N0 = N0_sim + N0_not
        N = N1 + N0
        treated_units, control_units  = range(N1), range(N1, N)
        T0, T1 = 2, 1
        T = T0 + T1 # unused
        te = 2
        #configs = [[[1, 0, 0], [0, 1, 1]], 
        #           [[0, 1, 0], [1, 0, 1]], 
        #           [[1, 0, 2], [0, 1, 1]], 
        #           [[0, 1, 2], [1, 0, 1]]]
        #configs = [[[2, 1, 0], [1, 2, 1]], 
        #           [[1, 2, 0], [2, 1, 1]], 
        #           [[2, 1, 2], [1, 2, 1]], 
        #           [[1, 2, 2], [2, 1, 1]]]
        configs = [[[1, 2, 2], [2, 1, 1]],
                   [[1, 2, 3], [2, 1, 1]]]
        for config in configs:
            proto_sim = np.array(config[0], ndmin=2)
            proto_not = np.array(config[1], ndmin=2)
            proto_tr = proto_sim + np.hstack((np.zeros((1, T0)), np.full((1, T1), te)))
            Y1 = np.matmul(np.ones((N1, 1)), proto_tr)
            Y0_sim = np.matmul(np.ones((N0_sim, 1)), proto_sim)
            Y0_not = np.matmul(np.ones((N0_not, 1)), proto_not)
        
            Y = np.vstack((Y1, Y0_sim, Y0_not))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit_res = fit(Y[:, :T0], Y[:, T0:], treated_units, model_type="retrospective", 
                              constrain="simplex", 
                              stopping_rule=4, progress=False, verbose=0, print_path=False)
            Y_sc = fit_res.predict(Y)
            treated_diff = (Y - Y_sc)[0, :]
            print("V: %s. Treated diff: %s" % (np.diag(fit_res.V), treated_diff))


if __name__ == "__main__":
    # t = TestFit()
    # t.setUp()
    # t.test_retrospective()
    unittest.main()
