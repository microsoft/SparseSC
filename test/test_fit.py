"""
Tests the fit methods
"""
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
import sys
import random
import unittest
import warnings
from scipy.optimize.linesearch import LineSearchWarning
import numpy as np
import traceback

try:
    import SparseSC
    from SparseSC.fit import fit
    from SparseSC.fit_fast import fit_fast
except ImportError:
    raise RuntimeError("SparseSC is not installed. use 'pip install -e .' from repo root to install in dev mode")
#import warnings
#warnings.simplefilter("error")

# pylint: disable=missing-docstring

class TestFitForErrors(unittest.TestCase):
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
    def run_test(cls, obj, model_type, verbose=False, w_pen_inner=False):
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
                    features=obj.X,
                    targets=obj.Y,
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
                    w_pen_inner=w_pen_inner,
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

    # def test_retrospective(self):
    #     TestFitForErrors.run_test(self, "retrospective")

    # def test_prospective(self):
    #     TestFitForErrors.run_test(self, "prospective")

    # def test_prospective_restrictive(self):
    #     TestFitForErrors.run_test(self, "prospective-restricted")

    # def test_full(self):
    #     TestFitForErrors.run_test(self, "full")

    def test_all(self):
        for model_type in ["retrospective", "prospective", "prospective-restricted", "full"]: #
            TestFitForErrors.run_test(self, model_type)

        model_type = "retrospective"
        TestFitForErrors.run_test(self, model_type, w_pen_inner=True) #default is, w_pen_inner=False

        ## How to combine with a match-space front-end (do retrospective)
        control_units = [u for u in range(self.Y.shape[0]) if u not in self.treated_units]
        # Separate
        match_space_maker = SparseSC.MTLassoCV_MatchSpace_factory()
        MatchSpace, _, _, _ = match_space_maker(self.X[control_units], self.Y[control_units], fit_model_wrapper=None)
        M = MatchSpace(self.X)
        X_orig = self.X
        self.X = M
        TestFitForErrors.run_test(self, model_type)
        self.X = X_orig
        # Mixed

        def _fit_model_wrapper(MatchSpace, *args, **kwargs): #allow room to pass in the V
            fit_obj = fit(
                features=MatchSpace(self.X),
                targets=self.Y,
                model_type=model_type,
                treated_units=self.treated_units,
                # KWARGS:
                print_path=False,
                stopping_rule=1,
                progress=False,
                grid_length=5,
                min_iter=-1,
                tol=1,
                verbose=0,
            )
            return fit_obj
        match_space_maker = SparseSC.MTLassoMixed_MatchSpace_factory(v_pens=[1,2])
        MatchSpace, _, _, _ = match_space_maker(self.X[control_units], self.Y[control_units], fit_model_wrapper=_fit_model_wrapper)
        M = MatchSpace(self.X)
        X_orig = self.X
        self.X = M
        TestFitForErrors.run_test(self, model_type)
        self.X = X_orig

class TestFitFastForErrors(unittest.TestCase):
    def setUp(self):

        random.seed(12345)
        np.random.seed(101101001)
        control_units = 50
        treated_units = 20
        features = 10
        targets = 5

        self.X = np.random.rand(control_units + treated_units, features)
        beta = np.array([[1,0,0,0,0,0,0,0,0,1]]).T
        yhat = self.X @ beta
        self.Y = yhat @ np.ones((1,targets)) + np.random.rand(control_units + treated_units, targets)
        self.treated_units = np.arange(treated_units)

    @classmethod
    def run_test(cls, obj, model_type="retrospective", match_maker=None, w_pen_inner=True):
        fit_fast(
            features=obj.X,
            targets=obj.Y,
            model_type=model_type,
            treated_units=obj.treated_units
            if model_type
            in ("retrospective", "prospective", "prospective-restricted")
            else None,
            match_space_maker=match_maker,
            w_pen_inner=w_pen_inner
        )

    def test_all(self):
        for match_maker in [None, SparseSC.MTLassoMixed_MatchSpace_factory(), SparseSC.MTLassoCV_MatchSpace_factory(), SparseSC.MTLSTMMixed_MatchSpace_factory(), SparseSC.Fixed_V_factory(np.full(self.X.shape[1], 1))]: #, 
            TestFitFastForErrors.run_test(self, "retrospective", match_maker)

        for model_type in ["prospective", "prospective-restricted", "full"]: #"retrospective", (tested above)
            TestFitFastForErrors.run_test(self, model_type, None)

        TestFitFastForErrors.run_test(self, "retrospective", w_pen_inner=False) #default is, w_pen_inner=True

class TestFitForCorrectness(unittest.TestCase):
    @staticmethod
    def simple_summ(fit_res, Y):
        #print("V_pen=%s, W_pen=%s" % (fit_res.fitted_v_pen, fit_res.fitted_w_pen))
        print("V=%s" % np.diag(fit_res.V))
        print("Treated weights: sim=%s, uns=%s, sum=%s" % ( fit_res.sc_weights[0, 49], fit_res.sc_weights[0, 99], sum(fit_res.sc_weights[0, :]),))
        print("Sim Con weights: sim=%s, uns=%s, sum=%s" % ( fit_res.sc_weights[1, 49], fit_res.sc_weights[1, 99], sum(fit_res.sc_weights[1, :]),))
        print("Uns Con weights: sim=%s, uns=%s, sum=%s" % ( fit_res.sc_weights[51, 49], fit_res.sc_weights[51, 99], sum(fit_res.sc_weights[51, :]),))
        Y_sc = fit_res.predict(Y)#[fit_res.control_units, :]
        print("Treated diff: %s" % (Y - Y_sc)[0, :])

    def test0s(self):
        N1, N0_sim, N0_not = 1, 50, 50
        N0 = N0_sim + N0_not
        #N = N1 + N0
        treated_units = range(N1)
        #control_units  = range(N1, N)
        T0, T1 = 2, 1
        #T = T0 + T1 # unused
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
    t = TestFitForErrors()
    t.setUp()
    t.test_all()
    
    t = TestFitFastForErrors()
    t.setUp()
    t.test_all()
    
    #unittest.main()
