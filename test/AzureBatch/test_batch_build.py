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
import numpy as np
import sys, os, random
import unittest
import warnings
from scipy.optimize.linesearch import LineSearchWarning

try:
    from SparseSC import fit
except ImportError:
    raise RuntimeError("SparseSC is not installed. use 'pip install -e .' to install")


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
        if verbose:
            print("Calling fit with `model_type  = '%s'`..." % (model_type,), end="")
        sys.stdout.flush()

        batchdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "batchTest")

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
                    batchDir=batchdir
                )
                if verbose:
                    print("DONE")
            except LineSearchWarning:
                pass
            except PendingDeprecationWarning: 
                pass
            except Exception as exc:  # pylint: disable=broad-except
                print(
                    "Failed with %s(%s)"
                    % (exc.__class__.__name__, getattr(exc, "message", ""))
                )

    def test_retrospective(self):
        TestFit.run_test(self, "retrospective")

#--     def test_prospective(self):
#--         TestFit.run_test(self, "prospective")
#-- 
#--     def test_prospective_restrictive(self):
#--         # Catch the LineSearchWarning silently, but allow others
#-- 
#--         TestFit.run_test(self, "prospective-restricted")
#-- 
#--     def test_full(self):
#--         TestFit.run_test(self, "full")


if __name__ == "__main__":
    # t = TestFit()
    # t.setUp()
    # t.test_retrospective()
    unittest.main()
