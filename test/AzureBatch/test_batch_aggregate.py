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
# pylint: disable=multiple-imports, missing-docstring
"""
usage 

az login

name="sparsesctest"
location="westus2"

export BATCH_ACCOUNT_NAME=$name
export BATCH_ACCOUNT_KEY=$(az batch account keys list -n $name -g $name --query primary)
export BATCH_ACCOUNT_URL="https://$name.$location.batch.azure.com"
export STORAGE_ACCOUNT_NAME=$name
export STORAGE_ACCOUNT_KEY=$(az storage account keys list -n $name --query [0].value)

"""


from __future__ import print_function  # for compatibility with python 2.7
import numpy as np
import sys, os, random
import unittest
import warnings

try:
    from SparseSC import fit
except ImportError:
    raise RuntimeError("SparseSC is not installed. use 'pip install -e .' to install")
from scipy.optimize.linesearch import LineSearchWarning
from SparseSC.utils.AzureBatch import aggregate_batch_results


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

        batchdir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "data", "batchTest"
        )
        assert os.path.exists(batchdir), "Batch Directory '{}' does not exist".format(
            batchdir
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
            warnings.filterwarnings("ignore", category=LineSearchWarning)
            try:
                verbose = 0
                model_a = fit(
                    features=obj.X,
                    targets=obj.Y,
                    model_type=model_type,
                    treated_units=obj.treated_units
                    if model_type
                    in ("retrospective", "prospective", "prospective-restricted")
                    else None,
                    # KWARGS:
                    print_path=verbose,
                    stopping_rule=1,
                    progress=0,
                    grid_length=5,
                    min_iter=-1,
                    tol=1,
                    verbose=0,
                )

                model_b = aggregate_batch_results(
                    batchDir=batchdir
                )  # , batch_client_config="sg_daemon"

                assert np.all(
                    np.abs(model_a.scores - model_b.scores) < 1e-14
                ), "model scores are not within rounding error"

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
                raise exc

    def test_retrospective(self):
        TestFit.run_test(self, "retrospective")


# --     def test_prospective(self):
# --         TestFit.run_test(self, "prospective")
# --
# --     def test_prospective_restrictive(self):
# --         # Catch the LineSearchWarning silently, but allow others
# --
# --         TestFit.run_test(self, "prospective-restricted")
# --
# --     def test_full(self):
# --         TestFit.run_test(self, "full")


if __name__ == "__main__":
    unittest.main()
