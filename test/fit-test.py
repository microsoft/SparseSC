# --------------------------------------------------------------------------------
# Programmer: Jason Thorpe
# Date    1/25/2019 3:34:02 PM
# Language:  Python (.py) Version 2.7 or 3.5
# Purpose:  simple test runner for SparseSC/fit.py
# Usage: 
# 
# Test all model types
# 
#     \SpasrseSC > python test/fit-test.py
# 
# Test a specific model type (e.g. "prospective-restricted"): 
# 
#     \SpasrseSC > python test/fit-test.py prospective-restricted
# 
# --------------------------------------------------------------------------------

from __future__ import print_function # for compatibility with python 2.7
import numpy as np
import sys, os

# Bootstrapping the system path
sys.path.insert(0,os.path.abspath('.'))

if len(sys.argv) > 1:
    _MODEL_TYPES = sys.argv[1:]
else: 
    _MODEL_TYPES = (  "prospective-restricted", "full", "prospective","retrospective", ) 


print("Importing fit module...",end="")
sys.stdout.flush()
try: 
    from SparseSC.fit import fit
    print("[Done]")
except Exception as exc: 
    print("Failed with %s: %s" % (exc.__class__.__name__,exc.message,) )
    exit()


np.random.seed(101101001)
control_units = 50
treated_units = 20
features = 10
targets = 5

_X = np.random.rand(control_units + treated_units,features)
_Y = np.random.rand(control_units + treated_units,targets)
_TREATED_UNITS = np.arange(treated_units)



for _MODEL_TYPE in _MODEL_TYPES:
    print("Calling fit with `model_type  = '%s'`..." % (_MODEL_TYPE, ),end="")
    sys.stdout.flush()
#--     try: 
    fit(X = _X,
        Y = _Y,
        model_type = _MODEL_TYPE,
        treated_units = _TREATED_UNITS if _MODEL_TYPE in ("retrospective", "prospective", "prospective-restricted") else None,
        # KWARGS: 
        print_path = False,
        progress = False,
        min_iter = -1,
        tol = 1,
        )
#--     except Exception as exc: 
#--         print("Failed with %s: %s" % (exc.__class__.__name__,exc.message,) )
#--         exit()
#--     else: 
    print("DONE")






