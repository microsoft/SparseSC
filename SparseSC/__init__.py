""" Public API for SparseSC
"""

# PRIMARY FITTING FUNCTIONS
from SparseSC.fit import fit
from SparseSC.estimate_effects import estimate_effects
from SparseSC.fit_loo import loo_v_matrix, loo_weights, loo_score
from SparseSC.fit_ct import ct_v_matrix, ct_weights, ct_score

# Public API
from SparseSC.cross_validation import (
    score_train_test,
    score_train_test_sorted_v_pens,
    CV_score,
)
from SparseSC.tensor import tensor
from SparseSC.weights import weights
from SparseSC.utils.penalty_utils import get_max_w_pen, get_max_v_pen, w_pen_guestimate

# The version as used in the setup.py
__version__ = "0.1.0"
