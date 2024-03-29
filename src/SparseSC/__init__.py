""" Public API for SparseSC
"""

# PRIMARY FITTING FUNCTIONS
from SparseSC.fit import fit, TrivialUnitsWarning
from SparseSC.fit_fast import fit_fast, _fit_fast_inner, _fit_fast_match
from SparseSC.utils.match_space import (
    keras_reproducible, MTLassoCV_MatchSpace_factory, MTLasso_MatchSpace_factory, MTLassoMixed_MatchSpace_factory, MTLSTMMixed_MatchSpace_factory, 
    Fixed_V_factory, D_LassoCV_MatchSpace_factory
)
from SparseSC.utils.penalty_utils import RidgeCVSolution
from SparseSC.fit_loo import loo_v_matrix, loo_weights, loo_score
from SparseSC.fit_ct import ct_v_matrix, ct_weights, ct_score

# ESTIMATION FUNCTIONS
from SparseSC.estimate_effects import estimate_effects, get_c_predictions_honest
from SparseSC.utils.dist_summary import SSC_DescrStat, Estimate
from SparseSC.utils.descr_sets import DescrSet, MatchingEstimate

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
__version__ = "0.2.0"
