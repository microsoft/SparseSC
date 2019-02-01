
# PRIMARY FITTING FUNCTIONS
from SparseSC.fit import fit, estimate_effects
from SparseSC.fit_loo import loo_v_matrix, loo_weights, loo_score
from SparseSC.fit_ct import ct_v_matrix, ct_weights, ct_score

# Public API
from SparseSC.cross_validation import score_train_test, score_train_test_sorted_lambdas, \
    CV_score
from SparseSC.tensor import tensor
from SparseSC.weights import weights
from SparseSC.lambda_utils import get_max_lambda, L2_pen_guestimate

# The version as used in the setup.py
__version__ = "0.1.0"
