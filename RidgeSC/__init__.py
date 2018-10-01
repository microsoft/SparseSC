
# PRIMARY FITTING FUNCTIONS
from RidgeSC.fit_loo import loo_v_matrix, loo_weights, loo_score
from RidgeSC.fit_ct import ct_v_matrix, ct_weights, ct_score

# Public API
from RidgeSC.cross_validation import score_train_test, score_train_test_sorted_lambdas, CV_score, joint_penalty_optimzation
from RidgeSC.tensor import tensor
from RidgeSC.weights import weights
from RidgeSC.lambda_utils import get_max_lambda, L2_pen_guestimate

# The version as used in the setup.py
__version__ = "0.1.0"
