""" 
Presents a unified API for the various weights methods
"""
from SparseSC.fit_loo import loo_weights
from SparseSC.fit_ct import ct_weights
from SparseSC.fit_fold import fold_weights
import numpy as np


def weights(X, X_treat=None, grad_splits=None, custom_donor_pool=None, **kwargs):
    """ Calculate synthetic control weights
    """

    # PARAMETER QC
    try:
        X = np.float64(X)
    except ValueError:
        raise TypeError("X is not coercible to float64")

    if X_treat is not None:
        # weight for the control units against the remaining controls:

        if X_treat.shape[1] == 0:
            raise ValueError("X_treat.shape[1] == 0")

        # PARAMETER QC
        try:
            X_treat = np.float64(X_treat)
        except ValueError:
            raise ValueError("X_treat is not coercible to float64")
        if X_treat.shape[1] == 0:
            raise ValueError("X_treat.shape[1] == 0")

        # FIT THE V-MATRIX AND POSSIBLY CALCULATE THE w_pen
        # note that the weights, score, and loss function value returned here
        # are for the in-sample predictions
        return ct_weights(
            X=np.vstack((X, X_treat)),
            control_units=np.arange(X.shape[0]),
            treated_units=np.arange(X_treat.shape[0]) + X.shape[0],
            custom_donor_pool=custom_donor_pool,
            **kwargs
        )

    # === X_treat is None: ===

    if grad_splits is not None:
        return fold_weights(X=X, grad_splits=grad_splits, **kwargs)

    # === X_treat is None and grad_splits is None: ===

    # weight for the control units against the remaining controls
    return loo_weights(
        X=X,
        control_units=np.arange(X.shape[0]),
        treated_units=np.arange(X.shape[0]),
        custom_donor_pool=custom_donor_pool,
        **kwargs
    )
