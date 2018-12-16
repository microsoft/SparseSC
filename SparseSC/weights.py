from SparseSC.fit_loo import loo_weights
from SparseSC.fit_ct import ct_weights
from SparseSC.fit_fold import fold_weights
import numpy as np

def weights(X, X_treat=None, grad_splits = None, **kwargs):
    """ Calculate synthetic control weights
    """

    # PARAMETER QC
    try:
        X = np.asmatrix(X)
    except ValueError:
        raise TypeError("X is not coercible to a matrix")

    if X_treat is not None:
        # weight for the control units against the remaining controls:

        if X_treat.shape[1] == 0:
            raise ValueError("X_treat.shape[1] == 0")

        # PARAMETER QC
        if not isinstance(X_treat, np.matrix):
            raise TypeError("X_treat is not a matrix")
        if X_treat.shape[1] == 0:
            raise ValueError("X_treat.shape[1] == 0")

        # FIT THE V-MATRIX AND POSSIBLY CALCULATE THE L2_PEN_W
        # note that the weights, score, and loss function value returned here are for the in-sample predictions
        return ct_weights(X = np.vstack((X,X_treat)),
                          control_units = np.arange(X.shape[0]),
                          treated_units = np.arange(X_treat.shape[0]) + X.shape[0],
                          **kwargs)

    else: 
        if grad_splits is not None:
            return fold_weights(X = X,
                                grad_splits = grad_splits,
                                **kwargs)

        # weight for the control units against the remaining controls
        else: 
            return loo_weights(X = X,
                               control_units = np.arange(X.shape[0]),
                               treated_units = np.arange(X.shape[0]),
                               **kwargs)
