from RidgeSC.fit_fold import fold_v_matrix
from RidgeSC.fit_loo import loo_v_matrix
from RidgeSC.fit_ct import ct_v_matrix
import numpy as np

def tensor(X, Y, X_treat=None, Y_treat=None, **kwargs):
    """ Presents a unified api for ct_v_matrix and loo_v_matrix
    """
    # PARAMETER QC
    if not isinstance(X, np.matrix): raise TypeError("X is not a matrix")
    if not isinstance(Y, np.matrix): raise TypeError("Y is not a matrix")
    if X.shape[1] == 0: raise ValueError("X.shape[1] == 0")
    if Y.shape[1] == 0: raise ValueError("Y.shape[1] == 0")
    if X.shape[0] != Y.shape[0]: raise ValueError("X and Y have different number of rows (%s and %s)" % (X.shape[0], Y.shape[0],))

    if X_treat is None != Y_treat is None: 
        raise ValueError("parameters `X_treat` and `Y_treat` must both be Matrices or None")

    if X_treat is not None:
        "Fit the Treated units to the control units; assuming that Y contains pre-intervention outcomes"

        # PARAMETER QC
        if not isinstance(X_treat, np.matrix): raise TypeError("X_treat is not a matrix")
        if not isinstance(Y_treat, np.matrix): raise TypeError("Y_treat is not a matrix")
        if X_treat.shape[1] == 0: raise ValueError("X_treat.shape[1] == 0")
        if Y_treat.shape[1] == 0: raise ValueError("Y_treat.shape[1] == 0")
        if X_treat.shape[0] != Y_treat.shape[0]: raise ValueError("X_treat and Y_treat have different number of rows (%s and %s)" % (X.shape[0], Y.shape[0],))

        # FIT THE V-MATRIX AND POSSIBLY CALCULATE THE L2_PEN_W
        # note that the weights, score, and loss function value returned here are for the in-sample predictions
        weights, v_mat, ts_score, loss, l2_pen_w, _ = \
                    ct_v_matrix(X = np.vstack((X,X_treat)),
                                Y = np.vstack((Y,Y_treat)),
                                control_units = np.arange(X.shape[0]),
                                treated_units = np.arange(X_treat.shape[0]) + X.shape[0],
                                **kwargs)

    else: 
        "Fit the control units to themselves; Y may contain post-intervention outcomes"

        weights, v_mat, ts_score, loss, l2_pen_w, _ = \
                loo_v_matrix(X = X,
                             Y = Y, 
                             control_units = np.arange(X.shape[0]),
                             treated_units = np.arange(X.shape[0]),
                             # treated_units = [X.shape[0] + i for i in  range(len(train))],
                             **kwargs)
    return v_mat

