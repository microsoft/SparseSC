""" 
Calculates the tensor (V) matrix which puts the metric on the covariate space
"""
from SparseSC.fit_fold import fold_v_matrix
from SparseSC.fit_loo import loo_v_matrix
from SparseSC.fit_ct import ct_v_matrix
import numpy as np


def tensor(X, Y, X_treat=None, Y_treat=None, grad_splits=None, **kwargs):
    """ Presents a unified api for ct_v_matrix and loo_v_matrix
    """
    # PARAMETER QC
    try:
        X = np.float64(X)
    except ValueError:
        raise ValueError("X is not coercible to float64")
    try:
        Y = np.float64(Y)
    except ValueError:
        raise ValueError("Y is not coercible to float64")

    Y = np.asmatrix(Y) # this needs to be deprecated properly -- bc Array.dot(Array) != matrix(Array).dot(matrix(Array)) -- not even close !!!
    X = np.asmatrix(X)

    if X.shape[1] == 0:
        raise ValueError("X.shape[1] == 0")
    if Y.shape[1] == 0:
        raise ValueError("Y.shape[1] == 0")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            "X and Y have different number of rows (%s and %s)"
            % (X.shape[0], Y.shape[0])
        )

    if (X_treat is None) != (Y_treat is None):
        raise ValueError(
            "parameters `X_treat` and `Y_treat` must both be Matrices or None"
        )

    if X_treat is not None:
        # Fit the Treated units to the control units; assuming that Y contains
        # pre-intervention outcomes:

        # PARAMETER QC
        try:
            X_treat = np.float64(X_treat)
        except ValueError:
            raise ValueError("X_treat is not coercible to float64")
        try:
            Y_treat = np.float64(Y_treat)
        except ValueError:
            raise ValueError("Y_treat is not coercible to float64")

        Y_treat = np.asmatrix(Y_treat) # this needs to be deprecated properly -- bc Array.dot(Array) != matrix(Array).dot(matrix(Array)) -- not even close !!!
        X_treat = np.asmatrix(X_treat)

        if X_treat.shape[1] == 0:
            raise ValueError("X_treat.shape[1] == 0")
        if Y_treat.shape[1] == 0:
            raise ValueError("Y_treat.shape[1] == 0")
        if X_treat.shape[0] != Y_treat.shape[0]:
            raise ValueError(
                "X_treat and Y_treat have different number of rows (%s and %s)"
                % (X_treat.shape[0], Y_treat.shape[0])
            )

        # FIT THE V-MATRIX AND POSSIBLY CALCULATE THE w_pen
        # note that the weights, score, and loss function value returned here
        # are for the in-sample predictions
        _, v_mat, _, _, _, _ = ct_v_matrix(
            X=np.vstack((X, X_treat)),
            Y=np.vstack((Y, Y_treat)),
            control_units=np.arange(X.shape[0]),
            treated_units=np.arange(X_treat.shape[0]) + X.shape[0],
            **kwargs
        )

    else:
        # Fit the control units to themselves; Y may contain post-intervention outcomes:

        adjusted=False
        if kwargs["w_pen"] < 1:
            adjusted=True

        if grad_splits is not None:
            _, v_mat, _, _, _, _ = fold_v_matrix(
                X=X,
                Y=Y,
                control_units=np.arange(X.shape[0]),
                treated_units=np.arange(X.shape[0]),
                grad_splits=grad_splits,
                **kwargs
            )

            #if adjusted:
            #    print("vmat: %s" % (np.diag(v_mat)))

        else:
            _, v_mat, _, _, _, _ = loo_v_matrix(
                X=X,
                Y=Y,
                control_units=np.arange(X.shape[0]),
                treated_units=np.arange(X.shape[0]),
                **kwargs
            )
    return v_mat
