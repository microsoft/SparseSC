
from fit_loo import  loo_v_matrix
from fit_ct import  ct_v_matrix
from fit_fold import fold_v_matrix
from optimizers.cd_line_search import cdl_search
import numpy as np

def L2_pen_guestimate(X):
    return np.mean(np.var(X, axis = 0))

def get_max_lambda(X,Y,L2_PEN_W=None,X_treat=None,Y_treat=None,**kwargs):
    """ returns the maximum value of the L1 penalty for which the elements of tensor matrix (V) are not all zero.
    """

    # PARAMETER QC
    if not isinstance(X, np.matrix): raise TypeError("X is not a matrix")
    if not isinstance(Y, np.matrix): raise TypeError("Y is not a matrix")
    if X_treat is None != Y_treat is None:
        raise ValueError("parameters `X_treat` and `Y_treat` must both be Matrices or None")
    if X.shape[1] == 0: raise ValueError("X.shape[1] == 0")
    if Y.shape[1] == 0: raise ValueError("Y.shape[1] == 0")
    if X.shape[0] != Y.shape[0]: raise ValueError("X and Y have different number of rows (%s and %s)" % (X.shape[0], Y.shape[0],))
    if L2_PEN_W is None: L2_PEN_W = np.mean(np.var(X, axis = 0))

    if X_treat is not None:

        # PARAMETER QC
        if not isinstance(X_treat, np.matrix): raise TypeError("X_treat is not a matrix")
        if not isinstance(Y_treat, np.matrix): raise TypeError("Y_treat is not a matrix")
        if X_treat.shape[1] == 0: raise ValueError("X_treat.shape[1] == 0")
        if Y_treat.shape[1] == 0: raise ValueError("Y_treat.shape[1] == 0")
        if X_treat.shape[0] != Y_treat.shape[0]: 
            raise ValueError("X_treat and Y_treat have different number of rows (%s and %s)" % (X.shape[0], Y.shape[0],))

        control_units = np.arange(X.shape[0])
        treated_units = np.arange(X.shape[0],X.shape[0] + X_treat.shape[0])

        try:
            _LAMBDA = iter(L2_PEN_W)
        except TypeError:
            # L2_PEN_W is a single value 
            return ct_v_matrix(X = np.vstack((X,X_treat)),
                               Y = np.vstack((Y,Y_treat)),
                               L2_PEN_W = L2_PEN_W,
                               control_units = control_units,
                               treated_units = treated_units,
                               max_lambda = True,  # this is terrible at least without documentation...
                               **kwargs)
            
        else:
            # L2_PEN_W is an iterable of values
            return [ ct_v_matrix(X = np.vstack((X,X_treat)),
                                 Y = np.vstack((Y,Y_treat)),
                                  control_units = control_units,
                                  treated_units = treated_units,
                                  max_lambda = True,  # this is terrible at least without documentation...
                                  L2_PEN_W = l2_pen,
                                  **kwargs)
                        for l2_pen in L2_PEN_W ] 

    else: 

        try:
            _LAMBDA = iter(L2_PEN_W)
        except TypeError:
            if "grad_splits" in kwargs:
                # L2_PEN_W is a single value 
                return fold_v_matrix(X = X,
                                   Y = Y,
                                   L2_PEN_W = L2_PEN_W,
                                   max_lambda = True,  # this is terrible at least without documentation...
                                   **kwargs)
            else:
                # L2_PEN_W is a single value 
                return loo_v_matrix(X = X,
                                   Y = Y,
                                   L2_PEN_W = L2_PEN_W,
                                   max_lambda = True,  # this is terrible at least without documentation...
                                   **kwargs)
        else:
            if "grad_splits" in kwargs:

                # L2_PEN_W is an iterable of values
                return [ fold_v_matrix(X = X,
                                       Y = Y,
                                       L2_PEN_W = l2_pen,
                                       max_lambda = True,  # this is terrible at least without documentation...
                                       **kwargs)
                            for l2_pen in L2_PEN_W ] 
            else:

                # L2_PEN_W is an iterable of values
                return [ loo_v_matrix(X = X,
                                       Y = Y,
                                       L2_PEN_W = l2_pen,
                                       max_lambda = True,  # this is terrible at least without documentation...
                                       **kwargs)
                            for l2_pen in L2_PEN_W ] 

