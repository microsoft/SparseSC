from numpy import dot, ones, diag, matrix, zeros, array,absolute, mean,var, linalg, prod,shape,sqrt
import numpy as np
import pandas as pd
import itertools
import timeit
import warnings
from utils.sub_matrix_inverse import subinv_k, all_subinverses
from optimizers.cd_line_search import cdl_search
warnings.filterwarnings('ignore')


def fold_v_matrix(X,
                 Y,
                 LAMBDA = 0,
                 treated_units = None,
                 control_units = None,
                 non_neg_weights = False,
                 start = None,
                 L2_PEN_W = None,
                 method = cdl_search, 
                 intercept = True,
                 max_lambda = False,  # this is terrible at least without documentation...
                 solve_method = "standard",
                 splits = 5,
                 **kwargs):
    '''
    Computes and sets the optimal v_matrix for the given moments and 
        penalty parameter.

    :param X: Matrix of Covariates
    :param Y: Matrix of Outcomes
    :param LAMBDA: penalty parameter used to shrink L1 norm of v/v.max() toward zero
    :param treated_units: a list containing the position (rows) of the treated units within X and Y
    :param control_units: a list containing the position (rows) of the control units within X and Y
    :param start: initial values for the diagonals of the tensor matrix
    :param L2_PEN_W: L2 penalty on the magnitude of the deviance of the weight vector from null. Optional.
    :param method: The name of a method to be used by scipy.optimize.minimize, or a callable with the same API as scipy.optimize.minimize
    :param intercept: If True, weights are penalized toward the 1 / the number of controls, else weights are penalized toward zero
    :;aram max_lambda: if True, the return value is the maximum L1 penalty for which at least one element of the tensor matrix is non-zero
    :;aram solve_method: Method for solving A.I.dot(B). Either "standard" or "step-down".
    :param **kwargs: additional arguments passed to the optimizer

    '''
    # (by default all the units are treated and all are controls)
    if treated_units is None: 
        if control_units is None: 
            # Neither provided; INCLUDE ALL SAMPLES AS BOTH TREAT AND CONTROL UNIT. 
            # (this is the typical controls-only Loo V-matrix estimation)
            control_units = list(range(X.shape[0]))
            treated_units = control_units 
        else:
            # Set the treated units to the not-control units
            treated_units = list(set(range(X.shape[0])) - set(control_units))  
    else:
        if control_units is None: 
            # Set the control units to the not-treated units
            control_units = list(set(range(X.shape[0])) - set(treated_units)) 
    control_units = np.array(control_units)
    treated_units = np.array(treated_units)

    # parameter QC
    if not isinstance(X, matrix): raise TypeError("X is not a matrix")
    if not isinstance(Y, matrix): raise TypeError("Y is not a matrix")
    if X.shape[1] == 0: raise ValueError("X.shape[1] == 0")
    if Y.shape[1] == 0: raise ValueError("Y.shape[1] == 0")
    if X.shape[0] != Y.shape[0]: raise ValueError("X and Y have different number of rows (%s and %s)" % (X.shape[0], Y.shape[0],))
    if not isinstance(LAMBDA, (float, int)): raise TypeError( "LAMBDA is not a number")
    if L2_PEN_W is None: L2_PEN_W = mean(var(X, axis = 0))
    if not isinstance(L2_PEN_W, (float, int)): raise TypeError( "L2_PEN_W is not a number")
    assert not non_neg_weights, "Bounds not implemented"

    if isinstance(splits, (int)) or np.issubdtype(splits, np.integer): 
        from sklearn.model_selection import KFold
        splits = KFold(splits).split(np.arange(X_treat.shape[0]))
    splits = list(splits)
    n_splits = len(grad_splits)

    # CONSTANTS
    C, N, K = len(control_units), len(treated_units), X.shape[1]
    if start is None: start = zeros(K) # formerly: .1 * ones(K) 
    assert N > 0; "No control units"
    assert C > 0; "No treated units"
    assert K > 0; "variables to fit (X.shape[1] == 0)"

    # CREATE THE INDEX THAT INDICATES THE ELIGIBLE CONTROLS FOR EACH TREATED UNIT
    in_controls = [list(set(control_units) - set([trt_unit])) for trt_unit in treated_units]
    in_controls2 = [np.ix_(i,i) for i in in_controls] # this is a much faster alternative to A[:,index][index,:]
    ctrl_rng = np.arange(len(control_units))
    out_controls = [ctrl_rng[control_units != trt_unit] for trt_unit in treated_units] 
    out_treated  = [ctrl_rng[control_units == trt_unit] for trt_unit in treated_units] # this is non-trivial when there control units are also being predicted.

    if intercept:
        for i, trt_unit in enumerate(treated_units):
            Y[trt_unit,:] -= Y[in_controls[i],:].mean(axis=0) 




def __old_fold_v_matrix(Xt,Xc,Yt,Yc,LAMBDA = .1,L2_PEN_W = .1,guess = None,method = 'Newton-CG',**kwargs):
    """ Calculates the V matrix with an explicit Treatment and Control folds,
        where the match is optimized on the error of the outcomes.

        don't use this. it hasn't been thoroughly been checked for bugs (compared
        to loo_v_matrix and ct_v_matrix) and lacks several of the optimizations of loo_v_matrix
    """
    C,K = Xc.shape; N,T = Yt.shape
    two_L2_PEN_W = 2*L2_PEN_W*diag(ones(C))
    def _score(diagV):
        """ calculates just the diagonal of dGamma0_dV
        """
        V = diag(diagV)
        A = Xc.dot(2*V).dot(Xc.T) + two_L2_PEN_W # 5
        B = Xc.dot(2*V).dot(Xt.T) # 6
        Pi = A.I.dot(B) 
        Ey = (Yt - Pi.T.dot(Yc)).getA()
        return (Ey **2).sum() + LAMBDA * absolute(Pi.T).sum()
    def _grad(diagV):
        """ calculates just the diagonal of dGamma0_dV
        """
        V = diag(diagV)
        assert Xc.shape[0] == Yc.shape[0]
        # CALCULATE CONSTANTS
        A = Xc.dot(2*V).dot(Xc.T) + two_L2_PEN_W # 5
        B = Xc.dot(2*V).dot(Xt.T) # 6
        Ai = A.I
        Pi = Ai.dot(B) 
        Ey = (Yt - Pi.T.dot(Yc)).getA()
        # INITIALIZE PARTIAL DERIVATIVES
        dGamma0_dV_term2 = zeros((K,K))
        for i in range(K):
            dA = Xc[:, i ].dot(Xc[:, i ].T) + Xc[:, i ].dot(Xc[:, i ].T) # 8
            dB = Xc[:, i ].dot(Xt [:, i ].T) + Xc[:, i ].dot(Xt [:, i ].T) # 9
            dPI_dV = Ai.dot(dB - dA.dot(Pi)) # 7
            dGamma0_dV_term2[i,i] = (Ey * Yc.T.dot(dPI_dV).T.getA()).sum()
        return diag(LAMBDA * (1 - 2*(V<0)) - 2 * dGamma0_dV_term2)
    def _weights(V):
        A = Xc.dot(2*V).dot(Xc.T) + two_L2_PEN_W # 5
        B = Xc.dot(2*V).dot(Xt.T) # 6
        #for col in treated_unit:
            #w_col = A[-col,-col].I.dot(B[:,col])
        return A.I.dot(B).T
    # DO THE OPTIMIZATION
    opt = minimize(_score, zeros(Xc.shape[1]) if guess is None else guess,
                   jac = _grad,
                   method = method, **kwargs)
    v_mat = diag(opt.x)
    ts_loss = opt.fun
    # CALCULATE weights AND ts_score
    weights = _weights(v_mat)
    errors = Yt - weights.dot(Yc)
    ts_score = linalg.norm(errors) / sqrt(prod(errors.shape))
    return weights, v_mat, ts_score, ts_loss, opt





def loo_weights(X, V, L2_PEN_W, treated_units = None, control_units = None, intercept = True, solve_method = "standard"):
    if treated_units is None: 
        if control_units is None: 
            # both not provided, include all samples as both treat and control unit.
            control_units = list(range(X.shape[0]))
            treated_units = control_units 
        else:
            # Set the treated units to the not-control units
            treated_units = list(set(range(X.shape[0])) - set(control_units))  
    else:
        if control_units is None: 
            # Set the control units to the not-treated units
            control_units = list(set(range(X.shape[0])) - set(treated_units)) 
    control_units = np.array(control_units)
    treated_units = np.array(treated_units)
    [C, N, K] = [len(control_units), len(treated_units), X.shape[1]]


    # index with positions of the controls relative to the incoming data
    in_controls = [list(set(control_units) - set([trt_unit])) for trt_unit in treated_units]
    in_controls2 = [np.ix_(i,i) for i in in_controls] # this is a much faster alternative to A[:,index][index,:]

    # index of the controls relative to the rows of the outgoing C x N matrix of weights
    ctrl_rng = np.arange(len(control_units))
    out_controls = [ctrl_rng[control_units != trt_unit] for trt_unit in treated_units] 
    out_treated  = [ctrl_rng[control_units == trt_unit] for trt_unit in treated_units] # this is non-trivial when there control units are also being predicted.

    # constants for indexing
    X_control = X[control_units,:]
    X_treat = X[treated_units,:]
    weights = zeros((C, N))

    if solve_method == "step-down":
        A = X_control.dot(V + V.T).dot(X_control.T) + 2 * L2_PEN_W * diag(ones(X_control.shape[0])) # 5
        B = X_treat.dot(  V + V.T).dot(X_control.T) # 6
        Ai = A.I
        for i, trt_unit in enumerate(treated_units):
            if trt_unit in control_units:
                (b) = subinv_k(Ai,_k).dot(B[out_controls[i],i])
            else:
                (b) = Ai.dot(B[:, i])

            weights[out_controls[i], i] = b.flatten()
            if intercept:
                weights[out_controls[i], i] += 1/len(out_controls[i])
    elif solve_method == "standard":
        A = X.dot(V + V.T).dot(X.T) + 2 * L2_PEN_W * diag(ones(X.shape[0])) # 5
        B = X.dot(V + V.T).dot(X.T).T # 6
        for i, trt_unit in enumerate(treated_units):
            (b) = linalg.solve(A[in_controls2[i]], B[in_controls[i], trt_unit])

            weights[out_controls[i], i] = b.flatten()
            if intercept:
                weights[out_controls[i], i] += 1/len(out_controls[i])
    else:
        raise ValueError("Unknown Solve Method: " + solve_method)
    return weights.T

def loo_score(Y, X, V, L2_PEN_W, LAMBDA = 0, treated_units = None, control_units = None,**kwargs):
    if treated_units is None: 
        if control_units is None: 
            # both not provided, include all samples as both treat and control unit.
            control_units = list(range(X.shape[0]))
            treated_units = control_units 
        else:
            # Set the treated units to the not-control units
            treated_units = list(set(range(X.shape[0])) - set(control_units))  
    else:
        if control_units is None: 
            # Set the control units to the not-treated units
            control_units = list(set(range(X.shape[0])) - set(treated_units)) 
    weights = loo_weights(X = X,
                           V = V,
                           L2_PEN_W = L2_PEN_W,
                           treated_units = treated_units,
                           control_units = control_units,
                           **kwargs)
    Y_tr = Y[treated_units, :]
    Y_c = Y[control_units, :]
    Ey = (Y_tr - weights.dot(Y_c)).getA()
    return (Ey **2).sum() + LAMBDA * V.sum()


