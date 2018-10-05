from numpy import ones, diag, matrix, zeros, mean,var, linalg, prod, sqrt, absolute
import numpy as np
import itertools
import warnings
#from RidgeSC.utils.sub_matrix_inverse import subinv_k, all_subinverses
from RidgeSC.optimizers.cd_line_search import cdl_search
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
                  grad_splits = 5,
                  random_state = 10101,
                  verbose = False,
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
    :param L2_PEN_W: L2 penalty on the magnitude of the deviance of the weight
                     vector from null. Optional.
    :param method: The name of a method to be used by scipy.optimize.minimize,
                   or a callable with the same API as scipy.optimize.minimize
    :param intercept: If True, weights are penalized toward the 1 / the number
                    of controls, else weights are penalized toward zero
    :param max_lambda: if True, the return value is the maximum L1 penalty for
                       which at least one element of the tensor matrix is
                       non-zero
    :param grad_splits: Splits for Fitted v.s. Control units in each gradient
                        descent step. An integer, or a list/generator of train
                        and test units in each fold of the gradient descent.
    :param random_state: Integer, used for setting the random state for consistency of fold splits across calls
    :param verbose: If true, print progress to the console (default: false)
    :param kwargs: additional arguments passed to the optimizer
    :param non_neg_weights: not implemented

    :raises ValueError: raised when parameter values are invalid
    :raises TypeError: raised when parameters are of the wrong type

    :return: something something
    :rtype: something something
    '''
    # (by default all the units are treated and all are controls)
    if treated_units is None: 
        if control_units is None: 
            # Neither provided; INCLUDE ALL SAMPLES AS BOTH TREAT AND CONTROL UNIT. 
            # (this is the typical controls-only fold V-matrix estimation)
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
    if not isinstance(X, matrix):
        raise TypeError("X is not a matrix")
    if not isinstance(Y, matrix):
        raise TypeError("Y is not a matrix")
    if X.shape[1] == 0:
        raise ValueError("X.shape[1] == 0")
    if Y.shape[1] == 0:
        raise ValueError("Y.shape[1] == 0")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y have different number of rows (%s and %s)" % (X.shape[0], Y.shape[0],))
    if not isinstance(LAMBDA, (float, int)):
        raise TypeError( "LAMBDA is not a number")
    if L2_PEN_W is None:
        L2_PEN_W = mean(var(X, axis = 0))
    if not isinstance(L2_PEN_W, (float, int)):
        raise TypeError( "L2_PEN_W is not a number")
    assert not non_neg_weights, "Bounds not implemented"

    splits = grad_splits # for readability...
    try:
        iter(splits)
    except TypeError: 
        from sklearn.model_selection import KFold
        splits = KFold(splits, shuffle=True, random_state = random_state).split(np.arange(len(treated_units)))
    splits = list(splits)

    for i, split in enumerate(splits):
        assert len(split[0]) + len(split[1]) == len(treated_units), \
                ("Splits for fold %s do not match the number of treated units.  Expected %s; got %s + %s" % 
                 (i, len(treated_units),len(split[0]), len(split[1]), ))

    # CONSTANTS
    N0, N1, K = len(control_units), len(treated_units), X.shape[1]
    if start is None: 
        start = zeros(K) # formerly: .1 * ones(K) 
    assert N1 > 0, "No control units"
    assert N0 > 0, "No treated units"
    assert K > 0, "variables to fit (X.shape[1] == 0)"

    # CREATE THE INDEX THAT INDICATES THE ELIGIBLE CONTROLS FOR EACH TREATED UNIT
    in_controls = [list(set(control_units) - set(treated_units[test])) for _,test in splits]
    in_controls2 = [np.ix_(i,i) for i in in_controls] # this is a much faster alternative to A[:,index][index,:]
    ctrl_rng = np.arange(len(control_units))
    out_controls = [ctrl_rng[np.logical_not(np.isin(control_units, treated_units[test]))] for _,test in splits] 

    # this is non-trivial when there control units are also being predicted:
    #out_treated  = [ctrl_rng[               np.isin(control_units, treated_units[test]) ] for train,test in splits]

    if intercept:
        Y = Y.copy()
        for in_ctrl, (_, test) in zip(in_controls,splits):
            Y[treated_units[test],:] -= Y[in_ctrl,:].mean(axis=0) 

    # handy constants (for speed purposes):
    Y_treated = Y[treated_units,:]
    Y_control = Y[control_units,:]

    # INITIALIZE PARTIAL DERIVATIVES
    dA_dV_ki = [ [None,] *N1 for i in range(K)]
    dB_dV_ki = [ [None,] *N1 for i in range(K)]
    b_i = [None,] *N1 
    for i, k in  itertools.product(range(len(splits)), range(K)): # TREATED unit i, moment k
        _, test = splits[i]
        Xc = X[in_controls[i], : ]
        Xt = X[treated_units[test], : ]
        dA_dV_ki [k][i] = Xc[:, k ].dot(Xc[:, k ].T) + Xc[:, k ].dot(Xc[:, k ].T) # 8
        dB_dV_ki [k][i] = Xc[:, k ].dot(Xt[:, k ].T) + Xc[:, k ].dot(Xt[:, k ].T) # 9

    k=0 # for linting...
    del Xc, Xt, i, k

    def _score(V):
        dv = diag(V)
        weights, _, _ = _weights(dv)
        Ey = (Y_treated - weights.T.dot(Y_control)).getA()
        # (...).copy() assures that x.flags.writeable is True
        return ((Ey **2).sum() + LAMBDA * absolute(V).sum()).copy() 

    def _grad(V):
        """ Calculates just the diagonal of dGamma0_dV

            There is an implementation that allows for all elements of V to be varied...
        """
        dv = diag(V)
        weights, A, _ = _weights(dv)
        Ey = (Y_treated - weights.T.dot(Y_control)).getA()
        dGamma0_dV_term2 = zeros(K)
        dPI_dV = zeros((N0, N1))
        for k in range(K):
            if verbose:  # for large sample sizes, linalg.solve is a huge bottle neck,
                print("Calculating gradient, for moment %s of %s" % (k ,K,))
            dPI_dV.fill(0) # faster than re-allocating the memory each loop.
            for i, (_, (_, test)) in enumerate(zip(in_controls,splits)):
                if verbose >=2:  # for large sample sizes, linalg.solve is a huge bottle neck,
                    print("Calculating gradient, linalg.solve() call %s of %s" % (i + k*len(splits) ,K*len(splits),))
                dA = dA_dV_ki[k][i]
                dB = dB_dV_ki[k][i]
                b = linalg.solve(A[in_controls2[i]],dB - dA.dot(b_i[i]))
                dPI_dV[np.ix_(in_controls[i], treated_units[test])] = b
            dGamma0_dV_term2[k] = (Ey * Y_control.T.dot(dPI_dV).T.getA()).sum()
        return LAMBDA - 2 * dGamma0_dV_term2 

    def _weights(V):
        weights = zeros((N0, N1))
        A = X.dot(V + V.T).dot(X.T) + 2 * L2_PEN_W * diag(ones(X.shape[0])) # 5
        B = X.dot(V + V.T).dot(X.T).T # 6
        for i, (_,test) in enumerate(splits):
            if verbose >=2:  # for large sample sizes, linalg.solve is a huge bottle neck,
                print("Calculating weights, linalg.solve() call %s of %s" % (i,len(splits),))
            (b) = b_i[i] = linalg.solve(A[in_controls2[i]], B[np.ix_(in_controls[i], treated_units[test])])
            weights[np.ix_(out_controls[i], test)] = b
        return weights, A, B

    if max_lambda:
        grad0 = _grad(zeros(K))
        return -grad0[grad0 < 0].min()

    # DO THE OPTIMIZATION
    if isinstance(method, str):
        from scipy.optimize import minimize
        opt = minimize(_score, start.copy(), jac = _grad, method = method, **kwargs)
    else:
        assert callable(method), "Method must be a valid method name for scipy.optimize.minimize or a minimizer"
        opt = method(_score, start.copy(), jac = _grad, **kwargs)
    v_mat = diag(opt.x)
    # CALCULATE weights AND ts_score
    weights, _, _ = _weights(v_mat)
    errors = Y_treated - weights.T.dot(Y_control)
    ts_loss = opt.fun
    ts_score = linalg.norm(errors) / sqrt(prod(errors.shape))

    if intercept:
        for i in range(len(splits)):
            weights[out_controls[i], i] += 1/len(out_controls[i])

    return weights, v_mat, ts_score, ts_loss, L2_PEN_W, opt



def fold_weights(X,
                 V,
                 L2_PEN_W,
                 treated_units = None,
                 control_units = None,
                 intercept = True,
                 grad_splits = 5,
                 random_state = 10101,
                 verbose=False):
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
    [N0, N1] = [len(control_units), len(treated_units)]

    splits = grad_splits # for readability...
    try:
        iter(splits)
    except TypeError: 
        from sklearn.model_selection import KFold
        splits = KFold(splits, shuffle=True, random_state = random_state).split(np.arange(len(treated_units)))
    splits = list(splits)

    # index with positions of the controls relative to the incoming data
    in_controls = [list(set(control_units) - set(treated_units[test])) for _,test in splits]
    in_controls2 = [np.ix_(i,i) for i in in_controls] # this is a much faster alternative to A[:,index][index,:]

    # index of the controls relative to the rows of the outgoing N0 x N1 matrix of weights
    ctrl_rng = np.arange(len(control_units))
    out_controls = [ctrl_rng[np.logical_not(np.isin(control_units, treated_units[test]))] for _,test in splits] 
    # this is non-trivial when there control units are also being predicted:
    #out_treated  = [ctrl_rng[               np.isin(control_units, treated_units[test]) ] for train,test in splits] 

    # constants for indexing
    # X_control = X[control_units,:]
    # X_treat = X[treated_units,:]
    weights = zeros((N0, N1))

    A = X.dot(V + V.T).dot(X.T) + 2 * L2_PEN_W * diag(ones(X.shape[0])) # 5
    B = X.dot(V + V.T).dot(X.T).T # 6

    for i, (_,test) in enumerate(splits):
        if verbose >=2:  # for large sample sizes, linalg.solve is a huge bottle neck,
            print("Calculating weights, linalg.solve() call %s of %s" % (i,len(splits),))
        (b) = linalg.solve(A[in_controls2[i]], B[np.ix_(in_controls[i], treated_units[test])])
        indx2 = np.ix_(out_controls[i], test)
        weights[indx2] = b
        if intercept:
            weights[indx2] += 1/len(out_controls[i])
    return weights.T



def fold_score(Y, X, V, L2_PEN_W, LAMBDA = 0, treated_units = None, control_units = None,**kwargs):
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
    weights = fold_weights(X = X,
                           V = V,
                           L2_PEN_W = L2_PEN_W,
                           treated_units = treated_units,
                           control_units = control_units,
                           **kwargs)
    Y_tr = Y[treated_units, :]
    Y_c = Y[control_units, :]
    Ey = (Y_tr - weights.dot(Y_c)).getA()
    return (Ey **2).sum() + LAMBDA * V.sum()
