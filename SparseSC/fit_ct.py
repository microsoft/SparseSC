from numpy import ones, diag, matrix, ndarray, zeros, absolute, mean,var, linalg, prod, sqrt
import numpy as np
import warnings
from SparseSC.optimizers.cd_line_search import cdl_search
warnings.filterwarnings('ignore')

def ct_v_matrix(X,
                Y,
                LAMBDA = 0,
                treated_units = None,
                control_units = None,
                start = None,
                L2_PEN_W = None,
                method = cdl_search, 
                intercept = True,
                max_lambda = False,  # this is terrible at least without documentation...
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
    :param L2_PEN_W: L2 penalty on the magnitude of the deviance of the weight vector from null. Optional.
    :param method: The name of a method to be used by scipy.optimize.minimize, 
        or a callable with the same API as scipy.optimize.minimize
    :param intercept: If True, weights are penalized toward the 1 / the number 
        of controls, else weights are penalized toward zero
    :param max_lambda: if True, the return value is the maximum L1 penalty for
        which at least one element of the tensor matrix is non-zero
    :param verbose: If true, print progress to the console (default: false)
    :param kwargs: additional arguments passed to the optimizer

    :raises ValueError: raised when parameter values are invalid
    :raises TypeError: raised when parameters are of the wrong type

    :return: something something
    :rtype: something something
    '''
    # DEFAULTS
    if treated_units is None: 
        if control_units is None: 
            raise ValueError("At least on of treated_units or control_units is required")
        # Set the treated units to the not-control units
        treated_units = list(set(range(X.shape[0])) - set(control_units))  
    if control_units is None: 
        control_units = list(set(range(X.shape[0])) - set(treated_units)) 

    # Parameter QC
    if set(treated_units).intersection(control_units):
        raise ValueError("Treated and Control units must be exclusive")
    try:
        X = np.asmatrix(X)
    except ValueError:
        raise ValueError("X is not coercible to a matrix")
    try:
        Y = np.asmatrix(Y)
    except ValueError:
        raise ValueError("Y is not coercible to a matrix")
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

    # CONSTANTS
    N0, N1, K = len(control_units), len(treated_units), X.shape[1]
    if start is None: 
        start = zeros(K) # formerly: .1 * ones(K) 
    Y_treated = Y[treated_units,:]
    Y_control = Y[control_units,:]
    X_treated = X[treated_units,:]
    X_control = X[control_units,:]

    if intercept: 
        Y = Y.copy()
        Y_treated -= Y_control.mean(axis=0) 

    # INITIALIZE PARTIAL DERIVATIVES
    dA_dV_ki = [ X_control[:, k ].dot(X_control[:, k ].T) +  # i,j are on the diagonal (both equal to k)
                 X_control[:, k ].dot(X_control[:, k ].T) for k in range(K)] # 8
    dB_dV_ki = [ X_control[:, k ].dot(X_treated[:, k ].T) +  # i,j are on the diagonal (both equal to k)
                 X_control[:, k ].dot(X_treated[:, k ].T) for k in range(K)] # 9

    def _score(V):
        dv = diag(V)
        weights, _, _ ,_ = _weights(dv)
        Ey = (Y_treated - weights.T.dot(Y_control)).getA()
        # note that (...).copy() assures that x.flags.writeable is True:
        return ((Ey **2).sum() + LAMBDA * absolute(V).sum()).copy() 

    def _grad(V):
        """ Calculates just the diagonal of dGamma0_dV

            There is an implementation that allows for all elements of V to be varied...
        """
        dv = diag(V)
        weights, A, _, AinvB = _weights(dv)
        Ey = (Y_treated - weights.T.dot(Y_control)).getA()
        dGamma0_dV_term2 = zeros(K)
        dPI_dV = zeros((N0, N1))
        #Ai = A.I
        for k in range(K):
            if verbose:  # for large sample sizes, linalg.solve is a huge bottle neck,
                print("Calculating gradient, linalg.solve() call %s of %s" % (k ,K,))
            dPI_dV.fill(0) # faster than re-allocating the memory each loop.
            dA = dA_dV_ki[k]
            dB = dB_dV_ki[k]
            dPI_dV = linalg.solve(A,(dB - dA.dot(AinvB))) 
            #dPI_dV = Ai.dot(dB - dA.dot(AinvB))
            dGamma0_dV_term2[k] = (Ey * Y_control.T.dot(dPI_dV).T.getA()).sum()
        return LAMBDA - 2 * dGamma0_dV_term2

    L2_PEN_W_mat = 2 * L2_PEN_W * diag(ones(X_control.shape[0]))
    def _weights(V):
        weights = zeros((N0, N1))
        A = X_control.dot(2*V).dot(X_control.T) + L2_PEN_W_mat # 5
        B = X_treated.dot(2*V).dot(X_control.T).T # 6
        b = linalg.solve(A,B)
        if intercept:
            weights = b + 1/N0
        else:
            weights = b
        return weights, A, B,b

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
    weights, _, _ ,_ = _weights(v_mat)
    errors = Y_treated - weights.T.dot(Y_control)
    ts_loss = opt.fun
    ts_score = linalg.norm(errors) / sqrt(prod(errors.shape))

    #if True:
    #    _do_gradient_check()
    if intercept: 
        # not above, b/c Y_treated was already offset at the start
        weights += 1/N0 
    return weights, v_mat, ts_score, ts_loss, L2_PEN_W, opt

def ct_weights(X, V, L2_PEN_W, treated_units = None, control_units = None, intercept = True):
    if treated_units is None: 
        if control_units is None: 
            raise ValueError("At least on of treated_units or control_units is required")
        # Set the treated units to the not-control units
        treated_units = list(set(range(X.shape[0])) - set(control_units))  
    if control_units is None: 
        control_units = list(set(range(X.shape[0])) - set(treated_units)) 

    N0 = len(control_units)
    X_treated = X[treated_units,:]
    X_control = X[control_units,:]

    A = X_control.dot(2*V).dot(X_control.T) + 2 * L2_PEN_W * diag(ones(X_control.shape[0])) # 5
    B = X_treated.dot(2*V).dot(X_control.T).T # 6

    weights = linalg.solve(A,B)
    if intercept:
        weights += 1/N0
    return weights.T

def ct_score(Y, X, V, L2_PEN_W, LAMBDA = 0, treated_units = None, control_units = None,**kwargs):
    if treated_units is None: 
        if control_units is None: 
            raise ValueError("At least on of treated_units or control_units is required")
        # Set the treated units to the not-control units
        treated_units = list(set(range(X.shape[0])) - set(control_units))  
    if control_units is None: 
        control_units = list(set(range(X.shape[0])) - set(treated_units)) 
    weights = ct_weights(X = X,
                         V = V,
                         L2_PEN_W = L2_PEN_W,
                         treated_units = treated_units,
                         control_units = control_units,
                         **kwargs)
    Y_tr = Y[treated_units, :]
    Y_c = Y[control_units, :]
    Ey = (Y_tr - weights.dot(Y_c)).getA()
    np.power(Y_tr - np.mean(Y_tr),2).sum()
    return (Ey **2).sum() + LAMBDA * V.sum()
