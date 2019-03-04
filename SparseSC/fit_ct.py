""" Implements cross-train gradient descent methods
"""
from numpy import ones, diag, zeros, absolute, mean,var, linalg, prod, sqrt
import numpy as np
import warnings
from .utils.print_progress import print_progress
from SparseSC.optimizers.cd_line_search import cdl_search
warnings.filterwarnings('ignore')

def ct_v_matrix(X,
                Y,
                v_pen = 0,
                treated_units = None,
                control_units = None,
                start = None,
                w_pen = None,
                method = cdl_search,
                max_lambda = False,  # this is terrible at least without documentation...
                verbose = False,
                gradient_message = "Calculating gradient",
                **kwargs):
    '''
    Computes and sets the optimal v_matrix for the given moments and
        penalty parameter.

    :param X: Matrix of Covariates
    :type X: coercible to :class:`numpy.matrix`

    :param Y: Matrix of Outcomes
    :type Y: coercible to :class:`numpy.matrix`

    :param v_pen: penalty parameter used to shrink L1 norm of v/v.max() toward zero
    :type v_pen: float

    :param treated_units: a list containing the position (rows) of the treated units within X and Y
    :type treated_units: int[] or numpy.ndarray

    :param control_units: a list containing the position (rows) of the control units within X and Y
    :type control_units: numpy.ndarray

    :param start: initial values for the diagonals of the tensor matrix
    :type start: float[] or numpy.ndarray

    :param w_pen: L2 penalty on the magnitude of the deviance of the weight
                     vector from null. Optional.
    :type w_pen: float

    :param method: The name of a method to be used by scipy.optimize.minimize,
                   or a callable with the same API as scipy.optimize.minimize
    :type method: str or callable

    :param max_lambda: (Internal API) If ``True``, the return value is the maximum L1 penalty for
                       which at least one element of the tensor matrix is
                       non-zero.
    :type max_lambda: boolean

    :param verbose: If true, print progress to the console (default: false)
    :type verbose: boolean

    :param gradient_message: Messaged prefixed to the progress bar when verbose = 1
    :type gradient_message: str

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
        raise ValueError("X and Y have different number of rows (%s and %s)"
                         % (X.shape[0], Y.shape[0],))
    if not isinstance(v_pen, (float, int)):
        raise TypeError( "v_pen is not a number")
    if w_pen is None:
        w_pen = mean(var(X, axis = 0))
    else:
        w_pen = float(w_pen)
    if not isinstance(w_pen, (float, int)):
        raise TypeError( "w_pen is not a number")

    # CONSTANTS
    N0, N1, K = len(control_units), len(treated_units), X.shape[1]
    if start is None:
        start = zeros(K) # formerly: .1 * ones(K)
    Y_treated = Y[treated_units,:]
    Y_control = Y[control_units,:]
    X_treated = X[treated_units,:]
    X_control = X[control_units,:]

    # INITIALIZE PARTIAL DERIVATIVES
    dA_dV_ki = [ 2 * X_control[:, k ].dot(X_control[:, k ].T) for k in range(K)] # 8
    dB_dV_ki = [ 2 * X_control[:, k ].dot(X_treated[:, k ].T) for k in range(K)] # 9

    def _score(V):
        dv = diag(V)
        weights, _, _ ,_ = _weights(dv)
        Ey = (Y_treated - weights.T.dot(Y_control)).getA()
        # note that (...).copy() assures that x.flags.writeable is True:
        # also einsum is faster than the equivalent (Ey **2).sum()
        return (np.einsum('ij,ij->',Ey,Ey) + v_pen * absolute(V).sum()).copy()

    def _grad(V):
        """ Calculates just the diagonal of dGamma0_dV

            There is an implementation that allows for all elements of V to be varied...
        """
        dv = diag(V)
        weights, A, _, AinvB = _weights(dv)
        Ey = (weights.T.dot(Y_control) - Y_treated).getA()
        dGamma0_dV_term2 = zeros(K)
        #dPI_dV = zeros((N0, N1)) # stupid notation: PI = W.T
        #Ai = A.I
        for k in range(K):
            if verbose:  # for large sample sizes, linalg.solve is a huge bottle neck,
                print_progress(k+1,K, prefix=gradient_message, decimals=1, bar_length=min(K,50))
            #dPI_dV.fill(0) # faster than re-allocating the memory each loop.
            dA = dA_dV_ki[k]
            dB = dB_dV_ki[k]
            try:
                dPI_dV = linalg.solve(A,(dB - dA.dot(AinvB)))
            except linalg.LinAlgError as exc:
                print("Unique weights not possible.")
                if w_pen==0:
                    print("Try specifying a very small w_pen rather than 0.")
                raise exc
            #dPI_dV = Ai.dot(dB - dA.dot(AinvB))
            # faster than the equivalent (Ey * Y_control.T.dot(dPI_dV).T.getA()).sum()
            dGamma0_dV_term2[k] = np.einsum("ij,kj,ki->",Ey, Y_control, dPI_dV)
        return v_pen + 2 * dGamma0_dV_term2

    w_pen_mat = 2 * w_pen * diag(ones(X_control.shape[0]))
    def _weights(V):
        weights = zeros((N0, N1))
        A = X_control.dot(2*V).dot(X_control.T) + w_pen_mat # 5
        B = X_treated.dot(2*V).dot(X_control.T).T + 2 * w_pen / X_control.shape[0] # 6
        try:
            b = linalg.solve(A,B)
        except linalg.LinAlgError as exc:
            print("Unique weights not possible.")
            if w_pen==0:
                print("Try specifying a very small w_pen rather than 0.")
            raise exc
        return weights, A, B,b

    if max_lambda:
        grad0 = _grad(zeros(K))
        return -grad0[grad0 < 0].min()

    # DO THE OPTIMIZATION
    if isinstance(method, str):
        from scipy.optimize import minimize
        opt = minimize(_score, start.copy(), jac = _grad, method = method, **kwargs)
    else:
        assert callable(method), "Method must be a valid method name for scipy.optimize.minimize or a minimizer" #pylint: disable=line-too-long
        opt = method(_score, start.copy(), jac = _grad, **kwargs)
    v_mat = diag(opt.x)

    # CALCULATE weights AND ts_score
    weights, _, _ ,_ = _weights(v_mat)
    errors = Y_treated - weights.T.dot(Y_control)
    ts_loss = opt.fun
    ts_score = linalg.norm(errors) / sqrt(prod(errors.shape))

    return weights, v_mat, ts_score, ts_loss, w_pen, opt

def ct_weights(X,
               V,
               w_pen,
               treated_units = None,
               control_units = None,
               custom_donor_pool = None):
    """ fit the weights using the cross-train gradient approach
    """
    if treated_units is None:
        if control_units is None:
            raise ValueError("At least on of treated_units or control_units is required")
        # Set the treated units to the not-control units
        treated_units = list(set(range(X.shape[0])) - set(control_units))
    if control_units is None:
        control_units = list(set(range(X.shape[0])) - set(treated_units))

    def _calc_W_ct(X_treated, X_control, V, w_pen):
        A = X_control.dot(2*V).dot(X_control.T)   + 2 * w_pen * diag(ones(X_control.shape[0])) # 5
        B = X_treated.dot(2*V).dot(X_control.T).T + 2 * w_pen / X_control.shape[0]# 6
        try:
            weights = linalg.solve(A,B).T
        except linalg.LinAlgError as exc:
            print("Unique weights not possible.")
            if w_pen==0:
                print("Try specifying a very small w_pen rather than 0.")
            raise exc
        return weights


    if custom_donor_pool is None:
        weights = _calc_W_ct(X[treated_units,:], X[control_units,:], V, w_pen)
    else:
        weights = np.zeros((len(treated_units),len(control_units)))
        for i, treated_unit in enumerate(treated_units):
            donors = np.where(custom_donor_pool[treated_unit,:])
            weights[i,donors] = _calc_W_ct(X[treated_unit,:], X[donors,:], V, w_pen)


    return weights

def ct_score(Y,
             X,
             V,
             w_pen,
             v_pen = 0,
             treated_units = None,
             control_units = None,
             **kwargs):
    """ in-sample residual error using the cross-train approach
    """
    if treated_units is None:
        if control_units is None:
            raise ValueError("At least on of treated_units or control_units is required")
        # Set the treated units to the not-control units
        treated_units = list(set(range(X.shape[0])) - set(control_units))
    if control_units is None:
        control_units = list(set(range(X.shape[0])) - set(treated_units))
    weights = ct_weights(X = X,
                         V = V,
                         w_pen = w_pen,
                         treated_units = treated_units,
                         control_units = control_units,
                         **kwargs)
    Y_tr = Y[treated_units, :]
    Y_c = Y[control_units, :]
    Ey = (Y_tr - weights.dot(Y_c)).getA()
    return np.einsum('ij,ij->',Ey,Ey) + v_pen * V.sum() # (Ey **2).sum() -> einsum

