""" Implements k-fold gradient descent methods
"""
from numpy import ones, diag, zeros, mean,var, linalg, prod, sqrt, absolute
import numpy as np
import itertools
import warnings
from .utils.print_progress import print_progress
from SparseSC.optimizers.cd_line_search import cdl_search
warnings.filterwarnings('ignore')


def fold_v_matrix(X,
                  Y,
                  v_pen = 0,
                  treated_units = None,
                  control_units = None,
                  non_neg_weights = False,
                  start = None,
                  w_pen = None,
                  method = cdl_search,
                  return_max_v_pen = False,  # this is terrible at least without documentation...
                  grad_splits = 5,
                  random_state = 10101,
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

    :param w_pen: weight penalty on the magnitude of the deviance of the weight
                     vector from null. Optional.
    :type w_pen: float

    :param method: The name of a method to be used by scipy.optimize.minimize,
                   or a callable with the same API as scipy.optimize.minimize
    :type method: str or callable

    :param return_max_v_pen: (Internal API) If ``True``, the return value is the maximum L1 penalty for
                       which at least one element of the tensor matrix is
                       non-zero.
    :type return_max_v_pen: boolean

    :param grad_splits: Splits for Fitted v.s. Control units in each gradient
                        descent step. An integer, or a list/generator of train
                        and test units in each fold of the gradient descent.
    :type grad_splits: int or int[][]

    :param random_state: Integer, used for setting the random state for
        consistency of fold splits across calls
    :type random_state:

    :param verbose: If true, print progress to the console (default: false)
    :type verbose: boolean

    :param gradient_message: Messaged prefixed to the progress bar when verbose = 1
    :type gradient_message: str

    :param kwargs: additional arguments passed to the optimizer
    :type kwargs:

    :param non_neg_weights: not implemented
    :type non_neg_weights:

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
        raise ValueError("X and Y have different number of rows (%s and %s)" %
                         (X.shape[0], Y.shape[0],))
    if not isinstance(v_pen, (float, int)):
        raise TypeError( "v_pen is not a number")
    if w_pen is None:
        w_pen = mean(var(X, axis = 0))
    else:
        w_pen = float(w_pen)
    if not isinstance(w_pen, (float, int)):
        raise TypeError( "w_pen is not a number")
    assert not non_neg_weights, "Bounds not implemented"

    splits = grad_splits # for readability...
    try:
        iter(splits)
    except TypeError:
        from sklearn.model_selection import KFold
        splits = KFold(splits,
                       shuffle=True,
                       random_state = random_state).split(np.arange(len(treated_units)))
    splits = list(splits)

    for i, split in enumerate(splits):
        assert len(split[0]) + len(split[1]) == len(treated_units), \
                ("Splits for fold %s do not match the number of treated units.  Expected %s; got %s + %s" %  #pylint: disable=line-too-long
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
    in_controls2 = [np.ix_(i,i) for i in in_controls]
    ctrl_rng = np.arange(len(control_units))
    out_controls = [ctrl_rng[np.logical_not(np.isin(control_units, treated_units[test]))] for _,test in splits]  #pylint: disable=line-too-long

    # this is non-trivial when there control units are also being predicted:
    #out_treated = [ctrl_rng[np.isin(control_units, treated_units[test]) ] for train,test in splits]

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
        dA_dV_ki [k][i] = 2 * Xc[:, k ].dot(Xc[:, k ].T) # 8
        dB_dV_ki [k][i] = 2 * Xc[:, k ].dot(Xt[:, k ].T) # 9

    k=0 # for linting...
    del Xc, Xt, i, k

    def _score(V):
        dv = diag(V)
        weights, _, _ = _weights(dv)
        Ey = (Y_treated - weights.T.dot(Y_control)).getA()
        # (...).copy() assures that x.flags.writeable is True
        # also einsum is faster than the equivalent (Ey **2).sum()
        return (np.einsum('ij,ij->',Ey,Ey) + v_pen * absolute(V).sum()).copy()

    def _grad(V):
        """ Calculates just the diagonal of dGamma0_dV

            There is an implementation that allows for all elements of V to be varied...
        """
        dv = diag(V)
        weights, A, _ = _weights(dv)
        #Ey = (weights.T.dot(Y_control) - Y_treated).getA()
        dGamma0_dV_term2 = zeros(K)
        dPI_dV = zeros((N0, N1)) # stupid notation: PI = W.T
        for k in range(K):
            if verbose:  # for large sample sizes, linalg.solve is a huge bottle neck,
                print_progress(k+1,K, prefix=gradient_message, decimals=1, bar_length=min(K,50))
            dPI_dV.fill(0) # faster than re-allocating the memory each loop.
            for i, (_, (_, test)) in enumerate(zip(in_controls,splits)):
                if verbose >=2:  # for large sample sizes, linalg.solve is a huge bottle neck,
                    print("Calculating gradient, linalg.solve() call %s of %s"
                          % (i + k*len(splits) ,K*len(splits),))
                dA = dA_dV_ki[k][i]
                dB = dB_dV_ki[k][i]
                try:
                    b = linalg.solve(A[in_controls2[i]],dB - dA.dot(b_i[i]))
                except linalg.LinAlgError as exc:
                    print("Unique weights not possible.")
                    if w_pen==0:
                        print("Try specifying a very small w_pen rather than 0.")
                    raise exc
                dPI_dV[np.ix_(in_controls[i], treated_units[test])] = b
            # einsum is faster than the equivalent (Ey * Y_control.T.dot(dPI_dV).T.getA()).sum()
            dGamma0_dV_term2[k] = 2 * np.einsum("ij,kj,ki->",
                                                (weights.T.dot(Y_control) - Y_treated),
                                                Y_control, dPI_dV)
        return v_pen + dGamma0_dV_term2

    def _weights(V):
        weights = zeros((N0, N1))
        A = X.dot(V + V.T).dot(X.T) + 2 * w_pen * diag(ones(X.shape[0])) # 5
        B = X.dot(V + V.T).dot(X.T).T # 6
        for i, (_,test) in enumerate(splits):
            if verbose >=2:  # for large sample sizes, linalg.solve is a huge bottle neck,
                print("Calculating weights, linalg.solve() call %s of %s" %
                      (i,len(splits),))
            try:
                b = b_i[i] = linalg.solve(A[in_controls2[i]],
                                          B[np.ix_(in_controls[i], treated_units[test])]
                                          + 2 * w_pen / len(in_controls[i]) )
            except linalg.LinAlgError as exc:
                print("Unique weights not possible.")
                if w_pen==0:
                    print("Try specifying a very small w_pen rather than 0.")
                raise exc
            weights[np.ix_(out_controls[i], test)] = b
        return weights, A, B

    if return_max_v_pen:
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
    weights, _, _ = _weights(v_mat)
    errors = Y_treated - weights.T.dot(Y_control)
    ts_loss = opt.fun
    ts_score = linalg.norm(errors) / sqrt(prod(errors.shape))

    return weights, v_mat, ts_score, ts_loss, w_pen, opt



def fold_weights(X,
                 V,
                 w_pen = None,
                 treated_units = None,
                 control_units = None,
                 grad_splits = 5,
                 random_state = 10101,
                 verbose=False):
    """ fit the weights using the k-fold gradient approach
    """
    if w_pen is None:
        w_pen = mean(var(X, axis = 0))
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
        splits = KFold(splits,
                       shuffle=True,
                       random_state = random_state).split(np.arange(len(treated_units)))
    splits = list(splits)

    # index with positions of the controls relative to the incoming data
    in_controls = [list(set(control_units) - set(treated_units[test])) for _,test in splits]
    in_controls2 = [np.ix_(i,i) for i in in_controls]

    # index of the controls relative to the rows of the outgoing N0 x N1 matrix of weights
    ctrl_rng = np.arange(len(control_units))
    out_controls = [ctrl_rng[np.logical_not(np.isin(control_units, treated_units[test]))] for _,test in splits]  #pylint: disable=line-too-long

    weights = zeros((N0, N1))
    A = X.dot(V + V.T).dot(X.T) + 2 * w_pen * diag(ones(X.shape[0])) # 5
    B = X.dot(V + V.T).dot(X.T).T # 6

    for i, (_,test) in enumerate(splits):
        if verbose >=2:  # for large sample sizes, linalg.solve is a huge bottle neck,
            print("Calculating weights, linalg.solve() call %s of %s" % (i,len(splits),)) #pylint: disable=line-too-long
        try:
            b = linalg.solve(A[in_controls2[i]],
                             B[np.ix_(in_controls[i], treated_units[test])] + 2 * w_pen / len(in_controls[i])) #pylint: disable=line-too-long
        except linalg.LinAlgError as exc:
            print("Unique weights not possible.")
            if w_pen==0:
                print("Try specifying a very small w_pen rather than 0.")
            raise exc

        indx2 = np.ix_(out_controls[i], test)
        weights[indx2] = b
    return weights.T


def fold_score(Y,
               X,
               V,
               w_pen,
               v_pen = 0,
               treated_units = None,
               control_units = None,
               **kwargs):
    """ in-sample residual error using the k=fold gradient approach
    """
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
                           w_pen = w_pen,
                           treated_units = treated_units,
                           control_units = control_units,
                           **kwargs)
    Y_tr = Y[treated_units, :]
    Y_c = Y[control_units, :]
    Ey = (Y_tr - weights.dot(Y_c)).getA()
    return np.einsum('ij,ij->',Ey,Ey) + v_pen * V.sum() # (Ey **2).sum() -> einsum
