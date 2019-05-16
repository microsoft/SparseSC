""" 
Implements leave-one-out gradient descent methods
"""

from numpy import ones, diag, zeros, absolute, mean, var, linalg, prod, sqrt
import numpy as np
import itertools

# only used by the step-down method (currently not implemented):
# from SparseSC.utils.sub_matrix_inverse import subinv_k, all_subinverses
from .utils.print_progress import print_progress
from SparseSC.optimizers.cd_line_search import cdl_search


def complete_treated_control_list(N, treated_units=None, control_units=None):
    """ 
    a utility function for calculating the ``treated_units`` from the
    ``control_units``, and vice versa
    """
    if treated_units is None:
        if control_units is None:
            # both not provided, include all samples as both treat and control unit.
            control_units = list(range(N))
            treated_units = control_units
        else:
            # Set the treated units to the not-control units
            treated_units = list(set(range(N)) - set(control_units))
    else:
        if control_units is None:
            # Set the control units to the not-treated units
            control_units = list(set(range(N)) - set(treated_units))
    return (treated_units, control_units)


def loo_v_matrix(
    X,
    Y,
    v_pen=0,
    treated_units=None,
    control_units=None,
    start=None,
    w_pen=None,
    method=cdl_search,
    return_max_v_pen=False,  # this is terrible at least without documentation...
    solve_method="standard",  # specific to fit_loo
    verbose=False,
    gradient_message="Calculating gradient",
    **kwargs
):
    """
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

    :param return_max_v_pen: (Internal API) If ``True``, the return value is
                the maximum L1 penalty for which at least one element of the
                tensor matrix is non-zero.
    :type return_max_v_pen: boolean

    :param solve_method: Method for solving A.I.dot(B). Either "standard" or
        "step-down". https://math.stackexchange.com/a/208021/252693
    :type solve_method: str

    :param verbose: If true, print progress to the console (default: false)
    :type verbose: boolean

    :param gradient_message: Messaged prefixed to the progress bar when verbose = 1
    :type gradient_message: str

    :param kwargs: additional arguments passed to the optimizer
    :type kwargs:

    :raises ValueError: raised when parameter values are invalid
    :raises TypeError: raised when parameters are of the wrong type

    :return: something something
    :rtype: something something
    """
    treated_units, control_units = complete_treated_control_list(
        X.shape[0], treated_units, control_units
    )
    control_units = np.array(control_units)
    treated_units = np.array(treated_units)

    # parameter QC
    try:
        X = np.asmatrix(X)
    except ValueError:
        raise TypeError("X is not coercible to a matrix")
    try:
        Y = np.asmatrix(Y)
    except ValueError:
        raise TypeError("Y is not coercible to a matrix")
    if X.shape[1] == 0:
        raise ValueError("X.shape[1] == 0")
    if Y.shape[1] == 0:
        raise ValueError("Y.shape[1] == 0")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            "X and Y have different number of rows (%s and %s)"
            % (X.shape[0], Y.shape[0])
        )
    if not isinstance(v_pen, (float, int)):
        raise TypeError("v_pen is not a number")
    if w_pen is None:
        w_pen = mean(var(X, axis=0))
    else:
        w_pen = float(w_pen)
    if not isinstance(w_pen, (float, int)):
        raise TypeError("w_pen is not a number")

    # CONSTANTS
    N0, N1, K = len(control_units), len(treated_units), X.shape[1]
    if start is None:
        start = zeros(K)  # formerly: .1 * ones(K)
    assert N1 > 0, "No control units"
    assert N0 > 0, "No treated units"
    assert K > 0, "variables to fit (X.shape[1] == 0)"

    # CREATE THE INDEX THAT INDICATES THE ELIGIBLE CONTROLS FOR EACH TREATED UNIT
    in_controls = [
        list(set(control_units) - set([trt_unit])) for trt_unit in treated_units
    ]
    in_controls2 = [np.ix_(i, i) for i in in_controls]
    ctrl_rng = np.arange(len(control_units))
    out_controls = [ctrl_rng[control_units != trt_unit] for trt_unit in treated_units]
    # this is non-trivial when there control units are also being predicted:
    # out_treated  = [ctrl_rng[control_units == trt_unit] for trt_unit in treated_units]

    # --     if intercept:
    # --         Y = Y.copy()
    # --         for i, trt_unit in enumerate(treated_units):
    # --             Y[trt_unit,:] -= Y[in_controls[i],:].mean(axis=0)

    # handy constants (for speed purposes):
    Y_treated = Y[treated_units, :]
    Y_control = Y[control_units, :]
    # only used by step-down method: X_treated = X[treated_units,:]
    # only used by step-down method: X_control = X[control_units,:]

    # INITIALIZE PARTIAL DERIVATIVES
    # note that this section can be quite memory intensive with lots of
    # controls: (1000 controls -> 8 MB per entry)
    dA_dV_ki = [[None] * N1 for i in range(K)]
    dB_dV_ki = [[None] * N1 for i in range(K)]
    b_i = [None] * N1
    for i, k in itertools.product(range(N1), range(K)):  # TREATED unit i, moment k
        Xc = X[in_controls[i], :]
        Xt = X[treated_units[i], :]
        dA_dV_ki[k][i] = 2 * Xc[:, k].dot(Xc[:, k].T)  # 8
        dB_dV_ki[k][i] = 2 * Xc[:, k].dot(Xt[:, k].T)  # 9

    k = 0  # for linting...
    del Xc, Xt

    # assert (dA_dV_ki [k][i] == X[index, k ].dot(X[index, k ].T) + X[index, k ].dot(X[index, k ].T)).all() #pylint: disable=line-too-long
    # https://math.stackexchange.com/a/1471836/252693

    def _score(V):
        dv = diag(V)
        weights, _, _ = _weights(dv)
        Ey = (Y_treated - weights.T.dot(Y_control)).getA()

        # (...).copy() assures that x.flags.writeable is True:
        # also einsum is faster than the equivalent (Ey **2).sum()
        return (np.einsum("ij,ij->", Ey, Ey) + v_pen * absolute(V).sum()).copy()  #

    def _grad(V):
        """ 
        Calculates just the diagonal of dGamma0_dV

        There is an implementation that allows for all elements of V to be varied...
        """
        dv = diag(V)
        weights, A, _ = _weights(dv)
        Ey = (weights.T.dot(Y_control) - Y_treated).getA()
        dGamma0_dV_term2 = zeros(K)
        dPI_dV = zeros((N0, N1))  # stupid notation: PI = W.T
        # if solve_method == "step-down": Ai_cache = all_subinverses(A)
        for k in range(K):
            if verbose:  # for large sample sizes, linalg.solve is a huge bottle neck,
                print_progress(
                    k + 1, K, prefix=gradient_message, decimals=1, bar_length=min(K, 50)
                )
            dPI_dV.fill(0)  # faster than re-allocating the memory each loop.
            for i, index in enumerate(in_controls):
                dA = dA_dV_ki[k][i]
                dB = dB_dV_ki[k][i]
                if solve_method == "step-down":  # pylint: disable=no-else-raise
                    raise NotImplementedError(
                        "The solve_method 'step-down' is currently not implemented"
                    )  # pylint: disable=line-too-long
                    # b = Ai_cache[i].dot(dB - dA.dot(b_i[i]))
                else:
                    if (
                        verbose >= 2
                    ):  # for large sample sizes, linalg.solve is a huge bottle neck,
                        print(
                            "Calculating weights, linalg.solve() call %s of %s"
                            % (i + k * K, K * len(in_controls))
                        )
                    try:
                        b = linalg.solve(A[in_controls2[i]], dB - dA.dot(b_i[i]))
                    except linalg.LinAlgError as exc:
                        print("Unique weights not possible.")
                        if w_pen == 0:
                            print("Try specifying a very small w_pen rather than 0.")
                        raise exc
                dPI_dV[index, i] = b.flatten()  # TODO: is the Transpose  an error???

            # einsum is faster than the equivalent (Ey * Y_control.T.dot(dPI_dV).T.getA()).sum()
            dGamma0_dV_term2[k] = 2 * np.einsum("ij,kj,ki->", Ey, Y_control, dPI_dV)
        return v_pen + dGamma0_dV_term2

    def _weights(V):
        weights = zeros((N0, N1))
        if solve_method == "step-down":  # pylint: disable=no-else-raise
            raise NotImplementedError(
                "The solve_method 'step-down' is currently not implemented"
            )
            # A = (X_control.dot(V + V.T).dot(X_control.T)
            #      + 2 * w_pen * diag(ones(X_control.shape[0]))) # 5
            # B = X_treated.dot(V + V.T).dot(X_control.T) # 6
            # Ai = A.I
            # for i, trt_unit in enumerate(treated_units):
            #     if trt_unit in control_units:
            #         (b) = subinv_k(Ai,_k).dot(B[out_controls[i],i])
            #     else:
            #         (b) = Ai.dot(B[:, i])
            #     b_i[i] = b
            #     weights[out_controls[i], i] = b.flatten()
        elif solve_method == "standard":
            A = X.dot(V + V.T).dot(X.T) + 2 * w_pen * diag(ones(X.shape[0]))  # 5
            B = X.dot(V + V.T).dot(X.T).T  # 6
            for i, trt_unit in enumerate(treated_units):
                if (
                    verbose >= 2
                ):  # for large sample sizes, linalg.solve is a huge bottle neck,
                    print(
                        "Calculating weights, linalg.solve() call %s of %s"
                        % (i, len(in_controls))
                    )
                try:
                    (b) = b_i[i] = linalg.solve(
                        A[in_controls2[i]],
                        B[in_controls[i], trt_unit] + 2 * w_pen / len(in_controls[i]),
                    )  # pylint: disable=line-too-long
                except linalg.LinAlgError as exc:
                    print("Unique weights not possible.")
                    if w_pen == 0:
                        print("Try specifying a very small w_pen rather than 0.")
                    raise exc
                weights[out_controls[i], i] = b.flatten()
        else:
            raise ValueError("Unknown Solve Method: " + solve_method)
        return weights, A, B

    if return_max_v_pen:
        grad0 = _grad(zeros(K))
        return -grad0[grad0 < 0].min()

    # DO THE OPTIMIZATION
    if isinstance(method, str):
        from scipy.optimize import minimize

        opt = minimize(_score, start.copy(), jac=_grad, method=method, **kwargs)
    else:
        assert callable(
            method
        ), "Method must be a valid method name for scipy.optimize.minimize or a minimizer"  # pylint: disable=line-too-long
        opt = method(_score, start.copy(), jac=_grad, **kwargs)
    v_mat = diag(opt.x)
    # CALCULATE weights AND ts_score
    weights, _, _ = _weights(v_mat)
    errors = Y_treated - weights.T.dot(Y_control)
    ts_loss = opt.fun
    ts_score = linalg.norm(errors) / sqrt(prod(errors.shape))

    return weights, v_mat, ts_score, ts_loss, w_pen, opt


def loo_weights(
    X,
    V,
    w_pen,
    treated_units=None,
    control_units=None,
    solve_method="standard",
    verbose=False,
    custom_donor_pool=None,
):
    """ 
    Fit the weights using the leave-one-out gradient approach
    """
    treated_units, control_units = complete_treated_control_list(
        X.shape[0], treated_units, control_units
    )
    control_units = np.array(control_units)
    treated_units = np.array(treated_units)
    [N0, N1] = [len(control_units), len(treated_units)]

    # index with positions of the controls relative to the incoming data
    in_controls = [
        list(set(control_units) - set([trt_unit])) for trt_unit in treated_units
    ]
    in_controls2 = [np.ix_(i, i) for i in in_controls]

    # index of the controls relative to the rows of the outgoing N0 x N1 matrix of weights
    ctrl_rng = np.arange(len(control_units))
    out_controls = [ctrl_rng[control_units != trt_unit] for trt_unit in treated_units]
    # this is non-trivial when there control units are also being predicted:
    # out_treated  = [ctrl_rng[control_units == trt_unit] for trt_unit in treated_units]

    # constants for indexing
    # > only used by the step-down method (currently not implemented) X_control = X[control_units,:]
    # > only used by the step-down method (currently not implemented) X_treat = X[treated_units,:]
    weights = zeros((N0, N1))

    if solve_method == "step-down":  # pylint: disable=no-else-raise
        raise NotImplementedError(
            "The solve_method 'step-down' is currently not implemented"
        )
        # A = (X_control.dot(V + V.T).dot(X_control.T)
        #      + 2 * w_pen * diag(ones(X_control.shape[0]))) # 5
        # B = X_treat.dot(  V + V.T).dot(X_control.T) # 6
        # Ai = A.I
        # for i, trt_unit in enumerate(treated_units):
        #     if trt_unit in control_units:
        #         (b) = subinv_k(Ai,_k).dot(B[out_controls[i],i])
        #     else:
        #         (b) = Ai.dot(B[:, i])
        #     weights[out_controls[i], i] = b.flatten()
    elif solve_method == "standard":
        if custom_donor_pool is None:
            A = X.dot(V + V.T).dot(X.T) + 2 * w_pen * diag(ones(X.shape[0]))  # 5
            B = X.dot(V + V.T).dot(X.T).T  # 6
            for i, trt_unit in enumerate(treated_units):
                if (
                    verbose >= 2
                ):  # for large sample sizes, linalg.solve is a huge bottle neck,
                    print(
                        "Calculating weights, linalg.solve() call %s of %s"
                        % (i, len(treated_units))
                    )  # pylint: disable=line-too-long
                try:
                    (b) = linalg.solve(
                        A[in_controls2[i]],
                        B[in_controls[i], trt_unit] + 2 * w_pen / len(in_controls[i]),
                    )
                except linalg.LinAlgError as exc:
                    print("Unique weights not possible.")
                    if w_pen == 0:
                        print("Try specifying a very small w_pen rather than 0.")
                    raise exc

                weights[out_controls[i], i] = b.flatten()
        else:
            for i, trt_unit in enumerate(treated_units):
                donors = np.where(custom_donor_pool[trt_unit, :])
                A = X[donors, :].dot(2 * V).dot(X[donors, :].T) + 2 * w_pen * diag(
                    ones(X[donors, :].shape[0])
                )  # 5
                B = (
                    X[trt_unit, :].dot(2 * V).dot(X[donors, :].T).T
                    + 2 * w_pen / X[donors, :].shape[0]
                )  # 6
                try:
                    weights[donors, i] = linalg.solve(A, B)
                except linalg.LinAlgError as exc:
                    print("Unique weights not possible.")
                    if w_pen == 0:
                        print("Try specifying a very small w_pen rather than 0.")
                    raise exc
    else:
        raise ValueError("Unknown Solve Method: " + solve_method)
    return weights.T


def loo_score(
    Y, X, V, w_pen, v_pen=0, treated_units=None, control_units=None, **kwargs
):
    """ 
    in-sample residual error using the leave-one-out gradient approach
    """
    treated_units, control_units = complete_treated_control_list(
        X.shape[0], treated_units, control_units
    )
    weights = loo_weights(
        X=X,
        V=V,
        w_pen=w_pen,
        treated_units=treated_units,
        control_units=control_units,
        **kwargs
    )
    Y_tr = Y[treated_units, :]
    Y_c = Y[control_units, :]
    Ey = (Y_tr - weights.dot(Y_c)).getA()
    return np.einsum("ij,ij->", Ey, Ey) + v_pen * V.sum()  # (Ey **2).sum() -> einsum
