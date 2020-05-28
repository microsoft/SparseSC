""" Fast API providing a single call for fitting SC Models

"""
import tracemalloc

import numpy as np
import scipy.linalg #superset of np.linalg and also optimized compiled

from .fit import SparseSCFit
from .utils.penalty_utils import RidgeCVSolution
from .utils.match_space import MTLassoCV_MatchSpace_factory
from .utils.misc import _ensure_good_donor_pool, _get_fit_units
from .utils.print_progress import print_memory_snapshot, log_if_necessary, print_progress

#not documenting the error for when trying to two function signatures (think of better way to do that)
def fit_fast(  # pylint: disable=unused-argument, missing-raises-doc
    features,
    targets,
    model_type="restrospective",
    treated_units=None,
    w_pens = None,
    custom_donor_pool=None,  
    match_space_maker = None,
    w_pen_inner=True,
    avoid_NxN_mats=False,
    verbose=0,
    targets_aux=None,
    **kwargs #keep so that calls can switch easily between fit() and fit_fast()
):
    r"""

    :param features: Matrix of features
    :type features: matrix of floats

    :param targets: Matrix of targets
    :type targets: matrix of floats

    :param model_type:  Type of model being
        fit. One of ``"retrospective"``, ``"prospective"``,
        ``"prospective-restricted"`` or ``"full"``
    :type model_type: str, default = ``"retrospective"``

    :param treated_units:  An iterable indicating the rows
        of `X` and `Y` which contain data from treated units.
    :type treated_units: int[], Optional
    
    :param w_pens:  Penalization values to try when searching for unit weights.
    :type w_pens: float[], default=np.logspace(start=-5, stop=5, num=40)
    
    :param treated_units:  An iterable indicating the rows
        of `X` and `Y` which contain data from treated units.
    :type treated_units: int[], default=np.logspace(start=-5, stop=5, num=40) (sklearn.RidgeCV can't automatically pick)

    :param custom_donor_pool: By default all control units are allowed to be donors
        for all units. There are cases where this is not desired and so the user
        can pass in a matrix specifying a unit-specific donor pool (NxC matrix
        of booleans).
        Common reasons for restricting the allowability:
        (a) When we would like to reduce interpolation bias by restricting the
        donor pool to those units similar along certain features.
        (b) If units are not completely independent (for example there may be
        contamination between neighboring units). This is a violation of the
        Single Unit Treatment Value Assumption (SUTVA).
        Note: These are not used in the fitting stage (of V and penalties) just
        in final unit weight determination.
    :type custom_donor_pool: boolean, default = ``None``

    :param match_space_maker: Function with signature
        MatchSpace_transformer, V_vector, best_v_pen, V desc = match_space_maker(X, Y, fit_model_wrapper)
        where we can call fit_model_wrapper(MatchSpace_transformer, V_vector).
        Default is MTLassoCV_MatchSpace_factory().

    :param avoid_NxN_mats: There are several points where typically a matrices on the order of NxN would
        be made (either N or N_c). With a large number of units these can be quite big. These can be avoided.
        One consequence is that the full unit-level weights will not be kept and just the built Synthetic Control
        outcome will be return.
    :type avoid_NxN_mats: bool, default=False

    :param verbose: Verbosity level. 0 means no printouts. 1 will note times of 
        completing each of the 3 main stages and some loop progress bars. 
        2 will print memory snapshots (Optionally out to a file if the env var SparseSC_log_file is set).
    :type verbose: int, default=0

    :param kwargs: Additional parameters so that one can easily switch between fit() and fit_fast()

    :returns: A :class:`SparseSCFit` object containing details of the fitted model.
    :rtype: :class:`SparseSCFit`

    :raises ValueError: when ``treated_units`` is not None and not an
            ``iterable``, or when model_type is not one of the allowed values
    """
    if verbose>1:
        tracemalloc.start()
    X = features
    Y = targets
    w_pens = np.logspace(start=-5, stop=5, num=40) if w_pens is None else w_pens
    try:
        X = np.float64(X)
    except ValueError:
        raise ValueError("X is not coercible to a numpy float64")
    try:
        Y = np.float64(Y)
    except ValueError:
        raise ValueError("Y is not coercible to a numpy float64")

    if treated_units is not None:
        control_units = [u for u in range(Y.shape[0]) if u not in treated_units]
        N0, N1 = len(control_units), len(treated_units)
    else:
        control_units = [u for u in range(Y.shape[0])]
        N0, N1 = Y.shape[0], 0
    N = N0 + N1

    D = np.full(N, True)
    D[control_units] = False
    
    if custom_donor_pool is not None:
        assert custom_donor_pool.shape == (N,N0)
    else:
        custom_donor_pool = np.full((N,N0), True)
    custom_donor_pool = _ensure_good_donor_pool(custom_donor_pool, control_units)
    match_space_maker = MTLassoCV_MatchSpace_factory() if match_space_maker is None else match_space_maker

    fit_units = _get_fit_units(model_type, control_units, treated_units, N)
    X_v = X[fit_units, :]
    Y_v = Y[fit_units,:]
    
    def _fit_fast_wrapper(MatchSpace, V):
        return _fit_fast_inner(X, MatchSpace.transform(X), Y, V, model_type, treated_units, w_pens=w_pens, custom_donor_pool=custom_donor_pool, w_pen_inner=w_pen_inner)
    MatchSpace, V, best_v_pen, MatchSpaceDesc = match_space_maker(X_v, Y_v, fit_model_wrapper=_fit_fast_wrapper, X_full=X, D_full=D)
    match_fit = None
    if isinstance(MatchSpaceDesc, tuple): #unpack if necessary
        MatchSpaceDesc, match_fit = MatchSpaceDesc

    M = MatchSpace.transform(X)
    log_if_necessary("Completed calculation of MatchSpace/V", verbose)

    return _fit_fast_inner(X, M, Y, V, model_type, treated_units, best_v_pen, w_pens, custom_donor_pool, 
                           MatchSpace, MatchSpaceDesc, w_pen_inner=w_pen_inner, avoid_NxN_mats=avoid_NxN_mats, 
                           verbose=verbose, Y_aux=targets_aux, match_fit=match_fit)


def _weights(V , X_treated, X_control, w_pen):
    V = np.diag(V) #make square
    #weights = np.zeros((X_control.shape[0], X_treated.shape[0]))
    w_pen_mat = 2 * w_pen * np.diag(np.ones(X_control.shape[0]))
    A = X_control.dot(2 * V).dot(X_control.T) + w_pen_mat  # 5
    B = (
        X_treated.dot(2 * V).dot(X_control.T).T + 2 * w_pen / X_control.shape[0]
    )  # 6
    try:
        b = scipy.linalg.solve(A, B)
    except scipy.linalg.LinAlgError as exc:
        print("Unique weights not possible.")
        if w_pen == 0:
            print("Try specifying a very small w_pen rather than 0.")
        raise exc
    return b

def _sc_weights_trad(M, M_c, V, N, N0, custom_donor_pool, best_w_pen, verbose=0):
    """ Traditional matrix solving. Requires making NxN0 matrices.
    """
    #Potentially could be decomposed to not build NxN0 matrix, but the RidgeSolution works fine for that.
    sc_weights = np.full((N,N0), 0.)
    weight_log_inc = max(int(N/100), 1)
    for i in range(N):
        if ((i % weight_log_inc) == 0 and verbose>0):
            print_progress(i+1, N)
            if verbose > 1:
                print_memory_snapshot(extra_str="Loop " + str(i))
        allowed = custom_donor_pool[i,:]
        sc_weights[i,allowed] = _weights(V, M[i,:], M_c[allowed,:], best_w_pen)
    if ((N-1) % weight_log_inc) != 0 and verbose > 0:
        print_progress(N, N)
    return sc_weights

def _RidgeSolution(M, control_units, V, w_pen, custom_donor_pool, ret_weights=True, Y_c=None, verbose=0):
    """ Newer ridge solution. Does not require making NxN0 matrices.
    """
    from sklearn.linear_model import Ridge
    #Could return the weights too
    M_c = M[control_units,:]
    N = M.shape[0]
    N_c = M_c.shape[0]
    weight_log_inc = max(int(N/100), 1)
    if ret_weights:
        weights = np.full((N,N_c), 0.)
    if Y_c is not None:
        Y_sc = np.full((N, Y_c.shape[1]), 0.)
    for i in range(N):
        if ((i % weight_log_inc) == 0 and verbose > 0):
            print_progress(i+1, N)
            if verbose > 1:
                print_memory_snapshot(extra_str="Loop " + str(i))
        if i in control_units:
            c_i = control_units.index(i)
            M_c_i = np.delete(M_c, c_i, axis=0)
            features_i = (M_c_i*np.sqrt(V)).T #K* x (N0-1) 
            targets_i = ((M_c[c_i,:]-M_c_i.mean(axis=0))*np.sqrt(V)).T #K*1
            offset = 1/(N_c-1)
        else:
            features_i = (M_c*np.sqrt(V)).T #K* x (N0-1) 
            targets_i = ((M[i,:]-M_c.mean(axis=0))*np.sqrt(V)).T #K*x1
            offset = 1/N_c
        weights_i = np.full((1,N_c), 0.)
        allowed = custom_donor_pool[i,:]
        ridgefit_i = Ridge(alpha=w_pen, fit_intercept=False, solver='cholesky').fit(features_i, targets_i) #'svd' is more stable but 'choleskly' is closed-form
        weights_i[0,allowed] = ridgefit_i.coef_ + offset
        if ret_weights:
            weights[i, :] = weights_i
        if Y_c is not None:
            Y_sc[i,:] = weights_i.dot(Y_c)

    if ((N-1) % weight_log_inc) != 0 and verbose > 0:
        print_progress(N, N)
    ret = ()
    if ret_weights:
        ret = (*ret, weights)
    if Y_c is not None:
        ret = (*ret, Y_sc)
    return ret

def _fit_fast_inner(
    X, 
    M,
    Y,
    V,
    model_type="restrospective",
    treated_units=None,
    best_v_pen = None,
    w_pens = None,
    custom_donor_pool=None,
    match_space_trans = None,
    match_space_desc = None,
    w_pen_inner=True,
    avoid_NxN_mats=False,
    verbose=0,
    Y_aux=None,
    match_fit=None
):
    #returns in-sample score
    if treated_units is not None:
        control_units = [u for u in range(Y.shape[0]) if u not in treated_units]
        N0, N1 = len(control_units), len(treated_units)
    else:
        control_units = [u for u in range(Y.shape[0])]
        N0, N1 = Y.shape[0], 0
    N = N0 + N1
    fit_units = _get_fit_units(model_type, control_units, treated_units, N)
    
    if custom_donor_pool is not None:
        assert custom_donor_pool.shape == (N,N0)
    else:
        custom_donor_pool = np.full((N,N0), True)
    custom_donor_pool = _ensure_good_donor_pool(custom_donor_pool, control_units)
    
    if len(V) == 0 or M.shape[1]==0:
        best_v_pen, best_w_pen, M = None, None, None
    else:
        M_c = M[control_units,:]
        separate_calcs = True if avoid_NxN_mats else None
        if w_pen_inner:
            if model_type=="retrospective":
                best_w_pen = RidgeCVSolution(M, control_units, True, None, V, w_pens, separate=separate_calcs)
            elif model_type=="prospective":
                best_w_pen = RidgeCVSolution(M, control_units, True, treated_units, V, w_pens, separate=separate_calcs)
            elif model_type=="prospective-restricted:":
                best_w_pen = RidgeCVSolution(M, control_units, False, treated_units, V, w_pens, separate=separate_calcs)
            else: #model_type=="full"
                best_w_pen = RidgeCVSolution(M, control_units, True, None, V, w_pens, separate=separate_calcs)
        else:
            best_w_pen = None
            best_w_pen_score = np.Inf
            for w_pen in w_pens:
                sc_weights = _sc_weights_trad(M, M_c, V, N, N0, custom_donor_pool, w_pen)
                Y_sc = sc_weights.dot(Y[control_units, :])
                mscore = np.sum(np.square(Y[fit_units,:] - Y_sc[fit_units,:]))
                if mscore<best_w_pen_score:
                    best_w_pen = w_pen
                    best_w_pen_score = mscore
    log_if_necessary("Completed calculation of best_w_pen", verbose)
            
    return _fit_fast_match(X, M, Y, V, model_type, treated_units, best_v_pen, best_w_pen, custom_donor_pool, 
                           match_space_trans, match_space_desc, w_pen_inner=w_pen_inner, avoid_NxN_mats=avoid_NxN_mats, 
                           verbose=verbose, Y_aux=Y_aux, match_fit=match_fit)

def _fit_fast_match(
    X, 
    M,
    Y,
    V,
    model_type="restrospective",
    treated_units=None,
    best_v_pen = None,
    best_w_pen = None,
    custom_donor_pool=None,
    match_space_trans = None,
    match_space_desc = None,
    w_pen_inner=True,
    avoid_NxN_mats=False,
    verbose=0,
    Y_aux=None,
    match_fit=None
):
    if treated_units is not None:
        control_units = [u for u in range(Y.shape[0]) if u not in treated_units]
        N0, N1 = len(control_units), len(treated_units)
    else:
        control_units = [u for u in range(Y.shape[0])]
        N0, N1 = Y.shape[0], 0
    N = N0 + N1
    fit_units = _get_fit_units(model_type, control_units, treated_units, N)
    if custom_donor_pool is not None:
        assert custom_donor_pool.shape == (N,N0)
    else:
        custom_donor_pool = np.full((N,N0), True)
    custom_donor_pool = _ensure_good_donor_pool(custom_donor_pool, control_units)

    Y_aux_sc = None

    if len(V) == 0 or M.shape[1]==0:
        sc_weights = None if avoid_NxN_mats else np.full((N,N0), 0.)
        Y_c = Y[control_units, :]
        Y_sc = np.full((N, Y_c.shape[1]), 0.)
        if Y_aux is not None:
            Y_aux_sc = np.full(Y_aux.shape, 0.)
            Y_aux_c = Y_aux[control_units,:]
        for i in range(N):
            weights_i = np.full((1,N0), 0.)
            allowed = custom_donor_pool[i,:]
            weights_i[0,allowed] = 1/np.sum(allowed)
            if not avoid_NxN_mats:
                sc_weights[i,:] = weights_i
            Y_sc[i,:] = weights_i.dot(Y_c)
            if Y_aux is not None:
                Y_aux_sc[i,:] = weights_i.dot(Y_aux_c)
        log_if_necessary("Completed calculation of sc_weights", verbose)
    else:
        M_c = M[control_units,:]
        Y_c = Y[control_units, :]
        if not avoid_NxN_mats:
            sc_weights = _sc_weights_trad(M, M_c, V, N, N0, custom_donor_pool, best_w_pen, verbose=verbose)
            log_if_necessary("Completed calculation of sc_weights", verbose)
            Y_sc = sc_weights.dot(Y_c)
        else:
            sc_weights = None
            Y_sc = _RidgeSolution(M, control_units, V, best_w_pen, custom_donor_pool, Y_c=Y_c, ret_weights=False, 
                                  verbose=verbose)[0]
            if Y_aux is not None:
                Y_aux_sc = _RidgeSolution(M, control_units, V, best_w_pen, custom_donor_pool, Y_c=Y_aux[control_units, :], ret_weights=False, 
                                      verbose=verbose)[0]
            log_if_necessary("Completed calculation of (temp.) sc_weights", verbose)


    log_if_necessary("Completed calculation of synthetic controls", verbose)
    mscore = np.sum(np.square(Y[fit_units,:] - Y_sc[fit_units,:]))

    fit_obj = SparseSCFit(
        features=X,
        targets=Y,
        control_units=control_units,
        treated_units=treated_units,
        model_type=model_type,
        fitted_v_pen=best_v_pen,
        fitted_w_pen=best_w_pen,
        V=np.diag(V),
        sc_weights=sc_weights,
        targets_sc=Y_sc,
        score=mscore, 
        match_space_trans = match_space_trans,
        match_space = M,
        match_space_desc = match_space_desc
    )
    if Y_aux is not None:
        fit_obj.Y_aux_sc = Y_aux_sc
    if match_fit is not None:
        fit_obj.match_fit = match_fit

    return fit_obj
