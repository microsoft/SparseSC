""" Fast API providing a single call for fitting SC Models

"""

import numpy as np
from sklearn.metrics import r2_score
import scipy.linalg #superset of np.linalg and also optimized compiled

from .fit import SparseSCFit
from .utils.penalty_utils import RidgeCVSolution
from .utils.match_space import MTLassoCV_MatchSpace_factory
from .utils.misc import _ensure_good_donor_pool, _get_fit_units


# To do:
# - Check weights are the same from RidgeCV solution

#not documenting the error for when trying to two function signatures (think of better way to do that)
def fit_fast(  # pylint: disable=unused-argument, missing-raises-doc
    features,
    targets,
    model_type="restrospective",
    treated_units=None,
    w_pens = np.logspace(start=-5, stop=5, num=40),
    custom_donor_pool=None,  
    match_space_maker = None,
    w_pen_inner=True,
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
    :type w_pens: float[], default=None
    
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
        MatchSpace_fn, V_vector, best_v_pen, V desc = match_space_maker(X, Y, fit_model_wrapper)
        where we can call fit_model_wrapper(MatchSpace_fn, V_vector).
        Default is MTLassoCV_MatchSpace_factory().

    :param kwargs: Additional parameters so that one can easily switch between fit() and fit_fast()

    :returns: A :class:`SparseSCFit` object containing details of the fitted model.
    :rtype: :class:`SparseSCFit`

    :raises ValueError: when ``treated_units`` is not None and not an
            ``iterable``, or when model_type is not one of the allowed values
    """
    X = features
    Y = targets
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
    
    if custom_donor_pool is not None:
        assert custom_donor_pool.shape == (N,N0)
    else:
        custom_donor_pool = np.full((N,N0), True)
    custom_donor_pool = _ensure_good_donor_pool(custom_donor_pool, control_units)
    match_space_maker = MTLassoCV_MatchSpace_factory() if match_space_maker is None else match_space_maker

    fit_units = _get_fit_units(model_type, control_units, treated_units, N)
    X_v = X[fit_units, :]
    Y_v = Y[fit_units,:]

    def _fit_model_wrapper(MatchSpace, V):
        return _fit_fast_inner(X, MatchSpace(X), Y, V, model_type, treated_units, w_pens=w_pens, custom_donor_pool=custom_donor_pool, w_pen_inner=w_pen_inner)
    MatchSpace, V, best_v_pen, MatchSpaceDesc = match_space_maker(X_v, Y_v, fit_model_wrapper=_fit_model_wrapper)

    M = MatchSpace(X)

    return _fit_fast_inner(X, M, Y, V, model_type, treated_units, best_v_pen, w_pens, custom_donor_pool, MatchSpace, MatchSpaceDesc, w_pen_inner=w_pen_inner)


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

def _sc_weights_trad(M, M_c, V, N, N0, custom_donor_pool, best_w_pen):
    sc_weights = np.full((N,N0), 0.)
    for i in range(N):
        allowed = custom_donor_pool[i,:]
        sc_weights[i,allowed] = _weights(V, M[i,:], M_c[allowed,:], best_w_pen)
    return sc_weights

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
    w_pen_inner=True
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
        sc_weights = np.full((N,N0), 0.)
        for i in range(N):
            allowed = custom_donor_pool[i,:]
            sc_weights[i,allowed] = 1/np.sum(allowed)
    else:
        M_c = M[control_units,:]
        if w_pen_inner:
            if model_type=="retrospective":
                best_w_pen = RidgeCVSolution(M, control_units, True, None, V, w_pens)
            elif model_type=="prospective":
                best_w_pen = RidgeCVSolution(M, control_units, True, treated_units, V, w_pens)
            elif model_type=="prospective-restricted:":
                best_w_pen = RidgeCVSolution(M, control_units, False, treated_units, V, w_pens)
            else: #model_type=="full"
                best_w_pen = RidgeCVSolution(M, control_units, True, None, V, w_pens)
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

        sc_weights = _sc_weights_trad(M, M_c, V, N, N0, custom_donor_pool, best_w_pen)

    Y_sc = sc_weights.dot(Y[control_units, :])
    
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
        score=mscore, 
        match_space_trans = match_space_trans,
        match_space = M,
        match_space_desc = match_space_desc
    )
    score_R2 = r2_score(Y[fit_units,:].flatten(), Y_sc[fit_units,:].flatten())
    setattr(fit_obj, 'score_R2', score_R2)

    return fit_obj
