""" Experimental but fast API providing a single call for fitting SC Models

"""
import numpy as np
from sklearn.linear_model import LassoCV, MultiTaskLassoCV, RidgeCV #Lasso, 
import scipy

from .fit import SparseSCFit

def MTLassoCV_MatchSpace(X, Y, v_pens=None, n_v_cv = 5):
    varselectorfit = MultiTaskLassoCV(normalize=True, cv=n_v_cv, alphas = v_pens).fit(X, Y)
    V = np.sqrt(np.sum(varselectorfit.coef_**2, axis=0)) #n_tasks x n_features -> n_feature
    best_v_pen = varselectorfit.alpha_
    m_sel = (V!=0)
    def _MT_Match(X):
        return(X[:,m_sel])
    return _MT_Match, V[m_sel], best_v_pen, m_sel
    
#def _FakeMTLassoCV_MatchSpace(X, Y, n_v_cv = 5, v_pens=None):
    #y_mean = Y.mean(axis=1)
    #y_mean_c = y_mean[control_units]
    #varselectorfit = LassoCV(normalize=True).fit(X_c, y_mean_c, cv=n_cv)
    #V = varselectorfit.coef_

def fit_fast(  # pylint: disable=differing-type-doc, differing-param-doc
    X,
    Y,
    model_type="restrospective",
    treated_units=None,
    v_pens = None,
    w_pens = np.logspace(start=-5, stop=5, num=40),
    custom_donor_pool=None,  
    #mixed = False,
    match_space_maker = MTLassoCV_MatchSpace,
    **kwargs #keep so that calls can switch easily between fit() and fit_fast()
):
    r"""

    :param X: Matrix of features
    :type X: matrix of floats

    :param Y: Matrix of targets
    :type Y: matrix of floats

    :param model_type:  Type of model being
        fit. One of ``"retrospective"``, ``"prospective"``,
        ``"prospective-restricted"`` or ``"full"``
    :type model_type: str, default = ``"retrospective"``

    :param treated_units:  An iterable indicating the rows
        of `X` and `Y` which contain data from treated units.
    :type treated_units: int[], Optional
    
    :param v_pens:  Penalization values to try when searching for variable weights.
    :type v_pens: float[], default=None (let sklearn.LassoCV pick list automatically)
    
    :param w_pens:  Penalization values to try when searching for unit weights.
    :type w_pens: float[], default=None (let sklearn.LassoCV pick list automatically)
    
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

    :returns: A :class:`SparseSCFit` object containing details of the fitted model.
    :rtype: :class:`SparseSCFit`

    :raises ValueError: when ``treated_units`` is not None and not an
            ``iterable``, or when model_type is not one of the allowed values
    """
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
        N0 = len(control_units)
        N1 = len(treated_units)
    else:
        control_units = [u for u in range(Y.shape[0])]
        N0 = Y.shape[0]
        N1 = 0
    N = N0 + N1
    
    if custom_donor_pool is not None:
        assert custom_donor_pool.shape == (N,N0)
    else:
        custom_donor_pool = np.full((N,N0), True)
    custom_donor_pool = _ensure_good_donor_pool(custom_donor_pool, control_units, N0)

    if model_type=="full":
        null_weights = (np.ones((N0, N0)) - np.eye(N0))* (1/(N0-1))
        X_v = X
        Y_v = Y
    else:
        null_weights = np.empty((N,N0))
        null_weights[control_units,:] = (np.ones((N0, N0)) - np.eye(N0))* (1/(N0-1))
        null_weights[treated_units,:] = np.ones((N1, N0))* (1/N0)
        if model_type=="retrospective":
            X_v = X[control_units, :]
            Y_v = Y[control_units,:]
        elif model_type=="prospective":
            X_v = X
            Y_v = Y
        elif model_type=="prospective-restricted:":
            X_v = X[treated_units, :]
            Y_v = Y[treated_units,:]
    MatchSpace, V, best_v_pen, m_sel = match_space_maker(X_v, Y_v, v_pens=v_pens)
    if m_sel is not None:
        V_full = np.full((X.shape[1]), 0.)
        V_full[m_sel] = V
    else:
        V_full = V
    #if mixed:
        #raise NotImplementedError()
        #lambdas = lassocvfit.alphas_
        #scores = [] * len(lambdas)
        #for l_i, l in enumerate(lambdas):
            #lassofit = Lasso(normalize=True, alpha=l).fit(X_c,y_c_mean)
            #params = lassocvfit.coef_
            #remove constant if necessary
            #scores[l_i] = scfit_pick_optimal_wpen(params, model_type=model_type)
    if len(V) == 0:
        return SparseSCFit(
            X=X,
            Y=Y,
            control_units=control_units,
            treated_units=treated_units,
            model_type=model_type,
            # fitting parameters
            fitted_v_pen=None,
            fitted_w_pen=None,
            initial_w_pen=None,
            initial_v_pen=None,
            V=np.diag(V_full),
            # Fitted Synthetic Controls
            sc_weights=null_weights,
            score=None, 
            scores=None,
            selected_score=None,
        )
    M = MatchSpace(X)

    return(fit_fast_inner(M, Y, V, model_type, treated_units, w_pens, custom_donor_pool, X, V_full))

def _ensure_good_donor_pool(custom_donor_pool, control_units, N0):
    custom_donor_pool_c = custom_donor_pool[control_units,:]
    for i in range(N0):
        custom_donor_pool_c[i, i] = False
    custom_donor_pool[control_units,:] = custom_donor_pool_c
    return(custom_donor_pool)

def _RidgeCVSolution(M, control_units, controls_as_goals, extra_goals, V, N0, w_pens):
    #Could return the weights too
    M_c = M[control_units,:]
    features = np.empty((0,0))
    targets = np.empty((0,))
    if controls_as_goals:
        for i in range(len(control_units)):
            M_c_i = np.delete(M_c, i, axis=0)
            features_i = (M_c_i*np.sqrt(V)).T #K* x (N0-1) 
            targets_i = ((M_c[i,:]-M_c_i.mean(axis=0))*np.sqrt(V)).T #K*x1

            features = scipy.linalg.block_diag(features, features_i)
            targets = np.hstack((targets, targets_i))
    if extra_goals is not None:
        for extra_goal in extra_goals:
            features_i = (M_c*np.sqrt(V)).T #K* x (N0-1) 
            targets_i = ((M[extra_goal,:]-M_c.mean(axis=0))*np.sqrt(V)).T #K*x1

            features = scipy.linalg.block_diag(features, features_i)
            targets = np.hstack((targets, targets_i))

    ridgecvfit = RidgeCV(alphas=w_pens, fit_intercept=False).fit(features, targets) #Use the generalized cross-validation
    return(ridgecvfit.alpha_)

def _weights(V , X_treated, X_control, w_pen):
    V = np.diag(V) #make square
    #weights = np.zeros((X_control.shape[0], X_treated.shape[0]))
    w_pen_mat = 2 * w_pen * np.diag(np.ones(X_control.shape[0]))
    A = X_control.dot(2 * V).dot(X_control.T) + w_pen_mat  # 5
    B = (
        X_treated.dot(2 * V).dot(X_control.T).T + 2 * w_pen / X_control.shape[0]
    )  # 6
    try:
        b = np.linalg.solve(A, B)
    except np.linalg.LinAlgError as exc:
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
    return(sc_weights)

def fit_fast_inner(
    M,
    Y,
    V,
    model_type="restrospective",
    treated_units=None,
    w_pens = np.logspace(start=-5, stop=5, num=40),
    custom_donor_pool=None,
    X=None,
    V_full=None,
):
    if treated_units is not None:
        control_units = [u for u in range(Y.shape[0]) if u not in treated_units]
        N0 = len(control_units)
        N1 = len(treated_units)
    else:
        control_units = [u for u in range(Y.shape[0])]
        N0 = Y.shape[0]
        N1 = 0
    N = N0 + N1

    if custom_donor_pool is not None:
        assert custom_donor_pool.shape == (N,N0)
    else:
        custom_donor_pool = np.full((N,N0), True)
    custom_donor_pool = _ensure_good_donor_pool(custom_donor_pool, control_units, N0)

    if model_type=="retrospective":
        best_w_pen = _RidgeCVSolution(M, control_units, True, None, V, N0, w_pens)
    elif model_type=="prospective":
        best_w_pen = _RidgeCVSolution(M, control_units, True, treated_units, V, N0, w_pens)
    elif model_type=="prospective-restricted:":
        best_w_pen = _RidgeCVSolution(M, control_units, False, treated_units, V, N0, w_pens)
    else: #model_type=="full"
        best_w_pen = _RidgeCVSolution(M, control_units, True, None, V, N0, w_pens)

    M_c = M[control_units,:]

    sc_weights = _sc_weights_trad(M, M_c, V, N, N0, custom_donor_pool, best_w_pen)

    return SparseSCFit(
        X=X,
        Y=Y,
        control_units=control_units,
        treated_units=treated_units,
        model_type=model_type,
        # fitting parameters
        fitted_v_pen=best_v_pen,
        fitted_w_pen=best_w_pen,
        initial_w_pen=None,
        initial_v_pen=None,
        V=np.diag(V),
        # Fitted Synthetic Controls
        sc_weights=sc_weights,
        score=None, 
        scores=None,
        selected_score=None,
    )
