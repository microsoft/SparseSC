""" Fast API providing a single call for fitting SC Models

"""
import numpy as np
from sklearn.linear_model import LassoCV, MultiTaskLassoCV, RidgeCV, MultiTaskLasso
from sklearn.metrics import r2_score
import scipy.linalg #superset of np.linalg and also optimized compiled

from .fit import SparseSCFit

# To do:
# - Check weights are the same from RidgeCV solution
# - For bigger data do the RidgeCV separately and find the best w_pen manually since block-diagonal
#   On small data (Y_c = 100x30) this is only roughly 2x faster
# - Implement the LSTM module

def MTLassoCV_MatchSpace(X, Y, v_pens=None, n_v_cv = 5):
    """
    Fit a MultiTaskLassoCV for Y ~ X

    :param X: Features N X K
    :param Y: Targets N X G (# of goals)
    :param v_pens: Penalties to evaluate (default is to automatically determince)
    :param n_v_cv: Number of Cross-Validation folds
    :returns: MatchSpace fn, V vector, best_v_pen, V
    """
    varselectorfit = MultiTaskLassoCV(normalize=True, cv=n_v_cv, alphas = v_pens).fit(X, Y)
    V = np.sqrt(np.sum(np.square(varselectorfit.coef_), axis=0)) #n_tasks x n_features -> n_feature
    best_v_pen = varselectorfit.alpha_
    m_sel = (V!=0)
    def _MT_Match(X):
        return(X[:,m_sel])
    return _MT_Match, V[m_sel], best_v_pen, V
    
def _FakeMTLassoCV_MatchSpace(X, Y, n_v_cv = 5, v_pens=None):
    y_mean = Y.mean(axis=1)
    varselectorfit = LassoCV(normalize=True, cv=n_v_cv, alphas = v_pens).fit(X, y_mean)
    V = varselectorfit.coef_
    best_v_pen = varselectorfit.alpha_
    m_sel = (V!=0)
    def _MT_Match(X):
        return(X[:,m_sel])
    return _MT_Match, V[m_sel], best_v_pen, V

def MTLassoMixed_MatchSpace(X, Y, fit_model_wrapper, v_pens=None, n_v_cv = 5):
    """
    Fit a MultiTaskLasso for Y ~ X, but evaluate each penalization based on using that to fit a SparseSC model (downstream estimation)

    :param X: Features N X K
    :param Y: Targets N X G (# of goals)
    :param fit_model_wrapper: Function that takes MatchSpace function and V vector and returns SparseSCFit object
    :param v_pens: Penalties to evaluate (default is to automatically determince)
    :param n_v_cv: Number of Cross-Validation folds
    :returns: MatchSpace fn, V vector, best_v_pen, V
    """
    #Note that MultiTaskLasso(CV).path with the same alpha doesn't produce same results as MultiTaskLasso(CV)
    mtlasso_cv_fit = MultiTaskLassoCV(normalize=True, cv=n_v_cv, alphas = v_pens).fit(X, Y)
    V_cv = np.sqrt(np.sum(np.square(mtlasso_cv_fit.coef_), axis=0)) #n_tasks x n_features -> n_feature
    v_pen_cv = mtlasso_cv_fit.alpha_
    m_sel_cv = (V_cv!=0)
    def _MT_Match_cv(X):
        return(X[:,m_sel_cv])
    sc_fit_cv = fit_model_wrapper(_MT_Match_cv, V_cv[m_sel_cv])

    v_pens = mtlasso_cv_fit.alphas_
    #fits_single = {}
    Vs_single = {}
    scores = np.zeros((len(v_pens)))
    for i in range(len(v_pens)):
        mtlasso_i_fit = MultiTaskLasso(alpha=v_pens[i], normalize=True).fit(X, Y)
        V_i = np.sqrt(np.sum(np.square(mtlasso_i_fit.coef_), axis=0))
        m_sel_i = (V_i!=0)
        def _MT_Match_i(X):
            return(X[:,m_sel_i])
        sc_fit_i = fit_model_wrapper(_MT_Match_i, V_i[m_sel_i])
        #fits_single[i] = sc_fit_i
        Vs_single[i] = V_i
        scores[i] = sc_fit_i.score

    i_best = np.argmin(scores)
    v_pen_best = v_pens[i_best]
    i_cv = np.where(v_pens==v_pen_cv)[0][0]
    #print("CV alpha: " + str(v_pen_cv) + " (" + str(scores[i_cv]) + "). Best alpha: " + str(v_pen_best) + " (" + str(scores[i_best]) + ") .")
    best_v_pen = v_pens[i_best]
    V_best = Vs_single[i_best]
    m_sel_best = (V_best!=0)
    def _MT_Match_best(X):
        return X[:,m_sel_best]
    return _MT_Match_best, V_best[m_sel_best], best_v_pen, V_best

def fit_fast(  # pylint: disable=unused-argument
    X,
    Y,
    model_type="restrospective",
    treated_units=None,
    w_pens = np.logspace(start=-5, stop=5, num=40),
    custom_donor_pool=None,  
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

    :param match_space_maker: Function that returns MatchSpace fn and V vector

    :param kwargs: Additional parameters so that one can easily switch between fit() and fit_fast()

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
    try:
        def _fit_model_wrapper(MatchSpace, V):
            return _fit_fast_inner(X, MatchSpace(X), Y, V, model_type, treated_units, w_pens=w_pens, custom_donor_pool=custom_donor_pool)
        MatchSpace, V, best_v_pen, MatchSpaceDesc = match_space_maker(X_v, Y_v, fit_model_wrapper=_fit_model_wrapper)
    except TypeError as te:
        MatchSpace, V, best_v_pen, MatchSpaceDesc = match_space_maker(X_v, Y_v)
    except Exception as e:
        raise e

    if len(V) == 0:
        return SparseSCFit(
            X=X,
            Y=Y,
            control_units=control_units,
            treated_units=treated_units,
            model_type=model_type,
            V=np.diag(V),
            sc_weights=null_weights,
            match_space_trans = MatchSpace,
            match_space_desc = MatchSpaceDesc
        )
    M = MatchSpace(X)

    return _fit_fast_inner(X, M, Y, V, model_type, treated_units, best_v_pen, w_pens, custom_donor_pool, MatchSpace, MatchSpaceDesc)


def _ensure_good_donor_pool(custom_donor_pool, control_units, N0):
    custom_donor_pool_c = custom_donor_pool[control_units,:]
    for i in range(N0):
        custom_donor_pool_c[i, i] = False
    custom_donor_pool[control_units,:] = custom_donor_pool_c
    return custom_donor_pool

def _RidgeCVSolution(M, control_units, controls_as_goals, extra_goals, V, w_pens):
    #Could return the weights too
    M_c = M[control_units,:]
    features = np.empty((0,0))
    targets = np.empty((0,))
    if controls_as_goals:
        for i in range(len(control_units)):
            M_c_i = np.delete(M_c, i, axis=0)
            features_i = (M_c_i*np.sqrt(V)).T #K* x (N0-1) 
            targets_i = ((M_c[i,:]-M_c_i.mean(axis=0))*np.sqrt(V)).T #K*x1

            features = scipy.linalg.block_diag(features, features_i) #pylint: disable=no-member
            targets = np.hstack((targets, targets_i))
    if extra_goals is not None:
        for extra_goal in extra_goals:
            features_i = (M_c*np.sqrt(V)).T #K* x (N0-1) 
            targets_i = ((M[extra_goal,:]-M_c.mean(axis=0))*np.sqrt(V)).T #K*x1

            features = scipy.linalg.block_diag(features, features_i) #pylint: disable=no-member
            targets = np.hstack((targets, targets_i))

    ridgecvfit = RidgeCV(alphas=w_pens, fit_intercept=False).fit(features, targets) #Use the generalized cross-validation
    return ridgecvfit.alpha_

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
    return sc_weights

def _fit_fast_inner(
    X, 
    M,
    Y,
    V,
    model_type="restrospective",
    treated_units=None,
    best_v_pen = None,
    w_pens = np.logspace(start=-5, stop=5, num=40),
    custom_donor_pool=None,
    match_space_trans = None,
    match_space_desc = None
):
    #returns in-sample score
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
        best_w_pen = _RidgeCVSolution(M, control_units, True, None, V, w_pens)
    elif model_type=="prospective":
        best_w_pen = _RidgeCVSolution(M, control_units, True, treated_units, V, w_pens)
    elif model_type=="prospective-restricted:":
        best_w_pen = _RidgeCVSolution(M, control_units, False, treated_units, V, w_pens)
    else: #model_type=="full"
        best_w_pen = _RidgeCVSolution(M, control_units, True, None, V, w_pens)

    M_c = M[control_units,:]

    sc_weights = _sc_weights_trad(M, M_c, V, N, N0, custom_donor_pool, best_w_pen)
    Y_sc = sc_weights.dot(Y[control_units, :])
    if model_type=="retrospective":
        mscore = np.sum(np.square(Y[control_units,:] - Y_sc[control_units,:]))
        score_R2 = r2_score(Y[control_units,:].flatten(), Y_sc[control_units,:].flatten())
    elif model_type=="prospective":
        mscore = np.sum(np.square(Y - Y_sc))
        score_R2 = r2_score(Y.flatten(), Y_sc.flatten())
    elif model_type=="prospective-restricted:":
        mscore = np.sum(np.square(Y[treated_units,:] - Y_sc[treated_units,:]))
        score_R2 = r2_score(Y[treated_units,:].flatten(), Y_sc[treated_units,:].flatten())
    else: #model_type=="full"
        mscore = np.sum(np.square(Y - Y_sc))
        score_R2 = r2_score(Y.flatten(), Y_sc.flatten())

    fit_obj = SparseSCFit(
        X=X,
        Y=Y,
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
    setattr(fit_obj, 'score_R2', score_R2)
    return fit_obj
