""" Experimental but fast API providing a single call for fitting SC Models

"""
import numpy as np


def fit_fast(  # pylint: disable=differing-type-doc, differing-param-doc
    X,
    Y,
    model_type="restrospective",
    treated_units=None,
    custom_donor_pool=None,  
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

    def _weights(V , X_treated, X_control, w_pen):
        V = np.diag(V)
        weights = np.zeros((X_control.shape[0], X_treated.shape[0]))
        w_pen_mat = 2 * w_pen * diag(np.ones(X_control.shape[0]))
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
        return weights

    if treated_units is not None:
        control_units = [u for u in range(Y.shape[0]) if u not in treated_units]
        N0 = len(control_units)
        N1 = len(treated_units)
    else:
        control_units = [u for u in range(Y.shape[0])]

    y_mean = Y.mean(axis=1)
    #
    from sklearn.linear_model import LassoCV, MultiTaskLassoCV, RidgeCV #Lasso, 
    import scipy
    if model_type=="retrospective":
        #For control units
        n_cv = 5
        X_c = X[control_units, :]
        y_mean_c = y_mean[control_units]
        #varselectorfit = LassoCV(normalize=True).fit(X_c, y_mean_c, cv=n_cv)
        #V = varselectorfit.coef_
        varselectorfit = MultiTaskLassoCV(normalize=True, cv=n_cv).fit(X_c,Y[control_units,:])
        V = np.sqrt(np.sum(varselectorfit.coef_**2, axis=1)) #if n_features x n_tasks
        if mixed:
            raise NotImplementedError()
            #lambdas = lassocvfit.alphas_
            #scores = [] * len(lambdas)
            #for l_i, l in enumerate(lambdas):
                #lassofit = Lasso(normalize=True, alpha=l).fit(X_c,y_c_mean)
                #params = lassocvfit.coef_
                #remove constant if necessary
                #scores[l_i] = scfit_pick_optimal_wpen(params, model_type=model_type)
        
        m_sel = (V!=0)
        M_c = X_c[:, m_sel]
        V_vec = V[m_sel]
        for i in range(N0):
            M_c_i = np.delete(M_c, i, axis=0)
            features_i = (M_c_i*np.sqrt(V_vec)).T #K* x (N0-1) 
            targets_i = ((M_c[i,:]-M_c_i.mean(axis=0))*np.sqrt(V_vec)).T #K*x1
            if i==0:
                features = features_i
                targets = targets_i
            else:
                features = scipy.linalg.block_diag(features, features_i)
                targets = np.concatenate(targets, targets_i)
        ridgecvfit = RidgeCV(fit_intercept=False).fit(features, targets) #Use the generalized cross-validation
        ridgecvfit.get_params() #make sure can get the w_pen.

    #elif model_type=="prospective":
        #same, but with X, y_mean
    #elif model_type=="prospective-restricted:":
        #same, but with X_t, y_t_mean
    #else: #model_type=="full"
        #same as prospective

    return SparseSCFit(
        X=X,
        Y=Y,
        control_units=control_units,
        treated_units=treated_units,
        model_type=model_type,
        # fitting parameters
        fitted_v_pen=None,
        fitted_w_pen=best_w_pen,
        initial_w_pen=None,
        initial_v_pen=None,
        V=V,
        # Fitted Synthetic Controls
        sc_weights=sc_weights,
        score=None,
        scores=None,
        selected_score=None,
    )
