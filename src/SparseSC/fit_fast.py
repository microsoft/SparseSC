""" Experimental but fast API providing a single call for fitting SC Models

"""
import numpy as np


def fit_fast(  # pylint: disable=differing-type-doc, differing-param-doc
    X,
    Y,
    treated_units=None,
    w_pen=None,  # Float
    v_pen=None,  # Float or an array of floats
    # PARAMETERS USED TO CONSTRUCT DEFAULT GRID COVARIATE_PENALTIES
    grid=None,  # USER SUPPLIED GRID OF COVARIATE PENALTIES
    grid_min=1e-6,
    grid_max=1,
    grid_length=20,
    stopping_rule=2,
    gradient_folds=10,
    **kwargs
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

    :param w_pen: Penalty applied to the difference
        between the current weights and the null weights (1/n). default
        provided by :func:``w_pen_guestimate``.
    :type w_pen: float | float[], optional

    :param v_pen: penalty
        (penalties) applied to the magnitude of the covariate weights.
        Defaults to ``[ Lambda_c_max * g for g in grid]``, where
        `Lambda_c_max` is determined via :func:`get_max_v_pen` .
    :type v_pen: float | float[], optional

    :param grid: only used when `v_pen` is not provided.
        Defaults to ``np.exp(np.linspace(np.log(grid_min),np.log(grid_max),grid_length))``
    :type grid: float | float[], optional

    :param grid_min: Lower bound for ``grid`` when
        ``v_pen`` and ``grid`` are not provided.  Must be in the
        range ``(0,1)``
    :type grid_min: float, default = 1e-6

    :param grid_max: Upper bound for ``grid`` when
        ``v_pen`` and ``grid`` are not provided.  Must be in the
        range ``(0,1]``
    :type grid_max: float, default = 1

    :param grid_length: number of points in the ``grid`` parameter when
        ``v_pen`` and ``grid`` are not provided
    :type grid_length: int, default = 20

    :param stopping_rule: A stopping rule less than one is interpreted as the
        percent improvement in the out-of-sample squared prediction error required
        between the current and previous iteration in order to continue with the
        coordinate descent. A stopping rule of one or greater is interpreted as
        the number of iterations of the coordinate descent (rounded down to the
        nearest Int).  Alternatively, ``stopping_rule`` may be a function which
        will be passed the current model fit, the previous model fit, and the
        iteration number (depending on it's signature), and should return a
        truthy value if the coordinate descent should stop and a falsey value
        if the coordinate descent should stop.
    :type stopping_rule: int, float, or function

    :param choice: Method for choosing from among the
        v_pen.  Only used when v_pen is an
        iterable.  Defaults to ``"min"`` which selects the v_pen parameter
        associated with the lowest cross validation error.
    :type choice: str or function. default = ``"min"``

    :param cv_folds: An integer number of Cross Validation folds passed to
        :func:`sklearn.model_selection.KFold`, or an explicit list of train
        validation folds. TODO: These folds are calculated with
        ``KFold(...,shuffle=False)``, but instead, it should be assigned a
        random state.
    :type cv_folds: int or (int[],int[])[], default = 10

    :param gradient_folds: (default = 10) An integer
        number of Gradient folds passed to
        :func:`sklearn.model_selection.KFold`, or an explicit list of train
        validation folds, to be used `model_type` is one either ``"foo"``
        ``"bar"``.
    :type gradient_folds: int or (int[],int[])[]


    :param cv_seed:  passed to :func:`sklearn.model_selection.KFold`
        to allow for consistent cross validation folds across calls
    :type cv_seed: int, default = 10101

    :param gradient_seed:  passed to :func:`sklearn.model_selection.KFold`
        to allow for consistent gradient folds across calls when
        `model_type` is one either ``"foo"`` ``"bar"`` with and
        `gradient_folds` is an integer.
    :type gradient_seed: int, default = 10101

    :param progress: Controls the level of verbosity.  If `True`, the
        messages indication the progress are printed to the console (stdout).
    :type progress: boolean, default = ``True``

    :param kwargs: Additional arguments passed to the optimizer (i.e.
        ``method`` or `scipy.optimize.minimize`).  See below.

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

    :Keyword Args:

        * **method** (str or callable) -- The method or function
            responsible for performing gradient  descent in the covariate
            space.  If a string, it is passed as the ``method`` argument to
            :func:`scipy.optimize.minimize`.  Otherwise, ``method`` must be
            a function with a signature compatible with
            :func:`scipy.optimize.minimize`
            (``method(fun,x0,grad,**kwargs)``) which returns an object
            having ``x`` and ``fun`` attributes. (Default =
            :func:`SparseSC.optimizers.cd_line_search.cdl_search`)

        * **learning_rate** *(float, Default = 0.2)*  -- The initial learning rate
            which determines the initial step size, which is set to
            ``learning_rate * null_model_error / gradient``. Must be between 0 and
            1.

        * **learning_rate_adjustment** *(float, Default = 0.9)* -- Adjustment factor
            applied to the learning rate applied between iterations when the
            optimal step size returned by :func:`scipy.optimize.line_search` is
            greater less than 1, else the step size is adjusted by
            ``1/learning_rate_adjustment``. Must be between 0 and 1,

        * **tol** *(float, Default = 0.0001)* -- Tolerance used for the stopping
            rule based on the proportion of the in-sample residual error
            reduced in the last step of the gradient descent.

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
    from sklearn.linear_model import LassoCV, RidgeCV #Lasso, 
    import scipy
    if model_type=="retrospective":
        #For control units
        n_cv = 5
        X_c = X[control_units, :]
        y_mean_c = y_mean[control_units]
        varselectorfit = LassoCV(normalize=True).fit(X_c, y_mean_c, cv=n_cv)
        if mixed:
            raise NotImplementedError()
            #lambdas = lassocvfit.alphas_
            #scores = [] * len(lambdas)
            #for l_i, l in enumerate(lambdas):
                #lassofit = Lasso(normalize=True, alpha=l).fit(X_c,y_c_mean)
                #params = lassocvfit.coef_
                #remove constant if necessary
                #scores[l_i] = scfit_pick_optimal_wpen(params, model_type=model_type)
        V = varselectorfit.coef_
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
