""" Main public API providing a single call for fitting SC Models

Implements round-robin fitting of Sparse Synthetic Controls Model for DGP based analysis
"""
import numpy as np
from os.path import join
from warnings import warn
from inspect import signature
from .utils.penalty_utils import get_max_w_pen, get_max_v_pen, w_pen_guestimate
from .cross_validation import CV_score
from .tensor import tensor
from .weights import weights
from .utils.warnings import SparseSCWarning

# pylint: disable=too-many-lines, inconsistent-return-statements, fixme


class SparseSCParameterWarning(
    SparseSCWarning
):  # pylint: disable=too-few-public-methods,missing-docstring
    pass


class TrivialUnitsWarning(
    SparseSCWarning
):  # pylint: disable=too-few-public-methods,missing-docstring
    pass


# TODO: Cleanup task 1:
#  random_state = gradient_seed, in the calls to CV_score() and tensor() are
#  only used when grad splits is not None... need to better control this...


def fit(  # pylint: disable=differing-type-doc, differing-param-doc
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
    # --------------------------------------------------
    # PARAMETER VALIDATION
    # --------------------------------------------------

    try:
        X = np.float64(X)
    except ValueError:
        raise ValueError("X is not coercible to a numpy float64")
    try:
        Y = np.float64(Y)
    except ValueError:
        raise ValueError("Y is not coercible to a numpy float64")

    Y = np.asmatrix(Y) # this needs to be deprecated properly -- bc Array.dot(Array) != matrix(Array).dot(matrix(Array)) -- not even close !!!
    X = np.asmatrix(X)

    w_pen_is_iterable = False
    try:
        iter(w_pen)
    except TypeError:
        pass
    else:
        if v_pen is None:
            raise ValueError("When v_pen is an iterable, v_pen must be provided")
        w_pen_is_iterable = True

    v_pen_is_iterable = False
    try:
        iter(v_pen)
    except TypeError:
        pass
    else:
        v_pen_is_iterable = True
        if w_pen is None:
            raise ValueError("When v_pen is an iterable, w_pen must be provided")

    if v_pen_is_iterable and w_pen_is_iterable:
        raise ValueError("Features and Weights penalties are both iterables")

    if (
        v_pen_is_iterable
        or w_pen_is_iterable
        or (v_pen is not None and w_pen is not None)
    ):
        return _fit(X, Y, treated_units, w_pen, v_pen, gradient_folds=gradient_folds,**kwargs)

    # Herein, either v_pen or w_pen is None (possibly both)

    # --------------------------------------------------
    # BUILD THE COORDINATE DESCENT PARAMETERS
    # --------------------------------------------------
    if grid is None:
        grid = np.exp(np.linspace(np.log(grid_min), np.log(grid_max), grid_length))

    if treated_units is not None:
        control_units = [u for u in range(Y.shape[0]) if u not in treated_units]
        _X, _Y = X[control_units, :], Y[control_units, :]
    else:
        _X, _Y = X, Y

    # --------------------------------------------------
    #  BUILD THE STOPPING RULE
    # --------------------------------------------------
    if callable(stopping_rule):
        parameters = len(signature(stopping_rule).parameters)
    else:
        assert stopping_rule > 0, "stopping_rule must be positive number or a function"
        if stopping_rule >= 1:
            iterations = [stopping_rule]
            parameters = 0

            def iterations_rule():
                iterations[0] -= 1
                return iterations[0] <= 0

            stopping_rule = iterations_rule
        else:
            parameters = 2
            _stopping_rule = stopping_rule

            def percent_reduction_rule(current, previous):
                if previous is None:
                    return False
                oss_error_reduction = 1 - current.score / previous.score
                return oss_error_reduction < _stopping_rule

            stopping_rule = percent_reduction_rule

    # --------------------------------------------------
    # ACTUAL WORK
    # --------------------------------------------------
    _iteration = 0
    model_fits = []
    last_axis = None
    previous_model_fit = None
    while True:

        v_pen, w_pen, axis = _build_penalties(
            _X, _Y, v_pen, w_pen, grid, gradient_folds, verbose=kwargs.get("verbose", 1)
        )
        if last_axis:
            assert axis != last_axis

        model_fit = _fit(X, Y, treated_units, w_pen, v_pen, gradient_folds=gradient_folds, **kwargs)

        if not model_fit:
            # this happens when only a batch file is being produced but not executed
            return

        if axis == "v_pen":
            v_pen, w_pen = model_fit.fitted_v_pen, None
        else:
            v_pen, w_pen = None, model_fit.fitted_w_pen
        last_axis = axis

        _params = [model_fit, previous_model_fit, _iteration][:parameters]
        model_fits.append(model_fit)

        if stopping_rule(*_params):
            break

        previous_model_fit = model_fit
        _iteration += 1

    model_fit.model_fits = model_fits
    return model_fit


def _build_penalties(X, Y, v_pen, w_pen, grid, gradient_folds, verbose):
    """ Build (sensible?) defaults for the v_pen and w_pen
    """

    if w_pen is None:
        if v_pen is None:
            # use the guestimate for w_pen and generate a grid based sequence for v_pen
            w_pen = w_pen_guestimate(X)
            v_pen_max = get_max_v_pen(
                X, Y, w_pen=w_pen, grad_splits=gradient_folds, verbose=verbose
            )
            axis = "v_pen"
            v_pen = grid * v_pen_max

        else:
            w_pen_max = get_max_w_pen(
                X, Y, v_pen=v_pen, grad_splits=gradient_folds, verbose=verbose
            )
            axis = "w_pen"
            w_pen = grid * w_pen_max

    else:  # w_pen is not None:

        v_pen_max = get_max_v_pen(
            X, Y, w_pen=w_pen, grad_splits=gradient_folds, verbose=verbose
        )
        axis = "v_pen"
        v_pen = grid * v_pen_max

    return v_pen, w_pen, axis


def _fit(
    X,
    Y,
    treated_units=None,
    w_pen=None,  # Float
    v_pen=None,  # Float or an array of floats
    # PARAMETERS USED TO CONSTRUCT DEFAULT GRID COVARIATE_PENALTIES
    choice="1se",
    cv_folds=10,
    gradient_folds=10,
    gradient_seed=10101,
    model_type="retrospective",
    custom_donor_pool=None,
    # VERBOSITY
    progress=True,
    batchDir=None,
    **kwargs
):
    assert X.shape[0] == Y.shape[0]

    if (not callable(choice)) and (choice not in ("min", "1se")):
        # Fail Faster (tm)
        raise ValueError("Unexpected value for choice parameter: %s" % choice)

    w_pen_is_iterable = False
    try:
        iter(w_pen)
    except TypeError:
        pass
    else:
        if v_pen is None:
            raise ValueError("When v_pen is an iterable, v_pen must be provided")
        w_pen_is_iterable = True

    v_pen_is_iterable = False
    try:
        iter(v_pen)
    except TypeError:
        pass
    else:
        v_pen_is_iterable = True
        if w_pen is None:
            raise ValueError("When v_pen is an iterable, w_pen must be provided")

    if v_pen_is_iterable and w_pen_is_iterable:
        raise ValueError("Features and Weights penalties are both iterables")

    if batchDir is not None:

        import pathlib
        from yaml import dump

        try:
            from yaml import CDumper as Dumper
        except ImportError:
            from yaml import Dumper

        pathlib.Path(batchDir).mkdir(parents=True, exist_ok=True)
        _fit_params = {
            "X": X,
            "Y": Y,
            "v_pen": v_pen,
            "w_pen": w_pen,
            "treated_units": treated_units,
            "choice": choice,
            "cv_folds": cv_folds,
            "gradient_folds": gradient_folds,
            "gradient_seed": gradient_seed,
            "model_type": model_type,
            "custom_donor_pool": custom_donor_pool,
            "kwargs": kwargs,
        }

        from .utils.AzureBatch.constants import _BATCH_FIT_FILE_NAME
        with open(join(batchDir, _BATCH_FIT_FILE_NAME), "w") as fp:
            fp.write(dump(_fit_params, Dumper=Dumper))

    def _choose(scores, scores_se):
        """ helper function which implements the choice of covariate weights penalty parameter

        Nested here for access to  v_pen, w_pe,n w_pen_is_iterable and
        v_pen_is_iterable, and choice, via Lexical Scoping
        """
        # GET THE INDEX OF THE BEST SCORE
        if w_pen_is_iterable:
            indx = _which(scores, scores_se, choice)
            return v_pen, w_pen[indx], scores[indx], indx
        if v_pen_is_iterable:
            indx = _which(scores, scores_se, choice)
            return v_pen[indx], w_pen, scores[indx], indx
        return v_pen, w_pen, scores, None

    if treated_units is not None:

        # --------------------------------------------------
        # Phase 0: Data wrangling
        # --------------------------------------------------

        try:
            iter(treated_units)
        except TypeError:
            raise ValueError("treated_units must be an iterable")

        # Coerce a mask of booleans in to a list of ints
        _t = [u for u in iter(treated_units)]
        if isinstance(_t[0], bool):
            treated_units = [i for i, t in enumerate(_t) if t]
        del _t

        assert len(set(treated_units)) == len(
            treated_units
        ), (
            "duplicated values in treated_units are not allowed"
        )  # pylint: disable=line-too-long
        assert all(unit < Y.shape[0] for unit in treated_units)
        assert all(unit >= 0 for unit in treated_units)

        control_units = [u for u in range(Y.shape[0]) if u not in treated_units]

        Xtrain = X[control_units, :]
        Xtest = X[treated_units, :]
        Ytrain = Y[control_units, :]
        Ytest = Y[treated_units, :]

        # --------------------------------------------------
        # Actual work
        # --------------------------------------------------

        if model_type == "retrospective":
            # Retrospective Treatment Effects:  ( *model_type = "prospective"*)

            # --------------------------------------------------
            # Phase 1: extract cross fold residual errors for each v_pen
            # --------------------------------------------------

            # SCORES FOR EACH VALUE OF THE GRID: very slow ( minutes to hours )
            ret = CV_score(
                X=Xtrain,
                Y=Ytrain,
                splits=cv_folds,
                v_pen=v_pen,
                w_pen=w_pen,
                progress=progress,
                grad_splits=gradient_folds,
                random_state=gradient_seed,  # TODO: Cleanup Task 1
                quiet=not progress,
                batchDir=batchDir,
                **kwargs
            )
            if not ret:
                # this happens when only a batch file is being produced but not executed
                return
            scores, scores_se = ret

            best_v_pen, best_w_pen, score, which = _choose(scores, scores_se)

            # --------------------------------------------------
            # Phase 2: extract V and weights: slow ( tens of seconds to minutes )
            # --------------------------------------------------
            best_V = tensor(
                X=Xtrain,
                Y=Ytrain,
                w_pen=best_w_pen,
                v_pen=best_v_pen,
                grad_splits=gradient_folds,
                random_state=gradient_seed,  # TODO: Cleanup Task 1
                **kwargs
            )

        elif model_type == "prospective":
            # we're doing in-sample "predictions" -- i.e. we're directly optimizing the
            # observed || Y_ctrl - W Y_ctrl ||

            try:
                iter(gradient_folds)
            except TypeError:
                from sklearn.model_selection import KFold

                gradient_folds = KFold(
                    gradient_folds, shuffle=True, random_state=gradient_seed
                ).split(np.arange(X.shape[0]))
                gradient_folds = [
                    [
                        list(set(train).union(treated_units)),
                        list(set(test).difference(treated_units)),
                    ]
                    for train, test in gradient_folds
                ]  # pylint: disable=line-too-long
                gradient_folds = [
                    [train, test]
                    for train, test in gradient_folds
                    if len(train) != 0 and len(test) != 0
                ]  # pylint: disable=line-too-long
                gradient_folds.append([control_units, treated_units])
            else:
                # user supplied gradient folds
                gradient_folds = list(gradient_folds)
                treated_units_set = set(treated_units)

                # TODO: this condition logic is untested:
                if not any(treated_units_set == set(gf[1]) for gf in gradient_folds):
                    warn(
                        "User supplied gradient_folds will be re-formed for compatibility with model_type 'prospective'",
                        SparseSCParameterWarning,
                    )  # pylint: disable=line-too-long
                    gradient_folds = [
                        [
                            list(set(train).union(treated_units)),
                            list(set(test).difference(treated_units)),
                        ]
                        for train, test in gradient_folds
                    ]  # pylint: disable=line-too-long
                    gradient_folds = [
                        [train, test]
                        for train, test in gradient_folds
                        if len(train) != 0 and len(test) != 0
                    ]  # pylint: disable=line-too-long
                    gradient_folds.append([control_units, treated_units])

            # --------------------------------------------------
            # Phase 1: extract cross fold residual errors for each v_pen
            # --------------------------------------------------

            # SCORES FOR EACH VALUE OF THE GRID: very slow ( minutes to hours )
            ret = CV_score(
                X=X,
                Y=Y,
                splits=cv_folds,
                v_pen=v_pen,
                w_pen=w_pen,
                progress=progress,
                grad_splits=gradient_folds,
                random_state=gradient_seed,  # TODO: Cleanup Task 1
                quiet=not progress,
                batchDir=batchDir,
                **kwargs
            )
            if not ret:
                # this happens when only a batch file is being produced but not executed
                return
            scores, scores_se = ret

            # GET THE INDEX OF THE BEST SCORE
            best_v_pen, best_w_pen, score, which = _choose(scores, scores_se)

            # --------------------------------------------------
            # Phase 2: extract V and weights: slow ( tens of seconds to minutes )
            # --------------------------------------------------

            best_V = tensor(
                X=X,
                Y=Y,
                w_pen=best_w_pen,
                v_pen=best_v_pen,
                grad_splits=gradient_folds,
                random_state=gradient_seed,  # TODO: Cleanup Task 1
                **kwargs
            )

        elif model_type == "prospective-restricted":
            # we're doing in-sample -- i.e. we're optimizing hold-out error in
            # the controls ( || Y_ctrl - W Y_ctrl || ) in the hopes that the
            # chosen penalty parameters and V matrix also optimizes the
            # unobserved ( || Y_treat - W Y_ctrl || ) in counter factual

            # --------------------------------------------------
            # Phase 1: extract cross fold residual errors for each v_pen
            # --------------------------------------------------

            # SCORES FOR EACH VALUE OF THE GRID: very slow ( minutes to hours )
            ret = CV_score(
                X=Xtrain,
                Y=Ytrain,
                X_treat=Xtest,
                Y_treat=Ytest,
                splits=cv_folds,
                v_pen=v_pen,
                w_pen=w_pen,
                progress=progress,
                quiet=not progress,
                batchDir=batchDir,
                **kwargs
            )
            if not ret:
                # this happens when only a batch file is being produced but not executed
                return
            scores, scores_se = ret

            # GET THE INDEX OF THE BEST SCORE
            best_v_pen, best_w_pen, score, which = _choose(scores, scores_se)

            # --------------------------------------------------
            # Phase 2: extract V and weights: slow ( tens of seconds to minutes )
            # --------------------------------------------------

            best_V = tensor(
                X=Xtrain,
                Y=Ytrain,
                X_treat=Xtest,
                Y_treat=Ytest,
                w_pen=best_w_pen,
                v_pen=best_v_pen,
                **kwargs
            )

        else:
            raise ValueError(
                "unexpected model_type '%s' or treated_units = None" % model_type
            )

        # GET THE BEST SET OF WEIGHTS
        sc_weights = np.empty((X.shape[0], Ytrain.shape[0]))
        if custom_donor_pool is None:
            custom_donor_pool_t = None
            custom_donor_pool_c = None
        else:
            custom_donor_pool_t = custom_donor_pool[treated_units, :]
            custom_donor_pool_c = custom_donor_pool[control_units, :]
        sc_weights[treated_units, :] = weights(
            Xtrain,
            Xtest,
            V=best_V,
            w_pen=best_w_pen,
            custom_donor_pool=custom_donor_pool_t,
        )
        sc_weights[control_units, :] = weights(
            Xtrain, V=best_V, w_pen=best_w_pen, custom_donor_pool=custom_donor_pool_c
        )

    else:

        if model_type != "full":
            raise ValueError(
                "Unexpected model_type ='%s' or treated_units is not None" % model_type
            )  # pylint: disable=line-too-long

        control_units = None

        # --------------------------------------------------
        # Phase 1: extract cross fold residual errors for each v_pen
        # --------------------------------------------------

        # SCORES FOR EACH VALUE OF THE GRID: very slow ( minutes to hours )
        ret = CV_score(
            X=X,
            Y=Y,
            splits=cv_folds,
            v_pen=v_pen,
            progress=progress,
            w_pen=w_pen,
            grad_splits=gradient_folds,
            random_state=gradient_seed,  # TODO: Cleanup Task 1
            quiet=not progress,
            batchDir=batchDir,
            **kwargs
        )
        if not ret:
            # this happens when only a batch file is being produced but not executed
            return
        scores, scores_se = ret

        # GET THE INDEX OF THE BEST SCORE
        best_v_pen, best_w_pen, score, which = _choose(scores, scores_se)

        # --------------------------------------------------
        # Phase 2: extract V and weights: slow ( tens of seconds to minutes )
        # --------------------------------------------------

        best_V = tensor(
            X=X,
            Y=Y,
            w_pen=best_w_pen,
            v_pen=best_v_pen,
            grad_splits=gradient_folds,
            random_state=gradient_seed,  # TODO: Cleanup Task 1
            **kwargs
        )

        # GET THE BEST SET OF WEIGHTS
        sc_weights = weights(
            X, V=best_V, w_pen=best_w_pen, custom_donor_pool=custom_donor_pool
        )

    return SparseSCFit(
        X=X,
        Y=Y,
        control_units=control_units,
        treated_units=treated_units,
        model_type=model_type,
        # fitting parameters
        fitted_v_pen=best_v_pen,
        fitted_w_pen=best_w_pen,
        initial_w_pen=w_pen,
        initial_v_pen=v_pen,
        V=best_V,
        # Fitted Synthetic Controls
        sc_weights=sc_weights,
        score=score,
        scores=scores,
        selected_score=which,
    )


class SparseSCFit(object):
    """ 
    A class representing the results of a Synthetic Control model instance.
    """

    model_fits = None

    def __init__(
        self,
        # Data:
        X,
        Y,
        control_units,
        treated_units,
        model_type,
        # fitting parameters:
        fitted_v_pen,
        fitted_w_pen,
        initial_v_pen,
        initial_w_pen,
        V,
        # Fitted Synthetic Controls:
        sc_weights,
        score,
        scores,
        selected_score,
    ):

        # DATA
        self.X = X
        self.Y = Y
        self.control_units = control_units
        self.treated_units = treated_units
        self.model_type = model_type

        # FITTING PARAMETERS
        self.fitted_w_pen = fitted_w_pen
        self.fitted_v_pen = fitted_v_pen
        self.initial_w_pen = initial_w_pen
        self.initial_v_pen = initial_v_pen
        self.V = V
        self.score = score
        self.scores = scores
        self.selected_score = selected_score

        # FITTED SYNTHETIC CONTROLS
        self._sc_weights = sc_weights

        # IDENTIFY TRIVIAL UNITS
        self.trivial_units = np.apply_along_axis(
            lambda x: (x == 0).all(), 1, np.hstack([X[:, np.diag(V) != 0], Y])
        )
        if self.trivial_units.any():
            warn(
                "Fitted Model contains %s trivial unit(s)" % self.trivial_units.sum(),
                TrivialUnitsWarning,
            )

    @property
    def sc_weights(self):
        """
        getter for the sc_weights. By default, the trivial
        """
        return self.get_weights()

    def get_weights(self, include_trivial_donors=True):
        """
        getter for the sc_weights. By default, the trivial
        """
        if include_trivial_donors or not self.trivial_units.any():
            return self._sc_weights

        if self.model_type != "full":
            trivial_donors = self.trivial_units[self.control_units]
        else:
            trivial_donors = self.trivial_units

        __weights = self._sc_weights.copy()
        __weights[np.ix_(np.logical_not(self.trivial_units), trivial_donors)] = 0
        return __weights

    def predict(self, Y=None, include_trivial_donors=True):
        """ 
        predict method

        :param Y: Matrix of targets
        :type Y:  (optional) matrix of floats

        :param include_trivial_donors: Should donors for whom selected
                predictors and all targets equal to zero be included in the weights for
                non-trivial units.  These units will typically have a weight of 1 /
                total number of units as they do not contribute to the gradient.
                Default = ```False```
        :type include_trivial_donors: boolean

        :returns: matrix of predicted outcomes
        :rtype: matrix of floats

        :raises ValueError: When ``Y.shape[0]`` is inconsistent with the fitted model.
        """
        if Y is None:
            Y = self.Y
        else:
            if Y.shape[0] != self.Y.shape[0]:
                raise ValueError(
                    "parameter Y must have the same number of rows as X and Y in the fitted model"
                )

        if self.model_type != "full":
            Y = Y[self.control_units, :]

        return self.get_weights(include_trivial_donors).dot(Y)

    def __str__(self):
        """ 
        Print details of the fit to the console
        """
        return _SparseFit_string_template % (
            self.model_type,
            self.fitted_v_pen,
            self.fitted_w_pen,
            np.diag(self.V),
        )

    def show(self):
        """ display goodness of figures illustrating goodness of fit
        """
        raise NotImplementedError()

    def summary(self):
        """
        A summary of the model fit / penalty selection

        This illustrates that (a) the gird function could / should be better,
        and (b) currently more than two iterations is typically useless.
        """
        try:
            import pandas as pd
        except ImportError:
            DF = dict
            # Pandas is not a core requirement, though it is popular and convenient:
            print("consider installing pandas for a better fit.summary() experience")
        else:
            DF = pd.DataFrame
        return [
            DF(
                {
                    "v_pen": _fit.initial_v_pen,
                    "w_pen": _fit.initial_w_pen,
                    "score": _fit.scores,
                    "min_score": _fit.scores == np.array(_fit.scores).min(),
                    "selected_score": np.arange(len(_fit.scores))
                    == _fit.selected_score,
                }
            )
            for _fit in self.model_fits
        ]


_SparseFit_string_template = """ Model type: %s"
V penalty: %s
W penalty: %s
V: %s
"""


def _which(x, se, f):
    """
    Return the index of the value which meets the selection rule
    """
    # GET THE INDEX OF THE BEST SCORE
    if callable(f):
        return f(x)
    if f == "min":
        return np.argmin(x)
    if f == "1se":
        """
        Return the first score which exceeds the remaining (higher penalized)
        scores by at least 1 standard error. If none of the values exceed the
        most penalized value, return the most penalized value.
        """
        x_1se = (np.array(x) + np.array(se))[:-1]
        # reversed cumulative minimum (excluding the first value)
        cum_min = np.minimum.accumulate(x[::-1])[-2::-1]  # pylint: disable=no-member
        return np.where(np.append(x_1se < cum_min, np.array((True,))))[0][0]
    raise ValueError("Unexpected value for choice parameter: %s" % f)


# TODO: CALCULATE ERRORS AND R-SQUARED'S
# ct_prediction_error = Y_SC_test - Ytest
# null_model_error = Ytest - np.mean(Xtest)
# betternull_model_error = (Ytest.T - np.mean(Xtest,1)).T
# print("#--------------------------------------------------")
# print("OUTER FOLD %s OF %s: Group Mean R-squared: %0.3f%%; Individual Mean R-squared: %0.3f%%" % (
#        i + 1,
#        100*(1 - np.power(ct_prediction_error,2).sum()  / np.power(null_model_error,2).sum()) ,
#        100*(1 - np.power(ct_prediction_error,2).sum()  /np.power(betternull_model_error,2).sum() )))
# print("#--------------------------------------------------")
