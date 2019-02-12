# --------------------------------------------------------------------------------
# Programmer: Jason Thorpe
# Date    1/11/2019 1:25:57 PM
# Purpose:   Implement round-robin fitting of Sparse Synthetic Controls Model for DGP based analysis
# Description:  Main public API providing a single call for fitting SC Models
# --------------------------------------------------------------------------------

from warnings import warn

import numpy as np
#import pandas as pd
from sklearn.model_selection import KFold
# import SparseSC as SC


# From the Public API
from SparseSC.lambda_utils import get_max_lambda, L2_pen_guestimate
from SparseSC.cross_validation import CV_score
from SparseSC.tensor import tensor
from SparseSC.weights import weights
from SparseSC.utils.metrics_utils import gen_placebo_stats_from_diffs


def fit(X,Y,
        treated_units = None,
        weight_penalty = None, # Float
        covariate_penalties = None, # Float or an array of floats
        # PARAMETERS USED TO CONSTRUCT DEFAULT GRID COVARIATE_PENALTIES 
        grid = None, # USER SUPPLIED GRID OF COVARIATE PENALTIES
        Lambda_min = 1e-6,
        Lambda_max = 1,
        grid_points = 20,
        choice = "min", # Method for choosing from best among the covariate_penalties.  (only used when covariate_penalties is an iterable)
        # fold tuning parameters: either a integer or list of test/train subsets such as the result of calling Kfold().split()
        cv_folds = 10,
        gradient_folds = 10,
        gradient_seed = 10101, # random state when fit_fold is invoked with an integer value for grad_splits -- this should be harmnonized with parameter names in cd_line_search and passed in via *args / **kwargs
        model_type = "retrospective",
        # VERBOSITY
        progress = True,
#--         # LINE SEARCH PARAMETERS
#--         learning_rate = 0.2, # TODO: this should be harmnonized with parameter names in cd_line_search and passed in via *args / **kwargs
#--         learning_rate_adjustment = 0.9, # TODO: this should be harmnonized with parameter names in cd_line_search and passed in via *args / **kwargs
        #*args,
        **kwargs):
    r"""

        :param X: Matrix of features
        :type X: matrix of floats

        :param Y:: Matrix of targets
        :type Y: matrix of floats

        :param model_type:  Type of model being
                fit. One of ``"retrospective"``, ``"prospective"``,
                ``"prospective-restricted"`` or ``"full"``
        :type model_type: str, default = ``"retrospective"``

        :param treated_units:  An iterable indicating the rows
                of `X` and `Y` which contain data from treated units.  
        :type treated_units: int[], Optional

        :param weight_penalty: Penalty applied to the difference
                between the current weights and the null weights (1/n). default
                provided by :func:``L2_pen_guestimate``.
        :type weight_penalty: float, Optional

        :param covariate_penalties: penalty
                (penalties) applied to the magnitude of the covariate weights.
                Defaults to ``[ Lambda_c_max * g for g in grid]``, where
                `Lambda_c_max` is determined via :func:`get_max_lambda` .
        :type covariate_penalties: float | float[], optional

        :param grid: only used when `covariate_penalties` is not provided.  Defaults to ``np.exp(np.linspace(np.log(Lambda_min),np.log(Lambda_max),grid_points))``
        :type grid: float | float[], optional

        :param Lambda_min: Lower bound for ``grid`` when ``covariate_penalties`` and ``grid`` are not provided.  Must be in the range ``(0,1)``
        :type Lambda_min: float, default = 1e-6

        :param Lambda_max: Upper bound for ``grid`` when ``covariate_penalties`` and ``grid`` are not provided.  Must be in the range ``(0,1]``
        :type Lambda_max: float, default = 1

        :param grid_points: number of points in the ``grid`` parameter when ``covariate_penalties`` and ``grid`` are not provided
        :type grid_points: int, default = 20

        :param choice: Method for choosing from among the
                covariate_penalties.  Only used when covariate_penalties is an
                iterable.  Defaults to ``"min"`` which selects the lambda parameter
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

        :param gradient_seed:  passed to :func:`sklearn.model_selection.KFold`
                to allow for consistent gradient folds across calls when
                `model_type` is one either ``"foo"`` ``"bar"`` with and
                `gradient_folds` is an integer.
        :type gradient_seed: int, default = 10101

        :param progress: Controls the level of verbosity.  If `True`, the
                messages indication the progress are printed to the console (stdout).
        :type progress: boolean, default = ``True``

        :param method: The method or function responsible for performing gradient 
            descent in the covariate space.  If a string, it is passed as the
            ``method`` argument to :func:`scipy.optimize.minimize`.  Otherwise,
            ``method`` must be a function with a signature compatible with
            :func:`scipy.optimize.minimize` (``method(fun,x0,grad,**kwargs)``)
            which returns an object having ``x`` and ``fun`` attributes. (Default
            = :func:`SparseSC.optimizers.cd_line_search.cdl_search`)
        :type method: str or function

        :param kwargs: Additional arguments passed to the optimizer (i.e. ``method`` or `scipy.optimize.minimize`).
            See below.

        :Keyword Args:
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
            # TODO: theses should be harmnonized with parameter names in cd_line_search and actually passed in via *args / **kwargs



    assert X.shape[0] == Y.shape[0]
    if "verbose" in kwargs:
        verbose = kwargs["verbose"]
    else:
        verbose = 1
    
    if treated_units is not None:

        # --------------------------------------------------
        # Phase 0: Data wrangling
        # --------------------------------------------------

        try:
            iter(treated_units)
        except TypeError:
            raise ValueError('treated_units must be an iterable' )

        assert len(set(treated_units)) == len(treated_units) , "duplicated values in treated_units are not allowed"
        assert all( unit < Y.shape[0] for unit in treated_units )
        assert all( unit >= 0 for unit in treated_units )

        control_units = [ u for u in range(Y.shape[0]) if u not in treated_units ]

        Xtrain = X[control_units,:]
        Xtest  = X[treated_units,:]
        Ytrain = Y[control_units,:]
        Ytest  = Y[treated_units,:]

        # --------------------------------------------------
        # (sensible?) defaults
        # --------------------------------------------------
        # Get the L2 penalty guestimate:  very quick ( milliseconds )
        if weight_penalty is None:
            weight_penalty  = L2_pen_guestimate(Xtrain) 
        if covariate_penalties is None: 
            if grid is None:
                grid = np.exp(np.linspace(np.log(Lambda_min),np.log(Lambda_max),grid_points))
            # GET THE MAXIMUM LAMBDAS: quick ~ ( seconds to tens of seconds )
            LAMBDA_max = get_max_lambda(
                        Xtrain,
                        Ytrain,
                        L2_PEN_W = weight_penalty,
                        grad_splits = gradient_folds,
                        verbose=verbose)
            covariate_penalties = grid * LAMBDA_max


        if model_type == "retrospective":
            # Retrospective Treatment Effects:  ( *model_type = "prospective"*)

            # --------------------------------------------------
            # Phase 1: extract cross fold residual errors for each lambda
            # --------------------------------------------------

            # SCORES FOR EACH VALUE OF THE GRID: very slow ( minutes to hours )
            scores = CV_score( X = Xtrain,
                               Y = Ytrain,
                               # *args, **kwargs
                               splits = cv_folds,
                               LAMBDA = covariate_penalties,
                               progress = progress,
                               L2_PEN_W = weight_penalty,
                               grad_splits = gradient_folds,
                               random_state = gradient_seed, # TODO: this is only used when grad splits is not None... need to better control this...
                               quiet = not progress, 
                               **kwargs)

            # GET THE INDEX OF THE BEST SCORE
            best_V_lambda = __choose(scores, covariate_penalties, choice)

            # --------------------------------------------------
            # Phase 2: extract V and weights: slow ( tens of seconds to minutes )
            # --------------------------------------------------

            best_V = tensor(X = Xtrain, 
                            Y = Ytrain,
                            # *args, **kwargs
                            LAMBDA = best_V_lambda,
                            grad_splits = gradient_folds,
                            random_state = gradient_seed, # TODO:  this is only used when grad splits is not None... need to better control this...
                            **kwargs)  # todo: this needs to be harmonized and passed in via *args or **kwargs

        elif model_type == "prospective":
            # we're doing in-sample "predictions" -- i.e. we're directly optimizing the
            # observed || Y_ctrl - W Y_ctrl ||


            try:
                iter(gradient_folds)
            except TypeError:
                gradient_folds = KFold(gradient_folds, shuffle=True, random_state = gradient_seed).split(np.arange(X.shape[0]))
                gradient_folds = [  [list(set(train).union(treated_units)), list(set(test).difference(treated_units))] for train, test in gradient_folds]
                gradient_folds = [ [train,test]  for train,test in gradient_folds if len(train) != 0 and len(test) != 0]
                gradient_folds.append([control_units, treated_units])
            else:
                # user supplied gradient folds
                gradient_folds = list(gradient_folds)
                treated_units_set = set(treated_units)
                if not any(treated_units_set == set(gf[1]) for gf in gradient_folds): # TODO: this condition logic is untested
                    warn("User supplied gradient_folds will be re-formed for compatibility with model_type 'prospective'")
                    gradient_folds = [  [list(set(train).union(treated_units)), list(set(test).difference(treated_units))] for train, test in gradient_folds]
                    gradient_folds = [ [train,test]  for train,test in gradient_folds if len(train) != 0 and len(test) != 0]
                    gradient_folds.append([control_units, treated_units])

            # --------------------------------------------------
            # Phase 1: extract cross fold residual errors for each lambda
            # --------------------------------------------------

            # SCORES FOR EACH VALUE OF THE GRID: very slow ( minutes to hours )
            scores = CV_score( X = X,
                               Y = Y,
                               # *args, **kwargs
                               splits = cv_folds,
                               LAMBDA = covariate_penalties,
                               progress = progress,
                               L2_PEN_W = weight_penalty,
                               grad_splits = gradient_folds,
                               random_state = gradient_seed, # TODO:  this is only used when grad splits is not None... need to better control this...
                               quiet = not progress, 
                               **kwargs)

            # GET THE INDEX OF THE BEST SCORE
            best_V_lambda = __choose(scores, covariate_penalties, choice)

            # --------------------------------------------------
            # Phase 2: extract V and weights: slow ( tens of seconds to minutes )
            # --------------------------------------------------

            best_V = tensor(X = X, 
                            Y = Y,
                            # *args, **kwargs
                            LAMBDA = best_V_lambda,
                            grad_splits = gradient_folds,
                            random_state = gradient_seed, # TODO:  this is only used when grad splits is not None... need to better control this...
                            **kwargs)  # todo: this needs to be harmonized and passed in via *args or **kwargs

        elif model_type == "prospective-restricted":
            # we're doing in-sample -- i.e. we're optimizing hold-out error in
            # the controls ( || Y_ctrl - W Y_ctrl || ) in the hopes that the
            # chosen penalty parameters and V matrix also optimizes the
            # unobserved ( || Y_treat - W Y_ctrl || ) in counter factual 

            # --------------------------------------------------
            # Phase 1: extract cross fold residual errors for each lambda
            # --------------------------------------------------

            # SCORES FOR EACH VALUE OF THE GRID: very slow ( minutes to hours )
            scores = CV_score( X = Xtrain,
                               Y = Ytrain,
                               X_treat = Xtest,
                               Y_treat = Ytest,
                               # *args, **kwargs
                               splits = cv_folds,
                               LAMBDA = covariate_penalties,
                               progress = progress,
                               L2_PEN_W = weight_penalty,
#--                                grad_splits = gradient_folds,
#--                                 random_state = gradient_seed, # TODO:  this is only used when grad splits is not None... need to better control this...
                               quiet = not progress, 
                               **kwargs)

            # GET THE INDEX OF THE BEST SCORE
            best_V_lambda = __choose(scores, covariate_penalties, choice)

            # --------------------------------------------------
            # Phase 2: extract V and weights: slow ( tens of seconds to minutes )
            # --------------------------------------------------

            best_V = tensor(X = Xtrain, 
                            Y = Ytrain,
                            X_treat = Xtest,
                            Y_treat = Ytest,
                            # *args, **kwargs
                            LAMBDA = best_V_lambda,
#--                             grad_splits = gradient_folds,
#--                                 random_state = gradient_seed, # TODO:  this is only used when grad splits is not None... need to better control this...
                            **kwargs)  # todo: this needs to be harmonized and passed in via *args or **kwargs


        else:
            raise ValueError("unexpected model_type '%s' or treated_units = None" % model_type)
        
        # GET THE BEST SET OF WEIGHTS
        sc_weights = np.empty((X.shape[0],Ytrain.shape[0]))
        sc_weights[treated_units,:] = weights(Xtrain,
                                                Xtest,
                                                V = best_V,
                                                L2_PEN_W = weight_penalty)
        sc_weights[control_units,:] = weights(Xtrain,
                                                V = best_V,
                                                L2_PEN_W = weight_penalty)
    else:

        if model_type != "full":
            raise ValueError( "Unexpected model_type ='%s' or treated_units is not None" % model_type) 

        control_units = None

        # --------------------------------------------------
        # (sensible?) defaults
        # --------------------------------------------------
        if covariate_penalties is None: 
            if grid is None:
                grid = np.exp(np.linspace(np.log(Lambda_min),np.log(Lambda_max),grid_points))
            # GET THE MAXIMUM LAMBDAS: quick ~ ( seconds to tens of seconds )
            LAMBDA_max = get_max_lambda(
                        X,
                        Y,
                        L2_PEN_W = weight_penalty,
                        grad_splits = gradient_folds,
                        verbose=verbose)
            covariate_penalties = grid * LAMBDA_max

        # Get the L2 penalty guestimate:  very quick ( milliseconds )
        if weight_penalty is None:
            weight_penalty  = L2_pen_guestimate(X) 

        # --------------------------------------------------
        # Phase 1: extract cross fold residual errors for each lambda
        # --------------------------------------------------

        # SCORES FOR EACH VALUE OF THE GRID: very slow ( minutes to hours )
        scores = CV_score( X = X,
                           Y = Y,
                           # *args, **kwargs
                           splits = cv_folds,
                           LAMBDA = covariate_penalties,
                           progress = progress,
                           L2_PEN_W = weight_penalty,
                           grad_splits = gradient_folds,
                                random_state = gradient_seed, # TODO:  this is only used when grad splits is not None... need to better control this...
                           quiet = not progress, 
                           **kwargs)

        # GET THE INDEX OF THE BEST SCORE
        best_V_lambda = __choose(scores, covariate_penalties, choice)

        # --------------------------------------------------
        # Phase 2: extract V and weights: slow ( tens of seconds to minutes )
        # --------------------------------------------------

        best_V = tensor(X = X, 
                        Y = Y,
                        LAMBDA = best_V_lambda,
                        grad_splits = gradient_folds,
                        random_state = gradient_seed, # TODO:  this is only used when grad splits is not None... need to better control this...
                        **kwargs)  # todo: this needs to be harmonized and passed in via *args or **kwargs

        # GET THE BEST SET OF WEIGHTS
        sc_weights = weights(X,
                             V = best_V,
                             L2_PEN_W = weight_penalty)

    return SparseSCFit( 
            # Data
            X,
            Y,
            control_units,
            treated_units,
            model_type,
            # fitting parameters
            best_V_lambda,
            weight_penalty,
            covariate_penalties,
            # Fitted Synthetic Controls
            sc_weights)

def __choose(scores, covariate_penalties, choice):
    """ helper function which implements the choice of covariate weights penalty parameter
    """
    # GET THE INDEX OF THE BEST SCORE
    try: 
        iter(covariate_penalties)
    except TypeError:
        best_lambda = scores
    else:
        if choice == "min":
            best_i = np.argmin(scores)
            best_lambda = (covariate_penalties)[best_i]
        elif callable(choice):
            best_lambda = choice(scores)
        else:
            # TODO: this is a terrible place to throw this error
            raise ValueError("Unexpected value for choice parameter: %s" % choice)

    return best_lambda


class SparseSCFit(object):
    """ A class representing the results of a Synthetic Control model instance.
    """
    def __init__(self,
            # Data
            X,
            Y,
            control_units,
            treated_units,
            model_type,
            # fitting parameters
            V_penalty,
            weight_penalty,
            covariate_penalties,
            # Fitted Synthetic Controls
            sc_weights,
            ):

        # DATA
        self.X = X
        self.Y = Y
        self.control_units = control_units
        self.treated_units = treated_units
        self.model_type = model_type

        # FITTING PARAMETERS
        self.V_penalty = V_penalty
        self.weight_penalty = weight_penalty
        self.covariate_penalties = covariate_penalties

        # FITTED SYNTHETIC CONTROLS
        self.sc_weights = sc_weights

    def predict(self, Ydonor = None):
        if Ydonor is None:
            if self.model_type != "full":
                Ydonor = self.Y[self.control_units,:]
            else:
                Ydonor = self.Y
        return self.sc_weights.dot(self.Y[self.control_units,:])

    def __str__(self):
        """ print details of the fit to the console
        """
        raise NotImplementedError()

        # CALCULATE ERRORS AND R-SQUARED'S
        #ct_prediction_error = Y_SC_test - Ytest
        #null_model_error = Ytest - np.mean(Xtest)
        #betternull_model_error = (Ytest.T - np.mean(Xtest,1)).T
        #print("#--------------------------------------------------")
        #print("OUTER FOLD %s OF %s: Group Mean R-squared: %0.3f%%; Individual Mean R-squared: %0.3f%%" % (
        #        i + 1,
        #        100*(1 - np.power(ct_prediction_error,2).sum()  / np.power(null_model_error,2).sum()) ,
        #        100*(1 - np.power(ct_prediction_error,2).sum()  /np.power(betternull_model_error,2).sum() )))
        #print("#--------------------------------------------------")

    def show(self):
        """ display goodness of figures illustrating goodness of fit
        """
        raise NotImplementedError()



def estimate_effects(X, Y_pre, Y_post, treated_units, max_n_pl = 1000000, ret_pl = False, ret_CI=False, level=0.95, 
                     weight_penalty = None, covariate_penalties=None, **kwargs):
    #TODO: Cleanup returning placebo distribution (incl pre?)
    #N1 = len(treated_units)
    N = Y_pre.shape[0]
    #N0 = N - N1
    #T1 = Y_post.shape[1]
    control_units = list(set(range(N)) - set(treated_units)) 
    all_units = list(range(N))
    
    fit_res = fit(X = np.hstack( ( X, Y_pre,) ), Y = Y_post, model_type = "retrospective",
                    treated_units = treated_units,
                    print_path = False, progress = False, verbose=0,
                    min_iter = -1, tol = 1)
    Y_pre_sc = fit_res.predict(Out_pre_control)
    Y_post_sc = fit_res.predict(Out_post_control)
    
    diff_pre = Y_pre - Y_pre_sc
    diff_post = Y_post - Y_post_sc
    # Get post effects
    effect_vecs = diff_post[treated_units, :]
    control_effect_vecs = diff_post[control_units, :]
    
    # Get pre match MSE (match quality)
    pre_tr_pes = diff_pre[treated_units, :]
    pre_c_pes = diff_post[control_units, :]
    pre_tr_rmspes = np.sqrt(np.mean(np.square(pre_tr_pes), axis=1))
    pre_c_rmspes = np.sqrt(np.mean(np.square(pre_c_pes), axis=1))

    pl_res = gen_placebo_stats_from_diffs(effect_vecs, control_effect_vecs, 
                                 pre_tr_rmspes, pre_c_rmspes,
                                 max_n_pl, ret_pl, ret_CI, level)
    pl_res.fit_res = fit_res
    return fit_res
