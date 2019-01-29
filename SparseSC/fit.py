# --------------------------------------------------------------------------------
# Programmer: Jason Thorpe
# Date    1/11/2019 1:25:57 PM
# Purpose:   Implement round-robin fitting of Sparse Synthetic Controls Model for DGP based analysis
# Description:  Main public API providing a single call for fitting SC Models
# --------------------------------------------------------------------------------

from warnings import warn

import pandas as pd
import numpy as np
# import SparseSC as SC


# From the Public API
from SparseSC.lambda_utils import get_max_lambda, L2_pen_guestimate
from SparseSC.cross_validation import CV_score
from SparseSC.tensor import tensor
from SparseSC.weights import weights

# Public API
from sklearn.model_selection import KFold


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
        # LINE SEARCH PARAMETERS
        learning_rate = 0.2, # TODO: this should be harmnonized with parameter names in cd_line_search and passed in via *args / **kwargs
        learning_rate_adjustment = 0.9, # TODO: this should be harmnonized with parameter names in cd_line_search and passed in via *args / **kwargs
        #*args,
        **kwargs):
    r"""

        :param X: Matrix of features
        :type X: matrix of floats

        :param Y:: Matrix of targets
        :type Y: matrix of floats

        :param model_type: (Default = ``"retrospective"``) Type of model being
                fit. One of ``"retrospective"``, ``"prospective"``,
                ``"prospective-restricted"`` or ``"full"``

        :param treated_units:  An iterable indicating the rows
                of `X` and `Y` which contain data from treated units.  
        :type treated_units: int[], Optional

        :param weight_penalty: Penalty applied to the difference
                between the current weights and the null weights (1/n). Default
                provided by :func:``L2_pen_guestimate``.
        :type weight_penalty: float, Optional

        :param covariate_penalties: penalty
                (penalties) applied to the magnitude of the covariate weights.
                Defaults to ``[ Lambda_c_max * g for g in grid]``, where
                `Lambda_c_max` is determined via :func:`get_max_lambda` .
        :type covariate_penalties: float | float[], optional

        :param grid: only used when
                `covariate_penalties` is not provided
        :type grid: float | float[], optional
        :param Lambda_min: (float, Default = 1e-6): only used when
                `covariate_penalties` and `grid` are not provided
        :param Lambda_max: (float, Default = 1): only used when
                `covariate_penalties` and `grid` are not provided
        :param grid_points: (int, Default = 20): only used when
                `covariate_penalties` and `grid` are not provided

        :param choice: ("min" or function) Method for choosing from among the
                covariate_penalties.  Only used when covariate_penalties is an
                iterable.  Defaults to ``"min"`` which selects the lambda parameter
                associated with the lowest cross validation error.

        :param cv_folds: (Default = 10) An integer number
                of Cross Validation folds passed to
                :func:`sklearn.model_selection.KFold`, or an explicit list of train
                validation folds. TODO: These folds are calculated with
                ``KFold(...,shuffle=False)``, but instead, it should be assigned a
                random state.
        :type cv_folds: int or (int[],int[])[] 

        :param gradient_folds: (Default = 10) An integer
                number of Gradient folds passed to
                :func:`sklearn.model_selection.KFold`, or an explicit list of train
                validation folds, to be used `model_type` is one either ``"foo"``
                ``"bar"``.
        :type gradient_folds: int or (int[],int[])[]

        :param gradient_seed: (default = 10101) passed to :func:`sklearn.model_selection.KFold`
                to allow for consistent gradient folds across calls when
                `model_type` is one either ``"foo"`` ``"bar"`` with and
                `gradient_folds` is an integer.
        :param gradient_seed: int

        :param progress: (Default = `True`)Controls the level of verbosity.  If
                `True`, the messages indication the progress are printed to the
                console (stdout).

        :param \**kwargs: See below

        :Keyword Args:

            Arguments passed on to :func:`cdl_search` which implements the
                gradient descent with adaptive step sizes

            * *learning_rate* (float, Default = 0.2)  -- The initial learning rate
                (alpha) which determines the initial step size, which is set to
                learning_rate * null_model_error / gradient. Must be between 0 and
                1.

            * *learning_rate_adjustment (float, Default = 0.9)* -- Adjustment factor
                applied to the learning rate applied between iterations when the
                optimal step size returned by :func:`scipy.optimize.line_search` is
                greater less than 1, else the step size is adjusted by
                ``1/learning_rate_adjustment``. Must be between 0 and 1,

            * *tol (float, Default = 1e-4)* -- Tolerance used for the stopping rule
    """
            # TODO: theses should be harmnonized with parameter names in cd_line_search and actually passed in via *args / **kwargs



    assert X.shape[0] == Y.shape[0]

    if treated_units is not None:

        # --------------------------------------------------
        # Phase 0: Data wrangling
        # --------------------------------------------------

        try:
            iter(treated_units)
        except:
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
                        aggressiveness = learning_rate, # initial learning rate # todo: this needs to be harmonized and passed in via *args or **kwargs
                        alpha_mult = learning_rate_adjustment, # todo: this needs to be harmonized and passed in via *args or **kwargs
                        verbose=1)
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
                               aggressiveness = learning_rate, # todo: this needs to be harmonized and passed in via *args or **kwargs
                               alpha_mult = learning_rate_adjustment,  # todo: this needs to be harmonized and passed in via *args or **kwargs
                               quiet = not progress, 
                               **kwargs)

            # GET THE INDEX OF THE BEST SCORE
            best_lambda = __choose(scores, covariate_penalties, choice)

            # --------------------------------------------------
            # Phase 2: extract V and weights: slow ( tens of seconds to minutes )
            # --------------------------------------------------

            best_V = tensor(X = Xtrain, 
                            Y = Ytrain,
                            # *args, **kwargs
                            LAMBDA = best_lambda,
                            grad_splits = gradient_folds,
                            random_state = gradient_seed, # TODO:  this is only used when grad splits is not None... need to better control this...
                            aggressiveness = learning_rate, # todo: this needs to be harmonized and passed in via *args or **kwargs
                            alpha_mult = learning_rate_adjustment,
                            **kwargs)  # todo: this needs to be harmonized and passed in via *args or **kwargs

            # GET THE BEST SET OF WEIGHTS
            # these are out-of-sample weights...
            sc_weights = weights(Xtrain,
                                 Xtest,
                                 V = best_V,
                                 L2_PEN_W = weight_penalty)

            synthetic_units = sc_weights.dot(Ytrain)

        elif model_type == "prospective":
            # we're doing in-sample "predictions" -- i.e. we're directly optimizing the
            # observed || Y_ctrl - W Y_ctrl ||


            try:
                iter(gradient_folds)
            except:
                from sklearn.model_selection import KFold
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
                               aggressiveness = learning_rate, # todo: this needs to be harmonized and passed in via *args or **kwargs
                               alpha_mult = learning_rate_adjustment,  # todo: this needs to be harmonized and passed in via *args or **kwargs
                               quiet = not progress, 
                               **kwargs)

            # GET THE INDEX OF THE BEST SCORE
            best_lambda = __choose(scores, covariate_penalties, choice)

            # --------------------------------------------------
            # Phase 2: extract V and weights: slow ( tens of seconds to minutes )
            # --------------------------------------------------

            best_V = tensor(X = X, 
                            Y = Y,
                            # *args, **kwargs
                            LAMBDA = best_lambda,
                            grad_splits = gradient_folds,
                            random_state = gradient_seed, # TODO:  this is only used when grad splits is not None... need to better control this...
                            aggressiveness = learning_rate, # todo: this needs to be harmonized and passed in via *args or **kwargs
                            alpha_mult = learning_rate_adjustment,
                            **kwargs)  # todo: this needs to be harmonized and passed in via *args or **kwargs

            # GET THE BEST SET OF WEIGHTS
            full_weights = weights(X,
                                   V = best_V,
                                   L2_PEN_W = weight_penalty)
            sc_weights = full_weights[np.ix_(treated_units,control_units)]
            synthetic_units = sc_weights.dot(Ytrain)

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
                               aggressiveness = learning_rate, # todo: this needs to be harmonized and passed in via *args or **kwargs
                               alpha_mult = learning_rate_adjustment,  # todo: this needs to be harmonized and passed in via *args or **kwargs
                               quiet = not progress, 
                               **kwargs)

            # GET THE INDEX OF THE BEST SCORE
            best_lambda = __choose(scores, covariate_penalties, choice)

            # --------------------------------------------------
            # Phase 2: extract V and weights: slow ( tens of seconds to minutes )
            # --------------------------------------------------

            best_V = tensor(X = Xtrain, 
                            Y = Ytrain,
                            X_treat = Xtest,
                            Y_treat = Ytest,
                            # *args, **kwargs
                            LAMBDA = best_lambda,
#--                             grad_splits = gradient_folds,
#--                                 random_state = gradient_seed, # TODO:  this is only used when grad splits is not None... need to better control this...
                            aggressiveness = learning_rate, # todo: this needs to be harmonized and passed in via *args or **kwargs
                            alpha_mult = learning_rate_adjustment,
                            **kwargs)  # todo: this needs to be harmonized and passed in via *args or **kwargs

            # GET THE BEST SET OF WEIGHTS
            # these are effectively in-sample weights
            sc_weights = weights(Xtrain,
                                            Xtest,
                                            V = best_V,
                                            L2_PEN_W = weight_penalty)

            synthetic_units = sc_weights.dot(Ytrain)

        else:
            raise ValueError("unexpected model_type '%s' or treated_units = None" % model_type)

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
                        aggressiveness = learning_rate, # initial learning rate # todo: this needs to be harmonized and passed in via *args or **kwargs
                        alpha_mult = learning_rate_adjustment, # todo: this needs to be harmonized and passed in via *args or **kwargs
                        verbose=1)
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
                           aggressiveness = learning_rate, # todo: this needs to be harmonized and passed in via *args or **kwargs
                           alpha_mult = learning_rate_adjustment,  # todo: this needs to be harmonized and passed in via *args or **kwargs
                           quiet = not progress, 
                           **kwargs)

        # GET THE INDEX OF THE BEST SCORE
        best_lambda = __choose(scores, covariate_penalties, choice)

        # --------------------------------------------------
        # Phase 2: extract V and weights: slow ( tens of seconds to minutes )
        # --------------------------------------------------

        best_V = tensor(X = X, 
                        Y = Y,
                        LAMBDA = best_lambda,
                        grad_splits = gradient_folds,
                        random_state = gradient_seed, # TODO:  this is only used when grad splits is not None... need to better control this...
                        aggressiveness = learning_rate, # todo: this needs to be harmonized and passed in via *args or **kwargs
                        alpha_mult = learning_rate_adjustment,
                        **kwargs)  # todo: this needs to be harmonized and passed in via *args or **kwargs

        # GET THE BEST SET OF WEIGHTS
        sc_weights = weights(X,
                             V = best_V,
                             L2_PEN_W = weight_penalty)
        synthetic_units = sc_weights.dot(Y)

    return SparseSCFit( 
            # Data
            X,
            Y,
            control_units,
            treated_units,
            model_type,
            # fitting parameters
            weight_penalty,
            covariate_penalties,
            # Fitted Synthetic Controls
            sc_weights,
            synthetic_units)

def __choose(scores, covariate_penalties, choice):
    """ helper function which implements the choice of covariate weights penalty parameter
    """
    # GET THE INDEX OF THE BEST SCORE
    try: 
        iter(covariate_penalties)
    except:
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
            weight_penalty,
            covariate_penalties,
            # Fitted Synthetic Controls
            sc_weights,
            synthetic_units,
            ):

        # DATA
        self.X = X
        self.Y = Y
        self.control_units = control_units
        self.treated_units = treated_units
        self.model_type = model_type

        # FITTING PARAMETERS
        self.weight_penalty = weight_penalty
        self.covariate_penalties = covariate_penalties

        # FITTED SYNTHETIC CONTROLS
        self.sc_weights = sc_weights
        self.synthetic_units = synthetic_units

    def __str__(self):
        """ print details of the fit to the console
        """
        raise NotImplementedError()

        # CALCULATE ERRORS AND R-SQUARED'S
        ct_prediction_error = Y_SC_test - Ytest
        null_model_error = Ytest - np.mean(Xtest)
        betternull_model_error = (Ytest.T - np.mean(Xtest,1)).T
        print("#--------------------------------------------------")
        print("OUTER FOLD %s OF %s: Group Mean R-squared: %0.3f%%; Individual Mean R-squared: %0.3f%%" % (
                i + 1,
                100*(1 - np.power(ct_prediction_error,2).sum()  / np.power(null_model_error,2).sum()) ,
                100*(1 - np.power(ct_prediction_error,2).sum()  /np.power(betternull_model_error,2).sum() )))
        print("#--------------------------------------------------")

    def show(self):
        """ display goodness of figures illustrating goodness of fit
        """
        raise NotImplementedError()

