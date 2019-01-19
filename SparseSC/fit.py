# --------------------------------------------------------------------------------
# Programmer: Jason Thorpe
# Date    1/11/2019 1:25:57 PM
# Purpose:   Implement round-robin fitting of Sparse Synthetic Controls Model for DGP based analysis
# Description:  Main public API providing a single call for fitting SC Models
# --------------------------------------------------------------------------------

from warnings import warn

import pandas as pd
import numpy as np
import L2_pen_guestimate from 
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
        choice = "min": # Method for choosing from best among the covariate_penalties.  (only used when covariate_penalties is an iterable)
        # fold tuning parameters: either a integer or list of test/train subsets such as the result of calling Kfold().split()
        cv_folds = 10,
        gradient_folds = 10,
        model_type == "retrospective"
        # VERBOSITY
        progress = True,
        # LINE SEARCH PARAMETERS
        learning_rate = 0.2, # TODO: this should be harmnonized with parameter names in cd_line_search and passed in via *args / **kwargs
        learning_rate_adjustment = 0.1, # TODO: this should be harmnonized with parameter names in cd_line_search and passed in via *args / **kwargs
        random_state = 10101, # random state when fit_fold is invoked with an integer value for grad_splits -- this should be harmnonized with parameter names in cd_line_search and passed in via *args / **kwargs
        #*args,
        #**kwargs,
        ):


    # --------------------------------------------------
    # (sensible?) defaults
    # --------------------------------------------------
    if covariate_penalties is none: 
        if grid is None:
            grid = np.exp(np.linspace(np.log(Lambda_min),np.log(Lambda_max),grid_points))
        # GET THE MAXIMUM LAMBDAS: quick ~ ( seconds to tens of seconds )
        LAMBDA_max = get_max_lambda(
                    Xtrain,
                    Ytrain,
                    L2_PEN_W = L2_PEN_W,
                    grad_splits = gradient_folds,
                    aggressiveness = learning_rate, # initial learning rate # todo: this needs to be harmonized and passed in via *args or **kwargs
                    alpha_mult = learning_rate_adjustment, # todo: this needs to be harmonized and passed in via *args or **kwargs
                    verbose=1)
        covariate_penalties = grid * LAMBDA_max

    assert X.shape[0] == Y.shape[0]

    if treated_units is not None:

        # --------------------------------------------------
        # Phase 0: Data wrangling
        # --------------------------------------------------

        assert len(set(treated_units)) == len(treated_units) , "duplicated values in treated_units are not allowed"
        assert all( unit < Y.shape[0] for unit in treated_units )
        assert all( unit >= 0 for unit in treated_units )

        control_units = [ u for u in range(Y.shape[0]) if u not in treated_units ]

        Xtrain = X[control_units,:]
        Xtest  = X[treated_units,:]
        Ytrain = Y[control_units,:]
        Ytest  = Y[treated_units,:]

        # Get the L2 penalty guestimate:  very quick ( milliseconds )
        if weight_penalty is None:
            weight_penalty  = L2_pen_guestimate(Xtrain) 

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
                                random_state = random_state, # TODO:  this is only used when grad splits is not None... need to better control this...
                               aggressiveness = learning_rate, # todo: this needs to be harmonized and passed in via *args or **kwargs
                               alpha_mult = learning_rate_adjustment)  # todo: this needs to be harmonized and passed in via *args or **kwargs

            # GET THE INDEX OF THE BEST SCORE
            best_lambda = __choose(covariate_penalties, choice)

            # --------------------------------------------------
            # Phase 2: extract V and weights: slow ( tens of seconds to minutes )
            # --------------------------------------------------

            best_V = tensor(X = Xtrain, 
                            Y = Ytrain,
                            # *args, **kwargs
                            random_state = random_state, # TODO: not sure if this is right
                            LAMBDA = best_lambda,
                            grad_splits = gradient_folds,
                                random_state = random_state, # TODO:  this is only used when grad splits is not None... need to better control this...
                            aggressiveness = learning_rate, # todo: this needs to be harmonized and passed in via *args or **kwargs
                            alpha_mult = learning_rate_adjustment)  # todo: this needs to be harmonized and passed in via *args or **kwargs

            # GET THE BEST SET OF WEIGHTS
            out_of_sample_weights = weights(Xtrain,
                                            Xtest,
                                            V = best_V,
                                            L2_PEN_W = weight_penalty)

            Synthetic_Units = out_of_sample_weights.dot(Ytrain)

        elif prediction_type = "prospective":
            # we're doing in-sample "predictions" -- i.e. we're directly optimizing the
            # observed || Y_ctrl - W Y_ctrl ||

            working here: need to generate cleaver gradient folds...
            try:
                iter(gradient_folds)
            except:
                from sklearn.model_selection import KFold
                np.arange(len(treated_units))
                control_units
                gradient_folds = list(KFold(splits, shuffle=True, random_state = random_state).split(control_units))
            else:
                # user supplied gradient folds
                warn("User supplied gradient_folds are being re-formed for compatibility with prediction_type 'prospective'")
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
                                random_state = random_state, # TODO:  this is only used when grad splits is not None... need to better control this...
                               aggressiveness = learning_rate, # todo: this needs to be harmonized and passed in via *args or **kwargs
                               alpha_mult = learning_rate_adjustment)  # todo: this needs to be harmonized and passed in via *args or **kwargs

            # GET THE INDEX OF THE BEST SCORE
            best_lambda = __choose(covariate_penalties, choice)

            # --------------------------------------------------
            # Phase 2: extract V and weights: slow ( tens of seconds to minutes )
            # --------------------------------------------------

            best_V = tensor(X = X, 
                            Y = Y,
                            # *args, **kwargs
                            LAMBDA = best_lambda,
                            grad_splits = gradient_folds,
                                random_state = random_state, # TODO:  this is only used when grad splits is not None... need to better control this...
                            aggressiveness = learning_rate, # todo: this needs to be harmonized and passed in via *args or **kwargs
                            alpha_mult = learning_rate_adjustment)  # todo: this needs to be harmonized and passed in via *args or **kwargs

            # GET THE BEST SET OF WEIGHTS
            full_weights = weights(Xtrain,
                                   Xtest,
                                   V = best_V,
                                   L2_PEN_W = weight_penalty)
            SC_weights = full_weights[treated_units,control_units]
            Synthetic_Units = out_of_sample_weights.dot(Ytrain)

        elif prediction_type = "prospective-restricted":
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
                               X_treat = Xtest
                               y_treat = ytest
                               # *args, **kwargs
                               splits = cv_folds,
                               LAMBDA = covariate_penalties,
                               progress = progress,
                               L2_PEN_W = weight_penalty,
#--                                grad_splits = gradient_folds,
#--                                 random_state = random_state, # TODO:  this is only used when grad splits is not None... need to better control this...
                               aggressiveness = learning_rate, # todo: this needs to be harmonized and passed in via *args or **kwargs
                               alpha_mult = learning_rate_adjustment)  # todo: this needs to be harmonized and passed in via *args or **kwargs

            # GET THE INDEX OF THE BEST SCORE
            best_lambda = __choose(covariate_penalties, choice)

            # --------------------------------------------------
            # Phase 2: extract V and weights: slow ( tens of seconds to minutes )
            # --------------------------------------------------

            best_V = tensor(X = Xtrain, 
                            Y = Ytrain,
                            X_treat = Xtest
                            y_treat = ytest
                            # *args, **kwargs
                            LAMBDA = best_lambda,
#--                             grad_splits = gradient_folds,
#--                                 random_state = random_state, # TODO:  this is only used when grad splits is not None... need to better control this...
                            aggressiveness = learning_rate, # todo: this needs to be harmonized and passed in via *args or **kwargs
                            alpha_mult = learning_rate_adjustment)  # todo: this needs to be harmonized and passed in via *args or **kwargs

            # GET THE BEST SET OF WEIGHTS
            in_smaple_weights = weights(Xtrain,
                                            Xtest,
                                            V = best_V,
                                            L2_PEN_W = weight_penalty)

            Synthetic_Units = in_smaple_weights.dot(Ytrain)

        else:
            raise ValueError("unexpected prediction_type '%s' or treated_units = None" % prediction_type)

    else:
        
        assert prediction_type = "full", ( "Unexpected prediction_type ='%s' or treated_units is not None" % prediction_type) 

        # Get the L2 penalty guestimate:  very quick ( milliseconds )
        if weight_penalty is None:
            weight_penalty  = L2_pen_guestimate(X) 

        # --------------------------------------------------
        # Phase 1: extract cross fold residual errors for each lambda
        # --------------------------------------------------

        # SCORES FOR EACH VALUE OF THE GRID: very slow ( minutes to hours )
        scores = CV_score( X = X
                           Y = Y
                           # *args, **kwargs
                           splits = cv_folds,
                           LAMBDA = covariate_penalties,
                           progress = progress,
                           L2_PEN_W = weight_penalty,
                           grad_splits = gradient_folds,
                                random_state = random_state, # TODO:  this is only used when grad splits is not None... need to better control this...
                           aggressiveness = learning_rate, # todo: this needs to be harmonized and passed in via *args or **kwargs
                           alpha_mult = learning_rate_adjustment)  # todo: this needs to be harmonized and passed in via *args or **kwargs

        # GET THE INDEX OF THE BEST SCORE
        best_lambda = __choose(covariate_penalties, choice)

        # --------------------------------------------------
        # Phase 2: extract V and weights: slow ( tens of seconds to minutes )
        # --------------------------------------------------

        best_V = tensor(X = X, 
                        Y = Y,
                        LAMBDA = best_lambda,
                        grad_splits = gradient_folds,
                                random_state = random_state, # TODO:  this is only used when grad splits is not None... need to better control this...
                        aggressiveness = learning_rate, # todo: this needs to be harmonized and passed in via *args or **kwargs
                        alpha_mult = learning_rate_adjustment)  # todo: this needs to be harmonized and passed in via *args or **kwargs

        # GET THE BEST SET OF WEIGHTS
        SC_weights = weights(Xtrain,
                             Xtest,
                             V = best_V,
                             L2_PEN_W = weight_penalty)

        synthetic_units = SC_weights.dot(Y)

    return SparseSCFit( X, Y, SC_weights, synthetic_units, control_units, treated_units, model_type,)

def __choose(covariate_penalties, choice):
    """ helper function which implements the choice of covariate weights penalty parameter
    """
        # GET THE INDEX OF THE BEST SCORE
        try: 
            len(covariate_penalties)
        except:
            if choice == "min":
                best_i = np.argmin(scores)
                best_lambda = (covariate_penalties)[best_i]
            elif callable(choice):
                best_lambda = choice(scores)
            else:
                # TODO: this is a terrible place to throw this error
                raise ValueError("Unexpected value for choice parameter: %s" % choice)
        else:
            best_lambda = scores

    return best_lambda


class SparseSCFit(object):
    """ A class representing the results of a Synthetic Control model instance.
    """
    def __init__(self,
            X,
            Y,
            weight_penalty,
            covariate_penalties,
            ):

        self.X = X
        self.Y = Y
        self.SC_weights = SC_weights
        self.synthetic_units = synthetic_units
        self.control_units = control_units
        self.treated_units = treated_units
        self.model_type = model_type

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

