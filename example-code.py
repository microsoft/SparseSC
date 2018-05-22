"""
USAGE: 

cd path/to/RidgeSC
python example-code.py
"""

import os
import sys
sys.path.append(os.path.join(os.getcwd(),".."))

import time
import RidgeSC as SC
import numpy as np
import random


# def setup(C,N,T,K,g,gs,bs ): # controls (C), treated units (N) , time periods (T), Predictors (K), groups (g), g-scale, b-scale

if __name__ == "__main__":

    # ------------------------------------------------------------
    # ------------------------------------------------------------
    # SECTION 1: GENERATE SOME TOY DATA
    # ------------------------------------------------------------
    # ------------------------------------------------------------

    # CONTROL PARAMETERS

    random.seed(12345)
    np.random.seed(10101)

    # Controls (per group), Treated (per group), pre-intervention Time points,  post intervention time-periods
    C,N,T0,T1 = 5,5,4,4
    # Causal Covariates, Confounders , Random Covariates 
    K,S,R = 7,4,5
    # Number of Groups, Scale of the Group Effect, (groups make the time series correlated for reasons other than the observed covariates...)
    groups,group_scale = 30,2
    # Scale of the Correlation of the Causal Covariates
    beta_scale,confounders_scale = 4,1

    # ------------------------------------------------------------

    # COVARIATE EFFECTS
    X_control = np.matrix(np.random.normal(0,1,((C)*groups, K+S+R)))
    X_treated = np.matrix(np.random.normal(0,1,((N)*groups, K+S+R)))

    # CAUSAL 
    b_cause = np.random.exponential(1,K)
    b_cause *= beta_scale / b_cause.max()

    # CONFOUNDERS 
    b_confound = np.random.exponential(1,S)
    b_confound *= confounders_scale / b_confound.max()

    beta_control = np.matrix(np.concatenate( ( b_cause ,b_confound,  np.zeros(R)) ) ).T
    beta_treated = np.matrix(np.concatenate( ( b_cause ,np.zeros(S), np.zeros(R)) ) ).T

    # GROUP EFFECTS (hidden)

    Y_pre_group_effects = np.random.normal(0,group_scale,(groups,T0))
    Y_pre_ge_control = Y_pre_group_effects[np.repeat(np.arange(groups),C)]
    Y_pre_ge_treated = Y_pre_group_effects[np.repeat(np.arange(groups),N)]

    Y_post_group_effects = np.random.normal(0,group_scale,(groups,T1))
    Y_post_ge_control = Y_post_group_effects[np.repeat(np.arange(groups),C)]
    Y_post_ge_treated = Y_post_group_effects[np.repeat(np.arange(groups),N)]

    # RANDOM ERRORS
    Y_pre_err_control = np.matrix(np.random.random( ( C*groups, T0, ) )) 
    Y_pre_err_treated = np.matrix(np.random.random( ( N*groups, T0, ) )) 

    Y_post_err_control = np.matrix(np.random.random( ( C*groups, T1, ) )) 
    Y_post_err_treated = np.matrix(np.random.random( ( N*groups, T1, ) )) 

    # THE DATA GENERATING PROCESS
    model = "full"

    if model == "full":
        """ In the full model, covariates (X) are correlated with pre and post
            outcomes, and variance of the outcomes pre- and post- outcomes is
            lower within groups which span both treated and control units.
        """
        Y_pre_control = X_control.dot(beta_control) + Y_pre_ge_control + Y_pre_err_control 
        Y_pre_treated = X_treated.dot(beta_treated) + Y_pre_ge_treated + Y_pre_err_treated 

        Y_post_control = X_control.dot(beta_control) + Y_post_ge_control + Y_post_err_control 
        Y_post_treated = X_treated.dot(beta_treated) + Y_post_ge_treated + Y_post_err_treated 

    elif model == "hidden":
        """ In the hidden model outcomes are independent of the covariates, but
            variance of the outcomes pre- and post- outcomes is lower within
            groups which span both treated and control units.
        """
        Y_pre_control = Y_pre_ge_control + Y_pre_err_control 
        Y_pre_treated = Y_pre_ge_treated + Y_pre_err_treated 

        Y_post_control = Y_post_ge_control + Y_post_err_control 
        Y_post_treated = Y_post_ge_treated + Y_post_err_treated 

    elif model == "null":
        "Purely random data" 
        Y_pre_control = Y_pre_err_control 
        Y_pre_treated = Y_pre_err_treated 

        Y_post_control = Y_post_err_control 
        Y_post_treated = Y_post_err_treated 

    else:
        raise ValueError("Unknown model type: "+model)

    # JOIN THE TREAT AND CONTROL DATA
    X = np.vstack( (X_control, X_treated,) )
    Y_pre  = np.vstack( (Y_pre_control,  Y_pre_treated, ) )
    Y_post = np.vstack( (Y_post_control, Y_post_treated,) )

    # in the leave-one-out scneario, the pre-treatment outcomes will be part of the covariates
    X_and_Y_pre = np.hstack( ( X, Y_pre,) )

    # IDENTIFIERS FOR TREAT AND CONTROL UNITS
    # control_units = np.arange( C * groups )
    # treated_units = np.arange( N * groups ) + C

    # ------------------------------------------------------------
    # ------------------------------------------------------------
    # find default penalties
    # ------------------------------------------------------------
    # ------------------------------------------------------------

    # get starting point for the L2 penalty 
    L2_pen_start_ct  = SC.L2_pen_guestimate(X_control)
    L2_pen_start_loo = SC.L2_pen_guestimate(X_and_Y_pre)

    # get the maximum value for the L1 Penalty parameter conditional on the guestimate for the L2 penalty
    L1_max_ct  = SC.get_max_lambda(X_control,Y_pre_control,X_treat=X_treated,Y_treat=Y_pre_treated)
    if False:
        L1_max_loo = SC.get_max_lambda(X_and_Y_pre[np.arange(100)],Y_post[np.arange(100)])
        print("Max L1 loo %s " % L1_max_loo)
    else:
        L1_max_loo = np.float(147975295.9121998)

    if False:
        "Demonstrate relations between the L1 and L2 penalties"

        # get the maximum value for the L1 Penalty parameter conditional on several L2 penalty parameter values
        L2_grid = (2.** np.arange(-1,2))

        L1_max_loo_grid = SC.get_max_lambda(X_and_Y_pre,
                                           Y_post, 
                                           L2_PEN_W = L2_pen_start_loo * L2_grid)
        L1_max_ct_grid = SC.get_max_lambda(X_control,
                                           Y_pre_control,
                                           X_treat=X_treated,
                                           Y_treat=Y_pre_treated,
                                           L2_PEN_W = L2_pen_start_ct * L2_grid)
        assert ( L1_max_loo / L1_max_loo_grid  == 1/L2_grid).all()
        assert ( L1_max_ct  / L1_max_ct_grid   == 1/L2_grid).all()


    # ------------------------------------------------------------
    # create a grid of penalties to try 
    # ------------------------------------------------------------
    n_points = 10 # number of points in the grid

    grid_type = "log-linear" # Usually the optimal penalties are quite Small

    if grid_type == "simple":
        # An equally spaced linear grid that does not include 0 or 1
        grid = ( 1 + np.arange(n_points) ) / ( 1 + n_points )
    elif grid_type == "linear":
        # Another equally spaced linear grid
        fmax = 1e-3 # lowest point in the grid relative to the max-lambda
        fmin = 1e-5 # highest point in the grid relative to the max-lambda
        grid = np.linspace(fmin,fmax,n_points)
    elif grid_type == "log-linear":
        # Another equally spaced linear grid
        fmax = 1e-2 # lowest point in the grid relative to the max-lambda
        fmin = 1e-4 # highest point in the grid relative to the max-lambda
        grid = np.exp(np.linspace(np.log(fmin),np.log(fmax),n_points))
    else:
        raise ValueError("Unknown grid type: %s" % grid_type)


    # ------------------------------------------------------------
    # get the Cross Validation Error over a grid of L1 penalty
    # parameters using both (a) Treat/Control and (b) the 
    # leave-one-out  controls only methods
    # ------------------------------------------------------------

    if False:

        print("starting grid scoring for treat / control scenario", grid*L1_max_ct)
        grid_scores_ct = SC.CV_score(
            X = X_control,
            Y = Y_pre_control,

            X_treat = X_treated,
            Y_treat = Y_pre_treated,

            # if LAMBDA is a single value, we get a single score, If it's an array of values, we get an array of scores.
            LAMBDA = grid * L1_max_ct,
            L2_PEN_W = L2_pen_start_ct,

            # CACHE THE V MATRIX BETWEEN LAMBDA PARAMETERS (generally faster, but path dependent)
            cache = False, # False by Default

            # Run each of the Cross-validation folds in parallel? Often slower
            # for large sample sizes because numpy.linalg.solve() already runs
            # in parallel for large matrices
            parallel=False,

            # ANNOUNCE COMPLETION OF EACH ITERATION
            progress = True)

        best_L1_penalty_ct = (grid * L1_max_ct)[np.argmin(grid_scores_ct)]

    if False:

        print("Starting grid scoring for Controls Only scenario with 5-fold gradient descent", grid*L1_max_ct)
        grid_scores_loo = SC.CV_score(
            X = X_and_Y_pre, # limit the amount of time...
            Y = Y_post     , # limit the amount of time...

            # this is what enables the k-fold gradient descent
            grad_splits = 5,
            random_state = 10101, # random_state for the splitting during k-fold gradient descent

            # L1 Penalty. if LAMBDA is a single value (value), we get a single score, If it's an array of values, we get an array of scores.
            LAMBDA = grid * L1_max_loo,

            # L2 Penalty (float)
            L2_PEN_W = L2_pen_start_loo,

            # CACHE THE V MATRIX BETWEEN LAMBDA PARAMETERS (generally faster, but path dependent)
            #cache = True, # False by Default

            # Run each of the Cross-validation folds in parallel? Often slower
            # for large sample sizes because numpy.linalg.solve() already runs
            # in parallel for large matrices
            parallel=False,

            # announce each call to `numpy.linalg.solve(A,B)` (the major bottleneck)
            verbose = False, # it's kind of obnoxious, but gives a sense of running time per gradient calculation

            # ANNOUNCE COMPLETION OF EACH ITERATION
            progress = True)

    if False:

        # even with smaller data, this takes a while.
        print("Starting grid scoring for Controls Only scenario with leave-one-out gradient descent", grid*L1_max_ct)
        grid_scores_loo = SC.CV_score(
            X = X_and_Y_pre [np.arange(100)], # limit the amount of time...
            Y = Y_post      [np.arange(100)], # limit the amount of time...

            # with `grad_splits = None` (the default behavior) we get leave-one-out gradient descent.
            grad_splits = None,

            # L1 Penalty. if LAMBDA is a single value (value), we get a single score, If it's an array of values, we get an array of scores.
            LAMBDA = grid * L1_max_loo,

            # L2 Penalty (float)
            L2_PEN_W = L2_pen_start_loo,

            # CACHE THE V MATRIX BETWEEN LAMBDA PARAMETERS (generally faster, but path dependent)
            #cache = True, # False by Default

            # Run each of the Cross-validation folds in parallel? Often slower
            # for large sample sizes because numpy.linalg.solve() already runs
            # in parallel for large matrices
            parallel=False,

            # announce each call to `numpy.linalg.solve(A,B)` (the major bottleneck)
            verbose = False, # it's kind of obnoxious, but gives a sense of running time per gradient calculation

            # ANNOUNCE COMPLETION OF EACH ITERATION
            progress = True)

    # ---------------------------------------------------------------------------
    # Calculate Synthetic Control weights for a fixed pair of penalty parameters
    # ---------------------------------------------------------------------------

    # This is a two-step process because in principle, we can estimate a 
    # tensor matrix (which contains the relative weights of the covariates and
    # possibly pre-treatment outcomes) in one population, and apply it to
    # another population.
    best_L1_penalty_ct = np.float(1908.9329)

    # -----------------------------------
    # Treat/Control:
    # -----------------------------------

    V_ct = SC.tensor(X = X_control,
                     Y = Y_pre_control,
                     X_treat = X_treated,
                     Y_treat = Y_pre_treated,
                     LAMBDA = best_L1_penalty_ct,
                     L2_PEN_W = L2_pen_start_ct)

    SC_weights_ct = SC.weights(X = X_control,
                               X_treat = X_treated,
                               V = V_ct,
                               L2_PEN_W = L2_pen_start_ct)

    Y_post_treated_synthetic_conrols_ct = SC_weights_ct.dot(Y_post_control)
    ct_prediction_error = Y_post_treated_synthetic_conrols_ct - Y_post_treated
    null_model_error = Y_post_treated - np.mean(Y_pre_treated)
    R_squared_post_ct = 1 - np.power(ct_prediction_error,2).sum() / np.power(null_model_error,2).sum()

    print( "C/T: Out of Sample post intervention R squared: %0.2f%% " % (100*R_squared_post_ct,))


    # -----------------------------------
    # Leave-One-Out with Control Only:
    # -----------------------------------

    if False: 
        # this takes a while:
        V_loo = SC.tensor(
                X = X_and_Y_pre [np.arange(100)], # limit the amount of time...
                Y = Y_post      [np.arange(100)], # limit the amount of time...
                LAMBDA = best_L1_penalty_ct,
                L2_PEN_W = L2_pen_start_loo)

        SC_weights_loo = SC.weights(X = X_control,
                                    V = V_ct,
                                    L2_PEN_W = L2_pen_start_loo)

        # in progress...
        import pdb; pdb.set_trace()

        Y_post_treated_synthetic_conrols_loo = SC_weights_ct.dot(Y_post_control)
        loo_prediction_error = Y_post_treated_synthetic_conrols_loo - Y_post_treated
        null_model_error = Y_post_treated - np.mean(Y_pre_treated)
        R_squared_post_ct = 1 - np.power(loo_prediction_error,2).sum() / np.power(null_model_error,2).sum()

        print( "LOO: Out of Sample post intervention R squared: %0.2f%% " % (100*R_squared_post_ct,))

        import pdb; pdb.set_trace()


    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    # Optimization of the L1 and L2 parameters together (second order)
    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------

    from scipy.optimize import fmin_l_bfgs_b, differential_evolution 
    import time

    # -----------------------------------------------------------------
    # Optimization of the L2 and L1 Penalties Simultaneously, keeping their
    # product constant.  Heuristically, this has been most efficient means of
    # optimizing the L2 Parameter.
    # -----------------------------------------------------------------


    # build the objective function to be minimized

    # cache for L2_obj_func
    n_calls = [0,]
    temp_results =[]
    SS = np.power(Y_pre_treated - np.mean(Y_pre_treated),2).sum()
    best_L1_penalty_ct = np.float(1908.9329)

    def L1_L2_obj_func (x): 
        n_calls[0] += 1
        t1 = time.time();
        score = SC.CV_score(X = X_control,
                            Y = Y_pre_control,
                            X_treat = X_treated,
                            Y_treat = Y_pre_treated,
                            # if LAMBDA is a single value, we get a single score, If it's an array of values, we get an array of scores.
                            LAMBDA = best_L1_penalty_ct * np.exp(x[0]),
                            L2_PEN_W = L2_pen_start_ct / np.exp(x[0]),
                            # suppress the analysis type message
                            quiet = True)
        t2 = time.time(); 
        temp_results.append((n_calls[0],x,score))
        print("calls: %s, time: %0.4f, x0: %0.4f, Cross Validation Error: %s, out-of-sample R-Squared: %s" % (n_calls[0], t2 - t1, x[0], score, 1 - score / SS ))
        #print("calls: %s, time: %0.4f, x0: %0.4f, x1: %0.4f, Cross Validation Error: %s, R-Squared: %s" % (n_calls[0], t2 - t1, x[0], x[1], score, 1 - score / SS ))
        return score

    # the actual optimization
    print("Starting L2 Penalty optimization")

    results = differential_evolution(L1_L2_obj_func, bounds = ((-6,6,),)  )
    #results = differential_evolution(L1_L2_obj_func, bounds = ((-6,6,),)*2  )

    import pdb; pdb.set_trace()
    NEW_best_L1_penalty_ct = best_L1_penalty_ct * np.exp(results.x[0])
    best_L2_penalty = L2_pen_start_ct * np.exp(results.x[1])

    print("DE optimized L2 Penalty: %s, DE optimized  L1 penalty: %s"  % (NEW_best_L1_penalty_ct, best_L2_penalty,) )

    # -----------------------------------------------------------------
    # Optimization of the L2 Parameter alone
    # -----------------------------------------------------------------

    # -----------------------
    # build the objective function to be minimized
    # -----------------------

    # OBJECTIVE FUNCTION(S) TO MINIMIZE USING DE

    # cache for L2_obj_func
    n_calls = [0,]
    temp_results =[]
    SS = np.power(Y_pre_treated - np.mean(Y_pre_treated),2).sum()
    best_L1_penalty_ct = np.float(1908.9329)

    def L2_obj_func (x): 
        n_calls[0] += 1
        t1 = time.time();

        score = SC.CV_score(X = X_control,
                            Y = Y_pre_control,
                            X_treat = X_treated,
                            Y_treat = Y_pre_treated,
                            # if LAMBDA is a single value, we get a single score, If it's an array of values, we get an array of scores.
                            LAMBDA = best_L1_penalty_ct,
                            L2_PEN_W = L2_pen_start_ct * np.exp(x[0]),
                            # suppress the analysis type message
                            quiet = True)
        
        t2 = time.time(); 
        temp_results.append((n_calls[0],x,score))
        print("calls: %s, time: %0.4f, x0: %0.4f, Cross Validation Error: %s, out-of-sample R-Squared: %s" %
                (n_calls[0], t2 - t1, x[0], score, 1 - score / SS ))
        return score

    # the actual optimization
    print("Starting L2 Penalty optimization")

    results = differential_evolution(L2_obj_func, bounds = ((-6,6,),))
    NEW_L2_pen_start_ct = L2_pen_start_ct * np.exp(results.x[1])
    print("DE optimized L2 Penalty: %s, using fixed L1 penalty: %s"  % (NEW_L2_pen_start_ct, best_L1_penalty_ct,) )

    # -----------------------------------------------------------------
    # Optimization of the L2 and L1 Penalties Simultaneously
    # -----------------------------------------------------------------

    # build the objective function to be minimized

    # cache for L2_obj_func
    n_calls = [0,]
    temp_results =[]
    SS = np.power(Y_pre_treated - np.mean(Y_pre_treated),2).sum()
    best_L1_penalty_ct = np.float(1908.9329)

    def L1_L2_obj_func (x): 
        n_calls[0] += 1
        t1 = time.time();
        score = SC.CV_score(X = X_control,
                            Y = Y_pre_control,
                            X_treat = X_treated,
                            Y_treat = Y_pre_treated,
                            # if LAMBDA is a single value, we get a single score, If it's an array of values, we get an array of scores.
                            LAMBDA = best_L1_penalty_ct * np.exp(x[0]),
                            L2_PEN_W = L2_pen_start_ct * np.exp(x[1]),
                            # suppress the analysis type message
                            quiet = True)
        t2 = time.time(); 
        temp_results.append((n_calls[0],x,score))
        print("calls: %s, time: %0.4f, x0: %0.4f, x1: %0.4f, Cross Validation Error: %s, out-of-sample R-Squared: %s" % (n_calls[0], t2 - t1, x[0], x[1], score, 1 - score / SS ))
        return score

    # the actual optimization
    print("Starting L2 Penalty optimization")

    results = differential_evolution(L1_L2_obj_func, bounds = ((-6,6,),)*2  )

    import pdb; pdb.set_trace()
    NEW_best_L1_penalty_ct = best_L1_penalty_ct * np.exp(results.x[0])
    best_L2_penalty = L2_pen_start_ct * np.exp(results.x[1])

    print("DE optimized L2 Penalty: %s, DE optimized  L1 penalty: %s"  % (NEW_best_L1_penalty_ct, best_L2_penalty,) )





