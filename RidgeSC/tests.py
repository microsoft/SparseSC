import unittest
import numpy as np
import random
import RidgeSC as SC

def ge_dgp(C,N,T0,T1,K,S,R,groups,group_scale,beta_scale,confounders_scale,model= "full"):
    """
    From example-code.py
    """

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

    return X_control, X_treated, Y_pre_control, Y_pre_treated, Y_post_control, Y_post_treated


def factor_dgp(C,N,T0,T1,K,R,F,beta_scale = 1):
    
    # COVARIATE EFFECTS
    X_control = np.matrix(np.random.normal(0,1,((C), K+R)))
    X_treated = np.matrix(np.random.normal(0,1,((N), K+R)))
 
    b_cause = np.random.exponential(1,K)
    b_cause *= b_cause.max()

    beta = np.matrix(np.concatenate( ( b_cause , np.zeros(R)) ) ).T

    # FACTORS
    Loadings_control = np.matrix(np.random.normal(0,1,((C), F)))
    Loadings_treated = np.matrix(np.random.normal(0,1,((N), F)))

    Factors_pre = np.matrix(np.random.normal(0,1,((F), T0)))
    Factors_post = np.matrix(np.random.normal(0,1,((F), T1)))

    
    
    # RANDOM ERRORS
    Y_pre_err_control = np.matrix(np.random.random( ( C, T0, ) )) 
    Y_pre_err_treated = np.matrix(np.random.random( ( N, T0, ) )) 
    Y_post_err_control = np.matrix(np.random.random( ( C, T1, ) )) 
    Y_post_err_treated = np.matrix(np.random.random( ( N, T1, ) )) 

    # OUTCOMES
    Y_pre_control = X_control.dot(beta) + Loadings_control.dot(Factors_pre) + Y_pre_err_control
    Y_pre_treated = X_treated.dot(beta) + Loadings_treated.dot(Factors_pre) + Y_pre_err_treated

    Y_post_control = X_control.dot(beta) + Loadings_control.dot(Factors_post) + Y_post_err_control
    Y_post_treated = X_treated.dot(beta) + Loadings_treated.dot(Factors_post) + Y_post_err_treated


    return X_control, X_treated, Y_pre_control, Y_pre_treated, Y_post_control, Y_post_treated

class TestDGPs(unittest.TestCase):
    def testFactorDGP(self):
        C,N = 100, 1
        T0,T1 = 20, 10
        K, R, F = 5, 5, 5
        X_control, X_treated, Y_pre_control, Y_pre_treated, Y_post_control, Y_post_treated = factor_dgp(C,N,T0,T1,K,R,F,beta_scale = 1)
        
        Y_post = np.vstack( (Y_post_control, Y_post_treated,) )
        X_and_Y_pre_control = np.hstack( ( X_control, Y_pre_control,) )

        
        # get the maximum value for the L1 Penalty parameter conditional on the guestimate for the L2 penalty
        L2_pen_start_loo = SC.L2_pen_guestimate(X_and_Y_pre_control)
        L1_max_loo = SC.get_max_lambda(X_and_Y_pre_control[np.arange(100)],Y_post[np.arange(100)]) ####
        
        # ------------------------------------------------------------
        # create a grid of penalties to try 
        # ------------------------------------------------------------
        n_points = 10 # number of points in the grid

        # Another equally spaced linear grid
        fmax = 1e-2 # lowest point in the grid relative to the max-lambda
        fmin = 1e-4 # highest point in the grid relative to the max-lambda
        grid = np.exp(np.linspace(np.log(fmin),np.log(fmax),n_points))
        
        print("Starting grid scoring for Controls Only scenario with 5-fold gradient descent", grid*L1_max_loo)
        grid_scores_loo = SC.CV_score(
            X = X_and_Y_pre_control, # limit the amount of time...
            Y = Y_post_control     , # limit the amount of time...

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
        best_LAMBDA = (grid * L1_max_loo)[np.argmin(grid_scores_loo)]
        V_loo = SC.tensor(X = X_and_Y_pre_control,
			  Y = Y_post_control,

			  LAMBDA = best_LAMBDA,

			  # Also optional
			  L2_PEN_W = L2_pen_start_loo)
        SC_weights_loo = SC.weights(X = X_and_Y_pre_control,
                            V = V_loo,
                            L2_PEN_W = L2_pen_start_loo)
        Y_pre  = np.vstack( (Y_pre_treated, Y_pre_control,  ) )
        Y_post = np.vstack( (Y_post_treated, Y_post_control, ) )
        Y = np.hstack( (Y_pre, Y_post) )

        est_res = SC.estimate_effects(Y_pre, Y_post, X_and_Y_pre_control, V_loo, [1], L2_pen_start_loo)


        #self.failUnlessEqual(calc, truth)

if __name__ == '__main__':
    random.seed(12345)
    np.random.seed(10101)

    unittest.main()
