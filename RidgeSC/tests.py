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


def factor_dgp(C,N,T0,T1,K,R,F):
    
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
        N, C = 1,100
        T0,T1 = 20, 10
        K, R, F = 5, 5, 5
        X_control, X_treated, Y_pre_control, Y_pre_treated, Y_post_control, Y_post_treated = factor_dgp(C,N,T0,T1,K,R,F)
        
        Y_post = np.vstack( (Y_post_treated,Y_post_control, ) )
        X = np.vstack( (X_treated, X_control, ) )
        Y_pre  = np.vstack( (Y_pre_treated, Y_pre_control, ) )
        treated_units = [0]
        #X_and_Y_pre = np.hstack( ( X, Y_pre,) )

        est_res = SC.estimate_effects(X, Y_pre, Y_post, treated_units)
        print(est_res)

        #self.failUnlessEqual(calc, truth)

if __name__ == '__main__':
    random.seed(12345)
    np.random.seed(10101)

    unittest.main()
