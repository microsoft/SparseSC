"""
Factor DGP
"""

import numpy as np

def factor_dgp(N0,N1,T0,T1,K,R,F):
    '''
    Factor DGP. 

    Covariates: Values are drawn from N(0,1) and coefficients are drawn from an
                exponential(1) and then scaled by the max.
    Factors and Loadings are from N(0,1)
    Errors are from N(0,1)

    :param N0:
    :param N1:
    :param T0:
    :param T1:
    :param K: Number of covariates that affect outcome
    :param R: Number of (noise) covariates to do not affect outcome
    :param F: Number of factors
    :returns: (X_control, X_treated, Y_pre_control, Y_pre_treated, Y_post_control, Y_post_treated)
    '''
    
    # COVARIATE EFFECTS
    X_control = np.matrix(np.random.normal(0,1,((N0), K+R)))
    X_treated = np.matrix(np.random.normal(0,1,((N1), K+R)))
 
    b_cause = np.random.exponential(1,K)
    b_cause *= 1 / b_cause.max()

    beta = np.matrix(np.concatenate( ( b_cause , np.zeros(R)) ) ).T

    # FACTORS
    Loadings_control = np.matrix(np.random.normal(0,1,((N0), F)))
    Loadings_treated = np.matrix(np.random.normal(0,1,((N1), F)))

    Factors_pre = np.matrix(np.random.normal(0,1,((F), T0)))
    Factors_post = np.matrix(np.random.normal(0,1,((F), T1)))

    # RANDOM ERRORS
    Y_pre_err_control = np.matrix(np.random.normal(0, 1, ( N0, T0, ) )) 
    Y_pre_err_treated = np.matrix(np.random.normal(0, 1, ( N1, T0, ) )) 
    Y_post_err_control = np.matrix(np.random.normal(0, 1, ( N0, T1, ) )) 
    Y_post_err_treated = np.matrix(np.random.normal(0, 1, ( N1, T1, ) )) 

    # OUTCOMES
    Y_pre_control = X_control.dot(beta) + Loadings_control.dot(Factors_pre) + Y_pre_err_control
    Y_pre_treated = X_treated.dot(beta) + Loadings_treated.dot(Factors_pre) + Y_pre_err_treated

    Y_post_control = X_control.dot(beta) + Loadings_control.dot(Factors_post) + Y_post_err_control
    Y_post_treated = X_treated.dot(beta) + Loadings_treated.dot(Factors_post) + Y_post_err_treated

    return X_control, X_treated, Y_pre_control, Y_pre_treated, Y_post_control, Y_post_treated

