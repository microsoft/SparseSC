"""
Factor DGP
"""

import numpy as np

def ge_dgp(N0,N1,T0,T1,K,S,R,groups,group_scale,beta_scale,confounders_scale,model= "full"):
    """
    From example-code.py
    """

    # COVARIATE EFFECTS
    X_control = np.matrix(np.random.normal(0,1,((N0)*groups, K+S+R)))
    X_treated = np.matrix(np.random.normal(0,1,((N1)*groups, K+S+R)))

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
    Y_pre_ge_control = Y_pre_group_effects[np.repeat(np.arange(groups),N0)]
    Y_pre_ge_treated = Y_pre_group_effects[np.repeat(np.arange(groups),N1)]

    Y_post_group_effects = np.random.normal(0,group_scale,(groups,T1))
    Y_post_ge_control = Y_post_group_effects[np.repeat(np.arange(groups),N0)]
    Y_post_ge_treated = Y_post_group_effects[np.repeat(np.arange(groups),N1)]

    # RANDOM ERRORS
    Y_pre_err_control = np.matrix(np.random.random( ( N0*groups, T0, ) )) 
    Y_pre_err_treated = np.matrix(np.random.random( ( N1*groups, T0, ) )) 

    Y_post_err_control = np.matrix(np.random.random( ( N0*groups, T1, ) )) 
    Y_post_err_treated = np.matrix(np.random.random( ( N1*groups, T1, ) )) 

    # THE DATA GENERATING PROCESS

    if model == "full":
        """ 
        In the full model, covariates (X) are correlated with pre and post
        outcomes, and variance of the outcomes pre- and post- outcomes is lower
        within groups which span both treated and control units.
        """
        Y_pre_control = X_control.dot(beta_control) + Y_pre_ge_control + Y_pre_err_control 
        Y_pre_treated = X_treated.dot(beta_treated) + Y_pre_ge_treated + Y_pre_err_treated 

        Y_post_control = X_control.dot(beta_control) + Y_post_ge_control + Y_post_err_control 
        Y_post_treated = X_treated.dot(beta_treated) + Y_post_ge_treated + Y_post_err_treated 

    elif model == "hidden":
        """ 
        In the hidden model outcomes are independent of the covariates, but
        variance of the outcomes pre- and post- outcomes is lower within groups
        which span both treated and control units.
        """
        Y_pre_control = Y_pre_ge_control + Y_pre_err_control 
        Y_pre_treated = Y_pre_ge_treated + Y_pre_err_treated 

        Y_post_control = Y_post_ge_control + Y_post_err_control 
        Y_post_treated = Y_post_ge_treated + Y_post_err_treated 

    elif model == "null":
        """
        Purely random data
        """
        Y_pre_control = Y_pre_err_control 
        Y_pre_treated = Y_pre_err_treated 

        Y_post_control = Y_post_err_control 
        Y_post_treated = Y_post_err_treated 

    else:
        raise ValueError("Unknown model type: "+model)

    return X_control, X_treated, Y_pre_control, Y_pre_treated, Y_post_control, Y_post_treated

