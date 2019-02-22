import unittest
import numpy as np
import random
import SparseSC as SC

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


def factor_dgp(N0,N1,T0,T1,K,R,F):
    '''
    Factor DGP. 
    Covariates: Values are drawn from N(0,1) and coefficients are drawn from an exponential(1) and then scaled by the max.
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

class TestDGPs(unittest.TestCase):
    def testSimpleTrendDGP(self):
        '''
        No X, just Y. half the donors are great, other half are bad
        '''
        N1,N0_sim,N0_not = 1,50,50
        N0 = N0_sim + N0_not
        N=N1+N0
        treated_units = [0]
        T0,T1 = 5, 5
        T=T0+T1
        proto_sim = np.array(range(0,T,1),ndmin=2)
        proto_not = np.array(range(0,2*T,2),ndmin=2)
        te = np.hstack((np.zeros((1,T0)), np.full((1,T0), 2)))
        Y1 = proto_sim + te
        Y0_sim = np.matmul(np.ones((N0_sim,1)), proto_sim)
        Y0_not = np.matmul(np.ones((N0_not,1)), proto_not)
        Y = np.vstack((Y1,Y0_sim,Y0_not))

        ret = SC.estimate_effects(Y[:,:T0], Y[:,T0:], treated_units)
        print(ret)



    def testFactorDGP(self):
        N1, N0  = 2,100
        treated_units = [0,1]
        T0,T1 = 20, 10
        K, R, F = 5, 5, 5
        Cov_control, Cov_treated, Out_pre_control, Out_pre_treated, Out_post_control, Out_post_treated = factor_dgp(N0,N1,T0,T1,K,R,F)
        
        Cov = np.vstack( (Cov_treated, Cov_control, ) )
        Out_pre  = np.vstack( (Out_pre_treated, Out_pre_control, ) )
        Out_post = np.vstack( (Out_post_treated,Out_post_control, ) )
        
        SC.estimate_effects(Out_pre, Out_post, treated_units, Cov)
        print(fit_res)
        #est_res = SC.estimate_effects(Cov, Out_pre, Out_post, treated_units, V_penalty = 0, W_penalty = 0.001)
        #print(est_res)

        #self.failUnlessEqual(calc, truth)

    #Simulations
    #1) As T0 and N0 increases do 
    ##a) SC match actuals in terms of the factor loadings
    ##b) our estimates look consistent and have good coverage
    ##c) Can we match a longer set of factor loadings
    # Other Counterfactual prediction:
    ## a) Compare to SC (big N0, small T0, then SC; or many factors; should do bad) to basic time-series model

if __name__ == '__main__':
    random.seed(12345)
    np.random.seed(10101)

    t = TestDGPs()
    t.testSimpleTrendDGP()
    #unittest.main()
