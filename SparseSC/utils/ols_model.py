import numpy as np
import statsmodels.api as sm

import SparseSC as SC
import SparseSC.utils.metrics_utils

#All data in the long format (rows mark id-time)
def OLS_avg_AA_simple(N=100, T=10, K=0, treat_ratio=.1, T0=None):
    #0-based index for i and t. each units panel is together (fine index is time, gross is unit)
    Y = np.random.normal(0,1,(N*T))
    id = np.tile(np.array(range(T)), N)
    time = np.repeat(np.array(range(N)), T)
    if T0 is None:
        T0 =  int(T/2)
    return OLS_avg_AA(Y, id, time, T0, N, T, treat_ratio)

def OLS_avg_AA(Y, id, time, post_start, N, T, treat_ratio, num_sim=1000, X=None, level=0.95):
    Const = np.ones((Y.shape[0], 1))
    Post = np.expand_dims((time>=post_start).astype(int), axis=1)
    X_base = np.hstack((Const, Post))
    #X_base = sm.add_constant(X_base)
    if X is not None:
        X_base = np.hstack(X_base, X)
    alpha = 1-level

    tes = np.empty((num_sim))
    ci_ls = np.empty((num_sim))
    ci_us = np.empty((num_sim))
    N1 = int(N*treat_ratio)
    sel_idx = np.concatenate((np.repeat(1,N1), np.repeat(0,N-N1)))
    for s in range(num_sim):
        np.random.shuffle(sel_idx)
        Treat = np.expand_dims(np.repeat(sel_idx, T), axis=1)
        D = Treat * Post
        X = np.hstack((X_base,Treat, D))
        model = sm.OLS(Y,X, hasconst=True)
        results = model.fit()
        tes[s] = results.params[3]
        [ci_ls[s], ci_us[s]] = results.conf_int(alpha, cols=[3])[0]

    stats = SC.utils.ols_model.simulation_eval(tes, ci_ls, ci_us, true_effect=0)
    print(stats)


#Do separate effects for each post treatment period?
#def OLS_AA_vec(Y, id, time, treat_ratio, post_times, X=None):
#    for post_time in post_times:
#        Post_ind_t = time==post_time
#        X_base = np.hstack((X_base, Post_ind_t))
#
#def OLS_AA_vec_specific(Y, X_base, sel_idx):
#    base_init_len = X_base.shape[1]
#    Treat = np.vstack(id_pre==sel_id, id_post==sel_id)
#    X_base = np.hstack(X_base, Treat)
#    for post_idx in 2:base_init_len:
#        D_t = Treat and X_base[:,post_idx]
#        X = np.hstack((X,D_t))
    
#    model = sm.OLS(Y,X, hasconst=True)
#    results = model.fit()
#    results.params[base_init_len+1:]
