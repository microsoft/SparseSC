import pickle
import random

import scipy
import numpy as np
import pandas as pd

try:
    import SparseSC as SC
except ImportError:
    raise RuntimeError("SparseSC is not installed. Use 'pip install -e .' or 'conda develop .' from repo root to install in dev mode")

random.seed(12345)
np.random.seed(101101001)

pkl_file = "../replication/smoking_fits.pkl"

smoking_df = pd.read_stata("../replication/smoking.dta")
smoking_df['year'] = smoking_df['year'].astype('int')
smoking_df = smoking_df.set_index(['state', 'year'])
Y = smoking_df[['cigsale']].unstack('year')
T0 = 19
i_t = 2 #unit 3, but zero-index
treated_units = [i_t]
control_units = [u for u in range(Y.shape[0]) if u not in treated_units]


Y_names = Y.columns.get_level_values('year')
Y_pre_names = ["cigsale(" + str(i) + ")" for i in Y_names[:T0]]
print(Y.isnull().sum().sum()) #0
Y = Y.values
T = Y.shape[1]
T1 = T-T0
Y_pre,Y_post = Y[:,:T0], Y[:,T0:]

# Stata: synth cigsale beer(1984(1)1988) lnincome retprice age15to24 cigsale(1988) cigsale(1980) cigsale(1975), xperiod(1980(1)1988)  trunit(3) trperiod(1989) 

year_ind = smoking_df.index.get_level_values('year')
beer_pre = smoking_df.loc[np.logical_and(year_ind>=1984, year_ind<=1988),["beer"]]
Xother_pre = smoking_df.loc[np.logical_and(year_ind>=1980, year_ind<=1988), ['lnincome', 'retprice', 'age15to24']]
X_avgs = pd.concat((beer_pre.groupby('state').mean(), 
                    Xother_pre.groupby('state').mean())
                   , axis=1)

#X_spot = pd.DataFrame({'cigsale_75':smoking_df.xs(1975, level='year')["cigsale"], 
#                       'cigsale_80':smoking_df.xs(1980, level='year')["cigsale"], 
#                       'cigsale_88':smoking_df.xs(1988, level='year')["cigsale"]})
#X_orig = pd.concat((X_avgs, X_spot), axis=1)
#X_orig.isnull().sum().sum() #0

X_full = pd.concat((X_avgs, beer_pre.unstack('year'), Xother_pre.unstack('year')), axis=1)
X_full_names = [c[0] + "(" + str(c[1]) + ")" if len(c)==2 else c for c in X_full.columns]
X_full.isnull().sum().sum() #0
X_full = X_full.values
X_Y_pre = np.concatenate((X_full, Y_pre), axis=1)
X_Y_pre_names = X_full_names + Y_pre_names
X_Y_pre_names_arr = np.array(X_Y_pre_names)

def print_summary(fit, Y_pre, Y_post, Y_sc):
    full_Y_pre_sc,full_Y_post_sc = Y_sc[:,:T0], Y_sc[:,T0:]
    print("V: " + str(np.diag(fit.V)))
    print("V>0: " + str(np.diag(fit.V)[np.diag(fit.V)>0]))
    print("#V>0: " + str(sum(np.diag(fit.V>0))))
    full_Y_pre_effect = Y_pre - full_Y_pre_sc
    full_Y_post_effect = Y_post - full_Y_post_sc

    print("Avg bias pre: " + str(full_Y_pre_effect[control_units, :].mean()))
    print(scipy.stats.ttest_1samp(full_Y_pre_effect[control_units, :].flatten(), popmean=0)) 
    full_pre_mse = np.mean(np.power(full_Y_pre_effect[control_units, :], 2))
    print("Avg MSE pre: " + str(full_pre_mse) )

    print("Avg effect post: " + str(full_Y_post_effect[control_units, :].mean()))
    print(scipy.stats.ttest_1samp(full_Y_post_effect[control_units, :].flatten(), popmean=0)) 
    full_post_mse = np.mean(np.power(full_Y_post_effect[control_units, :], 2))
    print("Avg MSE post: " + str(full_post_mse) )

    print(X_Y_pre_names_arr[np.diag(fit.V)>0])


# Fast  ----------------------#

if False:
    fast_fit = SC.fit_fast(X_Y_pre, Y_post, treated_units=[i_t])
    #print(len(np.diag(fast_fit.V)))
    #print(np.diag(fast_fit.V))
    #Y_post_sc = fast_fit.predict(Y_post)
    #Y_pre_sc = fast_fit.predict(Y_pre)
    #post_mse = np.mean(np.power(Y_post[control_units, :] - Y_post_sc[control_units, :], 2))
    #pre_mse = np.mean(np.power(Y_pre[control_units, :] - Y_pre_sc[control_units, :], 2))
    #print(pre_mse) #192.210632448
    #print(post_mse) #129.190437803
    #print(X_Y_pre_names_arr[fast_fit.match_space_desc>0])

# Full  ----------------------#

use_est_fn = True
#if use_est_fn:
#    est_fit = SC.estimate_effects(
#        outcomes=Y,
#        unit_treatment_periods=X,
#        covariates=None,
#        fast = False,
#        cf_folds = len(control_units), #sync with helper
#        **kwargs
#    )
#    full_fit = 
#else: 
full_fit = SC.fit(X_Y_pre, Y_post, treated_units=[i_t]) 
full_Y_sc = full_fit.predict(Y)
print_summary(full_fit, Y_pre, Y_post, full_Y_sc)
#Avg bias pre: 0.0775621971136
#Ttest_1sampResult(statistic=0.15231397793811494, pvalue=0.87898191261344316)
#Avg MSE pre: 186.969136939
#Avg effect post: 0.036591103495529125
#Ttest_1sampResult(statistic=0.065402542511047407, pvalue=0.94788222532433364)
#Avg MSE post: 142.422049057
print_summary(full_fit, Y_pre, Y_post, full_Y_sc)
Y_sc_c = get_c_preditions_honest(X_Y_pre[control_units,:], Y_post[control_units,:], Y[control_units,:], 
                                                       "retrospective", full_fit.initial_w_pen, full_fit.initial_v_pen, cf_folds, cf_seed)


# Full - Flat ----------------------#

full_fit_flat = SC._fit_fast_inner(X_Y_pre, X_Y_pre, Y_post, V=np.repeat(1,X_Y_pre.shape[1]), treated_units=[i_t])

full_flat_Y_sc = full_fit_flat.predict(Y)

print_summary(full_fit_flat, Y_pre, Y_post, full_flat_Y_sc)
#Avg bias pre: 0.00351336303993
#Ttest_1sampResult(statistic=0.050337513196736801, pvalue=0.95986737195005822)
#Avg MSE pre: 3.51236252311

#Avg effect post: -0.402068307329
#Ttest_1sampResult(statistic=-0.72604957692076211, pvalue=0.46818168129036042)
#Avg MSE post: 139.695171265


with open(pkl_file, "wb" ) as output_file:
    pickle.dump( (fast_fit, full_fit, full_fit_flat),  output_file)

with open(pkl_file, "rb" ) as input_file:
    fast_fit, full_fit, full_fit_flat = pickle.load(input_file)
