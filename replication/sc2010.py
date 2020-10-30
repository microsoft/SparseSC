import pickle

import numpy as np
import pandas as pd

import SparseSC as SC

random.seed(12345)
np.random.seed(101101001)

pkl_file = "smoking_fits.pkl"

smoking_df = pd.read_stata("../replication/smoking.dta")
smoking_df['year'] = smoking_df['year'].astype('int')
smoking_df = smoking_df.set_index(['state', 'year'])
T0 = 19
i_t = 2 #unit 3, but zero-index
treated_units = [i_t]
control_units = [u for u in range(Y.shape[0]) if u not in treated_units]


Y = smoking_df[['cigsale']].unstack('year')
Y_names = Y.columns.get_level_values('year')
Y_pre_names = ["cigsale(" + str(i) + ")" for i in Y_names[:T0]]
Y.isnull().sum().sum() #0
Y = Y.values
T = Y.shape[1]
T1 = T-T0
Y_pre = Y[:,:T0]
Y_post = Y[:,T0:]

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

fast_fit = SC.fit_fast(X_Y_pre, Y_post, treated_units=[i_t])
print(len(np.diag(fast_fit.V)))
print(np.diag(fast_fit.V))
Y_post_sc = fast_fit.predict(Y_post)
Y_pre_sc = fast_fit.predict(Y_pre)
post_mse = np.mean(np.power(Y_post[control_units, :] - Y_post_sc[control_units, :], 2))
pre_mse = np.mean(np.power(Y_pre[control_units, :] - Y_pre_sc[control_units, :], 2))
print(pre_mse) #192.210632448
print(post_mse) #129.190437803
X_Y_pre_names_arr[fast_fit.match_space_desc>0]


full_fit = SC.fit(X_Y_pre, Y_post, treated_units=[i_t])
print(np.diag(full_fit.V))
print(np.diag(full_fit.V)[np.diag(full_fit.V)>0])
print(sum(np.diag(full_fit.V>0)))
full_Y_post_sc = full_fit.predict(Y_post)
full_Y_pre_sc = full_fit.predict(Y_pre)
full_post_mse = np.mean(np.power(Y_post[control_units, :] - full_Y_post_sc[control_units, :], 2))
full_pre_mse = np.mean(np.power(Y_pre[control_units, :] - full_Y_pre_sc[control_units, :], 2))
print(full_pre_mse) #186.969136939
print(full_post_mse) #142.422049057
X_Y_pre_names_arr[np.diag(full_fit.V)>0]

with open(pkl_file, "wb" ) as output_file:
    pickle.dump( (fast_fit, full_fit),  output_file)

with open(pkl_file, "rb" ) as input_file:
    fast_fit, full_fit = pickle.dump(input_file)
