use smoking.dta, replace
tsset state year
set seed 1337

//The results from nested most closely match the paper (and allopt doesn't change anything) so use nested
/*
synth cigsale beer(1984(1)1988) lnincome retprice age15to24 cigsale(1988) cigsale(1980) cigsale(1975), xperiod(1980(1)1988)  trunit(3) trperiod(1989) 
synth cigsale beer(1984(1)1988) lnincome retprice age15to24 cigsale(1988) cigsale(1980) cigsale(1975), xperiod(1980(1)1988)  trunit(3) trperiod(1989) nested
synth cigsale beer(1984(1)1988) lnincome retprice age15to24 cigsale(1988) cigsale(1980) cigsale(1975), xperiod(1980(1)1988)  trunit(3) trperiod(1989) nested allopt
*/

// ----------- Nested Version --------------//
synth_runner cigsale beer(1984(1)1988) lnincome retprice age15to24 cigsale(1988) cigsale(1980) cigsale(1975), xperiod(1980(1)1988) trunit(3) trperiod(1989) nested gen_vars
mat li e(failed_opt_targets) //34
/*


//Utah=34 has an error with nested
//preserve
drop if state==3
synth cigsale beer(1984(1)1988) lnincome retprice age15to24 cigsale(1988) cigsale(1980) cigsale(1975), xperiod(1980(1)1988)  trunit(34) trperiod(1989) nested
//restore

Nested optimization requested
Starting nested optimization module
could not calculate numerical derivatives
flat or discontinuous region encountered
r(430);
*/

ttest effect==0 if state!=3 & year<1989 //mean=1.076577, t-stat: 2.6233 (dof=702)
gen pre_mspe = n_pre_rmspe^2
summ pre_mspe if (state!=3) //119.3197
summ pre_mspe //116.261
drop pre_mspe

ttest effect==0 if state!=3 & year>=1989 //mean=.5225635, t-stat: 0.8277 (dof=443)
gen post_mspe = post_rmspe^2
summ post_mspe if state!=3 //176.859
summ post_mspe //
drop post_mspe

rename (pre_rmspe post_rmspe lead effect cigsale_synth) (n_pre_rmspe n_post_rmspe n_lead n_effect n_cigsale_synth)

// ----------- Nested Version --------------//
/*
synth_runner cigsale beer(1984(1)1988) lnincome retprice age15to24 cigsale(1988) cigsale(1980) cigsale(1975), xperiod(1980(1)1988) trunit(3) trperiod(1989) gen_vars

gen post_mspe = post_rmspe^2
summ post_mspe if state!=3 //174.8333
summ post_mspe //
drop post_mspe

gen pre_mspe = pre_rmspe^2
summ pre_mspe if state!=3 //152.2567
summ pre_mspe //148.4591
drop pre_mspe
rename (pre_rmspe post_rmspe lead effect cigsale_synth) (f_pre_rmspe f_post_rmspe f_lead f_effect f_cigsale_synth)
*/

// ----------- Nested Version  V from SparseSC Fast --------------//
/*
loc sparseSC_fast_V = "0.51300713   22.96821089   66.11994103   20.37246761    0.39536433  146.42793943    0.4007635     1.09141769    1.6124491"
loc sparseSC_fast_preds = "beer(1986) lnincome(1985) lnincome(1987) lnincome(1988) retprice(1985) age15to24(1988) cigsale(1986) cigsale(1987) cigsale(1988)"
synth_runner cigsale `sparseSC_fast_preds', trunit(3) trperiod(1989) customV(`sparseSC_fast_V') nested gen_vars
mat li e(failed_opt_targets) //17, 34

summ effect if (state!=3)
gen post_mspe = post_rmspe^2
summ post_mspe if (state!=3) //251.1117
summ post_mspe //255.972
drop post_mspe

gen pre_mspe = pre_rmspe^2
summ pre_mspe if (state!=3) //217.9936
summ pre_mspe //212.6061
drop pre_mspe

rename (pre_rmspe post_rmspe lead effect cigsale_synth) (nf_pre_rmspe nf_post_rmspe nf_lead nf_effect nf_cigsale_synth)
*/

// ----------- Nested Version  V from SparseSC Full --------------//
loc sparseSC_full_V = "5.62581428e-06   1.67298271e-04   1.25447705e-04   2.83218864e-05 1.20684653e-04   2.90538206e-05   1.68471635e-04   1.91157505e-04 1.95644283e-04   2.87258604e-04   4.25739528e-04   4.62630652e-04 6.40482407e-04"
loc sparseSC_full_preds = "cigsale(1976) cigsale(1977) cigsale(1978) cigsale(1979) cigsale(1980) cigsale(1981) cigsale(1982) cigsale(1983) cigsale(1984) cigsale(1985) cigsale(1986) cigsale(1987) cigsale(1988)"
synth_runner cigsale `sparseSC_full_preds', trunit(3) trperiod(1989) customV(`sparseSC_full_V') nested gen_vars
mat li e(failed_opt_targets) //22, 34

ttest effect==0 if state!=3 & year<1989 //mean=-.1812726, t-stat:-0.9543  (dof=683)
gen pre_mspe = pre_rmspe^2
summ pre_mspe if (state!=3) //24.69864
summ pre_mspe //24.18579
drop pre_mspe

ttest effect==0 if state!=3 & year>=1989 //mean=.3479216, t-stat:0.5997  (dof=431)
gen post_mspe = post_rmspe^2
summ post_mspe if (state!=3) //145.1905
summ post_mspe //153.6692
drop post_mspe

rename (pre_rmspe post_rmspe lead effect cigsale_synth) (nn_pre_rmspe nn_post_rmspe nn_lead nn_effect nn_cigsale_synth)
