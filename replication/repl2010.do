cap log close _all
log using repl2010.log, replace name(repl2010)

use smoking.dta, replace
tsset state year
set seed 1337
loc vmatdir "vmats/"
loc dta_dir "dta_dir/"

cap program drop show_summ
program show_summ
	cap mat li e(failed_opt_targets)
	if _rc==0 {
		mat li e(failed_opt_targets)
	}
	di "Pre: Bias and MSE"
	ttest effect==0 if state!=3 & year<1989
	gen pre_mspe = pre_rmspe^2
	summ pre_mspe if (state!=3) 
	drop pre_mspe
	
	di "Pre: Bias and MSE (dropping NH outlier)"
	ttest effect==0 if state!=3 & state!=22 & year<1989
	gen pre_mspe = pre_rmspe^2
	summ pre_mspe if (state!=3 &  state!=22) 
	drop pre_mspe

	di "Post: Bias and MSE"
	ttest effect==0 if state!=3 & year>=1989
	gen post_mspe = post_rmspe^2
	summ post_mspe if state!=3
	drop post_mspe
	
	di "Post: Bias and MSE (dropping NH outlier)"
	ttest effect==0 if state!=3 &  state!=22 & year>=1989
	gen post_mspe = post_rmspe^2
	summ post_mspe if state!=3 &  state!=22
	drop post_mspe
end

cap program drop show_summ_xf
program show_summ_xf
	gen spe = effect^2
	di "Pre: Bias and MSE"
	ttest effect==0 if state!=3 & year<1989
	summ spe if year<1989 & state!=3 
	
	di "Pre: Bias and MSE (dropping NH outlier)"
	ttest effect==0 if state!=3 & state!=22 & year<1989
	summ spe if year<1989 & state!=3 & state!=22

	di "Post: Bias and MSE"
	ttest effect==0 if state!=3 & year>=1989
	summ spe if year>=1989 & state!=3
	
	di "Post: Bias and MSE (dropping NH outlier)"
	ttest effect==0 if state!=3 &  state!=22 & year>=1989
	summ spe if year>=1989 & state!=3 &  state!=22
	
	drop spe
end

cap prog drop run_synth_xf_Vs_file
program run_synth_xf_Vs_file
	syntax , fname(string)
	qui gen cigsale_synth = .
	file open myfile using "`fname'", read
	file read myfile line
	while r(eof)==0 {
		gettoken c_num V_names : line
		file read myfile line
		gettoken c_num V_values : line
		if `c_num'!=3 {
			//tempfile soutput //This was causing errors in my current setup for some reason
			local soutput "temp_gen.dta"
			preserve
			//di "Running for: `c_num'"
			qui drop if state==3
			qui synth cigsale `V_names', xperiod(1980(1)1988) trunit(`c_num') trperiod(1989) customV(`V_values') keep(`soutput') replace
			restore
			get_synth_cf , trunit(`c_num') fname(`soutput')
		}
		
		file read myfile line
	}
	file close myfile
	
	qui gen effect = cigsale - cigsale_synth
	//Don't make pre_rmspe post_rmspe lead (will get in summary)
end

//merge in the keep file from synth
cap prog drop get_synth_cf
prog get_synth_cf
	syntax, trunit(string) fname(string)
	
	preserve
	qui use `fname', clear
	qui drop if missing(_time)
	keep _time _Y_synthetic
	rename _time year
	gen state=`trunit'
	qui save `fname', replace
	restore
	
	qui merge 1:1 state year using `fname', keep(match master)
	qui replace cigsale_synth = _Y_synthetic if state==`trunit'
	drop _Y_synthetic _merge
end

//Get the V for a synth_runner execution from a file
cap program drop run_synth_V_file
program run_synth_V_file
	syntax, fname(string)

	file open myfile using "`fname'", read
	file read myfile V_names
	file read myfile V_values
	file close myfile

	synth_runner cigsale `V_names', trunit(3) trperiod(1989) customV(`V_values') gen_vars

end


//The results from nested most closely match the paper (and allopt doesn't change anything) so use nested
/*
synth cigsale beer(1984(1)1988) lnincome retprice age15to24 cigsale(1988) cigsale(1980) cigsale(1975), xperiod(1980(1)1988)  trunit(3) trperiod(1989) 
synth cigsale beer(1984(1)1988) lnincome retprice age15to24 cigsale(1988) cigsale(1980) cigsale(1975), xperiod(1980(1)1988)  trunit(3) trperiod(1989) nested
synth cigsale beer(1984(1)1988) lnincome retprice age15to24 cigsale(1988) cigsale(1980) cigsale(1975), xperiod(1980(1)1988)  trunit(3) trperiod(1989) nested allopt
*/


// ----------- Fast Version --------------//
/*
synth_runner cigsale beer(1984(1)1988) lnincome retprice age15to24 cigsale(1988) cigsale(1980) cigsale(1975), xperiod(1980(1)1988) trunit(3) trperiod(1989) gen_vars
show_summ
//174.8333

//152.2567
rename (pre_rmspe post_rmspe lead effect cigsale_synth) fast_=
*/

// ----------- V from SparseSC Fast --------------//
// Full-sample V
run_synth_V_file , fname("`vmatdir'fast_fit.txt")
show_summ
rename (pre_rmspe post_rmspe lead effect cigsale_synth) spfast_=
/*
loc sparseSC_fast_V = "0.51300713   22.96821089   66.11994103   20.37246761    0.39536433  146.42793943    0.4007635     1.09141769    1.6124491"
loc sparseSC_fast_preds = "beer(1986) lnincome(1985) lnincome(1987) lnincome(1988) retprice(1985) age15to24(1988) cigsale(1986) cigsale(1987) cigsale(1988)"
synth_runner cigsale `sparseSC_fast_preds', trunit(3) trperiod(1989) customV(`sparseSC_fast_V') gen_vars
show_summ
//17, 34

//251.1117

//217.9936
*/
// XF V
run_synth_xf_Vs_file , fname("`vmatdir'xf_fits_fast.txt")
show_summ_xf
rename (effect cigsale_synth) spfast_xf_=

// XF2 V
run_synth_xf_Vs_file , fname("`vmatdir'xf_fits_fast2.txt")
show_summ_xf
rename (effect cigsale_synth) spfast_xf2_=


// ----------- Flat V (I_55) V (all potential variables) --------------//
loc sparseSC_full_flatV = "1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"
loc sparseSC_full_flatpreds = "beer lnincome retprice age15to24 beer(1984) beer(1985) beer(1986) beer(1987) beer(1988) lnincome(1980) lnincome(1981) lnincome(1982) lnincome(1983) lnincome(1984) lnincome(1985) lnincome(1986) lnincome(1987) lnincome(1988) retprice(1980) retprice(1981) retprice(1982) retprice(1983) retprice(1984) retprice(1985) retprice(1986) retprice(1987) retprice(1988) age15to24(1980) age15to24(1981) age15to24(1982) age15to24(1983) age15to24(1984) age15to24(1985) age15to24(1986) age15to24(1987) age15to24(1988) cigsale(1970) cigsale(1971) cigsale(1972) cigsale(1973) cigsale(1974) cigsale(1975) cigsale(1976) cigsale(1977) cigsale(1978) cigsale(1979) cigsale(1980) cigsale(1981) cigsale(1982) cigsale(1983) cigsale(1984) cigsale(1985) cigsale(1986) cigsale(1987) cigsale(1988)"

synth_runner cigsale `sparseSC_full_flatpreds', trunit(3) trperiod(1989) customV(`sparseSC_full_flatV') gen_vars
show_summ
//No failed targets. mat li e(failed_opt_targets) //

//mean=-.3902378, t-stat:-0.6929  (dof=721)
//228.5252

//mean=.3554428, t-stat:0.4614 (dof=455)
//270.1838

rename (pre_rmspe post_rmspe lead effect cigsale_synth) flat55_=


// ----------- Flat (I_7) V (all hand-picked variables)  --------------//
/*
synth_runner cigsale beer(1984(1)1988) lnincome retprice age15to24 cigsale(1988) cigsale(1980) cigsale(1975), xperiod(1980(1)1988) trunit(3) trperiod(1989) gen_vars customV(1 1 1 1 1 1 1)
show_summ
// no failed targets. mat li e(failed_opt_targets)

//mean=-.7808893, t-stat: -1.3722 (dof=721)
//234.2147


//mean=.7111448, t-stat: 0.9186 (dof=455)
//273.183 

rename (pre_rmspe post_rmspe lead effect cigsale_synth) flat7_=
*/

// ----------- V from SparseSC Full --------------//
// Full-sample V
run_synth_V_file , fname("`vmatdir'full_fit.txt")
*loc sparseSC_full_V = "5.62581428e-06   1.67298271e-04   1.25447705e-04   2.83218864e-05 1.20684653e-04   2.90538206e-05   1.68471635e-04   1.91157505e-04 1.95644283e-04   2.87258604e-04   4.25739528e-04   4.62630652e-04 6.40482407e-04"
*loc sparseSC_full_preds = "cigsale(1976) cigsale(1977) cigsale(1978) cigsale(1979) cigsale(1980) cigsale(1981) cigsale(1982) cigsale(1983) cigsale(1984) cigsale(1985) cigsale(1986) cigsale(1987) cigsale(1988)"
*synth_runner cigsale `sparseSC_full_preds', trunit(3) trperiod(1989) customV(`sparseSC_full_V') gen_vars
show_summ
//22, 34

//mean=-.1812726, t-stat:-0.9543  (dof=683)
//24.69864

//mean=.3479216, t-stat:0.5997  (dof=431)
//145.1905

rename (pre_rmspe post_rmspe lead effect cigsale_synth) spfull_=

// XF V
run_synth_xf_Vs_file , fname("`vmatdir'xf_fits_full.txt")
show_summ_xf
rename (effect cigsale_synth) spfull_xf_=

// XF2 V
run_synth_xf_Vs_file , fname("`vmatdir'xf_fits_full2.txt")
show_summ_xf
rename (effect cigsale_synth) spfull_xf2_=

// ----------- Nested Version --------------//
synth_runner cigsale beer(1984(1)1988) lnincome retprice age15to24 cigsale(1988) cigsale(1980) cigsale(1975), xperiod(1980(1)1988) trunit(3) trperiod(1989) nested gen_vars
show_summ
//34
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
//mean=1.076577, t-stat: 2.6233 (dof=702)
//119.3197

//mean=.5225635, t-stat: 0.8277 (dof=443)
//176.859

rename (pre_rmspe post_rmspe lead effect cigsale_synth) nested_=

// ----------- End --------------//

//save out
keep state year cigsale *effect
save "`dta_dir'smoking_statafits.dta", replace

loc pe_varnames "nested flat55 spfast spfast_xf spfast_xf2 spfull spfull_xf spfull_xf2"
loc pe_titles `""Standard SC" "Standard SC (v=I_55)" "Standard SC (v=SparseSC Fast)" "Standard SC (v=SparseSC Fast XF)" "Standard SC (v=SparseSC Fast XF RE)" "Standard SC (v=SparseSC)" "Standard SC (v=SparseSC XF)" "Standard SC (v=SparseSC XF RE)""'

//Show the max by year
preserve
drop if state==3
foreach tname of local pe_varnames {
	qui replace `tname'_effect = abs(`tname'_effect)
}
collapse (max) *_effect, by(year)

summ *_effect if year<1989
summ *_effect if year>=1989
restore

erase temp_gen.dta
log close repl2010

// --------- Make some graphs ------------//
// Standard SC
/*
use "`dta_dir'smoking_statafits.dta", clear
*/
loc fig_dir "figs/"
loc y_range "r(-70, 155)"
*loc y_range "r(-200, 200)" //if you want include the 'fast' options
loc pe_n : word count `pe_varnames'
forvalues i=1/`pe_n' {
	loc pe_varname : word `i' of `pe_varnames'
	loc pe_title : word `i' of `pe_titles'
	xtline `pe_varname'_effect if state!=3, overlay xline(1989) legend(off) name("`pe_varname'", replace) title("Prediction errors: `pe_title'") ytitle("Cigarette sales (packs)/capita") xtitle("Year") caption("Control units only") yscale(`y_range')
	graph export "`fig_dir'standard_`pe_varname'_pe.pdf", as(pdf) replace
}

// SparseSC
local sparse_types "sparsesc_fast sparsesc_fast_xf sparsesc_fast_xf2 sparsesc_full sparsesc_full_xf sparsesc_full_xf2"
local sparse_names `""SparseSC Fast" "SparseSC Fast XF" "SparseSC Fast XF RE" "SparseSC Full" "SparseSC Full XF" "SparseSC Full XF RE""'
loc sparse_pe_n : word count `sparse_types'
forvalues i=1/`sparse_pe_n' {
	loc tname: word `i' of `sparse_types'
	loc pe_title : word `i' of `sparse_names'
	use "`dta_dir'smoking_`tname'.dta", clear
	xtset state year
	xtline cigsale if state!=3, overlay xline(1989) legend(off) name("`tname'", replace) title("Prediction errors: `pe_title'") ytitle("Cigarette sales (packs)/capita") xtitle("Year") caption("Control units only") yscale(`y_range')
	graph export "`fig_dir'`tname'_pe.pdf", as(pdf) replace
}

