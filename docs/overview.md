# Overview

SparseSC is a package that implements an ML-enhanced version of Synthetic Control Methodologies. Typically this is used to estimate causal effects from binary treatments on observational panel (longitudinal) data. The functions `fit()` and `fit_fast()` provide basic fitting of the model. If you are estimating treatment effects, fitting and diagnostic information can be done via `estimate_effects()`.

## Data
Suppose that we have panel data of an outcome variable `$Y$` for `$T$` time periods for `$N$` observation units (customer, computers, etc.). We may additionally have some baseline characteristics `$X$` about the units. In the treatment effect setting, we will also have a discrete change in treatment status (e.g. some policy change) at time, `$T0$`, for a select group of units. When there is treatment, we can think of the pre-treatment data as [`$X$`, `$Y_pre$`] and post-treatment data as [`$Y_post$`]. It should be

## Synthetic Controls

Synthetic Controls is matching estimator. For each unit it will find a weighted average of untreated units that is similar on key pre-treatment data (`$X$`, `$Y_pre$`). The goal of Synthetic controls is find out which variables are important to match on (let `V` be a diagonal matrix where element `$v_k$` is the weight for the `$k$`-th pre-treatment variable) and then given those a vector of per-unit weights that combine the control units into a synthtic control for said unit. The synthetic control acts as the counterfactual for a unit, and the estimate of a treatment effect is the difference between the observed outcome in the post-treatment period and the synthetic control outcome.

Specifically, W_i is set to minimize:


Given that SC needs to compute a diagonal matrix V of variable importance weights and per-unit W, these can be optimized jointly or separately.
* Jointly: We find V such that resulting SCs are optimal. In the original Stata synth command this is the 'nested' option.
* Separately: We perform some non-matching analysis to estimate V. In the original Stata synth command this is the default option.

One way to think of SC is as an improvement upon difference-in-difference (DiD) estimation. Typical DiD will compare a treated unit to the average of the control units. But often the treated unit does not look likea  typical control (e.g. it might have a different growth rate), in which case the 'parallel trend' assumption of DiD is not valid. SC remedies this by choosing a smarter linear combination, rather than the simple average, to weight more heavily similar units.

Inference is typically done by placebo tests (randomization inference). See details in the 

## SparseSC
SparseSC makes a number of changes to Synthetic Controls. To avoid overfitting, ensure uniqueness of the solution, and allow for estimation on large datasets we:
1. Automatically find a low-dimensional space to match in. This also removes ad-hoc decisions by analysist that can affect results. We have two strategies: 
1a. Feature selection - We impose an `$L_1$` penalization on the existing [`$X$`, `$Y_pre$`]
1b. Feature generation - We use generation a low dimensional space `$M$` to match on. This could be done with many methods but we provide one implementation using time-series neural-networks (LSTM).
2. Impose an `$L_2$` penalization on weights as they deviate from `$1/N_{controls}$`. That is, in the absernce of signal, we should resort to a simple comparison to the control group.

We also clearly separate features and targets.

We fit V once for the whole sample. This reduces overfitting and is faster.

## Custom donor pools
There are two main reasons why one might want to limit the per-unit donor pools: interference/contamination (violations of [SUTVA](https://en.wikipedia.org/wiki/Rubin_causal_model#Stable_unit_treatment_value_assumption_(SUTVA))) and limiting 'interpolation bias' (if there is a strong non-linearity in how features map to outcomes, then even if the synthetic control is similar in feature space, its outcome might not be). For all entry-points, one can specify the `custom_donor_pool` which is an `(N,N0)` boolean matrix of allowable donors for each unit.

Here is the default donor pool
```py
base_custom_donor_pool = np.ones((N,N0))
base_custom_donor_pool[control_units,:] = np.ones((N0,N0)) - np.eye(N0)
```

Interference: Suppose the units you were dealing with located in physical space and there is some concern that a treatment will spill over to adjacent units. Below is a way to construct the `custom_donor_pool` from an adjacency matrix `A`.
```py
custom_donor_pool = (1 - A[:,control_units]).astype('bool')
custom_donor_pool = np.logical_and(custom_donor_pool, base_custom_donor_pool)
#
```

Interpolation bias: Suppose we want to use 'calipers' to ensure that units are only matched to those with a maximum difference of `thresh` on a key metric `$B$` (which is `$N$`-vector) 
```
custom_donor_pool = (np.abs(B[:,np.newaxis] - B) < thresh)[:,control_units]
custom_donor_pool = np.logical_and(custom_donor_pool, base_custom_donor_pool)
```