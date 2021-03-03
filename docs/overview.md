# Overview

SparseSC is a package that implements an ML-enhanced version of Synthetic Control Methodologies. Typically this is used to estimate causal effects from binary treatments on observational panel (longitudinal) data. The functions `fit()` and `fit_fast()` provide basic fitting of the model. If you are estimating treatment effects, fitting and diagnostic information can be done via `estimate_effects()`.

## Data
Though the fitting methods do not require such structure, the typical setup is where we have panel data of an outcome variable `$Y$` for `$T$` time periods for `$N$` observation units (customer, computers, etc.). We may additionally have some baseline characteristics `$X$` about the units. In the treatment effect setting, we will also have a discrete change in treatment status (e.g. some policy change) at time, `$T_0$`, for a select group of units. When there is treatment, we can think of the pre-treatment data as [`$X$`, `$Y_{pre}$`] and post-treatment data as [`$Y_{post}$`].

## Synthetic Controls

[Synthetic Controls](https://en.wikipedia.org/wiki/Synthetic_control_method) was originally designed as a treatment effect estimator for a small number of regions (e.g. states). It is a type of matching estimator. For each unit it will find a weighted average of untreated units that is similar on key pre-treatment data (`$X$`, `$Y_{pre}$`). The goal of Synthetic controls is find out which variables are important to match on (specified by the diagonal matrix `V` where element `$v_{k,k}$` is the weight for the `$k$`-th pre-treatment variable) and then given those a vector of per-unit weights that combine the control units into its synthetic control. The synthetic control acts as the counterfactual for a unit, and the estimate of a treatment effect is the difference between the observed outcome in the post-treatment period and the synthetic control's outcome.

One way to think of SC is as an improvement upon difference-in-difference (DiD) estimation. Typical DiD will compare a treated unit to the average of the control units. But often the treated unit does not look like a  typical control (e.g. it might have a different growth rate), in which case the 'parallel trend' assumption of DiD is not valid. SC remedies this by choosing a smarter linear combination, rather than the simple average, to weight more heavily the more similar units.

The authors show if endogeneity of treatment is driven by a factor model with vectors components `$f_t\cdot\lambda_i$` where `$\lambda_i$` might be correlated with treatment (a simple example would be that there are groups with different typical time trends) and the synthetic control is able to reproduce the treated unit's pre-treatment history, then as the pre-history grows the size of the expected bias of the estimated treatment effect approaches zero (NB: this is not quite consistency). Essentially, if there are endogenous factors that affect treatment and future outcomes then you should be able to control for them by matching on past outcomes. The matching that SC provides can therefore deal with some problems in estimation that DiD can not handle.

### SparseSC Solution Structure
Given a specific set of variable weights, `$V$` and control variable matrix of data to match on `$M^C$`, unit-weights for unit `$i$` is `$W_i=\arg\min_{W}\sum_k(M_{ik}-W\cdot M^C_{\cdot,k})^2\cdot v_{kk}$`. Synthetic Controls typically also restricts the weight vector to be non-negative and sum to one. These restrictions may aid interpretability, though they are not econometrically necessary and may harm performance (e.g. they make it difficult to model units on the convex hull of the matching-space).

The determination of `$V$` can be done jointly or separately.
* Jointly: We find `$V$` such that resulting SCs have outcomes that are 'optimal' in some sense. They originally minimized squared prediction error on `$Y_{pre}$`. In the standard Stata `synth` command this is the `nested` option.
* Separately: We perform some non-matching analysis to determine `$V$`. They originally found those variables that were the best linear predictors of `$Y_{pre}$`. In the standard Stata `synth` command this is the default option.

The separate solution is much faster and often the only feasible method with larger data.

### Inference
Inference is typically done by placebo tests (randomization inference). The procedure is replicated for the control units and we get a distribution of placebo effects from the differences for the control units. We then compare main effect to this placebo distribution to evaluate statistical significance.

### Multiple treated units
If there are multiple (`N1`) treated units then the average treatment effect averages over these individual effects. The elements of the placebo distribution are each then averages over `N1` differences for untreated units. If the full placebo distribution is too large to compute then it can be randomly sampled to the desired precision.

If units experience treatment at different times, then we define consistent pre and post period (`T0`, `T1`) and compute difference for each event, then align these according to the standard windows and compute effects.

## SparseSC
SparseSC makes a number of changes to Synthetic Controls. To avoid overfitting, ensure uniqueness of the solution, and allow for estimation on large datasets we:
1. Automatically find a low-dimensional space to match in. We explicitly target a 'sparse' space so as to limit non-parametric bias that is typical in matching estimators. Using ML to find this space also removes ad-hoc decisions by analysts that can affect results. We provide two strategies: 
   * Feature selection - We impose an `$L_1$` penalization on the existing [`$X$`, `$Y_{pre}$`] variables.
   * Feature learning/generation - We generate a low dimensional space `$M$` to match on using time-series neural-networks (LSTM). This exploits the time-series nature of the outcome.
2. Impose an `$L_2$` penalization on weights as they deviate from `$1/N_{controls}$`. That is, in the absence of signal, we should resort to a simple comparison to the control group. This also implicitly penalizes weights whose sums is different than one.

Additional changes include:
1. We fit `$V$` once for the whole sample rather than for each unit. This reduces overfitting and speeds execution.
2. For optimizing `V` we target squared prediction error on `$Y_{post}$` for the control units, thus clearly separating features from targets. In the original formulation `$Y_{pre}$` were often features used to predict `$Y_{pre}$`. This caused problems as `$X$`'s could not be included with all `$Y_{pre}$`. Analysts were thus forced to hand-choose their matching varibles, which could introduce bias and lack of robustness.

### SparseSC Solution Structure
As with the original SC, there are choices about whether to calculate all parameters on the main matching objective or whether to get approximate/fast estimates of them using non-matching formulations. To complicate matters, we now typically have to solve for two new penalty parameters `v_pen` and `w_pen` in addition to `V` and `W`. The methods are:
* Full joint (done by `fit()`): We optimize over `v_pen`, `w_pen` and `V`, so that the resulting SC for controls have smallest squared prediction error on `$Y_{post}$`.
* Separate (done by `fit_fast()`): We note that we can efficiently estimate `w_pen` on main matching objective, since, given `V`, we can reformulate the finding problem into a Ridge Regression and use efficient LOO cross-validation (e.g. `RidgeCV`) to estimate `w_pen`. We will estimate `V` using an alternative, non-matching objective (such as a `MultiTaskLasso` of using `$X,Y_{pre}$` to predict `$Y_{post}$`). This setup also allows for feature generation to select the match space. There are two variants depending on how we handle `v_pen`:
  * Mixed. Choose `v_pen` based on the resulting down-stream main matching objective.
  * Full separate: Choose `v_pen` base on approximate objective (e.g., `MultiTaskLassoCV`).

The Fully Separate solution is fast and often quite good so we recommend starting there, and if need be, advancing to the Mixed and then Fully Joint optimizations.

## Custom donor pools
There are two main reasons why one might want to limit the per-unit donor pools: 
* Interference/contamination (violations of [SUTVA](https://en.wikipedia.org/wiki/Rubin_causal_model#Stable_unit_treatment_value_assumption_(SUTVA)))
* limiting 'interpolation bias' (if there is a strong non-linearity in how features map to outcomes, then even if the synthetic control is similar in feature space, its outcome might not be). 

For all entry-points, one can specify the `custom_donor_pool` which is an `(N,N0)` boolean matrix of allowable donors for each unit.

Here is the default donor pool
```py
base_custom_donor_pool = (1-np.eye(N))[:,control_units].astype('bool')
```

### Interference Example 
Suppose the units you were dealing with are located in physical space and there is some concern that a treatment will spill over to adjacent units. Below is a way to construct the `custom_donor_pool` from an adjacency matrix `A`.
```py
custom_donor_pool = (1 - A)[:,control_units].astype('bool')
custom_donor_pool = np.logical_and(custom_donor_pool, base_custom_donor_pool)
```

### Interpolation Bias Example
Suppose we want to use 'calipers' to ensure that units are only matched to those with a maximum difference of `thresh` on a key metric `$B$` (which is `$N$`-vector) 
```
custom_donor_pool = (np.abs(B[:,np.newaxis] - B) < thresh)[:,control_units].astype('bool')
custom_donor_pool = np.logical_and(custom_donor_pool, base_custom_donor_pool)
```

## Model types
There are two main model-types (corresponding to different cuts of the data) that can be used to estimate treatment effects.
1. Retrospective: The goal is to minimize squared prediction error of the control units on `$Y_{post}$` and the full-pre history of the outcome is used as features in fitting. This is the default and was used in the descriptive elements above.
2. Prospective: We make an artificial split in time before any treatment actually happens (`$Y_{pre}=[Y_{train},Y_{test}]$`). The goal is to minimize squared prediction error of all units on `$Y_{test}$` and `$Y_{train}$` for all units is used as features in fitting.

Given the same amount of features, the two will only differ when there are a non-trivial number of treated units. In this case the prospective model may provide lower prediction error for the treated units, though at the cost of less pre-history data used for fitting. When there are a trivial number of units, the retrospective design will be the most efficient.

See more details about these and two additional model types at [model-types](model-types).
