# [Sparse Synthetic Controls](https://sparsesc.readthedocs.io/en/latest/)

### TL;DR:

SparseSC is a package that implements an ML-enhanced version of Synthetic Control Methodologies. Typically this is used to estimate causal effects from binary treatments on observational panel (longitudinal) data. The functions `fit()` and `fit_fast()` provide basic fitting of the model. If you are estimating treatment effects, fitting and diagnostic information can be done via `estimate_effects()`.

Though the fitting methods do not require such structure, the typical setup is where we have panel data of an outcome variable `Y` for `T` time periods for `N` observation units (customer, computers, etc.). We may additionally have some baseline characteristics `X` about the units. In the treatment effect setting, we will also have a discrete change in treatment status (e.g. some policy change) at time, `T0`, for a select group of units. When there is treatment, we can think of the pre-treatment data as [`X`, `Y_pre`] and post-treatment data as [`Y_post`].

```py
import SparseSC

# Fit the model:
treatment_unit_size = np.full((N), np.NaN)
treatment_unit_size[treated_unit_idx] = T0
fitted_estimates = SparseSC.estimate_effects(Y,unit_treatment_periods,...)

# Print summary of the model including effect size estimates, 
# p-values, and confidendence intervals:
print(fitted_estimates)

# Extract model attributes:
fitted_estimates.pl_res_post.avg_joint_effect.p_value
fitted_estimates.pl_res_post.avg_joint_effect.CI

# access the fitted Synthetic Controls model:
fitted_model = fitted_estimates.fit
```

## Overview 

See [here](https://en.wikipedia.org/wiki/Synthetic_control_method) for more info on Synthetic Controls. In essence, it is a type of matching estimator. For each unit it will find a weighted average of untreated units that is similar on key pre-treatment data. The goal of Synthetic controls is find out which variables are important to match on (the `V` matrix) and then, given those, to find a vector of per-unit weights that combine the control units into its synthtic control. The synthetic control acts as the counterfactual for a unit, and the estimate of a treatment effect is the difference between the observed outcome in the post-treatment period and the synthetic control's outcome.

SparseSC makes a number of changes to Synthetic Controls. It uses regularization and feature learning to avoid overfitting, ensure uniqueness of the solution, automate researcher decisions, and allow for estimation on large datasets. See the docs for more details.

The main choices to make are:
1. The solution structure
2. The model-type

### SparseSC Solution Structure
The first choice is whether to calculate all of the high-level parameters (`V`, its  regularization parameter, and the regularization parameters for the weights) on the main matching objective or whether to get approximate/fast estimates of them using non-matching formulations. The options are:
* Full joint (done by `fit()`): We optimize over `v_pen`, `w_pen` and `V`, so that the resulting SC for controls have smallest squared prediction error on `Y_post`.
* Separate (done by `fit_fast()`): We note that we can efficiently estimate `w_pen` on main matching objective, since, given `V`, we can reformulate the finding problem into a Ridge Regression and use efficient LOO cross-validation (e.g. `RidgeCV`) to estimate `w_pen`. We will estimate `V` using an alternative, non-matching objective (such as a `MultiTaskLasso` of using `X,Y_pre` to predict `Y_post`). This setup also allows for feature generation to select the match space. There are two variants depending on how we handle `v_pen`:
  * Mixed. Choose `v_pen` based on the resulting down-stream main matching objective.
  * Full separate: Choose `v_pen` base on approximate objective (e.g., `MultiTaskLassoCV`).

The Fully Separate solution is fast and often quite good so we recommend starting there, and if need be, advancing to the Mixed and then Fully Joint optimizations.

### Model types
There are two main model-types (corresponding to different cuts of the data) that can be used to estimate treatment effects.
1. Retrospective: The goal is to minimize squared prediction error of the control units on `Y_post` and the full-pre history of the outcome is used as features in fitting. This is the default and was used in the descriptive elements above.
2. Prospective: We make an artificial split in time before any treatment actually happens (`Y_pre=[Y_train,Y_test]`). The goal is to minimize squared prediction error of all units on `Y_test` and `Y_train` for all units is used as features in fitting.

Given the same amount of features, the two will only differ when there are a non-trivial number of treated units. In this case the prospective model may provide lower prediction error for the treated units, though at the cost of less pre-history data used for fitting. When there are a trivial number of units, the retrospective design will be the most efficient.

See more details about these and two additional model types (Prospective-restrictive, and full) at the docs.

## Fitting a synthetic control model

### Documentation

You can read these online at [Read the
Docs](https://sparsesc.readthedocs.io/en/latest/). See there for:
* Custom Donor Pools
Parallelization
* Constraining the `V` matrix to be in the unit simplex
* Performance Notes for `fit()`
* Additional Performance Considerations for `fit()`
* Full parameter listings

To build the documentation see `docs/dev_notes.md`.

## Citation
Brian Quistorff, Matt Goldman, and Jason Thorpe (2020) [Sparse Synthetic Controls: Unit-Level Counterfactuals from High-Dimensional Data](https://drive.google.com/file/d/1lfH1CK_JZpc0ou7hP60FhQpkeoXhR6fC/view?usp=sharing), Microsoft Journal of Applied Research, 14, pp.155-170.

## Installation
Currently you need to download the repo and build and install the package locally. The commands for building the package are in the makefile.

## Contributing

This project welcomes contributions and suggestions.  Most contributions
require you to agree to a Contributor License Agreement (CLA) declaring
that you have the right to, and actually do, grant us the rights to use
your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine
whether you need to provide a CLA and decorate the PR appropriately (e.g.,
label, comment). Simply follow the instructions provided by the bot. You
will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of
Conduct](https://opensource.microsoft.com/codeofconduct/).  For more
information see the [Code of Conduct
FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact
[opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional
questions or comments.
