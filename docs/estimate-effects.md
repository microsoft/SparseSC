# Treatment Effects 

The `estimate_effects()` function can be used to conduct
[DID](https://en.wikipedia.org/wiki/Difference_in_differences) style
analyses where counter-factual observations are constructed using Sparse
Synthetic Controls.  

```py
import SparseSC

# Fit the model:
fitted_estimates = SparseSC.estimate_effects(outcomes,unit_treatment_periods,covariates=X,fast=True,...)

# Print summary of the model including effect size estimates, 
# p-values, and confidendence intervals:
print(fitted_estimates)

# Extract model attributes:
fitted_estimates.pl_res_post.avg_joint_effect.p_value
fitted_estimates.pl_res_post.avg_joint_effect.CI

# access the fitted Synthetic Controls model:
fitted_model = fitted_estimates.fit
```

The returned object is of class `SparseSCEstResults`.

#### Feature and Target Data

When estimating synthetic controls, units of observation are divided into
control and treated units. Data collected on these units may include
observations of the outcome of interest, as well as other characteristics
of the units (termed "covariates", herein). Outcomes may be observed both
before and after an intervention on the treated units.

To maintain independence of the fitted synthetic controls and the
post-intervention outcomes of interest of treated units, the
post-intervention outcomes from treated units are not used in the fitting
process. There are two cuts from the remaining data that may be used to
fit synthetic controls, and each has it's advantages and disadvantages.

In the call to `estimate_effects()`, `outcomes` should
be numeric matrices containing data on the target variables collected prior
to (after) the treatment / intervention ( respectively), and the optional
parameter `covariates` may be a matrix of additional features.  All matrices
should have one row per unit and one column per observation. 

In addition, the rows in `covariates` and `outcomes` which contain units that were affected
by the intervention ("treated units") should be indicated using the
`treated_units` parameter, which may be a vector of booleans or integers
indicating the rows which belong to treat units.

#### Statistical parameters

The confidence level may be specified with the `level` parameter, and the
maximum number of simulations used to produce the placebo distribution may
be set with the `max_n_pl` parameter.

#### Additional parameters

Additional keyword arguments are passed on to the call to `fit()`, which is
responsible for fitting the Synthetic Controls used to create the
counterfactuals. 
