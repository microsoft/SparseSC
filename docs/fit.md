# Fitting Sparse Synthetic Controls

The `fit()` and `fit_fast()` functions can be used to create a set of weights and returns a
fitted model which can be used to create synthetic units using it's
`.predict()` method:

```py
from SparseSC import fit

# Fit the model:
fitted_model = fit(features,targets,...)

# Get the fitted synthetic controls for `targets`:
in_sample_predictions = fitted_model.predict()

# Make predictions for a held out set of fetures (targets_hat) 
# using the fitted synthetic controls model:
additional_predictions = fitted_model.predict(targets_additional)
```

Note that `targets` and `features` here are depend on the model type and are not the
typical analysts outcome and covariates.

The two methods differ in terms of there  choices about whether to calculate all parameters on the main matching objective or whether to get approximate/fast estimates of them using non-matching formulations.
* Full joint (done by `fit()`): We optimize over `v_pen`, `w_pen` and `V`, so that the resulting SC for controls have smallest squared prediction error on `$Y_{post}$`.
* Separate (done by `fit_fast()`): We note that we can efficiently estimate `w_pen` on main matching objective, since, given `V`, we can reformulate the finding problem into a Ridge Regression and use efficient LOO cross-validation (e.g. `RidgeCV`) to estimate `w_pen`. We will estimate `V` using an alternative, non-matching objective (such as a `MultiTaskLasso` of using `$X,Y_{pre}$` to predict `$Y_{post}$`). This setup also allows for feature generation to select the match space. There are two variants depend on how we handle `v_pen`:
  * Mixed. Choose `v_pen` based on the resulting down-stream main matching objective.
  * Full separate: Choose `v_pen` base on approximate objective (e.g., `MultiTaskLassoCV`).
The Fully Separate solution is fast and often quite good so we recommend starting there, and if need be, advancing to the Mixed and then Fully Joint optimizations.

## Feature and Target Data

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

In the call to `fit()` and `fit_fast()`, parameters `features` and `targets` should be numeric matrices
containing data on the features and target variables, respectively, with
one row per unit of observation, and one column per feature or target
variable.

## Data and Model Type

There area 4 model types that can be fit using the fit functions which
can be selected by passing one of the following values to the `model_type` parameter: 

* `"retrospective"`: In this model, data are assumed to be collected
	retrospectively, sometime after an intervention or event has taken
	place in a subset of the subjects/units, typically with the intent of
	estimating the effect of the intervention. 
	
	In this model, `targets` should contain target variables recorded after the
	event of interest and `features` may contain a combination of target variables
	recorded prior to the event of interest and other predictors /
	covariates known prior to the event. In addition, the rows in `features` and
	`targets` which contain units that were affected by the intervention
	("treated units") should be indicated using the `treated_units`
	parameter.

* `"prospective"`: In a prospective analysis, a subset of units have been designated to
	receive a treatment but the treatment has not yet occurred and the
	designation of the treatment may be correlated with a (possibly unobserved)
	feature of the treatment units. In this scenario, all data are
	collected prior to the treatment intervention, and data on the outcome
	of interested are divided in two, typically divided in two subsets
	taken before and after a particular point in time. 
	
	In this model, `targets` should contain only target variables and `features` may
	contain a combination of target variables and other predictors /
	covariates. The parameters `treated_units` should be used to indicate
	the units which will or will not receive treatment.

* `"prospective-restricted"`: This is motivated by the same example as the 
	previous sample. It requires a larger set of treated units for similar
	levels of precision, with the benefit of substantially faster running
	time.

* `"full"`: This model is motivated by the need for prospective failure 
	detection, and is not used in the context of a historical event or
	treatment intervention. 

	like the `prospective` models, data on the outcome of interested are
	divided in two, typically divided in two subsets taken before and after
	a particular point in time, and `targets` should contain only target
	variables and `features` may contain a combination of target variables and
	other predictors / covariates. The parameter `treated_units` is unused.

A more through discussoin of the model types can be found
[Model Types](model-types) Page.


## Fit (Joint)

### Penalty Parameters

The fitted synthetic control weights depend on the penalties applied to the V and W
matrices (`v_pen` and `w_pen`, respectively), and the `fit()` function will
attempt to find an optimal pair of penalty parameters. Users can modify the selection
process or simply provide their own values for the penalty parameters, for
example to optimize these parameters on their own, with one of the
following methods:

#### 1. Passing `v_pen` and `w_pen` as floats:

When single values are passed in the to the `v_pen` and `w_pen`, a fitted
synthetic control model is returned using the provided penalties.

#### 2. Passing `v_pen` as a value and `w_pen` as a vector, or vice versa:

When either `v_pen` or `w_pen` are passed a vector of values, `fit()`
will iterate over the vector of values and return the model with an optimal
out of sample prediction error using cross validation. The choice of model
can be controlled with the `choice` parameter which has the options of
`"min"` (default) which selects the model with the smallest out of sample
error, `"1se"` which implements the 'one standard-error' rule, or a
function which implements a custom selection rule.

**Note that** passing vectors to both `v_pen` and `w_pen` is assumed to be
inefficient and `fit` will raise an error. If you wish to evaluate over a N x N
grid of penalties, use:

```py
from intertools import product
fitted_models = [ fit(..., v_pen=v, w_pen=w) for v,w in product(v_pen,w_pen)]
```

#### 3. Modifying the default search

By default `fit()` picks an arbitrary value for `w_pen` and creates a grid
of values for `v_pen` over which to search, picks the optimal for `v_pen`
from the set of parameters, and then repeats the process alternating
between a fixed `v_pen` and array of values `w_pen` and vice versa until
stopping rule is reached.

The grid over which each penalty parameter is searched is determined by the
value of the other (fixed) penalty parameter. For example, for a given
value of `w_pen` there is a maximum value of `v_pen` which does not result
in a null model (i.e. when the V matrix would be identically 0 and W would
be identically 1/N), and the same logic applies in both scenarios (i.e.
when `w_pen` is fixed).

The search grid is therefor bounded between 0 and the maximum referenced
above. By default the grid consists of 20 points log-linearly spaced
between 0 and the maximum. The number of points in the grid can be
controlled with the `grid_length` parameters, and the bounds are controlled
via the `grid_min` and `grid_max` parameters. Alternatively, an array of
values between 0 and 1 can be passed to the `grid` parameter and will be
multiplied by the relevant `grid_max` to determine the search grid at each
iteration of the alternating coordinate descent.

Finally, the parameter `stopping_rule` determines how long the coordinate
descent will alternate between searching over a grid of V and W penalties.
(see the [Big list of parameters](#big-list-of-parameters) for details)

### Advanced Topics

#### Constraining the V matrix

In the current implementation, the V matrix is a diagonal matrix, and the
individual elements of V are constrained to be positive, as negative values
would be interpreted as two units would considered to *more similar* when
their observed values for a particular feature are *more different*.

Additionally, the V matrix may be constrained to [the standard simplex](https://en.wikipedia.org/wiki/Simplex#The_standard_simplex).
which tends to minimize out of sample of error relative to the model
constrained to the [nonnegative
orthant](https://en.wikipedia.org/wiki/Orthant) in some cases. V is
constrained to the either the simplex or the nonnegative orthant by passing
either `"simplex"` or `"orthant"` to the `constrain` parameter.

Note that with both penalty parameters and V just constrained to the non-negative
orthant, then there is an extra degree of freedom. Typically then it is more
efficient to constrain V to the simplex. When solving using Azure Batch
then only a single penalty parameter is varied so then V should only be constrained
to the non-negative orthant.

#### Fold Parameters

The data are split into folds both purpose of calculating the cross fold
validation (out-of-sample) errors and for K-fold gradient descent, a
technique used to speed up the model fitting process. The parameters
`cv_fold` and `gradient_fold` can be passed either an integer number of
folds or an list-of-lists which indicate the units (rows) which are
allocated to each fold. 

In the case that an integer is passed, the scikit-learn function
[kfold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html)
is used internally to split the data into random folds. For consistency
across calls to fit, the `cv_seed` and `gradient_seed` parameters are
passed to `Kfold(..., random_state=seed)`.
### Performance Notes

The function `get_max_lambda()` requires a single calculation of the
gradient using all of the available data.  In contrast, ` SC.CV_score()`
performs gradient descent within each validation-fold of the data.
Furthermore, in the 'pre-only' scenario the gradient is calculated once for
each iteration of the gradient descent, whereas in the 'controls-only'
scenario the gradient is calculated once for each control unit.
Specifically, each control unit is excluded from the set of units that can
be used to predict it's own post-intervention outcomes, resulting in
leave-one-out gradient descent.

For large sample sizes in the 'controls-only' scenario, it may be
sufficient to divide the non-held out control units into "gradient folds", such
that controls within the same gradient-fold are not used to predict the
post-intervention outcomes of other control units in the same fold.  This
result's in K-fold gradient descent, which improves the speed of
calculating the overall gradient by a factor slightly greater than `c/k`
(where `c` is the number of control units) with an even greater reduction
in memory usage.

K-fold gradient descent is enabled by passing the parameter `grad_splits`
to `CV_score()`, and for consistency across calls to `CV_score()` it is
recommended to also pass a value to the parameter `random_state`, which is
used in selecting the gradient folds.

#### Parallelization

If you have the BLAS/LAPACK libraries installed and available to Python,
you should not need to do any further optimization to ensure that maximum
number of processors are used during the execution of `fit()`.  If
not, seting the parameter `paralell=True` when you call
`fit()` which will split the work across N - 2 sub-processes where N
is the [number of cores in your
machine](https://docs.python.org/2/library/multiprocessing.html#miscellaneous).

Note that setting `paralell=True` when the BLAS/LAPACK are available will
tend to increase running times. Also, this is considered an experimenatl
stub. While it works, parallel processing spends most of the time passing
repeatedly sending a relatively small amount of data, which could be (but
currently is not) initialized in each worker at the start. If this a
priority for your team, feel free to submit a PR or feature request.

#### Gradient Descent in feature space

Currently a custom gradient descent method called `cdl_search` (imported
from `SparseSC.optimizers.cd_line_search import`. ) is used which which
performs the constrained gradient descent. An alternate gradient descent
function may be supplied to the `method` parameter, and any additional
keyword arguments passed to `fit()` are passed along to whichever gradient
descent function is used. (see the [Big list of
parameters](#big-list-of-parameters) for details)


## Fit_fast (Separate)
The interface here is very similar except that `V` and `v_pen` are determined 
by modified problems. This is specified by the `match_space_maker` option and the main
choice is whether to use the fully separate solution method via the default `MTLassoCV_MatchSpace`
or to try the mixed method (slower, but better) via `MTLassoMixed_MatchSpace`

## Advanced Topics

#### Custom Donor Pools

By default all control units are allowed to be donors for all other units.
There are cases where this is not desired and so the user can pass in a
matrix specifying a unit-specific donor pool via a N x C matrix of booleans.
See [overview](overview) for an example.