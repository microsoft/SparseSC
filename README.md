# Sparse Synthetic Controls

### TL;DR:

The `fit()` function can be used to create a set of weights and returns a
fitted model which can be used to create synthetic units using it's
`.predict()` method:

```py
from SparseSC import fit

# fit the model:
fitted_model = fit(X,Y,...)

# make for the in-sample data
in_sample_predictions = fitted_model.predict()

# make predictions for a held out set of fetures (Y_hat) within the
# original set of units
additional_predictions = fitted_model.predict(Y_additional)
```

## Overview 

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

## Fitting a synthetic control model

### Data and Model Type

The parameters `X` and `Y` should be numeric matrices containing data on
the features and target variables, respectively, with one row per unit
of observation, and one column per feature or target variable.

There area 4 model types that can be fit using the `fit()` function which
can be selected by passing one of the following values to the `model_type` parameter: 

* `"retrospective"`: In this model, data are assumed to be collected
	retrospectively, sometime after an intervention or event has taken
	place in a subset of the subjects/units, typically with the intent of
	estimating the effect of the intervention. 
	
	In this model, `Y` should contain target variables recorded after the
	event of interest and `X` may contain a combination of target variables
	recorded prior to the event of interest and other predictors /
	covariates known prior to the event. In addition, the rows in `X` and
	`Y` which contain units that were affected by the intervention
	("treated units") should be indicated using the `treated_units`
	parameter.

* `"prospective"`: In a prospective analysis, a subset of units have been designated to
	receive a treatment but the treatment has not yet occurred and the
	designation of the treatment may be correlated with a (possibly unobserved)
	feature of the treatment units. In this scenario, all data are
	collected prior to the treatment intervention, and data on the outcome
	of interested are divided in two, typically divided in two subsets
	taken before and after a particular point in time. 
	
	In this model, `Y` should contain only target variables and `X` may
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
	a particular point in time, and `Y` should contain only target
	variables and `X` may contain a combination of target variables and
	other predictors / covariates. The parameter `treated_units` is unused.

More details on the above parameters can be found in file `fit.md` in the
root of this git repository.

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

## Advanced Topics

### Custom Donor Pools

By default all control units are allowed to be donors for all other units.
There are cases where this is not desired and so the user can pass in a
matrix specifying a unit-specific donor pool via a N x C matrix of booleans.

### Constraining the V matrix

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

### Fold Parameters

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

### Parallelization

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

### Gradient Descent in feature space

Currently a custom gradient descent method called `cdl_search` (imported
from `SparseSC.optimizers.cd_line_search import`. ) is used which which
performs the constrained gradient descent. An alternate gradient descent
function may be supplied to the `method` parameter, and any additional
keyword arguments passed to `fit()` are passed along to whichever gradient
descent function is used. (see the [Big list of
parameters](#big-list-of-parameters) for details)


# Big list of parameters


* `X` *(Matrix of flaots)*: Matrix of features variables with one row per unit of observation and
	one column per covariate / feature.

* `Y` *(Matrix of flaots)*: Matrix of targets variables with one row per unit of observation and
	one column per target variable.

* `model_type` *(string, default = `"retrospective"`)*: Type of model
	being fit. One of `"retrospective"`, `"prospective"`,
	`"prospective-restricted"` or `"full"` See [above](#data-and-model-type)
	for details.

* `treated_units` *(int[]|boolean[])*: A list of integers or array of
	booleans indicating the rows of `X` and `Y` which contain data from
	treated units.

* `w_pen` *(float | float[], optional)*: Penalty / penalties applied to the
	difference between the fitted weights (`W`) and the null weights (1/n),
	See [above](#penalty-parameters) for details.

* `v_pen` *(float | float[], optional)*: Penalty / penalties applied to the
	difference between the fitted weights (`W`) and the null weights (1/n).
	See [above](#penalty-parameters) for details.

* `grid`: (float[], optional). See [above](#3-modifying-the-default-search) for details.

* `grid_min` *(float, default = 1e-6)*: Lower bound for `grid` when
	`grid` are not provided. Must be in the range `(0,1)`

* `grid_max` *(float, default = 1)*: Upper bound for `grid` when
	`v_pen` and `grid` are not provided. Must be in the range `(0,1]`

* `grid_length` *(int, default = 20)*: number of points in the `grid`
	parameter when `v_pen` and `grid` are not provided

* `stopping_rule` *(int|float|function, optional)*: A stopping rule less
	than one is interpreted as the percent improvement in the out-of-sample
	squared prediction error required between the current and previous
	iteration in order to continue with the coordinate descent. A stopping
	rule of one or greater is interpreted as the number of iterations of
	the coordinate descent (rounded down to the nearest Int).
	Alternatively, `stopping_rule` may be a function which will be passed
	the current model fit, the previous model fit, and the iteration number
	(depending on it's signature), and should return a truthy value if the
	coordinate descent should stop and a falsey value if the coordinate
	descent should continue.

* `choice` *(string|function, default =`"min"`)*: Method for selecting the 
	optimal penalty parameter from an array of penalty parameters, from the
	out-of-sample error estimates and standard errors of the estimates.
	When either `v_pen` or `w_pen` are passed a vector of values, `fit()`
	will iterate over the vector of values and return the model with an
	optimal out of sample prediction error using cross validation. The
	choice of model can be controlled with the `choice` parameter which has
	the options of `"min"` (default) which selects the model with the
	smallest out of sample error, `"1se"` which implements the 'one
	standard-error' rule, or a function which implements a custom
	selection rule

* `cv_folds` *(int[]|int[][], default = 10)*: An integer number of Cross 
	Validation folds passed to `sklearn.model_selection.KFold`, or an
	explicit list of train validation folds 

* `gradient_folds` *(int[]|int[][], default = 10)*: An integer
	number of Gradient folds passed to `sklearn.model_selection.KFold`, or
	an explicit list of train validation folds. Not used when `model_type`
	is `"prospective-restricted"`

* `cv_seed` *(int, default = 10101)*: passed to `sklearn.model_selection.KFold`
	to allow for consistent cross validation folds across calls to `fit()`

* `gradient_seed` *(int, default = 110011)*: passed to `sklearn.model_selection.KFold`
	to allow for consistent gradient folds across calls to `fit()`

* `progress` *(boolean, default = `True`)*: Controls the level of
	verbosity. If `True`, the messages indication the progress are printed
	to the console at each iteration of the gradient descent in the feature
	space (stdout).

* `verbose` *(boolean, default = `False`)*: Controls the level of
	verbosity. If `True`, the messages indication the progress are printed
	to the console at each calculation of the partial gradient (stdout).
	partial gradients are  calculated `h * c` times in the leave-one-out
	gradient descent, and `h * k` times in the k-fold gradient descent,
	where `h` is the number of cross-validation folds , `c` is the number
	of controls, and `k` is the number of gradient folds. In short, this is
	level of messaging is typically excessive.

* `custom_donor_pool` *(boolean matrix, default = `None`)*: By default all 
	control units are allowed to be donors for all units. There are cases
	where this is not desired and so the user can pass in a matrix
	specifying a unit-specific donor pool (NxC matrix of booleans).

	Common reasons for restricting the allowability: (a) When we would like
	to reduce interpolation bias by restricting the donor pool to those
	units similar along certain features. (b) If units are not completely
	independent (for example there may be contamination between neighboring
	units). This is a violation of the Single Unit Treatment Value
	Assumption (SUTVA). Note: These are not used in the fitting stage (of
	V and penalties) just in final unit weight determination.

* `parallel` *(boolean, default=`false`)*: split the gradient descent
	across multiple sub-processes.  This is currently an experimental stub
	and tends to increase running time. See notes above.

* `method` *(string|function, default=`SparseSC.optimizers.cd_line_search.cdl_search`)*: 
	The method or function responsible for performing gradient descent in
	the feature space. If `method` is a string, it is passed as the
	`method` argument to `scipy.optimize.minimize`. Otherwise, `method`
	must be a function with a signature compatible with
	`scipy.optimize.minimize` (`method(fun,x0,grad,**kwargs)`) which
	returns an object having `x` and `fun` attributes.

* `kwargs`: Additional arguments passed to the optimizer (i.e. `method` or `scipy.optimize.minimize`). Additional arguments for the
	default optimizer include:
	
	* `constrain` *(string)*: The value `"orthant"` constrains `V`
	to the non-negative orthant, and `"simplex"` constrains V to the
	standard simplex.
	
	* `learning_rate` *(float, default = 0.2)*: The initial learning rate
	which determines the initial step size, which is set to `learning_rate * null_model_error / gradient`. Must be between 0 and 1.
	
	* `learning_rate_adjustment` *(float, default = 0.9)*: Adjustment factor
	applied to the learning rate applied between iterations when the
	optimal step size returned by `scipy.optimize.line_search` is greater
	less than 1, else the step size is adjusted by
	`1/learning_rate_adjustment`. Must be between 0 and 1,
	
	* `tol` *(float, default = 0.0001)*: Tolerance used for the stopping rule 
	based on the proportion of the in-sample residual error reduced in the
	last step of the gradient descent.

# The Model Object:

`fit()` returns an object of type `SparseSCFit` which contains the details
of the fitted model.

### Attributes:

##### Input Parameters:

* `X`: A reference to the input paremeter `X`
* `Y`: A reference to the input paremeter `Y`
* `control_units`: A reference to the input paremeter `control_units`
* `treated_units`: A reference to the input paremeter `treated_units`
* `model_type`: A reference to the input paremeter `model_type`
* `initial_w_pen`: A reference to the input paremeter `w_pen`
* `initial_v_pen`: A reference to the input paremeter `v_pen`

##### Fitted values:

* `fitted_w_pen`: The selected `w_pen` value.
* `fitted_v_pen`: The selected `v_pen` value.
* `V`: The fitted matrix of feature weights.
* `sc_weights`: The fitted synthetic control weights matrix `W`
* `score`: Squared out-of-sample error from cross validation of the
	for the model associated with the selected penalty parameters.
* `trivial_units`: An array of booleans indicating which (if any) units
	have zeros for all targets (outcomes) and all non-trivial features
	(features with a non-zero weight in the fitted V matrix). These are
	important as the penalties will tend to set their weights to `1/N` for
	all synthetics units for which they may be included. (*This is
	anticipated to be very rare in real life datasets*)

### Methods:

* model.**get_weights** *(include_trivial_donors=False)*: Returns the synthetic
	control weights, optionally setting the contributions of trivial units
	to the predicted values of non-trivial units to zero.

* model.**predict** *(Y=None,include_trivial_donors=False)*: Returns matrix
	of synthetic units, optionally applying the synthetic control weights
	to a new set of features `Y` (e.g. for prospective use-cases).

* model.**__str__**(): Brief summary of model fit.

* model.**summary**(): Provides a summary of the coordinate descent steps
	in the search for an optimal pair of penalty parameters.  Return a list
	with one pandas DataFrame (if installed) per direction of the
	coordinate descent.

# Developer Notes

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

### Additional Performance Considerations

If you have the BLAS/LAPACK libraries installed and available to Python,
you should not need to do any further optimization to ensure that maximum
number of processors are used during the execution of `CV_score()`.  If
not, you may wish to set the parameter `parallel=True` when you call
`CV_score()` which will split the work across N - 2 sub-processes where N
is the [number of cores in your
machine](https://docs.python.org/2/library/multiprocessing.html#miscellaneous).
(Note that setting `parallel=True` when the BLAS/LAPACK are available will
tend to increase running times.)

### Documentation

You can read these online at [Read the
Docs](https://sparsesc.readthedocs.io/en/latest/). 

To build the
documentation locally, you will need `sphinx`, `recommonmark`, and
`sphinx-markdown-tables` (to incorporate .md files)

The documentation can be built locally using the `(n)make` target
`htmldocs` and is generated in `docs/build/html/index.html`. 

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
