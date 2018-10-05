# Sparse Synthetic Controls

### Setup

This package requires [numpy](http://www.numpy.org/), 
[scipy](https://www.scipy.org/), and [scikit-learn](http://scikit-learn.org/) and has been tested with ( Python 2.7.14,
Numpy 1.14.1, and Scipy 1.0.0 )  and ( Python 3.5.5, Numpy 1.13.1, and
Scipy 1.0.1 )

To build the documentation you will need Sphinx and recommonmark (to incorporate .md files)

# Overview 

When estimating synthetic controls, units of observation are divided into
control and treated units.   Data collected on these units may include
observations of the outcome of interest, as well as other characteristics
of the units (termed "covariates", herein).  Outcomes may be observed both
before and after an intervention on the treated units.

To maintain independence of the fitted synthetic controls and the
post-intervention outcomes of interest of treated units, the
post-intervention outcomes from treated units are not used in the fitting
process.  There are two cuts from the remaining data that may be used to
fit synthetic controls, and each has it's advantages and disadvantages.

### Fitting to control observations

The first cut of the data involves only the control units, but **includes
both pre and post intervention outcomes**.  In general, any available
covariates along with some or all of the pre-treatment outcomes are
included in with the predictors (`X`), and cross validation is conducted by
dividing the control units into folds, fitting the model on all but a
hold-out fold, and creating synthetic controls for the held-out fold using
the fitted model. 

This cut is called the "controls-only" cut, and to implement this scenario,
we can use `CV_score()` to calculate out-of-sample prediction errors by
passing matrices containing (1) the covariates and some or all of the
pre-intervention outcomes and (2) the post-intervention outcomes for the
control units. 

```python
CV_score(X = x_and_y_pre, # Covariates and pre-intervention outcomes from the control units
		 Y = y_post_control, # Post-intervention outcomes from the control units
		 ...)
```

Note that the observations from the treated units are not used to optimize 
the penalty parameters in this scenario.

This scenario has the advantage that if shocks to the system which affect a
subset of factor loadings occur only in the post-intervention period, the
prediction accuracy will be superior to that of the later "pre-only" model.
However, this model has the disadvantage that it is computationally slower,
owing to the fact that individual components of the gradients must be
calculated for each control unit during gradient descent.  This is
discussed more thoroughly in the Optimization section below. 

### Fitting to pre-treatment observations

The second cut of the data includes all the pre-intervention data
**including both treated and control units**. This cut is called the
"pre-only" cut, and in this scenario, cross validation is performed by
holding out a single fold from the treated units, fitting a set of
penalized models (e.g.  distance metrics ) using the controls and
non-held-out treated units, and then using the fitted metric to create a
set of synthetic controls for the held out units.

In this scenario, the matrices of predictors for the control units (`X`)
and the treated units (`x_treat`) may contain covariates and some
observations of the outcome variable, and the outcome matrices (`Y` and
`Y_treat`) contain the remaining pre-intervention outcomes for the control
and treated units. 

Note that care should be taken to maintain as much independence between
outcomes included in the matrix of predictors and the outcomes matrices.
As an extreme example, if the outcomes formed a highly correlated time
series and every other observation was included in the matrices of predictors
(`X` and `x_treat`), then the estimated out of sample error would be biased
and the model would likely underperform when applied out of sample.

To implement the "pre-only" scenario, we will have to pass 4 values to
the `CV_score()`, which calculates the out-of-sample prediction error for one
or more value of the penalty parameters:

```python
CV_score(X = x_control, # Covariates from the control units
		 Y = y_pre_control, # Pre-intervention outcomes from the control units
		 x_treat = x_treated, # Covariates from the treated units
		 y_treat = y_pre_treated, # Pre-intervention outcomes from the treated units
		 ...)
```

This scenario has the advantage of being the fastest to compute, and may
have superior prediction (for example in A/A tests) if the treated units
vary systemically from the control units.

### Penalty Parameters

This method of fitting a Synthetic Controls model requires 2 penalty
parameters: The parameter `LAMBDA`, which is the penalty on the model
complexity, and the penalty `L2_PEN_W`, which is a penalty on the magnitude
of the fitted weights, relative to a simple (un-weighted) average of the
control units.

Aside: `L2_PEN_W` is really a stupid name.

The function `CV_score()` is optimized to calculate out-of sample errors
(scores) for a single value of `L2_PEN_W` and an iterable of `LAMBDA`'s.
Empirically, the product of `LAMBDA` and `L2_PEN_W` is the most important
axis for optimization, and, for example, halving the value of `L2_PEN_W`
doubles the minimum value of `LAMBDA` that results in the null model (i.e a
simple weighted average of the control units). 

Hence, if `L2_PEN_W` is not provided as a parameter to either `CV_score()`
or `get_max_lambda()`, `L2_PEN_W` defaults to a convenient guestimate equal
to the mean of the variances of the predictors, which is implemented in the
function `L2_pen_guestimate()`.

For a fixed value of the `L2_PEN_W`, the search space for `LAMBDA` penalty
is then bounded below by zero and above by the minimum value of `LAMBDA`
that forces all of distance parameters to zero (i.e. the null model).  The
upper bound for `LAMBDA` can be calculated using the function `get_max_lambda()`.

When performing a grid search over the range of possible values for
`LAMBDA`, there are a variety of distributions over which to search, such as
linear spacing (e.g. `np.linspace(fmin,fmax,n_points)`, a log-linear
spacing (e.g.  `np.exp(np.linspace(np.log(fmin),np.log(fmax),n_points))`),
or something else entirely.  An additional consideration is that optimal
values of `LAMBDA` may be quite small relative to `LAMBDA_max`, and we have
found values of `fmax =LAMBDA_max * 1e-2` to be a practical starting point.

### Putting it All Together

Assuming you have prepared your data into Numpy matrices (`X` and `Y`, and
optionally `X_treat` and `Y_treat`) with one row per unit of observation
and one column per covariate or observed outcome, scores for a grid of
parameters `LAMBDA` can be obtained as follows: 

```python
import SparseSC as SC
import numpy as np

# Obtain LAMBDA_max and a grid over which to search
fmin = 1e-5
fmax = 1e-2
grid = np.exp(np.linspace(np.log(fmin),np.log(fmax),n_points))

LAMBDA_max = SC.get_max_lambda(
					X, Y, 

					# OPTIONAL, used in the 'pre-only' scenario
					X_treat=X_treat, Y_treat=Y_treat, 

					# OPTIONAL. Defaults to SC.L2_pen_guestimate(X)
					# or SC.L2_pen_guestimate(np.vstack((X,X_treat))) as
					# appropriate
					L2_PEN_W = my_favorite_l2_penalty)

# get the scores for each value of `LAMBDA`
scores = SC.CV_score(
	X = X,
	Y = Y,

	# OPTIONAL, used in the pre-only scenario
	X_treat = X_treat,
	Y_treat = Y_treat,

	# if LAMBDA is a single value, we get a single score, If it's an array
	# of values, we get an array of scores.
	LAMBDA = grid * LAMBDA_max,

	# OPTIONAL, but should be present if used in the call to get_max_lambda()
	L2_PEN_W = my_favorite_l2_penalty)

# select the value of LAMBDA with the best out-of sample prediction error:
best_LAMBDA = (grid * LAMBDA_max)[np.argmin(scores)]


# Extract the V matrix which corresponds to the optimal value of LAMBDA
V = SC.tensor(X = X,
			  Y = Y,

			  # Optional
			  X_treat = X_treat,
			  Y_treat = Y_treat,

			  LAMBDA = best_LAMBDA,

			  # Also optional
			  L2_PEN_W = my_favorite_l2_penalty)

# extract the matrix of weights
weights = SC.weights(X = X_control,
					 X_treat = X_treated, # Optional
					 V = V_ct,
					 L2_PEN_W = my_favorite_l2_penalty) # Also optional

# create the matrix of Synthetic Controls 
synthetic_conrols = weights.dot(Y)
```

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
not, you may wish to set the parameter `paralell=True` when you call
`CV_score()` which will split the work across N - 2 sub-processes where N
is the [number of cores in your
machine](https://docs.python.org/2/library/multiprocessing.html#miscellaneous).
(Note that setting `paralell=True` when the BLAS/LAPACK are available will
tend to increase running times.)

### Logging

You may (and probably should) enable logging of the progress of the
`CV_score()`.  Setting `progress=True` enables logging of each
iteration of the gradient descent, and setting `verbose=True` enables
logging or each calculation of the (partial) gradient, which is calculated
`h * c` times in the leave-one-out gradient descent, and `h * k` times in
the k-fold gradient descent,  where `h` is the number of cross-validation
folds , `c` is the number of controls, and `k` is the number of gradient
folds.

### Examples

The file `./example-code.py` in this package examples and a simple data
generating process for use in understanding how to use this package.

# Contributing

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
