# Ridge Synthetic Controls

### Setup

This package requires [numpy](http://www.numpy.org/), 
[scipy](https://www.scipy.org/), and [scikit-learn](http://scikit-learn.org/) and has been tested with ( Python 2.7.14,
Numpy 1.14.1, and Scipy 1.0.0 )  and ( Python 3.5.5, Numpy 1.13.1, and
Scipy 1.0.1 )

# Overview 

When estimating synthetic controls, units of observation are divided into
control and treated units.   Data collected on these units may include
observations of the outcome of interest, as well as other characteristics
of the units (termed "covariates", herein).  Outcomes may be observed both
before and after an intervention on the treated units.

To maintain independence of the fitted synthetic controls and the outcomes
of interest within the treated units, post-intervention outcomes from
treated units are not used in the fitting process.  There are two cuts from
the remaining data that may be used to fit synthetic controls, and each has
it's advantages and disadvantages.

The first cut of the data includes all the pre-intervention data from the
treated and control units. This cut is called the "controls-only" cut, and
in this scenario, cross validation is performed by holding out a single
fold from the treated units and applying the fitted model to the held out
units. 

To implement this scenario, we will have to pass 4 values to
the `CV_score`, which calculates the out-of-sample prediction error for one
or more value of the penalty parameters:

```python
CV_score(X = x_control,           # Covariates from the control units
		 Y = y_pre_control,       # Pre-intervention outcomes from the control units
		 x_treat = x_treated,     # Covariates from the treated units
		 y_treat = y_pre_treated, # Pre-intervention outcomes from the treated units
		 ...)
```

This scenario has the advantage of being the fastest to compute, and may
have superior prediction (for example in A/A tests) if the treated units
vary systemically from the control units.

The second cut of the data involves only the control units, but includes
both pre and post intervention outcomes.  In general, some or all of the
pre-treatment outcomes are included in with the "covariates", and cross
validation is conducted by dividing the control units into folds, fitting
the model on all but a hold-out fold, and creating synthetic controls for
the held-out fold using the fitted model. 

To implement this scenario, we can use `CV_score` to calculate
out-of-sample prediction errors by passing matrices containing (1) the
covariates and some or all of the pre-intervention outcomes and (2) the
post-intervention outcomes for the control units. 

```python
CV_score(X = x_and_y_pre, # Covariates and pre-intervention outcomes from the control units
		 Y = y_post_control, # Post-intervention outcomes from the control units
		 ...)
```

Note that the parameters `x_treat` and `y_treat` are omitted, as treatment
observations are not used in this scenario.

This scenario has the advantage that if shocks to the system that affect a
subset of factor loadings only in the post-intervention period, the
prediction accuracy will be superior to that of the pre-only model.
However, this model has the disadvantage that it is computationally slower,
owing to the fact that partial gradients must be calculated for each
control unit during gradient descent.  This is discussed more thoroughly in
the Optimization section below. 

### Penalty Parameters

This method of fitting a Synthetic Controls model requires 2 penalty
parameters: The parameter `LAMBDA`, which is the penalty on the model
complexity, and the penalty `L2_PEN_W`, which is a penalty on the magnitude
of the fitted weights, relative to a simple (un-weighted) average of the
control units.

Aside: `L2_PEN_W` is really a stupid name.

The function `CV_score()` is optimized calculate out-of sample errors
(scores) for a single value of `L2_PEN_W` and an iterable of `LAMBDA`'s.
Empirically, the product of `LAMBDA` to `L2_PEN_W` is the most important
axis for optimization, and, for example, halving the value of `L2_PEN_W`
doubles the minimum value of `LAMBDA` which results in the null model. 

Hence, if `L2_PEN_W` is not provided, to either `CV_score`,
`get_max_lambda()`, it defaults to a convenient guestimate equal to the
mean of the variances of the predictors, which is implemented in the
function `L2_pen_guestimate()`.

For a fixed value of the `L2_PEN_W`, the search space for `LAMBDA` penalty
is then bounded below by zero and above by the minimum value of `LAMBDA`
which results in the null model, which can be calculated using the function
`get_max_lambda()`.

When performing a grid search over the range of possible values for
`LAMBDA`, the optimal distribution over which to search such as linear
spacing (e.g. `np.linspace(fmin,fmax,n_points)`, a log-linear grid (e.g.
`np.exp(np.linspace(np.log(fmin),np.log(fmax),n_points))`), or something
else entirely.  An additional consideration that optimal values of `LAMBDA`
may be quite small relative to `LAMBDA_max`, and we have found values of
`fmax =LAMBDA_max * 1e-2` to be a practical starting point.

### Putting it All Together

Assuming you have prepared your data into numpy matrices (`X` and `Y`, and
optionally `X_treat` and `Y_treat`) with one row per unit of observation
and one column per covariate or observed outcome, scores for a grid of
parameters `LAMBDA` can be obtained as follows: 

```python
import RidgeSC as SC
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
			  L2_PEN_W = L2_pen_start_ct)

# extract the matrix of weights
weights = SC.weights(X = X_control,
					 X_treat = X_treated, # Optional
					 V = V_ct,
					 L2_PEN_W = L2_pen_start_ct) # Also optional

# create the matrix of Synthetic Controls 
synthetic_conrols = weights.dot(Y)
```

### Performance Notes

The function `get_max_lambda()` requires a single calculation of the
gradient of the in-sample prediction error using all of the available data.
In contrast, ` SC.CV_score()` performs gradient descent within each fold of
the data.  Furthermore, in the 'pre-only' scenario the gradient is
calculated once for in each iteration of the gradient descent, whereas in
the 'controls-only' scenario the gradient is calculated once for each
control unit.  Specifically, each control unit is excluded from the set of
units that can be used to predict it's post-intervention outcomes,
resulting in leave-one-out gradient descent.

For large sample sizes in the 'controls-only' scenario, it may be
sufficient to divide the non-held out units into folds such that controls
within the same fold are not used to predict the post-intervention outcomes
of other control units in the same fold.  This resulting in K-fold gradient
descent, which improves the speed of calculating the overall gradient by a
factor slightly greater than `c/k` (where `c` is the number of control
units) with an even greater reduction in memory usage.

K-fold gradient descent is enabled by passing the parameter `grad_splits`
to `CV_score()`, and for consistency across calls, it is recommended to
also pass a value to the parameter `random_state`, which is used in
selecting the gradient folds.

### Additional Performance Considerations

If you have the BLAS / LAPACK libraries installed and available to python,
you should not need to do any further optimization to ensure that maximum
number of processors are used.  If not, you may wish to set the parameter
`paralell=True` when you call `CV_score()`.  (Note that setting
`paralell=True` when the BLAS / LAPACK are available will tend to increase
running times.)

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

