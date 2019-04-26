# Performance Notes

## Running time

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

## Additional Considerations

If you have the BLAS/LAPACK libraries installed and available to Python,
you should not need to do any further optimization to ensure that maximum
number of processors are used during the execution of `CV_score()`.  If
not, you may wish to set the parameter `parallel=True` when you call
`CV_score()` which will split the work across N - 2 sub-processes where N
is the [number of cores in your
machine](https://docs.python.org/2/library/multiprocessing.html#miscellaneous).
(Note that setting `parallel=True` when the BLAS/LAPACK are available will
tend to increase running times.)

