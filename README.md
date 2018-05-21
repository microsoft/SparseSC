
# Ridge Synthetic Controls

### Setup

This package requires [numpy](http://www.numpy.org/) and
[scipy](https://www.scipy.org/) and has been tested with ( Python 2.7.14,
Numpy 1.14.1, and Scipy 1.0.0 )  and ( Python 3.5.5, Numpy 1.13.1, and
Scipy 1.0.1 )

### API

For now, refer to the example code in `./example-code.py` for usage
details.

### Performance Notes

For larger sample sizes (number of controls), almost all of the running
time is spent calculating `np.linalg.solve(A,B)` where A is a `C x C`
matrix and `B` is a `C x T` matrix, where `C` is the number of control
units and `T` is the number of treated units.   Because the LAPACK library
is already parallelized, passing `parallel = true` to the `CV_score`
function (which runs each fold in a separate sub-process) will actually
tend to increase the running time. 

This is especially true for the leave-one-out gradient descent using only
controls as `np.linalg.solve(A,B)` will be called twice for each control
unit.  With larger sample sizes, it is more efficient to pass a value to
`grad_splits`, in order to use k-fold gradient descent, in which case
`np.linalg.solve(A,B)` will be called twice for each split.

