# Ridge Synthetic Controls

### Setup

This package requires [numpy](http://www.numpy.org/), 
[scipy](https://www.scipy.org/), and [scikit-learn](http://scikit-learn.org/) and has been tested with ( Python 2.7.14,
Numpy 1.14.1, and Scipy 1.0.0 )  and ( Python 3.5.5, Numpy 1.13.1, and
Scipy 1.0.1 )

### API

For now, refer to the example code in `./example-code.py` for usage
details.

### Performance Notes

For larger sample sizes (number of controls), almost all of the running
time is spent calculating `numpy.linalg.solve(A,B)` where A is a `C x C`
matrix and `B` is a `C x T` matrix, where `C` is the number of control
units and `T` is the number of treated units.  Note that
numpy.linalg.solve(A,B) has a computational runnint time of at least `O(
C^2.3 )`, and possibly as large as `O( T * C^2.3)`.  Because the LAPACK
library is already parallelized, passing `parallel = true` to the
`CV_score` function (which runs each fold in a separate sub-process) will
tend to **increase** the running time. 

This is especially true for the leave-one-out gradient descent using only
controls as `numpy.linalg.solve(A,B)` will be called K+1 for each control
unit on each calculation of the gradient (where K is the number of
moments).  With larger sample sizes, it is more efficient to pass a value
to `grad_splits`, in order to use k-fold gradient descent, in which case
`numpy.linalg.solve(A,B)` will be called twice for each split on each
calculation of the gradient.

