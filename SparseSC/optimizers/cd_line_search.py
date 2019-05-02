""" Optimizer for covariate weights restricted to the positive orthant.
"""
import numpy as np
from collections import namedtuple
from scipy.optimize import line_search
from scipy.optimize.linesearch import LineSearchWarning
from .simplex_step import simplex_step, simplex_step_proj_sort

import warnings
import locale

# A LineSearchWarning is raised occasionally by line_search(), but it's
# redundant to the return value and we're handling it appropriately
warnings.filterwarnings("ignore", category=LineSearchWarning)

locale.setlocale(locale.LC_ALL, "")

cd_res = namedtuple("cd_res", ["x", "fun"])


def cdl_step(
    score,
    guess,
    jac,
    val=None,
    learning_rate=0.2,
    zero_eps=1e2 * np.finfo(float).eps,
    print_path=True,
    decrement=0.9,
):
    """
    A wrapper of :func:`scipy.optimize.line_search` which restricts the
    gradient descent to the positive orthant and implements a dynamic step size

    PARAMETERS

    score: The objective function

    guess: Initial parameter for the objective function

    jac: Gradient function for the objective function

    val: Initial value for the objective function. Optional; defalts to ``score(guess)``

    learning_rate (float, Default = 0.2): The initial learning rate
        (alpha) which determines the initial step size, which is set to
        learning_rate * null_model_error / gradient. Must be between 0 and
        1.

    zero_eps (Optional, float):  Epsilon for determining if the gradient is
        effectively zero. Defaults to 100 * machine epsilon.

    print_path (boolean).  Optional Controls level of verbosity, Default = True.

    decrement (float, Default = 0.9): (learning_rate_adjustment ) Adjustment factor
        applied to the learning rate applied between iterations when the
        optimal step size returned by :func:`scipy.optimize.line_search` is
        greater less than 1, else the step size is adjusted by
        ``1/learning_rate_adjustment``. Must be between 0 and 1,
    """

    print_path=True
    if print_path:
        print("[FORCING FIRST STEP]")
    assert 0 < learning_rate < 1
    assert 0 < decrement < 1

    if val is None:
        val = score(guess)
    grad = jac(guess)
    # constrain to the positive orthant
    grad[grad > 0] = 0

    if (grad >= 0).all():
        # this happens when we're stuck at the origin and the gradient is
        # pointing in the all-negative direction
        raise RuntimeError("Failed to take a step")
        # I'm conflicted about what to do here. Another option is to:
        # return guess,val

    direction = -(learning_rate * val * grad) / grad.dot(grad.T)
    # THE ABOVE IS EQUIVALENT TO :
    # step_magnitude = learning_rate*val/np.linalg.norm(grad)
    # direction = -step_magnitude * (grad / np.linalg.norm(grad))

    while True:
        new_val = score(direction)
        if new_val < val:
            return direction, new_val
        direction *= decrement
        print("val: %s, new_val: %s, dir: %s",( val, new_val, sum(direction)))
        if sum(direction) < zero_eps:
            raise RuntimeError("Failed to take a step")


def cdl_search(
    score,
    guess,
    jac,
    tol=1e-4,
    learning_rate=0.2,
    learning_rate_adjustment=0.9,
    max_iter=3000,
    min_iter=3,
    # TODO: this is a stupid default (I'm using it out of laziness)
    zero_eps=1e2 * np.finfo(float).eps,
    print_path=True,
    print_path_verbose=False,
    constrain="orthant",
):
    """
    Implements coordinate descent with line search with the strong wolf
    conditions. Note, this tends to give nearly identical results as L-BFGS-B,
    and is *much* slower than that the super-fast 40 year old Fortran code
    wrapped by SciPy.

    score function
    """
    assert 0 < learning_rate < 1
    assert 0 < learning_rate_adjustment < 1
    assert (
        guess >= 0
    ).all(), "Initial guess (`guess`) should be in the closed positive orthant"

    if callable(constrain):
        constrain_factory = constrain
    elif constrain == "simplex":
        constrain_factory = simplex_restraint
    elif constrain == "orthant":
        constrain_factory = orthant_restraint
    else:
        raise ValueError("unknown option for `constrain` parameter")

    print_stop_iteration = print_path  # change to `1` for development purposes
    val_old = None
    grad = None
    x_curr = guess
    alpha_t = 0
    val = score(x_curr)
    if print_path:
        # this is only used for debugging...
        if (x_curr == np.zeros(x_curr.shape[0])).all():
            val0 = val
        else:
            val0 = score(np.zeros(x_curr.shape[0]))

    # if (x_curr == 0).all():
    #     # Force a single step away form the origin if it is at least a little
    #     # useful. Intuition: the curvature at the origin is typically
    #     # exceedingly sharp (becasue we're going from a state with "no
    #     # information" to "some information" in the covariate space, and as
    #     # result the strong wolf conditions will have a strong tendency to
    #     # fail. However, the origin is rarely optimal so forcing a step away
    #     # form the origin will be necessary in most cases.
    #     x_curr, val = cdl_step (score, guess, jac, val, learning_rate, zero_eps, print_path)

    for _i in range(max_iter):

        if grad is None:
            # (this happens when `constrained == True` or the next point falls
            # beyond zero due to rounding error)
            if print_path_verbose:
                print("[INITIALIZING GRADIENT]")
            grad = jac(x_curr)


        invalid_directions = np.logical_and(grad > 0, x_curr == 0)

        if (grad[np.logical_not(invalid_directions)] == 0).all():
            # this happens when we're stuck at the origin and the gradient is
            # pointing in the all-negative direction
            if print_stop_iteration:
                print("[STOP ITERATION: gradient is zero] i: %s" % (_i,))
            return cd_res(x_curr, val)



        if constrain == "simplex" and _i == 0 and (x_curr == np.zeros(x_curr.shape[0])).all():
            if print_path_verbose:
                print("[INITILALIZING V ON THE SIMPLEX]")
            # this *is* necessary to put x_curr on the constrained simplex:
            grad[invalid_directions] = 0
            x_curr = grad / grad.sum()
            grad = None
            continue


        direction = -(learning_rate * val * grad) / grad.dot(grad.T)
        # THE ABOVE IS EQUIVALENT TO :
        # step_magnitude = learning_rate*val/np.linalg.norm(grad)
        # direction = -step_magnitude * (grad / np.linalg.norm(grad))

        # adaptively adjust the step size:
        direction *= learning_rate_adjustment ** alpha_t

        if print_path_verbose:
            print("[STARTING LINE SEARCH]")
        _constraint = constrain_factory(x_curr)
        res = line_search(
            f=constraint_wrapper(score, _constraint),
            myfprime=constraint_wrapper(jac, _constraint),
            xk=x_curr,
            pk=direction,
            gfk=grad,
            old_fval=val,
            old_old_fval=val_old,
        )  #
        if print_path_verbose:
            print("[FINISHED LINE SEARCH]")
        alpha, _, _, _, _, _ = res
        if alpha is not None:
            # adjust the future step size
            if alpha >= 1:
                alpha_t -= 1
            else:
                alpha_t += 1
        else:
            # moving in the direction of the gradient yielded no improvement: stop
            if print_stop_iteration:
                print(
                    "[STOP ITERATION: alpha is None] i: %s, grad: %s, step: %s"
                    % (_i, grad, direction)
                )
            return cd_res(x_curr, val)

        # ITERATE
        # x_next = x_curr +        alpha *direction
        x_next = _constraint(x_curr + alpha * direction)

        # rounding error can get us to within rounding error of zero or even
        # across the coordinate plane:
        x_next[x_next < zero_eps] = 0

        x_old, x_curr, val_old, val, grad, old_grad = (
            x_curr,
            x_next,
            val,
            res[3],
            res[5],
            grad,
        )  # pylint: disable=line-too-long

        val_diff = val_old - val

        # rounding error can get us really close or even across the coordinate plane.
        # NOT SURE IF THIS IS NECESSARY NOW THAT THE GRAD IS WRAPPED IN ZED_WRAPPER
        # NOT SURE IF THIS IS NECESSARY NOW THAT THE GRAD IS WRAPPED IN ZED_WRAPPER
        # --         xtmp = x_curr.copy()
        # --         x_curr[abs(x_curr) < zero_eps] = 0
        # --         x_curr[x_curr < zero_eps] = 0
        # --         if (xtmp != x_curr).any():
        # --             if print_path_verbose:
        # --                 print('[CLEARING GRADIENT]')
        # --             grad = None
        # NOT SURE IF THIS IS NECESSARY NOW THAT THE GRAD IS WRAPPED IN ZED_WRAPPER
        # NOT SURE IF THIS IS NECESSARY NOW THAT THE GRAD IS WRAPPED IN ZED_WRAPPER

        if print_path:
            print(
                "[Path] i: %s, In Sample R^2: %0.6f, incremental R^2:: %0.6f, learning rate: %0.5f,  alpha: %0.5f, zeros: %s"
                % (  # pylint: disable=line-too-long
                    _i,
                    1 - val / val0,
                    (val_diff / val0),
                    learning_rate * (learning_rate_adjustment ** alpha_t),
                    alpha,
                    sum(x_curr == 0),
                )
            )  # pylint: disable=line-too-long
            if print_path_verbose:
                print("old_grad: %s,x_curr %s" % (old_grad, x_curr))

        if (x_curr == 0).all() and (x_old == 0).all():
            # this happens when we were at the origin and the gradient didn't
            # take us out of the range of zero_eps
            if _i == 0:
                x_curr, val = cdl_step(
                    score, guess, jac, val, learning_rate, zero_eps, print_path
                )
                if (x_curr == 0).all():
                    if print_stop_iteration:
                        print(
                            "[STOP ITERATION: Stuck at the origin] iteration: %s"
                            % (_i,)
                        )
            if (x_curr == 0).all():
                if print_stop_iteration:
                    print("[STOP ITERATION: Stuck at the origin] iteration: %s" % (_i,))
                return cd_res(x_curr, score(x_curr))  # tricky tricky...

        if (x_curr < 0).any():
            # This shouldn't ever happen if max_alpha is specified properly
            raise RuntimeError("An internal Error Occured: (x_curr < 0).any()")

        if val_diff / val < tol:
            # this a heuristic rule, to be sure, but seems to be useful.
            # TODO: this is kinda stupid without a minimum on the learning rate
            # (i.e. `learning_rate`).
            if _i > min_iter:
                if print_stop_iteration:
                    # this is kida stupid
                    print(
                        "[STOP ITERATION: val_diff/val < tol] i: %s, val: %s, val_diff: %s"
                        % (_i, val, val_diff)
                    )
                return cd_res(x_curr, val)

    # returns solution in for loop if successfully converges
    raise RuntimeError("Solution did not converge to default tolerance")


def orthant_restraint(x_curr):  # pylint: disable=unused-argument
    """
    Factory Function which builds a function which constrains the gradient step
    to the First Orthant
    """

    def inner(x):
        """
        Project x to the nearest point in the first orthant
        """
        return np.maximum(0, x)

    return inner


def simplex_restraint(x_curr):
    """
    Factory Function which builds a function which constrains the gradient step
    to the (constrained) simplex
    """

    def inner(x):
        """
        Project x to the nearest point in the first orthant
        """
        return simplex_step_proj_sort(x_curr, x_curr - x) #simplex_step

    return inner


def constraint_wrapper(fun, constraint):
    """
    Wrap a function such that it's first argument is constrained 
    """
    def inner(x, *args, **kwargs):
        """the wrapped function"""
        return fun(constraint(x), *args, **kwargs)

    return inner


def zed_wrapper(fun):
    """ a wrapper which implements the waterline algorithm (i.e. walk in
        direction of the gradient, and project to nearest point in the positive
        orthant.
    """

    def inner(x, *args, **kwargs):
        """the wrapped function"""
        return fun(np.maximum(0, x), *args, **kwargs)

    return inner
