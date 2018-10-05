import numpy as np
from scipy.optimize import line_search

import locale 
locale.setlocale(locale.LC_ALL, '')

class cd_res(object):
    def __init__(self, x, fun):
        self.x = x
        self.fun = fun

print_stop_iteration = 0

def cdl_search(score,
               guess,
               jac,
               tol = 1e-4,
               agressiveness = 0.01,# agressiveness
               alpha_mult = .9,
               max_iter = 3000,
               min_iter = 3,
               # TODO: this is a stupid default (I'm useing it out of lazyness)
               zero_eps = 1e2 * np.finfo(float).eps,
               print_path=False,
               print_path_verbose = False):
    '''
    Implements coordinate descent with line search with the strong wolf
    conditions. Note, this tends to give nearly identical results as L-BFGS-B,
    and is *much* slower than that the super-fast 40 year old fortran code
    wrapped by scipy.
    '''
    assert 0 < agressiveness < 1
    assert 0 < alpha_mult < 1
    assert (guess >=0).all(), "Initial guess (`guess`) should be in the closed positive orthant"
    x_curr = guess
    alpha_t = 1.
    val = score(x_curr)

    for _i in range(max_iter):

        grad = jac(x_curr)
        invalid_directions = np.logical_and(grad > 0,x_curr == 0)

        if (grad[np.logical_not(invalid_directions)] == 0).all():
            # this happens when we're stuck at the origin and the gradient is
            # pointing in the all-negative direction
            if print_stop_iteration: 
                print("[gradient is zero] i: %s" % (_i,))
            return cd_res(x_curr, val)

        # Expected Loss Reduction for a unit step in each of the valid directions
        elr = sum( abs(g) for g in grad[np.logical_not(invalid_directions)])
        naiive_step_size_multiplier = agressiveness*val/elr
        grad *= naiive_step_size_multiplier * alpha_t

        # constrain the gradient to being non-negative on axis where the
        # current guess is already zero

        # constrain to the positive orthant
        grad[invalid_directions] = 0

        if (grad>0).any():
            constrained = True
            alpha_ratios = grad[ grad >0 ] / x_curr[ grad >0 ]
            if (alpha_ratios > 1).any: 
                max_alpha = alpha_ratios.max()
            else:
                max_alpha = 1
        else:
            constrained = False
            max_alpha = 1

        res = line_search(f=score, myfprime=jac, xk=x_curr, pk=-grad/max_alpha) # 
        alpha, _, _, _, _, _ = res 
        if alpha is not None:
            if alpha > 1:
                alpha_t /= alpha_mult
            else:
                alpha_t *= alpha_mult
        elif constrained:
            for j in range(5): # formerly range(17), but that was excessive, 
                # in general, this succeeds happens when alpha >= 0.1 (super helpful) or alpha <= 1e-14 (super useless)
                if score(x_curr - (.3**j)*grad/max_alpha) < val:
                    # This can occur when the strong wolf condition insists that the
                    # current step size is too small (i.e. the gradient is too
                    # consistent with the function to think that a small step is 
                    # optimal for a global (unconstrained) optimization.
                    alpha = (.3**j)

                    # i secretly think this is stupid.
                    if print_stop_iteration: 
                        print("[simple line search worked :)] i: %s, alpha: 1e-%s" % (_i,j))
                    break
            else:
                # moving in the direction of the gradient yielded no improvement: stop
                if print_stop_iteration: 
                    print("[simple line search failed] i: %s" % (_i,))
                return cd_res(x_curr, val)
        else:
            # moving in the direction of the gradient yielded no improvement: stop
            if print_stop_iteration: 
                print("[alpha is None] i: %s, grad: %s" % (_i, grad/max_alpha, ))
            return cd_res(x_curr, val)

        # iterate
        if constrained:
            x_old, x_curr, val_old   =   x_curr, x_curr - min(1, alpha)*grad/max_alpha, val
        else:
            x_old, x_curr, val_old   =   x_curr, x_curr -      alpha*grad/max_alpha, val

        val = score(x_curr)
        val_diff = val_old - val

        if print_path: 
            print("i: %s, val: %s, val_diff: %0.2f, alpha: %0.5f, zeros: %s (constrained)"  % 
                  (_i, locale.format("%d", val, grouping=True), val_diff, alpha, sum( x_curr == 0)))
            if print_path_verbose:
                print("grad: %s,x_curr %s"  % (grad, x_curr, ))

        # rounding error can get us really close or even across the coordinate plane.
        x_curr[abs(x_curr) < zero_eps] = 0
        x_curr[x_curr < zero_eps] = 0

        if (x_curr == 0).all() and (x_old == 0).all():
            # this happens when we were at the origin and the gradient didn't
            # take us out of the range of zero_eps
            if print_stop_iteration: 
                print("stuck at the origin %s"% (_i,))
            return cd_res(x_curr, score(x_curr)) # tricky tricky...

        if (x_curr < 0).any():
            # This shouldn't ever happen if max_alpha is specified properly
            raise RuntimeError("An internal Error Occured: (x_curr < 0).any()")

        if val_diff/val < tol:
            # this a heuristic rule, to be sure, but seems to be useful.
            if _i > min_iter:
                if print_stop_iteration:
                    print("[val_diff/val < tol] i: %s, val: %s, val_diff: %s" % (_i, val, val_diff, ))
                return cd_res(x_curr, val)

    # returns solution in for loop if successfully converges
    raise RuntimeError('Solution did not converge to default tolerance')
