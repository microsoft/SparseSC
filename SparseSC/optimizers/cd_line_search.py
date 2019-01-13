import numpy as np
from scipy.optimize import line_search

import locale 
locale.setlocale(locale.LC_ALL, '')

try:
    runtime
except NameError:
    runtime = RuntimeError

class cd_res(object):
    def __init__(self, x, fun):
        self.x = x
        self.fun = fun

print_stop_iteration = 1

def cdl_step(score,
               guess,
               jac,
               val = None,
               aggressiveness = 0.1,
               zero_eps = 1e2 * np.finfo(float).eps,
               print_path = True,
               decrement = 1e-1):
               
    print("[FORCING FIRST STEP]")
    assert 0 < aggressiveness < 1
    assert 0 < decrement < 1

    if val is None: 
        val = score(guess)
    grad = jac(guess)
    grad_copy = grad.copy()
    # constrain to the positive orthant
    grad[grad > 0] = 0

    if (grad >= 0).all():
        # this happens when we're stuck at the origin and the gradient is
        # pointing in the all-negative direction
        raise runtime("Failed to take a step")
        # obviously I'm conflicted about what to do here...
        return guess,val

    direction = - (aggressiveness * val * grad) / grad.dot(grad.T)
    # THE ABOVE IS EQUIVALENT TO : 
    # step_magnitude = aggressiveness*val/np.linalg.norm(grad)
    # direction = -step_magnitude * (grad / np.linalg.norm(grad))

    while True: 
        new_val = score( direction)
        if new_val < val:
            return direction, new_val
        direction *= decrement
        if sum(direction) < zero_eps:  
            raise runtime("Failed to take a step")

def cdl_search(score,
               guess,
               jac,
               tol = 1e-4,
               aggressiveness = 0.1,# aggressiveness
               alpha_mult = .9,
               max_iter = 3000,
               min_iter = 3,
               # TODO: this is a stupid default (I'm using it out of laziness)
               zero_eps = 1e2 * np.finfo(float).eps,
               print_path = True,
               print_path_verbose = False,
               preserve_angle = False):
    '''
    Implements coordinate descent with line search with the strong wolf
    conditions. Note, this tends to give nearly identical results as L-BFGS-B,
    and is *much* slower than that the super-fast 40 year old Fortran code
    wrapped by SciPy.
    '''
    assert 0 < aggressiveness < 1
    assert 0 < alpha_mult < 1
    assert (guess >=0).all(), "Initial guess (`guess`) should be in the closed positive orthant"

    val_old = None
    grad = None
    x_curr = guess
    alpha_t = 0
    val = score(x_curr)
    if (x_curr == np.zeros(x_curr.shape[0])).all():
        val0 = val
    else: 
        val0 = score(np.zeros(x_curr.shape[0]))

#--     if (x_curr == 0).all():
#--         # Force a single step away form the origin if it is at least a little
#--         # useful. Intuition: the curvature at the origin is typically
#--         # exceedingly sharp (becasue we're going from a state with "no
#--         # information" to "some information" in the covariate space, and as
#--         # result the strong wolf conditions will have a strong tendency to
#--         # fail. However, the origin is rarely optimal so forcing a step away
#--         # form the origin will be necessary in most cases.
#--         x_curr, val = cdl_step (score, guess, jac, val, aggressiveness, zero_eps, print_path)

    for _i in range(max_iter):

        if grad is None:
            # (this happens when `constrained == True` or the next point falls beyond zero due to rounding error)
            if print_path_verbose:
                print("[INITIALIZING GRADIENT]")
            grad = jac(x_curr)
        invalid_directions = np.logical_and(grad > 0,x_curr == 0)

        if (grad[np.logical_not(invalid_directions)] == 0).all():
            # this happens when we're stuck at the origin and the gradient is
            # pointing in the all-negative direction
            if print_stop_iteration: 
                print("[STOP ITERATION: gradient is zero] i: %s" % (_i,))
            return cd_res(x_curr, val)


        # constrain to the positive orthant
        grad[invalid_directions] = 0

        direction = - (aggressiveness * val * grad) / grad.dot(grad.T)
        # THE ABOVE IS EQUIVALENT TO : 
        # step_magnitude = aggressiveness*val/np.linalg.norm(grad)
        # direction = -step_magnitude * (grad / np.linalg.norm(grad))

        # adaptively adjust the step size: 
        direction *= (alpha_mult ** alpha_t)

        # constrain the gradient to being non-negative on axis where the
        # current guess is already zero
        if (direction<0).any() and preserve_angle:
            constrained = True
            alpha_ratios = - direction[ direction <0 ] / x_curr[ direction <0 ]
            if (alpha_ratios > 1).any(): 
                max_alpha = alpha_ratios.max()
            else:
                max_alpha = 1
        else:
            constrained = False
            max_alpha = 1

        if print_path_verbose: 
            print("[STARTING LINE SEARCH]")
        res = line_search(f=zed_wrapper(score), myfprime=zed_wrapper(jac), xk=x_curr, pk= direction/max_alpha, gfk= grad, old_fval=val,old_old_fval=val_old) # 
        if print_path_verbose: 
            print("[FINISHED LINE SEARCH]")
        alpha, _, _, _, _, _ = res 
        if alpha is not None:
            # adjust the future step size
            if alpha >= 1:
                alpha_t -= 1
            else:
                alpha_t += 1
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
                        print("[STOP ITERATION: simple line search worked :)] i: %s, alpha: 1e-%s" % (_i,j))
                    break
            else:
                # moving in the direction of the gradient yielded no improvement: stop
                if print_stop_iteration: 
                    print("[STOP ITERATION: simple line search failed] i: %s" % (_i,))
                return cd_res(x_curr, val)
        else:
            # moving in the direction of the gradient yielded no improvement: stop
            if print_stop_iteration: 
                print("[STOP ITERATION: alpha is None] i: %s, grad: %s, step: %s" % (_i, grad, direction/max_alpha, ))
            return cd_res(x_curr, val)

        # iterate
        if constrained:
            x_next = x_curr + min(1, alpha)*direction/max_alpha
            x_old, x_curr, val_old, val, grad, old_grad   =   x_curr, x_next, val, score(x_next),  None, grad
        else:
            #x_next = x_curr +        alpha *direction/max_alpha
            x_next = np.maximum(x_curr +        alpha *direction/max_alpha,0)
            x_old, x_curr, val_old, val, grad, old_grad   =   x_curr, x_next, val,        res[3], res[5], grad

        val_diff = val_old - val

        # rounding error can get us really close or even across the coordinate plane.
        # NOT SURE IF THIS IS NECESSARY NOW THAT THE GRAD IS WRAPPED IN ZED_WRAPPER
        # NOT SURE IF THIS IS NECESSARY NOW THAT THE GRAD IS WRAPPED IN ZED_WRAPPER
#--         xtmp = x_curr.copy()
#--         x_curr[abs(x_curr) < zero_eps] = 0
#--         x_curr[x_curr < zero_eps] = 0
#--         if (xtmp != x_curr).any():
#--             if print_path_verbose: 
#--                 print('[CLEARING GRADIENT]')
#--             grad = None
        # NOT SURE IF THIS IS NECESSARY NOW THAT THE GRAD IS WRAPPED IN ZED_WRAPPER
        # NOT SURE IF THIS IS NECESSARY NOW THAT THE GRAD IS WRAPPED IN ZED_WRAPPER

        if print_path: 
            print("[Path] i: %s, In Sample R^2: %0.6f, incremental R^2:: %0.6f, learning rate: %0.5f,  alpha: %0.5f, zeros: %s"  % 
                    (_i,  1- val / val0, (val_diff/ val0), aggressiveness * (alpha_mult ** alpha_t), alpha, sum( x_curr == 0)))
            if print_path_verbose:
                print("old_grad: %s,x_curr %s"  % (old_grad, x_curr, ))


        if (x_curr == 0).all() and (x_old == 0).all():
            # this happens when we were at the origin and the gradient didn't
            # take us out of the range of zero_eps
            if _i == 0: 
                x_curr, val = cdl_step (score, guess, jac, val, aggressiveness, zero_eps, print_path)
                if (x_curr == 0).all():
                    if print_stop_iteration: 
                        print("[STOP ITERATION: Stuck at the origin] iteration: %s"% (_i,))
            if (x_curr == 0).all():
                if print_stop_iteration: 
                    print("[STOP ITERATION: Stuck at the origin] iteration: %s"% (_i,))
                return cd_res(x_curr, score(x_curr)) # tricky tricky...

        if (x_curr < 0).any():
            # This shouldn't ever happen if max_alpha is specified properly
            raise RuntimeError("An internal Error Occured: (x_curr < 0).any()")

        if val_diff/val < tol:
            # this a heuristic rule, to be sure, but seems to be useful. 
            # TODO: this is kinda stupid without a minimum on the learning rate (i.e. `aggressiveness`).
            if _i > min_iter:
                if print_stop_iteration:
                    # this is kida stupid
                    print("[STOP ITERATION: val_diff/val < tol] i: %s, val: %s, val_diff: %s" % (_i, val, val_diff, ))
                return cd_res(x_curr, val)

    # returns solution in for loop if successfully converges
    raise RuntimeError('Solution did not converge to default tolerance')

def zed_wrapper(fun):
    def inner(x,*args,**kwargs):
        return fun(np.maximum(0,x),*args,**kwargs)
    return inner

