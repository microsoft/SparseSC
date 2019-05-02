""" 
Gradient descent within the simplex

Method inspired by the qustion "what would a water droplet stuckin insise the
positive simplex go when pulled in the direction of the gradient which would
otherwise take the droplet outside of the simplex
"""
# pylint: disable=invalid-name
from numpy import (
    array,
    append,
    arange,
    cumsum,
    random,
    zeros,
    ones,
    logical_or,
    logical_not,
    logical_and,
    maximum,
    argmin,
    where,
    sort,
)


def _sub_simplex_project(d_hat, indx):
    """
    A utility function which projects the gradient into the subspace of the
    simplex which intersects the plane x[_index] == 0
    """
    # now project the gradient perpendicular to the edge we just came up against
    n = len(d_hat)
    _n = float(n)
    a_dot_a = (n - 1) / _n
    a_tilde = -ones(n) / _n
    a_tilde[indx] += 1  # plus a'
    proj_a_d = (d_hat.dot(a_tilde) / a_dot_a) * a_tilde
    d_tilde = d_hat - proj_a_d
    return d_tilde


def simplex_step(x, g, verbose=False):
    """
    follow the gradint as far as you can within the positive simplex
    """
    i = 0
    x, g = x.copy(), g.copy()
    # project the gradient into the simplex
    g = g - (g.sum() / len(x)) * ones(len(g))
    _g = g.copy()
    while True:
        if verbose:
            print("iter: %s, g: %s" % (i, g))
        # we can move in the direction of the gradient if either
        # (a) the gradient points away from the axis
        # (b) we're not yet touching the axis
        valid_directions = logical_or(g < 0, x > 0)
        if verbose:
            print(
                " valid_directions(%s, %s, %s): %s "
                % (
                    valid_directions.sum(),
                    (g < 0).sum(),
                    (x > 0).sum(),
                    ", ".join(str(x) for x in valid_directions),
                )
            )
        if not valid_directions.any():
            break
        if any(g[logical_not(valid_directions)] != 0):
            # TODO: make sure is is invariant on the order of operations
            n_valid = where(valid_directions)[0]
            W = where(logical_not(valid_directions))[0]
            for i, _w in enumerate(W):
                # TODO: Project the invalid directions into the current (valid) subspace of
                # the simplex
                mask = append(array(_w), n_valid)
                # print("work in progress")
                g[mask] = _sub_simplex_project(g[mask], 0)
                g[_w] = 0  # may not be exactly zero due to rounding error
        # rounding error can take us out of the simplex positive orthant:
        g = maximum(0, g)
        if (g == zeros(len(g))).all():
            # we've arrived at a corner and the gradient points outside the constrained simplex
            break
        # HOW FAR CAN WE GO?
        limit_directions = logical_and(valid_directions, g > 0)
        xl = x[limit_directions]
        gl = g[limit_directions]
        ratios = xl / gl
        try:
            c = ratios.min()
        except:
            import pdb

            pdb.set_trace()
        if c > 1:
            x = x - g
            # pdb.set_trace()
            break
        arange(len(g))
        indx = argmin(ratios)
        # MOVE
        # there's gotta be a better way...
        _indx = where(limit_directions)[0][indx]
        tmp = -ones(len(x))
        tmp[valid_directions] = arange(valid_directions.sum())
        __indx = int(tmp[_indx])
        # get the index
        del xl, gl, ratios
        x = x - c * g
        # PROJECT THE GRADIENT
        d_tilde = _sub_simplex_project(g[valid_directions] * (1 - c), __indx)
        if verbose:
            print(
                "i: %s, which: %s, g.sum(): %f, x.sum(): %f, x[i]: %f, g[i]: %f, d_tilde[i]: %f"
                % (
                    i,
                    indx,
                    g.sum(),
                    x.sum(),
                    x[valid_directions][__indx],
                    g[valid_directions][__indx],
                    d_tilde[__indx],
                )
            )
        g[valid_directions] = d_tilde
        # handle rounding error...
        x[_indx] = 0
        g[_indx] = 0
        # INCREMENT THE COUNTER
        i += 1
        if i > len(x):
            raise RuntimeError("something went wrong")
    return x

def simplex_step_proj_sort(x, g, verbose=False):
    x_new = simplex_proj_sort(x-g)
    return x_new

#There's a fast version which uses the median finding algorithm rather than full sorting, but more complicated
#See https://en.wikipedia.org/wiki/Simplex#Projection_onto_the_standard_simplex
# and https://gist.github.com/mblondel/6f3b7aaad90606b98f71
def simplex_proj_sort(v, verbose=False):
    k = v.shape[0]
    if k == 1:
        return 1

    u = sort(v)[::-1] #switches the order
    ind = arange(1, k+1) #shift to 1-indexing
    pis = (cumsum(u) - 1) / ind
    rho = ind[(u - pis) > 0][-1]
    theta = pis[rho-1] #shift back to 0-indexing
    v_new = maximum(v - theta, 0)

    return v_new
