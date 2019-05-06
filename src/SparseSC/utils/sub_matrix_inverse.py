""" In the leave-one-out method with larger sample sizes most of the time is spent in calculating 
    A.I.dot(B) using np.linalg.solve (which is much faster than the brute force method i.e. `A.I.dot(B)`). 
    For example, with 200 units, about about 95% of the time is spent in this line.  

    However we can take advantage of the fact that we're calculating the
    inverse of N matrices which are all sub-matrices of a common matrix by inverting the common matrix 
    and calculating each subset inverse from there.

    https://math.stackexchange.com/a/208021/252693


    TODO: this could be made a bit faster by passing in the indexes (k_rng,
          k_rng2) instead of re-building them
"""
# pylint: skip-file
import numpy as np

def subinv(x,eps=None):
    """ Given an matrix (x), calculate all the inverses of leave-one-out sub-matrices.
    
    :param x: a square matrix for which to find the inverses of all it's leave one out sub-matrices.
    :param eps: If not None, used to assert that the each calculated
           sub-matrix-inverse is within eps of the brute force calculation.
           Testing only, this slows the process way down since the inverse of
           each sub-matrix is calculated by the brute force method. Typically
           set to a multiple of `np.finfo(float).eps`
    """
    # handy constant for indexing
    xi = x.I
    N = x.shape[0]
    rng = np.arange(N)
    out = [None,] * N
    for k in range(N):
        k_rng = rng[rng != k]
        out[k] = xi[np.ix_(k_rng,k_rng)] - xi[k_rng,k].dot(xi[k,k_rng])/xi[k,k]
        if eps is not None:
            if not (abs(out[k] - x[np.ix_(k_rng,k_rng)].I) < eps).all():
                raise RuntimeError("Fast and brute force methods were not within epsilon (%s) for sub-matrix k = %s; max difference = %s" % 
                                   (eps, k,  abs(out[k] - x[np.ix_(k_rng,k_rng)].I).max(), ) )
    return out

def subinv_k(xi,k,eps=None):
    """ Given an matrix (x), calculate all the inverses of leave-one-out sub-matrices. 

    :param x: a square matrix for which to find the inverses of all it's leave one out sub-matrices.
    :param k: the column and row to leave out
    :param eps: If not None, used to assert that the each calculated
           sub-matrix-inverse is within eps of the brute force calculation.
           Testing only, this slows the process way down since the inverse of
           each sub-matrix is calculated by the brute force method. Typically
           set to a multiple of `np.finfo(float).eps`
    """
    # handy constant for indexing
    N = xi.shape[0]
    rng = np.arange(N)
    k_rng = rng[rng != k]
    out = xi[np.ix_(k_rng,k_rng)] - xi[k_rng,k].dot(xi[k,k_rng])/xi[k,k]
    if eps is not None:
        if not (abs(out[k] - x[np.ix_(k_rng,k_rng)].I) < eps).all():
            raise RuntimeError("Fast and brute force methods were not within epsilon (%s) for sub-matrix k = %s; max difference = %s" % (eps, k,  abs(out[k] - x[np.ix_(k_rng,k_rng)].I).max(), ) )
    return out



# ---------------------------------------------
# single sub-matrix
# ---------------------------------------------

if __name__ == "__main__":

    import time

    n = 200
    B = np.matrix(np.random.random((n,n,)))


    n = 5
    p = 3
    a = np.matrix(np.random.random((n,p,)))
    v = np.diag(np.random.random((p,)))
    x = a.dot(v).dot(a.T)
    x.dot(x.I)

    x = np.matrix(np.random.random((n,n,)))
    x.dot(x.I)

    xi = x.I


    B = np.matrix(np.random.random((n,n,)))

    k = np.arange(2)
    N = xi.shape[0]
    rng = np.arange(N)
    k_rng = rng[np.logical_not(np.isin(rng,k))]

    out = xi[np.ix_(k_rng,k_rng)] - xi[np.ix_(k_rng,k)].dot(xi[np.ix_(k,k_rng)])/np.linalg.det(xi[np.ix_(k,k)])

    for i in range(100):
        # create a sub-matrix that meets the matching criteria
        x = np.matrix(np.random.random((n,n,)))
        try:
            zz = subinv(x,10e-10)
            break
        except:
            pass
    else:
        print("Failed to generate a %sx%s matrix whose inverses are all within %s of the quick method")


    k = 5
    n_tests = 1000

    # =======================
    t0 = time.time()
    for i in range(n_tests): 
        _N = xi.shape[0]
        rng = np.arange(_N)
        k_rng = rng[rng != k]
        k_rng2 = np.ix_(k_rng,k_rng)
        zz = x[k_rng2].I.dot(B[k_rng2])

    t1 = time.time()
    slow_time = t1 - t0
    print("A.I.dot(B): brute force time (N = %s): %s"% (n,t1 - t0))

    # =======================
    t0 = time.time()
    for i in range(n_tests): 
        # make the comparison fair
        if i % n == 0:
            xi = x.I
        zz = subinv_k(xi,k).dot(B[k_rng2])

    t1 = time.time()
    fast_time = t1 - t0
    print("A.I.dot(B): quick time (N = %s): %s"% (n,t1 - t0))

    # =======================
    t0 = time.time()
    for i in range(n_tests): 
        _N = xi.shape[0]
        rng = np.arange(_N)
        k_rng = rng[rng != k]
        k_rng2 = np.ix_(k_rng,k_rng)
        zz = np.linalg.solve(x[k_rng2],B[k_rng2])

    t1 = time.time()
    fast_time = t1 - t0
    print("A.I.dot(B): np.linalg.solve time (N = %s): %s"% (n,t1 - t0))

    # ---------------------------------------------
    # ---------------------------------------------

    t0 = time.time()
    for i in range(100): 
        zz = subinv(x,10e-10)

    t1 = time.time()
    slow_time = t1 - t0
    print("Full set of inverses: brute force time (N = %s): %s", (n,t1 - t0))

    t0 = time.time()
    for i in range(100): 
        zz = subinv(x)

    t1 = time.time()
    fast_time = t1 - t0
    print("Full set of inverses: quick time (N = %s): %s", (n,t1 - t0))

    # ---------------------------------------------
    # ---------------------------------------------

