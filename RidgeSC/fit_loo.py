from numpy import ones, diag, matrix, zeros, absolute, mean,var, linalg, prod, sqrt
import numpy as np
import itertools
import warnings
import math
from collections import namedtuple
# only used by the step-down method (currently not implemented):
# from RidgeSC.utils.sub_matrix_inverse import subinv_k, all_subinverses
from RidgeSC.optimizers.cd_line_search import cdl_search
warnings.filterwarnings('ignore')

def complete_treated_control_list(C_N, treated_units = None, control_units = None):
    if treated_units is None: 
        if control_units is None: 
            # both not provided, include all samples as both treat and control unit.
            control_units = list(range(C_N))
            treated_units = control_units 
        else:
            # Set the treated units to the not-control units
            treated_units = list(set(range(C_N)) - set(control_units))  
    else:
        if control_units is None: 
            # Set the control units to the not-treated units
            control_units = list(set(range(C_N)) - set(treated_units)) 
    return(treated_units, control_units)

def loo_v_matrix(X,
                 Y,
                 LAMBDA = 0,
                 treated_units = None,
                 control_units = None,
                 non_neg_weights = False,
                 start = None,
                 L2_PEN_W = None,
                 method = cdl_search, 
                 intercept = True,
                 max_lambda = False,  # this is terrible at least without documentation...
                 solve_method = "standard",
                 verbose = False,
                 **kwargs):
    '''
    Computes and sets the optimal v_matrix for the given moments and 
        penalty parameter.

    :param X: Matrix of Covariates
    :param Y: Matrix of Outcomes
    :param LAMBDA: penalty parameter used to shrink L1 norm of v/v.max() toward zero
    :param treated_units: a list containing the position (rows) of the treated units within X and Y
    :param control_units: a list containing the position (rows) of the control units within X and Y
    :param start: initial values for the diagonals of the tensor matrix
    :param L2_PEN_W: L2 penalty on the magnitude of the deviance of the weight
        vector from null. Optional.
    :param method: The name of a method to be used by scipy.optimize.minimize,
        or a callable with the same API as scipy.optimize.minimize
    :param intercept: If True, weights are penalized toward the 1 / the number
        of controls, else weights are penalized toward zero
    :param max_lambda: if True, the return value is the maximum L1 penalty for
        which at least one element of the tensor matrix is non-zero
    :param solve_method: Method for solving A.I.dot(B). Either "standard" or
        "step-down". https://math.stackexchange.com/a/208021/252693
    :param verbose: If true, print progress to the console (default: false)
    :param kwargs: additional arguments passed to the optimizer
    :param non_neg_weights: not implemented

    :raises ValueError: raised when parameter values are invalid
    :raises TypeError: raised when parameters are of the wrong type

    :return: something something
    :rtype: something something
    '''
    treated_units, control_units = complete_treated_control_list(X.shape[0], treated_units, control_units)
    control_units = np.array(control_units)
    treated_units = np.array(treated_units)

    # parameter QC
    if not isinstance(X, matrix):
        raise TypeError("X is not a matrix")
    if not isinstance(Y, matrix):
        raise TypeError("Y is not a matrix")
    if X.shape[1] == 0:
        raise ValueError("X.shape[1] == 0")
    if Y.shape[1] == 0:
        raise ValueError("Y.shape[1] == 0")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y have different number of rows (%s and %s)" % (X.shape[0], Y.shape[0],))
    if not isinstance(LAMBDA, (float, int)):
        raise TypeError( "LAMBDA is not a number")
    if L2_PEN_W is None:
        L2_PEN_W = mean(var(X, axis = 0))
    if not isinstance(L2_PEN_W, (float, int)):
        raise TypeError( "L2_PEN_W is not a number")
    assert not non_neg_weights, "Bounds not implemented"

    # CONSTANTS
    C, N, K = len(control_units), len(treated_units), X.shape[1]
    if start is None:
        start = zeros(K) # formerly: .1 * ones(K) 
    assert N > 0, "No control units"
    assert C > 0, "No treated units"
    assert K > 0, "variables to fit (X.shape[1] == 0)"

    # CREATE THE INDEX THAT INDICATES THE ELIGIBLE CONTROLS FOR EACH TREATED UNIT
    in_controls = [list(set(control_units) - set([trt_unit])) for trt_unit in treated_units]
    in_controls2 = [np.ix_(i,i) for i in in_controls] # this is a much faster alternative to A[:,index][index,:]
    ctrl_rng = np.arange(len(control_units))
    out_controls = [ctrl_rng[control_units != trt_unit] for trt_unit in treated_units] 
    # this is non-trivial when there control units are also being predicted:
    #out_treated  = [ctrl_rng[control_units == trt_unit] for trt_unit in treated_units] 

    if intercept:
        Y = Y.copy()
        for i, trt_unit in enumerate(treated_units):
            Y[trt_unit,:] -= Y[in_controls[i],:].mean(axis=0) 

    # handy constants (for speed purposes):
    Y_treated = Y[treated_units,:]
    Y_control = Y[control_units,:]
    # only used by step-down method: X_treated = X[treated_units,:]
    # only used by step-down method: X_control = X[control_units,:]

    # INITIALIZE PARTIAL DERIVATIVES
    dA_dV_ki = [ [None,] *N for i in range(K)]
    dB_dV_ki = [ [None,] *N for i in range(K)]
    b_i = [None,] *N 
    for i, k in  itertools.product(range(N), range(K)): # TREATED unit i, moment k
        Xc = X[in_controls[i], : ]
        Xt = X[treated_units[i], : ]
        dA_dV_ki [k][i] = Xc[:, k ].dot(Xc[:, k ].T) + Xc[:, k ].dot(Xc[:, k ].T) # 8
        dB_dV_ki [k][i] = Xc[:, k ].dot(Xt[:, k ].T) + Xc[:, k ].dot(Xt[:, k ].T) # 9

    k=0 # for linting...
    del Xc, Xt, i, k

        #assert (dA_dV_ki [k][i] == X[index, k ].dot(X[index, k ].T) + X[index, k ].dot(X[index, k ].T)).all()
        # https://math.stackexchange.com/a/1471836/252693

    def _score(V):
        dv = diag(V)
        weights, _, _ = _weights(dv)
        Ey = (Y_treated - weights.T.dot(Y_control)).getA()
        # (...).copy() assures that x.flags.writeable is True:
        return ((Ey **2).sum() + LAMBDA * absolute(V).sum()).copy() 

    def _grad(V):
        """ Calculates just the diagonal of dGamma0_dV

            There is an implementation that allows for all elements of V to be varied...
        """
        dv = diag(V)
        weights, A, _ = _weights(dv)
        Ey = (Y_treated - weights.T.dot(Y_control)).getA()
        dGamma0_dV_term2 = zeros(K)
        dPI_dV = zeros((C, N))
        # if solve_method == "step-down": Ai_cache = all_subinverses(A)
        for k in range(K):
            if verbose:  # for large sample sizes, linalg.solve is a huge bottle neck,
                print("Calculating gradient, for moment %s of %s" % (k ,K,))
            dPI_dV.fill(0) # faster than re-allocating the memory each loop.
            for i, index in enumerate(in_controls):
                dA = dA_dV_ki[k][i]
                dB = dB_dV_ki[k][i]
                if solve_method == "step-down":
                    raise NotImplementedError("The solve_method 'step-down' is currently not implemented")
                    # b = Ai_cache[i].dot(dB - dA.dot(b_i[i]))
                else:
                    if verbose >=2:  # for large sample sizes, linalg.solve is a huge bottle neck,
                        print("Calculating weights, linalg.solve() call %s of %s" % 
                              (i + k*K , 
                               K * len(in_controls),))
                    b = linalg.solve(A[in_controls2[i]],dB - dA.dot(b_i[i]))
                dPI_dV[index, i] = b.flatten() # TODO: is the Transpose  an error???
            dGamma0_dV_term2[k] = (Ey * Y_control.T.dot(dPI_dV).T.getA()).sum()
        return LAMBDA - 2 * dGamma0_dV_term2 

    def _weights(V):
        weights = zeros((C, N))
        if solve_method == "step-down":
            raise NotImplementedError("The solve_method 'step-down' is currently not implemented")
            # A = X_control.dot(V + V.T).dot(X_control.T) + 2 * L2_PEN_W * diag(ones(X_control.shape[0])) # 5
            # B = X_treated.dot(V + V.T).dot(X_control.T) # 6
            # Ai = A.I
            # for i, trt_unit in enumerate(treated_units):
            #     if trt_unit in control_units:
            #         (b) = subinv_k(Ai,_k).dot(B[out_controls[i],i])
            #     else:
            #         (b) = Ai.dot(B[:, i])
            #     b_i[i] = b
            #     weights[out_controls[i], i] = b.flatten()
        elif solve_method == "standard":
            A = X.dot(V + V.T).dot(X.T) + 2 * L2_PEN_W * diag(ones(X.shape[0])) # 5
            B = X.dot(V + V.T).dot(X.T).T # 6
            for i, trt_unit in enumerate(treated_units):
                if verbose >= 2:  # for large sample sizes, linalg.solve is a huge bottle neck,
                    print("Calculating weights, linalg.solve() call %s of %s" % (i,len(in_controls),))
                (b) = b_i[i] = linalg.solve(A[in_controls2[i]], B[in_controls[i], trt_unit])
                weights[out_controls[i], i] = b.flatten()
        else:
            raise ValueError("Unknown Solve Method: " + solve_method)
        return weights, A, B

    if max_lambda:
        grad0 = _grad(zeros(K))
        return -grad0[grad0 < 0].min()

    # DO THE OPTIMIZATION
    if isinstance(method, str):
        from scipy.optimize import minimize
        opt = minimize(_score, start.copy(), jac = _grad, method = method, **kwargs)
    else:
        assert callable(method), "Method must be a valid method name for scipy.optimize.minimize or a minimizer"
        opt = method(_score, start.copy(), jac = _grad, **kwargs)
    v_mat = diag(opt.x)
    # CALCULATE weights AND ts_score
    weights, _, _ = _weights(v_mat)
    errors = Y_treated - weights.T.dot(Y_control)
    ts_loss = opt.fun
    ts_score = linalg.norm(errors) / sqrt(prod(errors.shape))

    #if True:
    #    _do_gradient_check()

    if intercept:
        Y = Y.copy()
        for i, trt_unit in enumerate(treated_units):
            weights[out_controls[i], i] += 1/len(out_controls[i])
    return weights, v_mat, ts_score, ts_loss, L2_PEN_W, opt

def loo_weights(X, V, L2_PEN_W, treated_units = None, control_units = None, intercept = True, solve_method = "standard", verbose = False):
    treated_units, control_units = complete_treated_control_list(X.shape[0], treated_units, control_units)
    control_units = np.array(control_units)
    treated_units = np.array(treated_units)
    [C, N] = [len(control_units), len(treated_units)]


    # index with positions of the controls relative to the incoming data
    in_controls = [list(set(control_units) - set([trt_unit])) for trt_unit in treated_units]
    in_controls2 = [np.ix_(i,i) for i in in_controls] # this is a much faster alternative to A[:,index][index,:]

    # index of the controls relative to the rows of the outgoing C x N matrix of weights
    ctrl_rng = np.arange(len(control_units))
    out_controls = [ctrl_rng[control_units != trt_unit] for trt_unit in treated_units] 
    # this is non-trivial when there control units are also being predicted:
    #out_treated  = [ctrl_rng[control_units == trt_unit] for trt_unit in treated_units] 

    # constants for indexing
    # > only used by the step-down method (currently not implemented) X_control = X[control_units,:]
    # > only used by the step-down method (currently not implemented) X_treat = X[treated_units,:]
    weights = zeros((C, N))

    if solve_method == "step-down":
        raise NotImplementedError("The solve_method 'step-down' is currently not implemented")
        # A = X_control.dot(V + V.T).dot(X_control.T) + 2 * L2_PEN_W * diag(ones(X_control.shape[0])) # 5
        # B = X_treat.dot(  V + V.T).dot(X_control.T) # 6
        # Ai = A.I
        # for i, trt_unit in enumerate(treated_units):
        #     if trt_unit in control_units:
        #         (b) = subinv_k(Ai,_k).dot(B[out_controls[i],i])
        #     else:
        #         (b) = Ai.dot(B[:, i])
        #     weights[out_controls[i], i] = b.flatten()
        #     if intercept:
        #         weights[out_controls[i], i] += 1/len(out_controls[i])
    elif solve_method == "standard":
        A = X.dot(V + V.T).dot(X.T) + 2 * L2_PEN_W * diag(ones(X.shape[0])) # 5
        B = X.dot(V + V.T).dot(X.T).T # 6
        for i, trt_unit in enumerate(treated_units):
            if verbose >= 2:  # for large sample sizes, linalg.solve is a huge bottle neck,
                print("Calculating weights, linalg.solve() call %s of %s" % (i,len(treated_units),))
            (b) = linalg.solve(A[in_controls2[i]], B[in_controls[i], trt_unit])

            weights[out_controls[i], i] = b.flatten()
            if intercept:
                weights[out_controls[i], i] += 1/len(out_controls[i])
    else:
        raise ValueError("Unknown Solve Method: " + solve_method)
    return weights.T


def loo_score(Y, X, V, L2_PEN_W, LAMBDA = 0, treated_units = None, control_units = None,**kwargs):
    treated_units, control_units = complete_treated_control_list(X.shape[0], treated_units, control_units)
    weights = loo_weights(X = X,
                          V = V,
                          L2_PEN_W = L2_PEN_W,
                          treated_units = treated_units,
                          control_units = control_units,
                          **kwargs)
    Y_tr = Y[treated_units, :]
    Y_c = Y[control_units, :]
    Ey = (Y_tr - weights.dot(Y_c)).getA()
    return (Ey **2).sum() + LAMBDA * V.sum()


def _ncr(n, r):
    #https://stackoverflow.com/questions/4941753/is-there-a-math-ncr-function-in-python
    import operator as op
    r = min(r, n-r)
    numer = reduce(op.mul, xrange(n, n-r, -1), 1)
    denom = reduce(op.mul, xrange(1, r+1), 1)
    return numer//denom

def random_combination(iterable, r):
    "Random selection from itertools.combinations(iterable, r)"
    #https://stackoverflow.com/questions/22229796/choose-at-random-from-combinations
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), r))
    return tuple(pool[i] for i in indices)

def repeatfunc(func, times=None, *args):
    """Repeat calls to func with specified arguments.

    Example:  repeatfunc(random.random)
    """
    if times is None:
        return starmap(func, repeat(args))
    return starmap(func, repeat(args, times))

def estimate_effects(Y_pre, Y_post, X, V, treated_units, L2_PEN_W, max_n_pl = 1000000, ret_pl = False, ret_CI=False, ret_p1s=False, level=0.95, **kwargs):
    #TODO: Add pre-treatment match quality filter
    #TODO: Cleanup returning placebo distribution (incl pre?)
    keep_pl = ret_pl or ret_CI
    N = len(treated_units)
    C_N = X.shape[0]
    C = C_N - N
    T1 = Y_post.shape[1]
    control_units = list(set(range(C_N)) - set(treated_units)) 
    all_units = list(range(C_N))
    weights = loo_weights(X = X,
                           V = V,
                           L2_PEN_W = L2_PEN_W,
                           treated_units = all_units,
                           control_units = control_units,
                           **kwargs)
    # Get post effects
    Y_post_tr = Y_post[treated_units, :]
    Y_post_c = Y_post[control_units, :]
    Y_post_sc = weights.dot(Y_post_c)
    Y_post_tr_sc = Y_post_sc[treated_units, :]
    Y_post_c_sc = Y_post_sc[control_units, :]
    effect_vecs = Y_post_tr - Y_post_tr_sc
    joint_effects = np.sqrt(np.mean(effect_vecs^2, axis=1))
    control_effect_vecs = Y_post_c - Y_post_c_sc
    control_joint_effects = np.sqrt(np.mean(control_effect_vecs^2, axis=1))
    
    # Get pre match MSE (match quality)
    Y_pre_tr = Y_pre[treated_units, :]
    Y_pre_c = Y_pre[control_units, :]
    Y_pre_sc = weights.dot(Y_pre_c)
    Y_pre_tr_sc = Y_pre_sc[treated_units, :]
    Y_pre_c_sc = Y_pre_sc[control_units, :]
    pre_tr_pes = Y_pre_tr - Y_pre_tr_sc
    pre_c_pes = Y_pre_c - Y_pre_c_sc
    pre_tr_rmspes = np.sqrt(np.mean(pre_tr_pes^2, axis=1))
    pre_c_rmspes = np.sqrt(np.mean(pre_c_pes^2, axis=1))


    control_std_effect_vecs = control_effect_vecs / pre_c_rmspes
    control_joint_std_effect = control_joint_effects / pre_c_rmspes

    effect_vec = np.mean(effect_vecs, 2)
    std_effect_vec = np.mean(effect_vecs / pre_tr_rmspes, 2)
    joint_effect = np.mean(joint_effects)
    joint_std_effect = np.mean(joint_effects / pre_tr_rmspes)

    effect_vec_sgn = np.sign(effect_vec)
    n_pl = _ncr(C, N)
    if (max_n_pl > 0 & n_pl > max_n_pl): #randomize
        comb_iter = itertools.combinations(range(C), N)
        comb_len = max_n_pl
    else:
        comb_iter = repeatfunc(random_combination, n_pl, range(C), N)
        comb_len = n_pl
    n_bigger = 0
    placebo_effect_vecs = None
    if keep_pl:
        placebo_effect_vecs = np.empty((comb_len,T1))
    p1s = np.zero((1,T1))
    p2s = np.zero((1,T1))
    p1s_std = np.zero((1,T1))
    p2s_std = np.zero((1,T1))
    joint_p = 0
    joint_std_p = 0
    for idx, comb in enumerate(comb_iter):
        placebo_effect_vec = np.mean(control_effect_vecs[comb,:], 2)
        placebo_std_effect_vec = np.mean(control_std_effect_vecs[comb,:], 2)
        placebo_joint_effect = np.mean(control_joint_effects[comb,:])
        placebo_joint_std_effect = np.mean(control_joint_std_effects[comb,:])

        p1s += (effect_vec_sgn*placebo_effect_vec >= effect_vec_sgn*effect_vec)
        p2s += (abs(placebo_effect_vec) >= abs(effect_vec))
        p1s_std += (effect_vec_sgn*placebo_std_effect_vec >= effect_vec_sgn*std_effect_vec)
        p2s_std += (abs(placebo_std_effect_vec) >= abs(std_effect_vec))
        joint_p += (placebo_joint_effect >= joint_effect)
        joint_std_p += (placebo_joint_std_effect >= joint_std_effect)
        if keep_pl:
            placebo_effect_vecs[idx,:] = placebo_effect_vec
    p1s = p1s/comb_len
    p2s = p2s/comb_len
    p1s_std = p1s_std/comb_len
    p2s_std = p2s_std/comb_len
    joint_p = joint_p/comb_len
    joint_std_p = joint_std_p/comb_len
    #p2s = 2*p1s #Ficher 2-sided p-vals (less common)
    if ret_CI:
        #CI - All hypothetical true effects (beta0) that would not be reject at the certain level
        # To test non-zero beta0, apply beta0 to get unexpected deviation beta_hat-beta0 and compare to permutation distribution
        # This means that we take the level-bounds of the permutation distribution then "flip it around beta_hat"
        # To make the math a bit nicer, I will reject a hypothesis if pval<=(1-level)
        assert level<=1; "Use a level in [0,1]"
        alpha = (1-level)
        p2min = 2/n_pl
        alpha_ind = max((1,round(alpha/p2min)))
        alpha = alpha_ind* p2min
        CIs = np.empty((2,T1))
        for t in range(T1):
            sorted = sort(placebo_effect_vecs[:,t])
            low_effect = sorted[alpha_ind]
            high_effect = sorted[(n_avgs+1)-alpha_ind]
            if np.sign(low_effect)==np.sign(high_effect):
                warnings.warn("CI doesn't containt effect. You might not have enough placebo effects.")
            CIs[:,t] = (mean_effect[t] - high_effect, mean_effect[t] - low_effect) 
    else:
        CIs = None
    if not ret_p1s:
        p1s = None
        p1s_std = None
    
    RidgeSCEstResults = namedtuple('RidgeSCEstResults', 'effect_vec p2s std_p2s joint_p joint_std_p N_placebo placebo_effect_vecs CIs p1s std_p1s')
    ret_struct = RidgeSCEstResults(effect_vec, p2s, p2s_std, joint_p, joint_std_p, comb_len, placebo_effect_vecs, CIs, p1s, p1s_std)
    return ret_struct

