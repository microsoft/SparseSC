from RidgeSC.fit_fold import  fold_v_matrix, fold_score
from RidgeSC.fit_loo import  loo_v_matrix, loo_score, loo_weights
from RidgeSC.fit_ct import  ct_v_matrix, ct_score
from RidgeSC.optimizers.cd_line_search import cdl_search
from RidgeSC.lambda_utils import get_max_lambda, L2_pen_guestimate
import atexit
import numpy as np
import itertools
from concurrent import futures
import warnings
from collections import namedtuple

def score_train_test(X, 
                     Y,
                     train,
                     test,
                     X_treat=None,
                     Y_treat=None,
                     FoldNumber=None, # For consistency with score_train_test_sorted_lambdas()
                     grad_splits=None, #  If present, use  k fold gradient descent. See fold_v_matrix for details
                     **kwargs):
    """ presents a unified api for ct_v_matrix and loo_v_matrix
        and returns the v_mat, l2_pen_w (possibly calculated, possibly a parameter), and the score 
    """
    # to use `pdb.set_trace()` here, set `parallel = False` above
    if X_treat is None != Y_treat is None:
        raise ValueError("parameters `X_treat` and `Y_treat` must both be Matrices or None")

    if X_treat is not None:
        # >> K-fold validation on the Treated units; assuming that Y and Y_treat are pre-intervention outcomes

        # PARAMETER QC
        if not isinstance(X_treat, np.matrix):
            raise TypeError("X_treat is not a matrix")
        if not isinstance(Y_treat, np.matrix):
            raise TypeError("Y_treat is not a matrix")
        if X_treat.shape[1] == 0:
            raise ValueError("X_treat.shape[1] == 0")
        if Y_treat.shape[1] == 0:
            raise ValueError("Y_treat.shape[1] == 0")
        if X_treat.shape[0] != Y_treat.shape[0]:
            raise ValueError("X_treat and Y_treat have different number of rows (%s and %s)" % 
                             (X.shape[0], Y.shape[0],))

        # FIT THE V-MATRIX AND POSSIBLY CALCULATE THE L2_PEN_W
        # note that the weights, score, and loss function value returned here are for the in-sample predictions
        _, v_mat, _, _, l2_pen_w, _ = \
                    ct_v_matrix(X = np.vstack((X,X_treat[train, :])),
                                Y = np.vstack((Y,Y_treat[train, :])),
                                treated_units = [X.shape[0] + i for i in  range(len(train))],
                                method = cdl_search,
                                **kwargs)

                    
        # GET THE OUT-OF-SAMPLE PREDICTION ERROR
        s = ct_score(X = np.vstack((X,X_treat[test, :])),
                     Y = np.vstack((Y,Y_treat[test, :])), 
                     treated_units = [X.shape[0] + i for i in  range(len(test))],
                     V = v_mat,
                     L2_PEN_W = l2_pen_w)

        
#--         if False:
#--             print("LAMBDA: %0.1f, zeros %s (of %s), Score: %0.1f / %0.1f " % 
#--                   (kwargs["LAMBDA"],
#--                    sum(np.diag(v_mat == 0)),
#--                    v_mat.shape[0],
#--                    s,
#--                    np.power(Y_treat[test, :] - np.mean(Y_treat[test, :]),2).sum() ))

    else: 
        # >> K-fold validation on the only control units; assuming that Y contains post-intervention outcomes 

        if grad_splits is not None:

            try:
                iter(grad_splits)
                # grad_splits may be a generator...
                grad_splits_1, grad_splits_2  = itertools.tee(grad_splits)
            except TypeError:
                # grad_splits is an integer (most common)
                grad_splits_1, grad_splits_2  = (grad_splits,grad_splits)

            # FIT THE V-MATRIX AND POSSIBLY CALCULATE THE L2_PEN_W
            # note that the weights, score, and loss function value returned here are for the in-sample predictions
            _, v_mat, _, _, l2_pen_w, _ = \
                    fold_v_matrix(X = X[train, :],
                                  Y = Y[train, :], 
                                  # treated_units = [X.shape[0] + i for i in  range(len(train))],
                                  method = cdl_search,
                                  grad_splits = grad_splits_1,
                                  **kwargs)

            # GET THE OUT-OF-SAMPLE PREDICTION ERROR (could also use loo_score, actually...)
            s = fold_score(X = X, Y = Y, 
                           treated_units = test,
                           V = v_mat,
                           grad_splits = grad_splits_2,
                           L2_PEN_W = l2_pen_w)
            

        else:

            # FIT THE V-MATRIX AND POSSIBLY CALCULATE THE L2_PEN_W
            # note that the weights, score, and loss function value returned here are for the in-sample predictions
            _, v_mat, _, _, l2_pen_w, _ = \
                    loo_v_matrix(X = X[train, :],
                                 Y = Y[train, :], 
                                 # treated_units = [X.shape[0] + i for i in  range(len(train))],
                                 method = cdl_search,
                                 **kwargs)

            # GET THE OUT-OF-SAMPLE PREDICTION ERROR
            s = loo_score(X = X, Y = Y, 
                          treated_units = test,
                          V = v_mat,
                          L2_PEN_W = l2_pen_w)

    return v_mat, l2_pen_w, s


def score_train_test_sorted_lambdas(LAMBDA,
                                    start=None,
                                    cache=False,
                                    progress=False,
                                    FoldNumber=None,
                                    **kwargs):
    """ a wrapper which calls  score_train_test() for each element of an
        array of `LAMBDA`'s, optionally caching the optimized v_mat and using it
        as the start position for the next iteration.
    """

    # DEFAULTS
    values = [None]*len(LAMBDA)

    if progress > 0:
        import time
        t0 = time.time()

    for i,Lam in enumerate(LAMBDA):
        v_mat, _, _ = values[i] = score_train_test( LAMBDA = Lam, start = start, **kwargs)

        if cache: 
            start = np.diag(v_mat)
        if progress > 0 and (i % progress) == 0:
            t1 = time.time() 
            if FoldNumber is None:
                print("iteration %s of %s time: %0.4f ,lambda: %0.4f" % 
                      (i+1, len(LAMBDA), t1 - t0, Lam,))
                #print("iteration %s of %s time: %0.4f ,lambda: %0.4f, diags: %s" % 
                #      (i+1, len(LAMBDA), t1 - t0, Lam, np.diag(v_mat),))
            else:
                print("Fold %s, iteration %s of %s, time: %0.4f ,lambda: %0.4f" % 
                      (FoldNumber, i+1, len(LAMBDA), t1 - t0, Lam, ))
                #print("Fold %s, iteration %s of %s, time: %0.4f ,lambda: %0.4f, diags: %s" % 
                #      (FoldNumber, i+1, len(LAMBDA), t1 - t0, Lam, np.diag(v_mat),))
            t0 = time.time() 

    return list(zip(*values))


def CV_score(X,Y,
             LAMBDA,
             X_treat=None,
             Y_treat=None,
             splits=5,
             sub_splits=None, # ignore pylint -- this is here for consistency...
             quiet=False,
             parallel=False,
             max_workers=None,
             **kwargs):
    """ Cross fold validation for 1 or more L1 Penalties, holding the L2 penalty fixed. 
    """

    # PARAMETER QC
    if not isinstance(X, np.matrix):
        raise TypeError("X is not a matrix")
    if not isinstance(Y, np.matrix):
        raise TypeError("Y is not a matrix")
    if X_treat is None != Y_treat is None:
        raise ValueError("parameters `X_treat` and `Y_treat` must both be Matrices or None")
    if X.shape[1] == 0:
        raise ValueError("X.shape[1] == 0")
    if Y.shape[1] == 0:
        raise ValueError("Y.shape[1] == 0")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y have different number of rows (%s and %s)" % (X.shape[0], Y.shape[0],))

    try:
        _LAMBDA = iter(LAMBDA)
    except TypeError:
        # Lambda is a single value 
        multi_lambda = False
        __score_train_test__ = score_train_test
    else:
        # Lambda is an iterable of values
        multi_lambda = True
        __score_train_test__ = score_train_test_sorted_lambdas

    if X_treat is not None:

        # PARAMETER QC
        if not isinstance(X_treat, np.matrix):
            raise TypeError("X_treat is not a matrix")
        if not isinstance(Y_treat, np.matrix):
            raise TypeError("Y_treat is not a matrix")
        if X_treat.shape[1] == 0:
            raise ValueError("X_treat.shape[1] == 0")
        if Y_treat.shape[1] == 0:
            raise ValueError("Y_treat.shape[1] == 0")
        if X_treat.shape[0] != Y_treat.shape[0]: 
            raise ValueError("X_treat and Y_treat have different number of rows (%s and %s)" % 
                             (X.shape[0], Y.shape[0],))

        try:
            iter(splits)
        except TypeError: 
            from sklearn.model_selection import KFold
            splits = KFold(splits).split(np.arange(X_treat.shape[0]))
        train_test_splits = list(splits)
        n_splits = len(train_test_splits)

        # MESSAGING
        if not quiet: 
            print("%s-fold validation with %s control and %s treated units %s predictors and %s outcomes, holding out one fold among Treated units; Assumes that `Y` and `Y_treat` are pre-intervention outcomes" % 
                  (n_splits, X.shape[0] , X_treat.shape[0],X.shape[1],Y.shape[1],))

        if parallel: 

            if max_workers is None:
                # CALCULATE A DEFAULT FOR MAX_WORKERS
                import multiprocessing
                multiprocessing.cpu_count()
                if n_splits == 1:
                    print("WARNING: Using Parallel options with a single split is expected reduce performance")
                max_workers = min(max(multiprocessing.cpu_count() - 2,1),len(train_test_splits))
                if max_workers == 1 and n_splits > 1:
                    print("WARNING: Default for max_workers is 1 on a machine with %s cores is 1.")

            _initialize_Global_worker_pool(max_workers)

            try:

                promises = [ _worker_pool.submit(__score_train_test__,
                                                 X = X,
                                                 Y = Y,
                                                 LAMBDA = LAMBDA,
                                                 X_treat = X_treat, 
                                                 Y_treat = Y_treat, 
                                                 train = train,
                                                 test = test,
                                                 FoldNumber = fold,
                                                 **kwargs)
                             for fold, (train,test) in enumerate(train_test_splits) ] 
                results = [ promise.result() for promise in futures.as_completed(promises)]

            finally:

                _clean_up_worker_pool()

        else:

            results = [ __score_train_test__(X = X,
                                             Y = Y,
                                             X_treat = X_treat, 
                                             Y_treat = Y_treat, 
                                             LAMBDA = LAMBDA,
                                             train = train,
                                             test = test,
                                             FoldNumber = fold,
                                             **kwargs)
                        for fold, (train,test) in enumerate(train_test_splits) ] 


    else: # X_treat *is* None

        try:
            iter(splits)
        except TypeError: 
            from sklearn.model_selection import KFold
            splits = KFold(splits).split(np.arange(X.shape[0]))
        train_test_splits = [ x for x in splits ]
        n_splits = len(train_test_splits)

        # MESSAGING
        if not quiet: 
            print("%s-fold Cross Validation with %s control units, %s predictors and %s outcomes; Y may contain post-intervention outcomes" % 
                  (n_splits, X.shape[0],X.shape[1],Y.shape[1],) )

        if parallel: 

            if max_workers is None:
                # CALCULATE A DEFAULT FOR MAX_WORKERS
                import multiprocessing
                multiprocessing.cpu_count()
                if n_splits == 1:
                    print("WARNING: Using Parallel options with a single split is expected reduce performance")
                max_workers = min(max(multiprocessing.cpu_count() - 2,1),len(train_test_splits))
                if max_workers == 1 and n_splits > 1:
                    print("WARNING: Default for max_workers is 1 on a machine with %s cores is 1.")

            _initialize_Global_worker_pool(max_workers)

            try:

                promises = [ _worker_pool.submit(__score_train_test__,
                                                 X = X,
                                                 Y = Y,
                                                 LAMBDA = LAMBDA,
                                                 train = train,
                                                 test = test,
                                                 FoldNumber = fold,
                                                 **kwargs)
                             for fold, (train,test) in enumerate(train_test_splits) ] 

                results = [ promise.result() for promise in futures.as_completed(promises)]

            finally:

                _clean_up_worker_pool()

        else:
            results = [ __score_train_test__(X = X,
                                             Y = Y,
                                             LAMBDA = LAMBDA,
                                             train = train,
                                             test = test,
                                             FoldNumber = fold,
                                             **kwargs)
                        for fold, (train,test) in enumerate(train_test_splits) ] 

    # extract the score.
    _, _, scores = list(zip(* results))

    if multi_lambda:
        total_score = [sum(s) for s in zip(*scores)]
    else:
        total_score = sum(scores)

    return total_score

def joint_penalty_optimzation(X, Y, L1_pen_start = None, L2_pen_start = None, bounds = ((-6,6,),)*2, X_treat = None, Y_treat = None):
    #TODO: Default bounds?
    # -----------------------------------------------------------------
    # Optimization of the L2 and L1 Penalties Simultaneously
    # -----------------------------------------------------------------
    from scipy.optimize import fmin_l_bfgs_b, differential_evolution 
    import time

    if L2_pen_start is None:
        L2_pen_start = L2_pen_guestimate(X)

    L1_pen_start  = get_max_lambda(X,Y,X_treat=X_treat,Y_treat=Y_treat) #TODO: is this right?

    # build the objective function to be minimized
    n_calls = [0,]
    temp_results =[]

    def L1_L2_obj_func (x): 
        n_calls[0] += 1
        t1 = time.time()
        score = CV_score(X = X, Y = Y,
                            X_treat = X_treat, Y_treat = Y_treat,
                            # if LAMBDA is a single value, we get a single score, If it's an array of values, we get an array of scores.
                            LAMBDA = L1_pen_start * np.exp(x[0]),
                            L2_PEN_W = L2_pen_start * np.exp(x[1]),
                            # suppress the analysis type message
                            quiet = True)
        t2 = time.time()
        temp_results.append((n_calls[0],x,score))
        print("calls: %s, time: %0.4f, x0: %0.4f, Cross Validation Error: %s" % (n_calls[0], t2 - t1, x[0], score))
        #print("calls: %s, time: %0.4f, x0: %0.4f, x1: %0.4f, Cross Validation Error: %s, R-Squared: %s" % (n_calls[0], t2 - t1, x[0], x[1], score, 1 - score / SS ))
        return score

    # the actual optimization
    diff_results = differential_evolution(L1_L2_obj_func, bounds = bounds)
    diff_results.x[0] = L1_pen_start * np.exp(diff_results.x[0])
    diff_results.x[1] = L2_pen_start * np.exp(diff_results.x[1])
    return diff_results


def _ncr(n, r):
    #https://stackoverflow.com/questions/4941753/is-there-a-math-ncr-function-in-python
    import operator as op
    import functools
    r = min(r, n-r)
    numer = functools.reduce(op.mul, range(n, n-r, -1), 1) #from py2 xrange()
    denom = functools.reduce(op.mul, range(1, r+1), 1) #from py2 xrange()
    return numer//denom

def random_combination(iterable, r):
    "Random selection from itertools.combinations(iterable, r)"
    #https://stackoverflow.com/questions/22229796/choose-at-random-from-combinations
    import random

    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), r))
    return tuple(pool[i] for i in indices)

def repeatfunc(func, times=None, *args):
    """Repeat calls to func with specified arguments.

    Example:  repeatfunc(random.random)
    """

    if times is None:
        return itertools.starmap(func, itertools.repeat(args))
    return itertools.starmap(func, itertools.repeat(args, times))

def _gen_placebo_stats_from_diffs(N1, effect_vec, std_effect_vec, joint_effect, joint_std_effect,
                                 control_effect_vecs, control_std_effect_vecs, control_joint_effects, control_joint_std_effects,
                                 max_n_pl = 1000000, ret_pl = False, ret_CI=False, level=0.95):
    #ret_p1s=False
    keep_pl = ret_pl or ret_CI
    N0 = control_effect_vecs.shape[0]
    T1 = len(effect_vec)
    n_pl = _ncr(N0, N1)
    if (max_n_pl > 0 & n_pl > max_n_pl): #randomize
        comb_iter = itertools.combinations(range(N0), N1)
        comb_len = max_n_pl
    else:
        comb_iter = repeatfunc(random_combination, n_pl, range(N0), N1)
        comb_len = n_pl
    placebo_effect_vecs = None
    if keep_pl:
        placebo_effect_vecs = np.empty((comb_len,T1))
    p2s = np.zero((1,T1))
    p2s_std = np.zero((1,T1))
    #p1s = np.zero((1,T1))
    #p1s_std = np.zero((1,T1))
    #effect_vec_sgn = np.sign(effect_vec)
    joint_p = 0
    joint_std_p = 0
    for idx, comb in enumerate(comb_iter):
        placebo_effect_vec = np.mean(control_effect_vecs[comb,:], 2)
        placebo_std_effect_vec = np.mean(control_std_effect_vecs[comb,:], 2)
        placebo_joint_effect = np.mean(control_joint_effects[comb,:])
        placebo_joint_std_effect = np.mean(control_joint_std_effects[comb,:])

        p2s += (abs(placebo_effect_vec) >= abs(effect_vec))
        p2s_std += (abs(placebo_std_effect_vec) >= abs(std_effect_vec))
        #p1s += (effect_vec_sgn*placebo_effect_vec >= effect_vec_sgn*effect_vec)
        #p1s_std += (effect_vec_sgn*placebo_std_effect_vec >= effect_vec_sgn*std_effect_vec)
        joint_p += (placebo_joint_effect >= joint_effect)
        joint_std_p += (placebo_joint_std_effect >= joint_std_effect)
        if keep_pl:
            placebo_effect_vecs[idx,:] = placebo_effect_vec
    p2s = p2s/comb_len
    p2s_std = p2s_std/comb_len
    #p1s = p1s/comb_len
    #p1s_std = p1s_std/comb_len
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
            sorted_eff = np.sort(placebo_effect_vecs[:,t]) #TODO: check with Stata about sort order
            low_effect = sorted_eff[alpha_ind]
            high_effect = sorted_eff[(comb_len+1)-alpha_ind]
            if np.sign(low_effect)==np.sign(high_effect):
                warnings.warn("CI doesn't containt effect. You might not have enough placebo effects.")
            CIs[:,t] = (effect_vec[t] - high_effect, effect_vec[t] - low_effect) 
    else:
        CIs = None

    EstResultCI = namedtuple('EstResults', 'effect p ci')
    
    RidgeSCEstResults = namedtuple('RidgeSCEstResults', 'effect_vec_res std_p joint_p joint_std_p N_placebo placebo_effect_vecs')
    ret_struct = RidgeSCEstResults(EstResultCI(effect_vec, p2s, CIs), p2s_std, joint_p, joint_std_p, comb_len, placebo_effect_vecs)
    return ret_struct

def estimate_effects(X, Y_pre, Y_post, treated_units, max_n_pl = 1000000, ret_pl = False, ret_CI=False, level=0.95, **kwargs):
    #TODO: Cleanup returning placebo distribution (incl pre?)
    N1 = len(treated_units)
    X_and_Y_pre = np.hstack( ( X, Y_pre,) )
    N = X_and_Y_pre.shape[0]
    #N0 = N - N1
    #T1 = Y_post.shape[1]
    control_units = list(set(range(N)) - set(treated_units)) 
    all_units = list(range(N))
    Y_post_c = Y_post[control_units, :]
    Y_post_tr = Y_post[treated_units, :]
    X_and_Y_pre_c = X_and_Y_pre[control_units, :]
    
    results = joint_penalty_optimzation(X = X_and_Y_pre_c, Y = Y_post_c, **kwargs)

    best_L1_penalty = results.x[0]
    best_L2_penalty = results.x[1]

    V = loo_v_matrix(X = X_and_Y_pre_c, 
                     Y = Y_post_c,
                     LAMBDA = best_L1_penalty, L2_PEN_W = best_L2_penalty)

    weights = loo_weights(X = X_and_Y_pre,
                          V = V,
                          L2_PEN_W = best_L2_penalty,
                          treated_units = all_units,
                          control_units = control_units)
    Y_post_sc = weights.dot(Y_post_c)
    # Get post effects
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
    control_joint_std_effects = control_joint_effects / pre_c_rmspes

    effect_vec = np.mean(effect_vecs, 2)
    std_effect_vec = np.mean(effect_vecs / pre_tr_rmspes, 2)
    joint_effect = np.mean(joint_effects)
    joint_std_effect = np.mean(joint_effects / pre_tr_rmspes)

    return _gen_placebo_stats_from_diffs(N1, effect_vec, std_effect_vec, joint_effect, joint_std_effect,
                                 control_effect_vecs, control_std_effect_vecs, control_joint_effects, control_joint_std_effects,
                                 max_n_pl, ret_pl, ret_CI, level)

# ------------------------------------------------------------
# utilities for maintaining a worker pool
# ------------------------------------------------------------

_worker_pool = None

def _initialize_Global_worker_pool(n_workers):
    global _worker_pool

    if _worker_pool is not None:
        return # keep it itempotent, please

    _worker_pool = futures.ProcessPoolExecutor(max_workers=n_workers)

def _clean_up_worker_pool():
    global _worker_pool

    if _worker_pool is not None:
        _worker_pool.shutdown()
        _worker_pool = None

atexit.register(_clean_up_worker_pool)
