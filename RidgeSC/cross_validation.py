
from RidgeSC.fit_fold import  fold_v_matrix, fold_score
from RidgeSC.fit_loo import  loo_v_matrix, loo_score
from RidgeSC.fit_ct import  ct_v_matrix, ct_score
from RidgeSC.optimizers.cd_line_search import cdl_search
import atexit
import numpy as np
import itertools
from concurrent import futures

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

def joint_penalty_optimzation(X, Y, L1_pen_start, L2_pen_start, bounds, X_treat = None, Y_treat = None):
    from scipy.optimize import fmin_l_bfgs_b, differential_evolution 
    import time

    # -----------------------------------------------------------------
    # Optimization of the L2 and L1 Penalties Simultaneously, keeping their
    # product constant.  Heuristically, this has been most efficient means of
    # optimizing the L2 Parameter.
    # -----------------------------------------------------------------


    # build the objective function to be minimized

    # cache for L2_obj_func
    n_calls = [0,]
    temp_results =[]

    def L1_L2_obj_func (x): 
        n_calls[0] += 1
        t1 = time.time();
        score = CV_score(X = X, Y = Y,
                            X_treat = X_treat, Y_treat = Y_treat,
                            # if LAMBDA is a single value, we get a single score, If it's an array of values, we get an array of scores.
                            LAMBDA = L1_pen_start * np.exp(x[0]),
                            L2_PEN_W = L2_pen_start * np.exp(x[1]),
                            # suppress the analysis type message
                            quiet = True)
        t2 = time.time(); 
        temp_results.append((n_calls[0],x,score))
        print("calls: %s, time: %0.4f, x0: %0.4f, Cross Validation Error: %s" % (n_calls[0], t2 - t1, x[0], score))
        #print("calls: %s, time: %0.4f, x0: %0.4f, x1: %0.4f, Cross Validation Error: %s, R-Squared: %s" % (n_calls[0], t2 - t1, x[0], x[1], score, 1 - score / SS ))
        return score

    # the actual optimization
    diff_results = differential_evolution(L1_L2_obj_func, bounds = bounds)
    diff_results.x[0] = L1_pen_start * np.exp(diff_results.x[0])
    diff_results.x[1] = L2_pen_start * np.exp(diff_results.x[1])
    return diff_results

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
