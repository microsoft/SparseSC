""" 
Implements the cross-fold Validation and parallelization methods
"""

from SparseSC.fit_fold import fold_v_matrix
from SparseSC.fit_loo import loo_v_matrix
from SparseSC.fit_ct import ct_v_matrix, ct_score
import atexit
import numpy as np
from concurrent import futures


def score_train_test(
    X,
    Y,
    train,
    test,
    X_treat=None,
    Y_treat=None,
    FoldNumber=None,  # pylint: disable=unused-argument
    grad_splits=None,
    **kwargs
):
    """ 
    Presents a unified api for ct_v_matrix and loo_v_matrix
    and returns the v_mat, w_pen (possibly calculated, possibly a parameter), and the score

    :param X: Matrix of covariates for untreated units
    :type X: coercible to :class:`numpy.matrix`

    :param Y: Matrix of outcomes for untreated units
    :type Y: coercible to :class:`numpy.matrix`

    :param train: List of rows in the current training set
    :type train: int[]

    :param test: LIst of rows in the current test set
    :type test: int[]

    :param X_treat: Optional matrix of covariates for treated units
    :type X_treat: coercible to :class:`numpy.matrix`

    :param Y_treat: Optional matrix of outcomes for treated units
    :type Y_treat: ``None`` or coercible to :class:`numpy.matrix`

    :param FoldNumber: Unused, for API compatibility only.
    :type FoldNumber: ``None``

    :param grad_splits: Splits for Fitted v.s. Control units in each gradient
                       descent step. An integer, or a list/generator of train
                       and test units in each fold of the gradient descent.
    :type grad_splits: int or int[][], optional

    :param kwargs: additional arguments passed to the underlying matrix method

    :raises ValueError: when X, Y, X_treat, or Y_treat are not coercible to a
       :class:`numpy.matrix` or have incompatible dimensions

    :raises RuntimeError: When a MemoryError is raised and grad_splits
        (which reduces memory requirements) is not used.

    :returns: tuple containing the matrix of covariate weights, the unit
        weights penalty, and the out-of-sample score
    :rtype: tuple
    """
    # to use `pdb.set_trace()` here, set `parallel = False` above
    if (X_treat is None) != (Y_treat is None):
        raise ValueError(
            "parameters `X_treat` and `Y_treat` must both be Matrices or None"
        )

    if X_treat is not None:
        # >> K-fold validation on the Treated units; assuming that Y and
        # Y_treat are pre-intervention outcomes

        # PARAMETER QC
        try:
            X = np.asmatrix(X)
        except ValueError:
            raise ValueError("X is not coercible to a matrix")
        try:
            Y = np.asmatrix(Y)
        except ValueError:
            raise ValueError("Y is not coercible to a matrix")
        if X_treat.shape[1] == 0:
            raise ValueError("X_treat.shape[1] == 0")
        if Y_treat.shape[1] == 0:
            raise ValueError("Y_treat.shape[1] == 0")
        if X_treat.shape[0] != Y_treat.shape[0]:
            raise ValueError(
                "X_treat and Y_treat have different number of rows (%s and %s)"
                % (X.shape[0], Y.shape[0])
            )

        # FIT THE V-MATRIX AND POSSIBLY CALCULATE THE w_pen
        # note that the weights, score, and loss function value returned here
        # are for the in-sample predictions
        _, v_mat, _, _, w_pen, _ = ct_v_matrix(
            X=np.vstack((X, X_treat[train, :])),
            Y=np.vstack((Y, Y_treat[train, :])),
            treated_units=[X.shape[0] + i for i in range(len(train))],
            **kwargs
        )

        # GET THE OUT-OF-SAMPLE PREDICTION ERROR
        s = ct_score(
            X=np.vstack((X, X_treat[test, :])),
            Y=np.vstack((Y, Y_treat[test, :])),
            treated_units=[X.shape[0] + i for i in range(len(test))],
            V=v_mat,
            w_pen=w_pen,
        )

    else:  # X_treat *is* None
        # >> K-fold validation on the only control units; assuming that Y
        # contains post-intervention outcomes
        if grad_splits is not None:

            try:
                iter(grad_splits)
            except TypeError:
                # not iterable
                pass
            else:
                # TRIM THE GRAD SPLITS NEED TO THE TRAINING SET

                # inspired by R's match() function
                match = lambda a, b: np.concatenate([np.where(a == x)[0] for x in b])

                grad_splits = [
                    (match(train, _X), match(train, _Y)) for _X, _Y in grad_splits
                ]

            # FIT THE V-MATRIX AND POSSIBLY CALCULATE THE w_pen
            # note that the weights, score, and loss function value returned
            # here are for the in-sample predictions
            _, v_mat, _, _, w_pen, _ = fold_v_matrix(
                X=X[train, :],
                Y=Y[train, :],
                # treated_units = [X.shape[0] + i for i in  range(len(train))],
                grad_splits=grad_splits,
                **kwargs
            )

            # GET THE OUT-OF-SAMPLE PREDICTION ERROR (could also use loo_score, actually...)
            s = ct_score(
                X=X,
                Y=Y,  # formerly: fold_score
                treated_units=test,
                V=v_mat,
                w_pen=w_pen,
            )

        else:

            # FIT THE V-MATRIX AND POSSIBLY CALCULATE THE w_pen
            # note that the weights, score, and loss function value returned
            # here are for the in-sample predictions
            try:
                _, v_mat, _, _, w_pen, _ = loo_v_matrix(
                    X=X[train, :],
                    Y=Y[train, :],
                    # treated_units = [X.shape[0] + i for i in  range(len(train))],
                    **kwargs
                )
            except MemoryError:
                raise RuntimeError(
                    "MemoryError encountered.  Try setting `grad_splits` "
                    + "parameter to reduce memory requirements."
                )

            # GET THE OUT-OF-SAMPLE PREDICTION ERROR
            s = ct_score(X=X, Y=Y, treated_units=test, V=v_mat, w_pen=w_pen)

    return v_mat, w_pen, s


def score_train_test_sorted_v_pens(
    v_pen, start=None, cache=False, progress=False, FoldNumber=None, **kwargs
):
    """ a wrapper which calls  score_train_test() for each element of an
        array of `v_pen`'s, optionally caching the optimized v_mat and using it
        as the start position for the next iteration.
    """

    # DEFAULTS
    values = [None] * len(v_pen)

    if progress > 0:
        import time

        t0 = time.time()

    for i, Lam in enumerate(v_pen):
        v_mat, _, _ = values[i] = score_train_test(v_pen=Lam, start=start, **kwargs)

        if cache:
            start = np.diag(v_mat)
        if progress > 0 and (i % progress) == 0:
            t1 = time.time()
            if FoldNumber is None:
                print(
                    "v_pen: %0.4f, value %s of %s, time elapsed: %0.4f sec."
                    % (Lam, i + 1, len(v_pen), t1 - t0)
                )
                # print("iteration %s of %s time: %0.4f ,v_pen: %0.4f, diags: %s" %
                #      (i+1, len(v_pen), t1 - t0, Lam, np.diag(v_mat),))
            else:
                print(
                    "Fold %s,v_pen: %0.4f, value %s of %s, time elapsed: %0.4f sec."
                    % (FoldNumber, Lam, i + 1, len(v_pen), t1 - t0)
                )
                # print("Fold %s, iteration %s of %s, time: %0.4f ,v_pen: %0.4f, diags: %s" %
                #      (FoldNumber, i+1, len(v_pen), t1 - t0, Lam, np.diag(v_mat),))
            t0 = time.time()

    return list(zip(*values))


def CV_score(
    X,
    Y,
    v_pen,
    X_treat=None,
    Y_treat=None,
    splits=5,
    # sub_splits=None, # ignore pylint -- this is here for consistency...
    quiet=False,
    parallel=False,
    max_workers=None,
    progress=None,
    **kwargs
):
    """ 
    Cross fold validation for 1 or more v Penalties, holding the w penalty fixed.
    """

    # PARAMETER QC
    try:
        X = np.asmatrix(X)
    except ValueError:
        raise ValueError("X is not coercible to a matrix")
    try:
        Y = np.asmatrix(Y)
    except ValueError:
        raise ValueError("X is not coercible to a matrix")
    if (X_treat is None) != (Y_treat is None):
        raise ValueError(
            "parameters `X_treat` and `Y_treat` must both be Matrices or None"
        )
    if X.shape[1] == 0:
        raise ValueError("X.shape[1] == 0")
    if Y.shape[1] == 0:
        raise ValueError("Y.shape[1] == 0")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            "X and Y have different number of rows (%s and %s)"
            % (X.shape[0], Y.shape[0])
        )

    try:
        _v_pen = iter(v_pen)
    except TypeError:
        # v_pen is a single value
        multi_v_pen = False
        __score_train_test__ = score_train_test
    else:
        # v_pen is an iterable of values
        multi_v_pen = True
        __score_train_test__ = score_train_test_sorted_v_pens

    if X_treat is not None:

        # PARAMETER QC
        try:
            X_treat = np.asmatrix(X_treat)
        except ValueError:
            raise ValueError("X_treat is not coercible to a matrix")
        try:
            Y_treat = np.asmatrix(Y_treat)
        except ValueError:
            raise ValueError("Y_treat is not coercible to a matrix")
        if X_treat.shape[1] == 0:
            raise ValueError("X_treat.shape[1] == 0")
        if Y_treat.shape[1] == 0:
            raise ValueError("Y_treat.shape[1] == 0")
        if X_treat.shape[0] != Y_treat.shape[0]:
            raise ValueError(
                "X_treat and Y_treat have different number of rows (%s and %s)"
                % (X_treat.shape[0], Y_treat.shape[0])
            )

        try:
            iter(splits)
        except TypeError:
            from sklearn.model_selection import KFold

            splits = KFold(splits, True).split(np.arange(X_treat.shape[0]))
        train_test_splits = list(splits)
        n_splits = len(train_test_splits)

        # MESSAGING
        if not quiet:
            print(
                "%s-fold validation with %s control and %s treated units %s "
                "predictors and %s outcomes, holding out one fold among "
                "Treated units; Assumes that `Y` and `Y_treat` are pre-intervention outcomes"
                % (  # pylint: disable=line-too-long
                    n_splits,
                    X.shape[0],
                    X_treat.shape[0],
                    X.shape[1],
                    Y.shape[1],
                )
            )

        if parallel:

            if max_workers is None:
                # CALCULATE A DEFAULT FOR MAX_WORKERS
                import multiprocessing

                multiprocessing.cpu_count()
                if n_splits == 1:
                    print(
                        "WARNING: Using Parallel options with a single "
                        "split is expected reduce performance"
                    )  # pylint: disable=line-too-long
                max_workers = min(
                    max(multiprocessing.cpu_count() - 2, 1), len(train_test_splits)
                )
                if max_workers == 1 and n_splits > 1:
                    print(
                        "WARNING: Default for max_workers is 1 on a machine with %s cores is 1."
                    )  # pylint: disable=line-too-long

            _initialize_Global_worker_pool(max_workers)

            try:

                promises = [
                    _worker_pool.submit(
                        __score_train_test__,
                        X=X,
                        Y=Y,
                        v_pen=v_pen,
                        X_treat=X_treat,
                        Y_treat=Y_treat,
                        train=train,
                        test=test,
                        FoldNumber=fold,
                        **kwargs
                    )
                    for fold, (train, test) in enumerate(train_test_splits)
                ]
                results = [
                    promise.result() for promise in futures.as_completed(promises)
                ]

            finally:

                _clean_up_worker_pool()

        else:

            results = [
                __score_train_test__(
                    X=X,
                    Y=Y,
                    X_treat=X_treat,
                    Y_treat=Y_treat,
                    v_pen=v_pen,
                    train=train,
                    test=test,
                    FoldNumber=fold,
                    **kwargs
                )
                for fold, (train, test) in enumerate(train_test_splits)
            ]

    else:  # X_treat *is* None

        try:
            iter(splits)
        except TypeError:
            from sklearn.model_selection import KFold

            splits = KFold(splits).split(np.arange(X.shape[0]))
        train_test_splits = [x for x in splits]
        n_splits = len(train_test_splits)

        # MESSAGING
        if not quiet:
            print(
                "%s-fold Cross Validation with %s control units, "
                "%s predictors and %s outcomes; Y may contain "
                "post-intervention outcomes"
                % (n_splits, X.shape[0], X.shape[1], Y.shape[1])
            )

        if parallel:

            if max_workers is None:
                # CALCULATE A DEFAULT FOR MAX_WORKERS
                import multiprocessing

                multiprocessing.cpu_count()
                if n_splits == 1:
                    print(
                        "WARNING: Using Parallel options with a "
                        "single split is expected reduce performance"
                    )  # pylint: disable=line-too-long
                max_workers = min(
                    max(multiprocessing.cpu_count() - 2, 1), len(train_test_splits)
                )
                if max_workers == 1 and n_splits > 1:
                    print(
                        "WARNING: Default for max_workers is 1 on a machine with %s cores is 1."
                    )  # pylint: disable=line-too-long

            _initialize_Global_worker_pool(max_workers)

            try:

                promises = [
                    _worker_pool.submit(
                        __score_train_test__,
                        X=X,
                        Y=Y,
                        v_pen=v_pen,
                        train=train,
                        test=test,
                        FoldNumber=fold,
                        **kwargs
                    )
                    for fold, (train, test) in enumerate(train_test_splits)
                ]

                results = [
                    promise.result() for promise in futures.as_completed(promises)
                ]

            finally:

                _clean_up_worker_pool()

        else:
            results = [
                __score_train_test__(
                    X=X,
                    Y=Y,
                    v_pen=v_pen,
                    train=train,
                    test=test,
                    FoldNumber=fold,
                    **kwargs
                )
                for fold, (train, test) in enumerate(train_test_splits)
            ]

    # extract the score.
    _, _, scores = list(zip(*results))

    if multi_v_pen:
        total_score = [sum(s) for s in zip(*scores)]
    else:
        total_score = sum(scores)

    return total_score


# ------------------------------------------------------------
# utilities for maintaining a worker pool
# ------------------------------------------------------------

_worker_pool = None


def _initialize_Global_worker_pool(n_workers):
    global _worker_pool  # pylint: disable=global-statement

    if _worker_pool is not None:
        return  # keep it itempotent, please

    _worker_pool = futures.ProcessPoolExecutor(max_workers=n_workers)


def _clean_up_worker_pool():
    global _worker_pool  # pylint: disable=global-statement
    if _worker_pool is not None:
        _worker_pool.shutdown()
        _worker_pool = None


atexit.register(_clean_up_worker_pool)
