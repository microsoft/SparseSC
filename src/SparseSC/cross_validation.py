""" 
Implements the cross-fold Validation and parallelization methods
"""

from os.path import join
import atexit
import numpy as np
from concurrent import futures

from SparseSC.fit_fold import fold_v_matrix
from SparseSC.fit_loo import loo_v_matrix
from SparseSC.fit_ct import ct_v_matrix, ct_score


def score_train_test(
    X,
    Y,
    train,
    test,
    X_treat=None,
    Y_treat=None,
    FoldNumber=None,  # pylint: disable=unused-argument
    grad_splits=None,
    progress=None,  # pylint: disable=unused-argument
    **kwargs
):
    """ 
    Presents a unified api for ct_v_matrix and loo_v_matrix
    and returns the v_mat, w_pen (possibly calculated, possibly a parameter), and the score

    :param X: Matrix of covariates for untreated units
    :type X: coercible to :class:`numpy.float64`

    :param Y: Matrix of outcomes for untreated units
    :type Y: coercible to :class:`numpy.float64`

    :param train: List of rows in the current training set
    :type train: int[]

    :param test: LIst of rows in the current test set
    :type test: int[]

    :param X_treat: Optional matrix of covariates for treated units
    :type X_treat: coercible to :class:`numpy.float64`

    :param Y_treat: Optional matrix of outcomes for treated units
    :type Y_treat: ``None`` or coercible to :class:`numpy.float64`

    :param FoldNumber: Unused, for API compatibility only.
    :type FoldNumber: ``None``

    :param grad_splits: Splits for Fitted v.s. Control units in each gradient
                       descent step. An integer, or a list/generator of train
                       and test units in each fold of the gradient descent.
    :type grad_splits: int or int[][], optional

    :param progress: Should progress messages be printed to the console?
    :type progress: boolean

    :param kwargs: additional arguments passed to the underlying matrix method

    :raises ValueError: when X, Y, X_treat, or Y_treat are not coercible to a
       :class:`numpy.float64` or have incompatible dimensions

    :raises RuntimeError: When a MemoryError is raised and grad_splits
        (which reduces memory requirements) is not used.

    :returns: tuple containing the matrix of covariate weights, the unit
        weights penalty, and the out-of-sample score
    :rtype: tuple
    """
    if (X_treat is None) != (Y_treat is None):
        raise ValueError(
            "parameters `X_treat` and `Y_treat` must both be Matrices or None"
        )

    if X_treat is not None:
        # >> K-fold validation on the Treated units; assuming that Y and
        # Y_treat are pre-intervention outcomes

        # PARAMETER QC
        try:
            X = np.float64(X)
        except ValueError:
            raise ValueError("X is not coercible to numpy float64")
        try:
            Y = np.float64(Y)
        except ValueError:
            raise ValueError("Y is not coercible to numpy float64")

        Y = np.asmatrix(Y) # this needs to be deprecated properly -- bc Array.dot(Array) != matrix(Array).dot(matrix(Array)) -- not even close !!!
        X = np.asmatrix(X)

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


def score_train_test_sorted_w_pens(
    w_pen, start=None, cache=False, progress=False, FoldNumber=None, **kwargs
):
    """ a wrapper which calls  score_train_test() for each element of an
        array of `w_pen`'s, optionally caching the optimized v_mat and using it
        as the start position for the next iteration.
    """

    # DEFAULTS
    values = [None] * len(w_pen)

    if progress > 0:
        import time

        t0 = time.time()

    for i, _w_pen in enumerate(w_pen):
        v_mat, _, _ = values[i] = score_train_test(w_pen=_w_pen, start=start, **kwargs)

        if cache:
            start = np.diag(v_mat)
        if progress > 0 and (i % progress) == 0:
            t1 = time.time()
            if FoldNumber is None:
                print(
                    "w_pen: %0.4f, value %s of %s, time elapsed: %0.4f sec."
                    % (_w_pen, i + 1, len(w_pen), t1 - t0)
                )
                # print("iteration %s of %s time: %0.4f ,w_pen: %0.4f, diags: %s" %
                #      (i+1, len(w_pen), t1 - t0, _w_pen, np.diag(v_mat),))
            else:
                print(
                    "Fold %s,w_pen: %0.4f, value %s of %s, time elapsed: %0.4f sec."
                    % (FoldNumber, _w_pen, i + 1, len(w_pen), t1 - t0)
                )
                # print("Fold %s, iteration %s of %s, time: %0.4f ,w_pen: %0.4f, diags: %s" %
                #      (FoldNumber, i+1, len(w_pen), t1 - t0, _w_pen, np.diag(v_mat),))
            t0 = time.time()

    return list(zip(*values))


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

    for i, _v_pen in enumerate(v_pen):
        v_mat, _, _ = values[i] = score_train_test(v_pen=_v_pen, start=start, **kwargs)

        if cache:
            start = np.diag(v_mat)
        if progress > 0 and (i % progress) == 0:
            t1 = time.time()
            if FoldNumber is None:
                print(
                    "v_pen: %0.4f, value %s of %s, time elapsed: %0.4f sec."
                    % (_v_pen, i + 1, len(v_pen), t1 - t0)
                )
                # print("iteration %s of %s time: %0.4f ,v_pen: %0.4f, diags: %s" %
                #      (i+1, len(v_pen), t1 - t0, _v_pen, np.diag(v_mat),))
            else:
                print(
                    "Fold %s,v_pen: %0.4f, value %s of %s, time elapsed: %0.4f sec."
                    % (FoldNumber, _v_pen, i + 1, len(v_pen), t1 - t0)
                )
                # print("Fold %s, iteration %s of %s, time: %0.4f ,v_pen: %0.4f, diags: %s" %
                #      (FoldNumber, i+1, len(v_pen), t1 - t0, _v_pen, np.diag(v_mat),))
            t0 = time.time()

    return list(zip(*values))


def CV_score(
    X,
    Y,
    v_pen,
    w_pen,
    X_treat=None,
    Y_treat=None,
    splits=5,
    # sub_splits=None,
    quiet=False,
    parallel=False,
    batchDir=None,
    max_workers=None,
    cv_seed=110011,
    # this is here for API consistency:
    progress=None,  # pylint: disable=unused-argument
    **kwargs
):
    """ 
    Cross fold validation for 1 or more v Penalties, holding the w penalty fixed.
    """

    # PARAMETER QC
    try:
        X = np.float64(X)
    except ValueError:
        raise ValueError("X is not coercible to float64")
    try:
        Y = np.float64(Y)
    except ValueError:
        raise ValueError("X is not coercible to float64")

    Y = np.asmatrix(Y) # this needs to be deprecated properly -- bc Array.dot(Array) != matrix(Array).dot(matrix(Array)) -- not even close !!!
    X = np.asmatrix(X)

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

    __score_train_test__ = score_train_test
    try:
        iter(w_pen)
    except TypeError:
        w_pen_is_iterable = False
    else:
        w_pen_is_iterable = True
        __score_train_test__ = score_train_test_sorted_w_pens

    try:
        iter(v_pen)
    except TypeError:
        v_pen_is_iterable = False
    else:
        v_pen_is_iterable = True
        __score_train_test__ = score_train_test_sorted_v_pens

    if v_pen_is_iterable and w_pen_is_iterable:
        raise ValueError("v_pen and w_pen must not both be iterable")

    if X_treat is not None:

        # PARAMETER QC
        try:
            X_treat = np.float64(X_treat)
        except ValueError:
            raise ValueError("X_treat is not coercible to float64")
        try:
            Y_treat = np.float64(Y_treat)
        except ValueError:
            raise ValueError("Y_treat is not coercible to float64")

        Y_treat = np.asmatrix(Y_treat) # this needs to be deprecated properly -- bc Array.dot(Array) != matrix(Array).dot(matrix(Array)) -- not even close !!!
        X_treat = np.asmatrix(X_treat)

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

            splits = KFold(splits, shuffle=True, random_state=cv_seed).split(
                np.arange(X_treat.shape[0])
            )
        train_test_splits = list(splits)
        n_splits = len(train_test_splits)

        # MESSAGING
        if not quiet:
            print(
                "%s-fold validation with %s control and %s treated units %s "
                "predictors and %s outcomes, holding out one fold among "
                "Treated units; Assumes that `Y` and `Y_treat` are pre-intervention outcomes"
                % (n_splits, X.shape[0], X_treat.shape[0], X.shape[1], Y.shape[1])
            )

        if batchDir is not None:
            from yaml import load, dump

            try:
                from yaml import CLoader as Loader, CDumper as Dumper
            except ImportError:
                from yaml import Loader, Dumper

            _params = kwargs.copy()
            _params.update(
                {
                    "X": X,
                    "Y": Y,
                    "v_pen": v_pen,
                    "w_pen": w_pen,
                    "X_treat": X_treat,
                    "Y_treat": Y_treat,
                    "folds": train_test_splits,
                }
            )
            with open(join(batchDir,"cv_parameters.yaml"), "w") as fp:
                fp.write(dump(_params, Dumper=Dumper))
            return

        if parallel:

            if max_workers is None:
                # CALCULATE A DEFAULT FOR MAX_WORKERS
                import multiprocessing

                multiprocessing.cpu_count()
                if n_splits == 1:
                    print(
                        "WARNING: Using Parallel options with a single "
                        "split is expected reduce performance"
                    )
                max_workers = min(
                    max(multiprocessing.cpu_count() - 2, 1), len(train_test_splits)
                )
                if max_workers == 1 and n_splits > 1:
                    print(
                        "WARNING: Default for max_workers is 1 on a machine with %s cores is 1."
                    )

            _initialize_Global_worker_pool(max_workers)

            try:

                promises = [
                    _worker_pool.submit(
                        __score_train_test__,
                        X=X,
                        Y=Y,
                        v_pen=v_pen,
                        w_pen=w_pen,
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
                    w_pen=w_pen,
                    train=train,
                    test=test,
                    FoldNumber=fold,
                    progress=progress,
                    **kwargs
                )
                for fold, (train, test) in enumerate(train_test_splits)
            ]

    else:  # X_treat *is* None

        try:
            iter(splits)
        except TypeError:
            from sklearn.model_selection import KFold

            splits = KFold(splits, shuffle=True, random_state=cv_seed).split(
                np.arange(X.shape[0])
            )
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

        if batchDir is not None:
            from yaml import load, dump

            try:
                from yaml import CLoader as Loader, CDumper as Dumper
            except ImportError:
                from yaml import Loader, Dumper

            _params = kwargs.copy()
            _params.update(
                {
                    "X": X,
                    "Y": Y,
                    "v_pen": v_pen,
                    "w_pen": w_pen,
                    "folds": train_test_splits,
                }
            )
            with open(join(batchDir,"cv_parameters.yaml"), "w") as fp:
                fp.write(dump(_params, Dumper=Dumper))
            return

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
                    )

            _initialize_Global_worker_pool(max_workers)

            try:

                promises = [
                    _worker_pool.submit(
                        __score_train_test__,
                        X=X,
                        Y=Y,
                        v_pen=v_pen,
                        w_pen=w_pen,
                        train=train,
                        test=test,
                        FoldNumber=fold,
                        progress=progress,
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
                    w_pen=w_pen,
                    train=train,
                    test=test,
                    FoldNumber=fold,
                    progress=progress,
                    **kwargs
                )
                for fold, (train, test) in enumerate(train_test_splits)
            ]

    # extract the score.
    _, _, scores = list(zip(*results))

    # TODO: np.sqrt(len(scores)) * np.std(scores) is a quick and dirty hack for
    # calculating the standard error of the sum from the partial sums.  It's
    # assumes the samples are equal size and randomly allocated (which is true
    # in the default settings).  However, it could be made more formal with a
    # fixed effects framework, and leveraging the individual errors.
    # https://stats.stackexchange.com/a/271223/67839

    if v_pen_is_iterable or w_pen_is_iterable:
        total_score = [sum(s) for s in zip(*scores)]
        se = [np.sqrt(len(s)) * np.std(s) for s in zip(*scores)]
    else:
        total_score = sum(scores)
        se = np.sqrt(len(scores)) * np.std(scores)

    return total_score, se

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
