"""
aggregate batch results, and optionally use batch to compute the final gradient descent.
"""
import numpy as np
import os
#$ from ...cross_validation import _score_from_batch
from ...fit import _which, SparseSCFit
from ...weights import weights
from ...tensor import tensor
from . import _BATCH_CV_FILE_NAME, _BATCH_FIT_FILE_NAME

def aggregate_batch_results(batchDir,batch_client_config=None, choice=None):
    """
    Aggregate results from a batch run 
    """

    from yaml import load

    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader

    with open(os.path.join(batchDir, _BATCH_CV_FILE_NAME), "r") as fp:
        _cv_params = load(fp, Loader=Loader)

    with open(os.path.join(batchDir, _BATCH_FIT_FILE_NAME), "r") as fp:
        _fit_params = load(fp, Loader=Loader)

    # https://stackoverflow.com/a/17074606/1519199
    pluck = lambda d, *args: (d[arg] for arg in args)

    X_cv, Y_cv, grad_splits, random_state, v_pen, w_pen = pluck(
        _cv_params, "X", "Y", "grad_splits", "random_state", "v_pen", "w_pen"
    )

    choice = choice if choice is not None else _fit_params["choice"]
    X, Y, treated_units, custom_donor_pool, model_type, kwargs = pluck(
        _fit_params, "X", "Y", "treated_units", "custom_donor_pool", "model_type" , "kwargs"
    )

    # this is on purpose (allows for debugging remote sessions at no cost to the local console user)
    kwargs["print_path"] = 1

    scores, scores_se = _score_from_batch(batchDir, _cv_params)

    try:
        iter(w_pen)
    except TypeError:
        w_pen_is_iterable = False
    else:
        w_pen_is_iterable = True

    try:
        iter(v_pen)
    except TypeError:
        v_pen_is_iterable = False
    else:
        v_pen_is_iterable = True

    # GET THE INDEX OF THE BEST SCORE
    def _choose(scores, scores_se):
        """ helper function which implements the choice of covariate weights penalty parameter

        Nested here for access to  v_pen, w_pe,n w_pen_is_iterable and
        v_pen_is_iterable, and choice, via Lexical Scoping
        """
        # GET THE INDEX OF THE BEST SCORE
        if w_pen_is_iterable:
            indx = _which(scores, scores_se, choice)
            return v_pen, w_pen[indx], scores[indx], indx
        if v_pen_is_iterable:
            indx = _which(scores, scores_se, choice)
            return v_pen[indx], w_pen, scores[indx], indx
        return v_pen, w_pen, scores, None

    best_v_pen, best_w_pen, score, which = _choose(scores, scores_se)

    # --------------------------------------------------
    # Phase 2: extract V and weights: slow ( tens of seconds to minutes )
    # --------------------------------------------------
    if treated_units is not None:
        control_units = [u for u in range(Y.shape[0]) if u not in treated_units]
        Xtrain = X[control_units, :]
        Xtest = X[treated_units, :]
        Ytrain = Y[control_units, :]
        Ytest = Y[treated_units, :]


    if model_type == "prospective-restricted":
        best_V = tensor(
            X=X_cv,
            Y=Y_cv,
            w_pen=best_w_pen,
            v_pen=best_v_pen,
            #
            X_treat=Xtest,
            Y_treat=Ytest,
            #
            batch_client_config= batch_client_config, # TODO: not sure if this makes sense...
            **_fit_params["kwargs"]
        )
    else:
        best_V = tensor(
            X=X_cv,
            Y=Y_cv,
            w_pen=best_w_pen,
            v_pen=best_v_pen,
            #
            grad_splits=grad_splits,
            random_state=random_state,
            #
            batch_client_config= batch_client_config,
            **_fit_params["kwargs"]
        )

    if treated_units is not None:

        # GET THE BEST SET OF WEIGHTS
        sc_weights = np.empty((X.shape[0], Ytrain.shape[0]))
        if custom_donor_pool is None:
            custom_donor_pool_t = None
            custom_donor_pool_c = None
        else:
            custom_donor_pool_t = custom_donor_pool[treated_units, :]
            custom_donor_pool_c = custom_donor_pool[control_units, :]
        sc_weights[treated_units, :] = weights(
            Xtrain,
            Xtest,
            V=best_V,
            w_pen=best_w_pen,
            custom_donor_pool=custom_donor_pool_t,
        )
        sc_weights[control_units, :] = weights(
            Xtrain, V=best_V, w_pen=best_w_pen, custom_donor_pool=custom_donor_pool_c
        )

    else:
        # GET THE BEST SET OF WEIGHTS
        sc_weights = weights(
            X, V=best_V, w_pen=best_w_pen, custom_donor_pool=custom_donor_pool
        )

    return SparseSCFit(
        X=X,
        Y=Y,
        control_units=control_units,
        treated_units=treated_units,
        model_type=model_type,
        # fitting parameters
        fitted_v_pen=best_v_pen,
        fitted_w_pen=best_w_pen,
        initial_w_pen=w_pen,
        initial_v_pen=v_pen,
        V=best_V,
        # Fitted Synthetic Controls
        sc_weights=sc_weights,
        score=score,
        scores=scores,
        selected_score=which,
    )




def _score_from_batch(batchDir, config):
    """
    read in the results from a batch run
    """
    from yaml import load

    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader

    try:
        v_pen = tuple(config["v_pen"])
    except TypeError:
        v_pen = (config["v_pen"],)

    try:
        w_pen = tuple(config["w_pen"])
    except TypeError:
        w_pen = (config["w_pen"],)

    n_folds = len(config["folds"]) * len(v_pen) * len(w_pen)
    n_pens = np.max((len(v_pen), len(w_pen)))
    n_cv_folds = n_folds // n_pens

    scores = np.empty((n_pens, n_cv_folds))
    for i in range(n_folds):
        # i_fold, i_v, i_w = pluck(res, "i_fold", "i_v", "i_w", )
        i_fold = i % len(config["folds"])
        i_pen = i // len(config["folds"])
        with open(os.path.join(batchDir, "fold_{}.yaml".format(i)), "r") as fp:
            res = load(fp, Loader=Loader)
            assert (
                res["batch"] == i
            ), "Batch File Import Error Inconsistent batch identifiers"
            scores[i_pen, i_fold] = res["results"][2]

    # TODO: np.sqrt(len(scores)) * np.std(scores) is a quick and dirty hack for
    # calculating the standard error of the sum from the partial sums.  It's
    # assumes the samples are equal size and randomly allocated (which is true
    # in the default settings).  However, it could be made more formal with a
    # fixed effects framework, and leveraging the individual errors.
    # https://stats.stackexchange.com/a/271223/67839

    if len(v_pen) > 0 or len(w_pen):
        n_pens = np.max((len(v_pen), len(w_pen)))
        n_cv_folds = n_folds // n_pens
        total_score = scores.sum(axis=1)
        se = np.sqrt(n_cv_folds) * scores.std(axis=1)
    else:
        total_score = sum(scores)
        se = np.sqrt(len(scores)) * np.std(scores)

    return total_score, se




