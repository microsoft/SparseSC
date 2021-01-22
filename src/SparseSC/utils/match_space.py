""" Utils for getting match spaces (Matching features and potentially feature wegihts)

"""
# To do:
# - Hve the factories return partial objects rather than anonymous functions. That way they can be pickled and parallelized in get_c_predictions_honest
# - Implement Post-lasso versions. Could do MT OLS as fully separate then aggregate coefs like with Lasso.
#   (Though some coefficients aren't well estimated we don't want to just take t-stats as we still want to be fit-based.
#   Ideally we'd have something like marginal R2, but the initial method is probably fine for most uses. (We could standardize input features).)
import numpy as np
from sklearn.linear_model import MultiTaskLassoCV, MultiTaskLasso, LassoCV
from .misc import capture_all


def keras_reproducible(seed=1234, verbose=0, TF_CPP_MIN_LOG_LEVEL="3"):
    """
    https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
    """
    import random
    import pkg_resources
    import os

    random.seed(seed)
    np.random.seed(seed)

    os.environ["PYTHONHASHSEED"] = "0"  # might need to do this outside the script

    if verbose == 0:
        os.environ[
            "TF_CPP_MIN_LOG_LEVEL"
        ] = TF_CPP_MIN_LOG_LEVEL  # 2 will print warnings

    try:
        import tensorflow
    except ImportError:
        raise ImportError("Missing required package 'tensorflow'")

    # Use the TF 1.x API
    if pkg_resources.get_distribution("tensorflow").version.startswith("1."):
        tf = tensorflow
    else:
        tf = tensorflow.compat.v1

    if verbose == 0:
        # https://github.com/tensorflow/tensorflow/issues/27023
        try:
            from tensorflow.python.util import deprecation

            deprecation._PRINT_DEPRECATION_WARNINGS = False
        except ImportError:
            try:
                from tensorflow.python.util import module_wrapper as deprecation
            except ImportError:
                from tensorflow.python.util import deprecation_wrapper as deprecation
            deprecation._PER_MODULE_WARNING_LIMIT = 0

        # this was deprecated in 1.15 (maybe earlier)
        tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR)

    ConfigProto = tf.ConfigProto

    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
    )

    with capture_all():  # doesn't have quiet option
        try:
            from tensorflow.python.keras import backend as K
        except ImportError:
            raise ImportError("Missing required module 'keras'")

    tf.set_random_seed(seed)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)


def Fixed_V_factory(V):
    """
    Return a MatchSpace function with user-supplied V over raw X.

    :param V: V Matrix on the raw features
    :returns: a function with the signature 
        MatchSpace fn, V vector, best_v_pen, V = function(X,Y)
    """

    def _Fixed_V_MatchSpace_wrapper(X, Y, **kwargs):
        return _Fixed_V_MatchSpace(X, Y, V=V, **kwargs)

    return _Fixed_V_MatchSpace_wrapper


def _Fixed_V_MatchSpace(X, Y, V, **kwargs):  # pylint: disable=unused-argument
    return IdTransformer(), V, np.nan, V


class IdTransformer:
    def transform(self, X):
        return X

def _neg_se_rule(lasso_fit=None, mse_path=None, alphas=None, alpha_min=None, factor = 1):
    from statsmodels.stats.weightstats import DescrStatsW
    if lasso_fit is not None:
        mse_path = lasso_fit.mse_path_
        alphas = lasso_fit.alphas_
        alpha_min = lasso_fit.alpha_
    alpha_min_i = np.where(alphas == alpha_min)
    dw = DescrStatsW(mse_path[alpha_min_i,:].T)
    mse_mean = mse_path.mean(axis=1)
    allowed = mse_mean<=(mse_mean[alpha_min_i] + factor*dw.std_mean[0])
    new_alpha_i = max(np.where(allowed)[0])
    return alphas[new_alpha_i]

def _block_summ_cols(Y, Y_col_block_size):
    """Block averages a Y np.array. E.g., convert 150->5 where each is a 30-day average.  so that MTLasso could be faster
    """
    if Y_col_block_size is not None:
        if (Y.shape[1] % Y_col_block_size) == 0:
            # Can just change the dim on the ndarray (which doesn't shuffle data) to add a new dim and then average across it.
            return Y.reshape(Y.shape[0], Y.shape[1]//Y_col_block_size, Y_col_block_size).mean(axis=2)
        print("Can only average target across columns blocks if blocks fit evenly")
    return Y

def MTLassoCV_MatchSpace_factory(v_pens=None, n_v_cv=5, sample_frac=1, Y_col_block_size=None, se_factor=None, normalize=True):
    """
    Return a MatchSpace function that will fit a MultiTaskLassoCV for Y ~ X

    :param v_pens: Penalties to evaluate (default is to automatically determince)
    :param n_v_cv: Number of Cross-Validation folds
    :param sample_frac: Fraction of the data to sample
    :param se_factor: Allows taking a different penalty than the min mse. Similar to the lambda.1se rule,
        if not None, it will take the max lambda that has mse < min_mse + se_factor*(MSE standard error).
    :returns: MatchSpace fn, V vector, best_v_pen, V
    """

    def _MTLassoCV_MatchSpace_wrapper(X, Y, **kwargs):
        return _MTLassoCV_MatchSpace(
            X, Y, v_pens=v_pens, n_v_cv=n_v_cv, sample_frac=sample_frac, Y_col_block_size=Y_col_block_size, se_factor=se_factor, normalize=normalize, **kwargs
        )

    return _MTLassoCV_MatchSpace_wrapper


def _MTLassoCV_MatchSpace(
    X, Y, v_pens=None, n_v_cv=5, sample_frac=1, Y_col_block_size=None, se_factor=None, normalize=True, **kwargs
):  # pylint: disable=missing-param-doc, unused-argument
    # A fake MT would do Lasso on y_mean = Y.mean(axis=1)
    if sample_frac < 1:
        N = X.shape[0]
        sample = np.random.choice(N, int(sample_frac * N), replace=False)
        X = X[sample, :]
        Y = Y[sample, :]
    if Y_col_block_size is not None:
        Y = _block_summ_cols(Y, Y_col_block_size)
    varselectorfit = MultiTaskLassoCV(normalize=normalize, cv=n_v_cv, alphas=v_pens).fit(
        X, Y
    )
    best_v_pen = varselectorfit.alpha_
    if se_factor is not None:
        best_v_pen = _neg_se_rule(varselectorfit, factor=se_factor)
        varselectorfit = MultiTaskLasso(alpha=best_v_pen, normalize=normalize).fit(X, Y)
    V = np.sqrt(
        np.sum(np.square(varselectorfit.coef_), axis=0)
    )  # n_tasks x n_features -> n_feature
    m_sel = V != 0
    transformer = SelMatchSpace(m_sel)
    return transformer, V[m_sel], best_v_pen, (V, varselectorfit)

def MTLasso_MatchSpace_factory(v_pen, sample_frac=1, Y_col_block_size=None, se_factor=None, normalize=True):
    """
    Return a MatchSpace function that will fit a MultiTaskLasso for Y ~ X

    :param v_pen: Penalty
    :param sample_frac: Fraction of the data to sample
    :param se_factor: Allows taking a different penalty than the min mse. Similar to the lambda.1se rule,
        if not None, it will take the max lambda that has mse < min_mse + se_factor*(MSE standard error).
    :returns: MatchSpace fn, V vector, best_v_pen, V
    """

    def _MTLasso_MatchSpace_wrapper(X, Y, **kwargs):
        return _MTLasso_MatchSpace(
            X, Y, v_pen=v_pen, sample_frac=sample_frac, Y_col_block_size=Y_col_block_size, se_factor=se_factor, normalize=normalize, **kwargs
        )

    return _MTLasso_MatchSpace_wrapper


def _MTLasso_MatchSpace(
    X, Y, v_pen, sample_frac=1, Y_col_block_size=None, se_factor=None, normalize=True, **kwargs
):  # pylint: disable=missing-param-doc, unused-argument
    # A fake MT would do Lasso on y_mean = Y.mean(axis=1)
    if sample_frac < 1:
        N = X.shape[0]
        sample = np.random.choice(N, int(sample_frac * N), replace=False)
        X = X[sample, :]
        Y = Y[sample, :]
    if Y_col_block_size is not None:
        Y = _block_summ_cols(Y, Y_col_block_size)
    varselectorfit = MultiTaskLasso(normalize=normalize, alpha=v_pen).fit(
        X, Y
    )
    V = np.sqrt(
        np.sum(np.square(varselectorfit.coef_), axis=0)
    )  # n_tasks x n_features -> n_feature
    m_sel = V != 0
    transformer = SelMatchSpace(m_sel)
    return transformer, V[m_sel], v_pen, (V, varselectorfit)

def D_LassoCV_MatchSpace_factory(v_pens=None, n_v_cv=5, sample_frac=1, y_V_share=0.5):
    """
    Return a MatchSpace function that will fit a MultiTaskLassoCV for Y ~ X and Lasso of D_full ~ X_full
    and then combines the coefficients into weights using y_V_share

    :param v_pens: Penalties to evaluate (default is to automatically determince)
    :param n_v_cv: Number of Cross-Validation folds
    :param sample_frac: Fraction of the data to sample
    :param y_V_share: The fraction of the V weight that goes to the variables weights from the Y~X problem.
    :returns: MatchSpace fn, V vector, best_v_pen, V
    """

    def _D_LassoCV_MatchSpace_wrapper(X, Y, **kwargs):
        return _D_LassoCV_MatchSpace(
            X,
            Y,
            v_pens=v_pens,
            n_v_cv=n_v_cv,
            sample_frac=sample_frac,
            y_V_share=y_V_share,
            **kwargs
        )

    return _D_LassoCV_MatchSpace_wrapper


def _D_LassoCV_MatchSpace(
    X, Y, X_full, D_full, v_pens=None, n_v_cv=5, sample_frac=1, y_V_share=0.5, **kwargs
):  # pylint: disable=missing-param-doc, unused-argument
    if sample_frac < 1:
        N_y = X.shape[0]
        sample_y = np.random.choice(N_y, int(sample_frac * N_y), replace=False)
        X = X[sample_y, :]
        Y = Y[sample_y, :]
        N_d = D_full.shape[0]
        sample_d = np.random.choice(N_d, int(sample_frac * N_d), replace=False)
        X_full = X_full[sample_d, :]
        D_full = D_full[sample_d]
    y_varselectorfit = MultiTaskLassoCV(normalize=True, cv=n_v_cv, alphas=v_pens).fit(
        X, Y
    )
    y_V = np.sqrt(
        np.sum(np.square(y_varselectorfit.coef_), axis=0)
    )  # n_tasks x n_features -> n_feature
    best_y_v_pen = y_varselectorfit.alpha_

    d_varselectorfit = LassoCV(normalize=True, cv=n_v_cv, alphas=v_pens).fit(
        X_full, D_full
    )
    d_V = np.abs(d_varselectorfit.coef_)
    best_d_v_pen = d_varselectorfit.alpha_

    m_sel = (y_V + d_V) != 0
    transformer = SelMatchSpace(m_sel)
    if y_V.sum() == 0:
        V = d_V
    elif d_V.sum() == 0:
        V = y_V
    else:
        V = y_V_share * y_V / (y_V.sum()) + (1 - y_V_share) * d_V / (2 * d_V.sum())
    return transformer, V[m_sel], (best_y_v_pen, best_d_v_pen), V


class SelMatchSpace:
    def __init__(self, m_sel):
        self.m_sel = m_sel

    def transform(self, X):
        return X[:, self.m_sel]


def MTLSTMMixed_MatchSpace_factory(
    T0=None,
    K_fixed=0,
    M_sizes=None,
    dropout_rate=0.2,
    epochs=2,
    verbose=0,
    hidden_length=100,
):
    """
    Return a MatchSpace function that will fit an LSTM of [X_fixed, X_time_varying, Y_pre] ~ Y with the hidden-layer size 
    optimized to reduce errors on goal units

    :param T0: length of Y_pre
    :param K_fixed: Number of fixed unit-covariates (rest will assume to be time-varying)
    :param M_sizes: list of sizes of hidden layer (match-space) sizes to try. Default is range(1, 2*int(np.log(Y.shape[0])))
    :param dropout_rate:
    :param epochs:
    :param verbose:
    :param hidden_length:
    :returns: MatchSpace fn, V vector, best_M_size, V
    """

    def _MTLSTMMixed_MatchSpace_wrapper(X, Y, fit_model_wrapper, **kwargs):
        return _MTLSTMMixed_MatchSpace(
            X,
            Y,
            fit_model_wrapper,
            T0=T0,
            K_fixed=K_fixed,
            M_sizes=M_sizes,
            dropout_rate=dropout_rate,
            epochs=epochs,
            verbose=verbose,
            hidden_length=hidden_length,
            **kwargs
        )

    return _MTLSTMMixed_MatchSpace_wrapper


def _split_LSTM_x_data(X, T0, K_fixed=0):
    N, K = X.shape

    Cov_F = X[:, :K_fixed]

    Cov_TV0 = X[:, K_fixed : (K - T0)]
    assert Cov_TV0.shape[1] % T0 == 0, "Time-varying covariates not the right shape"
    K_TV = int(Cov_TV0.shape[1] / T0)
    Cov_TV = np.empty((N, T0, K_TV))
    for i in range(K_TV):
        Cov_TV[:, :, i] = Cov_TV0[:, (i * K_TV) : ((i + 1) * K_TV)]

    Out_pre = X[:, (K - T0) :]
    return Cov_F, Cov_TV, Out_pre


def _shape_LSTM_x_data(Cov_F, Cov_TV, Out_pre):
    N, K_fixed = Cov_F.shape
    T0 = Out_pre.shape[1]
    K_TV = Cov_TV.shape[2]
    LSTM_K = K_fixed + K_TV + 1

    LSTM_x = np.empty((N, T0, LSTM_K))
    for t in range(T0):
        LSTM_x[:, t, :K_fixed] = Cov_F
        LSTM_x[:, t, K_fixed : (K_fixed + K_TV)] = Cov_TV[:, t, :]
        LSTM_x[:, t, (K_fixed + K_TV) :] = Out_pre[:, t, np.newaxis]
    return LSTM_x


def _shape_LSTM_y_data(Y_pre, Y_post, T0):
    _, T1 = Y_post.shape
    Y = np.hstack((Y_pre, Y_post))

    LSTM_y = []
    for t in range(T1):
        LSTM_y.append(Y[:, (t + 1) : (T0 + t + 1), np.newaxis])
    return LSTM_y


def _MTLSTMMixed_MatchSpace(
    X,
    Y,
    fit_model_wrapper,
    T0=None,
    K_fixed=0,
    M_sizes=None,
    dropout_rate=0.2,
    epochs=2,
    verbose=0,
    hidden_length=100,
    **kwargs
):
    # could have just used the LSTM state units direclty, but having this be big and then timeDistributed to narrow down is more expressive/powerful
    with capture_all():  # doesn't have quiet option
        import keras
    if M_sizes is None:
        M_sizes = range(1, 2 * int(np.log(Y.shape[0])))

    if T0 is None:
        T0 = X.shape[1]

    if verbose == 0:
        import os

        if (
            "TF_CPP_MIN_LOG_LEVEL" in os.environ
            and os.environ["TF_CPP_MIN_LOG_LEVEL"] != "2"
            and os.environ["TF_CPP_MIN_LOG_LEVEL"] != "3"
        ):
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
            # Otherwise prints random info about CPU instruction sets

    Cov_F, Cov_TV, Out_pre = _split_LSTM_x_data(X, T0, K_fixed=K_fixed)
    LSTM_x = _shape_LSTM_x_data(Cov_F, Cov_TV, Out_pre)
    LSTM_y = _shape_LSTM_y_data(Out_pre, Y, T0)
    LSTM_K = LSTM_x.shape[2]
    T1 = Y.shape[1]

    # fits_single = {}
    int_layer_single = {}
    Vs_single = {}
    scores = np.zeros((len(M_sizes)))
    for i, M_size in enumerate(M_sizes):
        inp = keras.Input(
            batch_shape=(1, T0, LSTM_K), name="input"
        )  # batch_shape=(1,..) ensures trained on one case at time
        with capture_all():  # doesn't have quiet option
            x1 = keras.layers.LSTM(units=hidden_length, return_sequences=True)(inp)  ##
            x2 = keras.layers.Dropout(rate=dropout_rate)(x1)  ##
        core = keras.layers.TimeDistributed(
            keras.layers.Dense(units=M_size, activation="elu"), name="embedding"
        )(x2)
        output_vec = []
        for t in range(T1):
            new_output = keras.layers.Dense(
                units=1, activation="linear", name="yp%s" % (t)
            )(core)
            output_vec.append(new_output)
        model = keras.models.Model(inputs=inp, outputs=output_vec)

        model.compile(loss="mse", optimizer="Adam", metrics=["mean_squared_error"])
        model.fit(x=LSTM_x, y=LSTM_y, batch_size=1, epochs=epochs, verbose=verbose)

        outputs_fit = model.get_layer(
            name="embedding"
        ).output  # .get_output_at(node_index = 1)
        intermediate_layer_model = keras.models.Model(
            inputs=model.input, outputs=outputs_fit
        )

        final_weights = np.empty((T1, M_size))
        n_layers_w = len(model.get_weights())
        for t in range(T1):
            l_i = n_layers_w - 2 - 2 * t
            final_weights[t, :] = model.get_weights()[l_i][:, 0]
        V_i = np.mean(np.abs(final_weights), axis=0)

        transformer_i = LSTMTransformer(T0, K_fixed, intermediate_layer_model)

        sc_fit_i = fit_model_wrapper(transformer_i, V_i)
        # fits_single[i] = sc_fit_i
        int_layer_single[i] = intermediate_layer_model
        Vs_single[i] = V_i
        scores[i] = sc_fit_i.score  # = sc_fit_i.score_R2

    i_best = np.argmin(scores)
    best_M_size = M_sizes[i_best]
    V_best = Vs_single[i_best]
    intermediate_layer_model = int_layer_single[i_best]
    transformer = LSTMTransformer(T0, K_fixed, intermediate_layer_model)
    return transformer, V_best, best_M_size, V_best


class LSTMTransformer:
    def __init__(self, T0, K_fixed, intermediate_layer_model):
        self.T0 = T0
        self.K_fixed = K_fixed
        self.intermediate_layer_model = intermediate_layer_model

    def transform(self, X):
        Cov_F, Cov_TV, Out_Pre = _split_LSTM_x_data(X, self.T0, K_fixed=self.K_fixed)
        LSTM_x = _shape_LSTM_x_data(Cov_F, Cov_TV, Out_Pre)
        M = self.intermediate_layer_model.predict(LSTM_x, batch_size=1)[
            :, self.T0 - 1, :
        ]
        return M


def MTLassoMixed_MatchSpace_factory(v_pens=None, n_v_cv=5):
    """
    Return a MatchSpace function that will fit a MultiTaskLassoCV for Y ~ X with the penalization optimized to reduce errors on goal units

    :param v_pens: Penalties to evaluate (default is to automatically determince)
    :param n_v_cv: Number of Cross-Validation folds
    :returns: MatchSpace fn, V vector, best_v_pen, V
    """

    def _MTLassoMixed_MatchSpace_wrapper(X, Y, fit_model_wrapper, **kwargs):
        return _MTLassoMixed_MatchSpace(
            X, Y, fit_model_wrapper, v_pens=v_pens, n_v_cv=n_v_cv, **kwargs
        )

    return _MTLassoMixed_MatchSpace_wrapper


def _MTLassoMixed_MatchSpace(
    X, Y, fit_model_wrapper, v_pens=None, n_v_cv=5, **kwargs
):  # pylint: disable=missing-param-doc, unused-argument
    # Note that MultiTaskLasso(CV).path with the same alpha doesn't produce same results as MultiTaskLasso(CV)
    mtlasso_cv_fit = MultiTaskLassoCV(normalize=True, cv=n_v_cv, alphas=v_pens).fit(
        X, Y
    )
    # V_cv = np.sqrt(np.sum(np.square(mtlasso_cv_fit.coef_), axis=0)) #n_tasks x n_features -> n_feature
    # v_pen_cv = mtlasso_cv_fit.alpha_
    # m_sel_cv = (V_cv!=0)
    # sc_fit_cv = fit_model_wrapper(SelMatchSpace(m_sel_cv), V_cv[m_sel_cv])

    v_pens = mtlasso_cv_fit.alphas_
    # fits_single = {}
    Vs_single = {}
    scores = np.zeros((len(v_pens)))
    # R2s = np.zeros((len(v_pens)))
    for i, v_pen in enumerate(v_pens):
        mtlasso_i_fit = MultiTaskLasso(alpha=v_pen, normalize=True).fit(X, Y)
        V_i = np.sqrt(np.sum(np.square(mtlasso_i_fit.coef_), axis=0))
        m_sel_i = V_i != 0
        sc_fit_i = fit_model_wrapper(SelMatchSpace(m_sel_i), V_i[m_sel_i])
        # fits_single[i] = sc_fit_i
        Vs_single[i] = V_i
        scores[i] = sc_fit_i.score
        # R2s[i] = sc_fit_i.score_R2

    i_best = np.argmin(scores)
    # v_pen_best = v_pens[i_best]
    # i_cv = np.where(v_pens==v_pen_cv)[0][0]
    # print("CV alpha: " + str(v_pen_cv) + " (" + str(R2s[i_cv]) + ")." + " Best alpha: " + str(v_pen_best) + " (" + str(R2s[i_best]) + ") .")
    best_v_pen = v_pens[i_best]
    V_best = Vs_single[i_best]
    m_sel_best = V_best != 0
    return SelMatchSpace(m_sel_best), V_best[m_sel_best], best_v_pen, V_best
