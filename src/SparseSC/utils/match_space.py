""" Utils for getting match spaces (Matching features and potentially feature wegihts)

"""
import numpy as np
from sklearn.linear_model import MultiTaskLassoCV, MultiTaskLasso
from .misc import capture_all

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

def _Fixed_V_MatchSpace(X, Y, V, **kwargs): #pylint: disable=unused-argument
    def _MT_Match(X):
        return X
    return _MT_Match, V, np.nan, V

def MTLassoCV_MatchSpace_factory(v_pens=None, n_v_cv = 5):
    """
    Return a MatchSpace function that will fit a MultiTaskLassoCV for Y ~ X

    :param v_pens: Penalties to evaluate (default is to automatically determince)
    :param n_v_cv: Number of Cross-Validation folds
    :returns: MatchSpace fn, V vector, best_v_pen, V
    """
    def _MTLassoCV_MatchSpace_wrapper(X, Y, **kwargs):
        return _MTLassoCV_MatchSpace(X, Y, v_pens=v_pens, n_v_cv = n_v_cv, **kwargs)
    return _MTLassoCV_MatchSpace_wrapper

def _MTLassoCV_MatchSpace(X, Y, v_pens=None, n_v_cv = 5, **kwargs): #pylint: disable=missing-param-doc, unused-argument
    #A fake MT would do Lasso on y_mean = Y.mean(axis=1)
    varselectorfit = MultiTaskLassoCV(normalize=True, cv=n_v_cv, alphas = v_pens).fit(X, Y)
    V = np.sqrt(np.sum(np.square(varselectorfit.coef_), axis=0)) #n_tasks x n_features -> n_feature
    best_v_pen = varselectorfit.alpha_
    m_sel = (V!=0)
    def _MT_Match(X):
        return(X[:,m_sel])
    return _MT_Match, V[m_sel], best_v_pen, V

def MTLSTMMixed_MatchSpace_factory(T0=None, K_fixed=0, M_sizes=None, dropout_rate=0.2, epochs=2, verbose=0, hidden_length=100):
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
    def _MTLSTMMixed_MatchSpace_wrapper(X, Y, fit_model_wrapper):
        return _MTLSTMMixed_MatchSpace(X, Y, fit_model_wrapper, T0=T0, K_fixed=K_fixed, M_sizes=M_sizes, dropout_rate=dropout_rate, epochs=epochs, verbose=verbose, hidden_length=hidden_length)
    return _MTLSTMMixed_MatchSpace_wrapper

def _MTLSTMMixed_MatchSpace(X, Y, fit_model_wrapper, T0=None, K_fixed=0, M_sizes=None, dropout_rate=0.2, epochs=2, verbose=0, hidden_length=100):
    # could have just used the LSTM state units direclty, but having this be big and then timeDistributed to narrow down is more expressive/powerful
    with capture_all() as _: #doesn't have quiet option
        import keras
    if M_sizes is None:
        M_sizes = range(1, 2*int(np.log(Y.shape[0])))
    
    if T0 is None:
        T0 = X.shape[1]

    if verbose==0:
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        #Otherwise prints random info about CPU instruction sets

    def _split_LSTM_x_data(X, T0, K_fixed=0):
        N, K = X.shape

        Cov_F = X[:,:K_fixed]

        Cov_TV0 = X[:,K_fixed:(K-T0)]
        assert Cov_TV0.shape[1] % T0 == 0, "Time-varying covariates not the right shape"
        K_TV = int(Cov_TV0.shape[1]/T0)
        Cov_TV = np.empty((N, T0, K_TV))
        for i in range(K_TV):
            Cov_TV[:,:,i] = Cov_TV0[:,(i*K_TV):((i+1)*K_TV)]

        Out_pre = X[:,(K-T0):]
        return Cov_F, Cov_TV, Out_pre

    def _shape_LSTM_x_data(Cov_F, Cov_TV, Out_pre):
        N, K_fixed = Cov_F.shape
        T0 = Out_pre.shape[1]
        K_TV = Cov_TV.shape[2]
        LSTM_K = K_fixed + K_TV + 1

        LSTM_x = np.empty((N, T0, LSTM_K))
        for t in range(T0):
            LSTM_x[:,t,:K_fixed] = Cov_F
            LSTM_x[:,t,K_fixed:(K_fixed+K_TV)] = Cov_TV[:,t,:]
            LSTM_x[:,t,(K_fixed+K_TV):] = Out_pre[:,t, np.newaxis]
        return LSTM_x
        
    def _shape_LSTM_y_data(Y_pre, Y_post, T0):
        _, T1 = Y_post.shape
        Y = np.hstack((Y_pre, Y_post))

        LSTM_y = []
        for t in range(T1):
            LSTM_y.append(Y[:,(t+1):(T0+t+1), np.newaxis] )
        return LSTM_y

    Cov_F, Cov_TV, Out_pre = _split_LSTM_x_data(X, T0, K_fixed=K_fixed)
    LSTM_x = _shape_LSTM_x_data(Cov_F, Cov_TV, Out_pre)
    LSTM_y = _shape_LSTM_y_data(Out_pre, Y, T0)
    LSTM_K = LSTM_x.shape[2]
    T1 = Y.shape[1]

    #fits_single = {}
    int_layer_single = {}
    Vs_single = {}
    scores = np.zeros((len(M_sizes)))
    for i, M_size in enumerate(M_sizes):
        inp = keras.Input(batch_shape=(1, T0, LSTM_K), name='input') #batch_shape=(1,..) ensures trained on one case at time
        with capture_all() as _: #doesn't have quiet option
            x1 = keras.layers.LSTM(units = hidden_length, return_sequences = True)(inp) ##
            x2 = keras.layers.Dropout(rate = dropout_rate)(x1) ##
        core = keras.layers.TimeDistributed(keras.layers.Dense(units = M_size, activation = 'elu'), name = 'embedding')(x2)
        output_vec = []
        for t in range(T1):
            new_output = keras.layers.Dense(units = 1, activation = 'linear', name = 'yp%s' % (t))(core)
            output_vec.append(new_output)
        model = keras.models.Model(inputs=inp, outputs=output_vec)

        model.compile(loss = 'mse', optimizer = 'Adam', metrics = ['mean_squared_error'])
        model.fit(x = LSTM_x, y = LSTM_y, batch_size = 1, epochs = epochs, verbose = verbose)

        outputs_fit = model.get_layer(name = 'embedding').output #.get_output_at(node_index = 1)
        intermediate_layer_model = keras.models.Model(inputs = model.input, outputs = outputs_fit)

        final_weights = np.empty((T1, M_size))
        n_layers_w = len(model.get_weights())
        for t in range(T1):
            l_i = n_layers_w-2-2*t
            final_weights[t,:] = model.get_weights()[l_i][:,0]
        V_i = np.mean(np.abs(final_weights), axis=0)
        
        def _MT_Match_i(X): #evaluated at call-time, not definition, but I call immediately so no problem
            Cov_F, Cov_TV, Out_Pre = _split_LSTM_x_data(X, T0, K_fixed=K_fixed)
            LSTM_x = _shape_LSTM_x_data(Cov_F, Cov_TV, Out_Pre)
            M = intermediate_layer_model.predict(LSTM_x, batch_size=1)[:,T0-1,:] #pylint: disable=cell-var-from-loop
            return M

        sc_fit_i = fit_model_wrapper(_MT_Match_i, V_i)
        #fits_single[i] = sc_fit_i
        int_layer_single[i] = intermediate_layer_model
        Vs_single[i] = V_i
        scores[i] = sc_fit_i.score # = sc_fit_i.score_R2

    i_best = np.argmin(scores)
    best_M_size = M_sizes[i_best]
    V_best = Vs_single[i_best]
    intermediate_layer_model = int_layer_single[i_best]
    
    def _MT_Match(X):
        Cov_F, Cov_TV, Out_Pre = _split_LSTM_x_data(X, T0, K_fixed=K_fixed)
        LSTM_x = _shape_LSTM_x_data(Cov_F, Cov_TV, Out_Pre)
        M = intermediate_layer_model.predict(LSTM_x, batch_size=1)[:,T0-1,:]
        return M
    return _MT_Match, V_best, best_M_size, V_best

def MTLassoMixed_MatchSpace_factory(v_pens=None, n_v_cv = 5):
    """
    Return a MatchSpace function that will fit a MultiTaskLassoCV for Y ~ X with the penalization optimized to reduce errors on goal units

    :param v_pens: Penalties to evaluate (default is to automatically determince)
    :param n_v_cv: Number of Cross-Validation folds
    :returns: MatchSpace fn, V vector, best_v_pen, V
    """
    def _MTLassoMixed_MatchSpace_wrapper(X, Y, fit_model_wrapper):
        return _MTLassoMixed_MatchSpace(X, Y, fit_model_wrapper, v_pens=v_pens, n_v_cv = n_v_cv)
    return _MTLassoMixed_MatchSpace_wrapper

def _MTLassoMixed_MatchSpace(X, Y, fit_model_wrapper, v_pens=None, n_v_cv = 5, **kwargs): #pylint: disable=missing-param-doc, unused-argument
    #Note that MultiTaskLasso(CV).path with the same alpha doesn't produce same results as MultiTaskLasso(CV)
    mtlasso_cv_fit = MultiTaskLassoCV(normalize=True, cv=n_v_cv, alphas = v_pens).fit(X, Y)
    V_cv = np.sqrt(np.sum(np.square(mtlasso_cv_fit.coef_), axis=0)) #n_tasks x n_features -> n_feature
    #v_pen_cv = mtlasso_cv_fit.alpha_
    m_sel_cv = (V_cv!=0)
    def _MT_Match_cv(X):
        return(X[:,m_sel_cv])
    #sc_fit_cv = fit_model_wrapper(_MT_Match_cv, V_cv[m_sel_cv])

    v_pens = mtlasso_cv_fit.alphas_
    #fits_single = {}
    Vs_single = {}
    scores = np.zeros((len(v_pens)))
    #R2s = np.zeros((len(v_pens)))
    for i, v_pen in enumerate(v_pens):
        mtlasso_i_fit = MultiTaskLasso(alpha=v_pen, normalize=True).fit(X, Y)
        V_i = np.sqrt(np.sum(np.square(mtlasso_i_fit.coef_), axis=0))
        m_sel_i = (V_i!=0)
        def _MT_Match_i(X): #evaluated at call-time, not definition, but I call immediately so no problem
            return(X[:,m_sel_i]) #pylint: disable=cell-var-from-loop
        sc_fit_i = fit_model_wrapper(_MT_Match_i, V_i[m_sel_i])
        #fits_single[i] = sc_fit_i
        Vs_single[i] = V_i
        scores[i] = sc_fit_i.score
        #R2s[i] = sc_fit_i.score_R2

    i_best = np.argmin(scores)
    #v_pen_best = v_pens[i_best]
    #i_cv = np.where(v_pens==v_pen_cv)[0][0]
    #print("CV alpha: " + str(v_pen_cv) + " (" + str(R2s[i_cv]) + ")." + " Best alpha: " + str(v_pen_best) + " (" + str(R2s[i_best]) + ") .")
    best_v_pen = v_pens[i_best]
    V_best = Vs_single[i_best]
    m_sel_best = (V_best!=0)
    def _MT_Match_best(X):
        return X[:,m_sel_best]
    return _MT_Match_best, V_best[m_sel_best], best_v_pen, V_best
