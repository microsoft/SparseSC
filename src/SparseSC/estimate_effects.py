"""
Effect estimation routines
"""
# To do:
# - Rename the pl_res
# - Allow pass in in label (rather than #idx) for treated units. 
import numpy as np
import pandas as pd
from .utils.metrics_utils import _gen_placebo_stats_from_diffs
from .fit import fit
from .fit_fast import fit_fast
from .utils.misc import par_map

#Note https://stackoverflow.com/questions/31917964/
# - that pandas can only store datetime[ns]
# - that iterating through it normally with give Timestamps, so use .values first to stay in datetime[ns]
# - Allow honest control predictions from full refitting
# - When using main hyper-parameter values for cross-fitting, remove potential w_pen, v_pen from kwargs


def _convert_dt_to_idx(dt, dt_index): 
    #pylint seems to be confused by the ufunc
    if np.isnat(dt): #pylint: disable=no-member
        return np.nan
    else:
        idx_list = np.where(dt_index==dt)[0]
        return np.nan if len(idx_list)==0 else idx_list[0]

#Used by par_map so needs f to be the first argument
def _fit_p_wrapper(f, test_folds, fit_fn, **kwargs):
    test = test_folds[f]

    return (fit_fn(treated_units=test, **kwargs), test)
    
def get_c_predictions_honest(X_and_Y_pre_c, Y_post_c, Y_c, model_type= "retrospective", cf_folds = 10, cf_seed=110011, fast = True, verbose=1, progress=False, print_path=False, n_multi=0, **kwargs):
    r"""
    Cross-fits the model across the controls for single considered treatment period

    :param X_and_Y_pre_c:
    :param Y_post_c:
    :param Y_c:
    :param model_type: Model type
    :param cf_folds: Number of cross-fit folds for getting honest placebo effects
    :param cf_seed: Seed for cross-fit fold splitting    
    :param fast: Whether to use the fast approximate solution (fit_fast() rather than fit())
    :type fast: bool
    :param n_multi: Number of processes use (0=single threaded)
    :param kwargs: Additional parameters passed to fit() or fit_fast()

    :returns: Y_local_c_sc_honest, [(f_train_fit, f_test_idxs) for f in folds]
    """
    # TODO: Maybe build this into the FitResults object or the fit methods?
    from functools import partial
    fit_fn = fit_fast if fast else fit
    try:
        iter(cf_folds)
    except TypeError:
        from sklearn.model_selection import KFold
        cf_folds = KFold(cf_folds, shuffle=True, random_state=cf_seed).split(np.arange(Y_c.shape[0]))
    train_test_splits = list(cf_folds)
    F = len(train_test_splits)
    test_folds = [test for (_, test) in train_test_splits]
    part_fn = partial(_fit_p_wrapper, test_folds=test_folds, fit_fn=fit_fn, features=X_and_Y_pre_c,
    targets=Y_post_c,
    model_type=model_type,
    verbose=verbose-1,
    print_path=print_path,
    progress=progress,
    **kwargs)

    fits = par_map(part_fn, range(F), F, verbose, n_multi=n_multi, header="CROSS-FITTING")
        
    Y_c_sc_honest = Y_c
    for fold, (_, test) in enumerate(train_test_splits):
        fit_k, _ = fits[fold]
        Y_c_sc_honest[test,:] = fit_k.predict(Y_c)[test,:]

    return Y_c_sc_honest, fits


def estimate_effects(
    outcomes,
    unit_treatment_periods,
    T0=None,
    T1=None,
    covariates=None,
    max_n_pl=10000,
    ret_pl=False,
    ret_CI=False,
    level=0.95,
    fast = True,
    model_type = "retrospective",
    T2 = None,
    cf_folds = 10, #sync with helper
    cf_seed=110011, #sync with helper
    **kwargs
):
    r"""
    Determines statistical significance for average and individual effects

    :param outcomes: Outcomes
    :type outcomes: np.array or pd.DataFrame with shape (N,T)
    :param unit_treatment_periods: Vector of treatment periods for each unit
        (if a unit is never treated then use np.NaN if vector refers to time periods by numerical index
        and np.datetime64('NaT') if using DateTime to refer to time periods (and thne Y must be pd.DataFrame with columns in DateTime too))
        If using a prospective-based design this is the true treatment periods (and fit will be called with
        pseudo-treatment periods that are T1 periods earlier).
    :type unit_treatment_periods: np.array or pd.Series with shape (N)
    :param T0: pre-history length to match over. 
    :type T0: int, Optional (default is pre-period for first treatment)
    :param T1: post-history length to fit over. 
    :type T1: int, Optional (Default is post-period for last treatment)
    :param covariates: Additional pre-treatment features
    :type covariates: np.array or pd.DataFrame with shape (N,K), Optional
    :param max_n_pl: The full number of placebos is choose(N0,N1). If N1=1
            then this is N0. This number grows quickly with N1 so we can
            set a maximum that we compute and this is drawn at random.
    :type max_n_pl: int, Optional
    :param ret_pl: Return the matrix of placebos (different from the SC of
            the controls when N1>1)
    :type ret_pl: bool
    :param ret_CI: Whether to return confidence intervals (requires more memory during execution)
    :param level: Level for confidence intervals
    :type level: float (between 0 and 1)
    :param fast: Whether to use the fast approximate solution (fit_fast() rather than fit())
    :type fast: bool
    :param model_type: Model type
    :param T2: If model='prospective' then the period of which to evaluate the effect
    :param kwargs: Additional parameters passed to fit() or fit_fast()

    :returns: An instance of SparseSCEstResults with the fitted results
    :raises ValueError:  when invalid parameters are passed

    :Keyword Args: Passed on to fit() or fit_fast()
    """
    Y = outcomes
    X = covariates
    Y_df = None
    if isinstance(Y, pd.DataFrame):
        Y_df = Y
        Y = Y.values
        if isinstance(X, pd.DataFrame):
            X = X.reindex(Y_df.index)
        if isinstance(unit_treatment_periods, pd.Series):
            unit_treatment_periods = unit_treatment_periods.reindex(Y_df.index)
    X_df = None
    if X is not None and isinstance(X, pd.DataFrame):
        X_df = X
        X = X.values
    using_dt_index = (unit_treatment_periods.dtype.kind=='M')
    if using_dt_index:
        assert (Y_df is not None), "Can't determine time period of treatment"
        assert (Y_df.columns.dtype.kind=='M'), "Can't determine time period of treatment"


        def _convert_dts_to_idx(datetimes, dt_index):
            if isinstance(datetimes, pd.Series):
                datetimes = datetimes.values
            #is_a_nat = np.isnat(datetimes)
            dt_idx = np.empty((len(datetimes)))
            for i, val in enumerate(datetimes):
                dt_idx[i] = _convert_dt_to_idx(val, dt_index)
            return dt_idx
        dt_index = Y_df.columns.values
        unit_treatment_periods_idx = _convert_dts_to_idx(unit_treatment_periods, dt_index)
    else:
        unit_treatment_periods_idx = unit_treatment_periods

    if model_type == 'full': 
        raise ValueError("parameter 'model_type' can't be 'full'" )
    N,T = Y.shape #pylint: disable=unused-variable
    finite_t_idx = unit_treatment_periods_idx[np.isfinite(unit_treatment_periods_idx)].astype('int')
    t_max_before = min(finite_t_idx[finite_t_idx>=1])
    t_max_after = T-max(finite_t_idx[finite_t_idx<=(T-1)])
    if model_type == "retrospective":
        if T0 is None:
            T0 = t_max_before
        else:
            assert (T0<=t_max_before), "T0 too large to accomodate all treated units. Drop them or change T0."
        if T1 is None:
            T1 = t_max_after
        else:
            assert (T1<=t_max_after), "T1 too large to accomodate all treated units. Drop them or change T1."
        finite_t_idx_fit = finite_t_idx
        unit_treatment_periods_idx_fit = unit_treatment_periods_idx
    else:
        assert (t_max_before>=2), "Prospective-based designs need at least 2 periods before the first treatment"
        if (T0 is None and T1 is None):
            T1 = max(1, int(t_max_before/5))
            T0 = t_max_before - T1
        elif T0 is None:
            assert t_max_before>=(T1+1)
            T0 = t_max_before - T1
        elif T1 is None:
            assert t_max_before>=(T0+1)
            T1 = t_max_before - T0
        else:
            assert t_max_before>=(T0+T1)
        
        if T2 is None:
            T2 = t_max_after

        finite_t_idx_fit = finite_t_idx - T1
        unit_treatment_periods_idx_fit = unit_treatment_periods_idx - T1

    if model_type != 'retrospective' and T2 is None:
        raise ValueError("Must specificy 'T2' for non-retrospective designs.")
    fit_fn = fit_fast if fast else fit

    #Set default parameters for fit()
    if 'print_path' not in kwargs:
        kwargs['print_path'] = False
    if 'progress' not in kwargs:
        kwargs['progress'] = False
    if 'verbose' not in kwargs:
        kwargs['verbose'] = 0
    if 'min_iter' not in kwargs:
        kwargs['min_iter'] = -1
    if 'tol' not in kwargs:
        kwargs['tol'] = 1
    if 'choice' not in kwargs:
        kwargs['choice'] = "min"
    
    #pullout hidden ones
    if 'treatment_unit_size' in kwargs and kwargs['treatment_unit_size'] is not None:
        treatment_unit_size = kwargs['treatment_unit_size']
        del kwargs['treatment_unit_size']
    else:
        treatment_unit_size = None
    
    if model_type=='retrospective':
        Tpost = T1
        Teval = T1
    else:
        Tpost = T1+T2
        Teval = T2
    #Teval_offset = Tpost-Teval

    treatment_periods_idx_fit = np.unique(finite_t_idx_fit[np.logical_and(finite_t_idx_fit>=T0,finite_t_idx_fit<=(T-Tpost))]) #sorts
    fits = {}
    ind_CI = {} if ret_CI else None
    diffs_pre_c = np.empty((0,T0))
    diffs_pre_t = np.empty((0,T0))
    if model_type!="retrospective":
        diffs_post_fit_c = np.empty((0,T1))
        diffs_post_fit_t = np.empty((0,T1))
    diffs_post_eval_c = np.empty((0,Teval))
    diffs_post_eval_t = np.empty((0,Teval))
    diffs_post_eval_scaled_c = np.empty((0,Teval))
    diffs_post_eval_scaled_t = np.empty((0,Teval))

    for treatment_period_idx_fit in treatment_periods_idx_fit:
        #Get the local values
        treatment_period_idx = treatment_period_idx_fit if model_type=='retrospective' else treatment_period_idx_fit + T1
        if using_dt_index:
            treatment_period = dt_index[treatment_period_idx]
        user_index = treatment_period if using_dt_index else treatment_period_idx
        c_units_mask_full, t_units_mask_full, ct_units_mask_full = get_sample_masks(unit_treatment_periods_idx_fit, treatment_period_idx_fit, Tpost)
        n_treated = np.sum(t_units_mask_full)
        n_control = np.sum(c_units_mask_full)
        Y_local = Y[ct_units_mask_full,(treatment_period_idx_fit-T0):(treatment_period_idx_fit+Tpost)]
        if Y_df is not None:
            col_index = Y_df.columns[(treatment_period_idx_fit-T0):(treatment_period_idx_fit+Tpost)]
        else:
            col_index = None
        treated_units = t_units_mask_full[ct_units_mask_full].nonzero()[0]
        control_units = c_units_mask_full[ct_units_mask_full].nonzero()[0]
        if treatment_unit_size is not None:
            doses = treatment_unit_size[t_units_mask_full]
            doses[np.isnan(doses)] = 1

        Y_pre = Y_local[:,:T0]
        #Y_post = Y_local[:,T0:]
        Y_post_fit = Y_local[:,T0:(T0+T1)]
        if X is None:
            X_and_Y_pre = Y_pre
        else:
            X_and_Y_pre = np.hstack((X[ct_units_mask_full,:], Y_pre))

        fit_res = fit_fn(
            features=X_and_Y_pre,
            targets=Y_post_fit,
            model_type=model_type,
            treated_units=treated_units,
            cv_folds=cv_folds,
            cv_seed=cv_seed,
            **kwargs
        )
        fits[user_index] = fit_res

        #Get the fit on match variables. Nothing was fit so that these would fit well, so don't worry about overfitting
        M = fit_res.match_space if fit_res.match_space is not None else fit_res.features
        if M is None or M.shape[1]==0 or len(fit_res.V) == 0: #think last test is redundant
            rmspe_M_w_p = np.nan
        else:
            M_diffs_2 = np.square(M - fit_res.predict(M))
            #rmspe_M_unw = np.sqrt(np.mean(M_diff_2, axis=1))
            V_fit = np.diag(fit_res.V)
            V_fit_norm = V_fit / np.sum(V_fit)
            rmspe_M_w = np.sqrt(np.mean(np.asarray(M_diffs_2) * V_fit_norm, axis=1))
            rmspe_M_w_p = _gen_placebo_stats_from_diffs(rmspe_M_w[control_units,None], rmspe_M_w[treated_units,None], max_n_pl, False, True).rms_joint_effect.p
        setattr(fit_res, 'rmspe_M_w_p', rmspe_M_w_p)

        #Get honest predictions (for honest placebo effects)
        Y_sc = fit_res.predict(Y_local) #doesn't have honest ones for the control units
        Y_sc[control_units:], _ = get_c_predictions_honest(X_and_Y_pre[control_units,:], Y_post_fit[control_units,:], Y_local[control_units,:], 
                                                       model_type, cf_folds, cf_seed, w_pen=fit_res.initial_w_pen, v_pen=fit_res.initial_v_pen,
                                                       cv_folds=cv_folds, cv_seed=cv_seed, **kwargs)


        #Get statistical significance
        diffs = Y_local - Y_sc
        rmspes_pre = np.sqrt(np.mean(np.square(diffs[:,:T0]), axis=1))
        diffs_post_eval_scaled = np.diagflat(1/rmspes_pre).dot(diffs[:,T0+Tpost-Teval:T0+Tpost])

        diffs_pre_t = np.vstack((diffs_pre_t,diffs[treated_units,:T0]))
        if model_type!="retrospective":
            diffs_post_fit_t = np.vstack((diffs_post_fit_t,diffs[treated_units,T0:(T0+T1)]))
        diffs_post_eval_t_i = diffs[treated_units,T0+Tpost-Teval:T0+Tpost]
        diffs_post_eval_scaled_t_i = diffs_post_eval_scaled[treated_units,:]
        if treatment_unit_size is not None:
            diffs_post_eval_t_i = diffs_post_eval_t_i / doses[:, np.newaxis] #scale rows
            diffs_post_eval_scaled_t_i = diffs_post_eval_scaled_t_i / doses[:, np.newaxis] #scale rows
        diffs_post_eval_t = np.vstack((diffs_post_eval_t,diffs_post_eval_t_i))
        diffs_post_eval_scaled_t = np.vstack((diffs_post_eval_scaled_t,diffs_post_eval_scaled_t_i))

        for t_i in range(n_treated): #technically only need to do if len(treatment_periods)>1, but consistency is nice
            diffs_pre_c = np.vstack((diffs_pre_c, diffs[control_units,:T0]))
            if model_type!="retrospective":
                diffs_post_fit_c = np.vstack((diffs_post_fit_c, diffs[control_units,T0:(T0+T1)]))
            diffs_post_eval_c_i = diffs[control_units,T0+Tpost-Teval:T0+Tpost]
            diffs_post_eval_scaled_c_i = diffs_post_eval_scaled[control_units,:]
            if treatment_unit_size is not None:
                diffs_post_eval_c_i = diffs_post_eval_c_i / doses[t_i]
                diffs_post_eval_scaled_c_i = diffs_post_eval_scaled_c_i / doses[t_i]
            diffs_post_eval_c = np.vstack((diffs_post_eval_c, diffs_post_eval_c_i))
            diffs_post_eval_scaled_c = np.vstack((diffs_post_eval_scaled_c, diffs_post_eval_scaled_c_i))


        if ret_CI:
            Y_sc_full = fit_res.predict(Y[ct_units_mask_full,:])
            diffs_full = Y[ct_units_mask_full,:] - Y_sc_full
            ind_ci_vals = _gen_placebo_stats_from_diffs(diffs_full[control_units,:], np.zeros((1,diffs_full.shape[1])),
                                                                     max_n_pl, False, True, level, vec_index=col_index).effect_vec.ci
            ind_CI[user_index] = ind_ci_vals

    if len(treatment_periods_idx_fit) == 1 and Y_df is not None:
        treatment_period_idx_fit = treatment_periods_idx_fit[0]
        pre_index = Y_df.columns[(treatment_period_idx_fit-T0):treatment_period_idx_fit]
        #post_index = Y_df.columns[treatment_period:(treatment_period+Tpost)]
        post_eval_index = Y_df.columns[(treatment_period_idx_fit+Tpost-Teval):(treatment_period_idx_fit+Tpost)]
        if model_type!="retrospective":
            post_fit_index = Y_df.columns[treatment_period_idx_fit:(treatment_period_idx_fit+T1)]
    else:
        pre_index, post_fit_index, post_eval_index = None, None, None
    # diagnostics
    pl_res_pre = _gen_placebo_stats_from_diffs(
        diffs_pre_c,
        diffs_pre_t,
        max_n_pl,
        ret_pl,
        ret_CI,
        level,
        vec_index = pre_index
    )

    if model_type!="retrospective":
        pl_res_post_fit = _gen_placebo_stats_from_diffs(
            diffs_post_fit_c,
            diffs_post_fit_t,
            max_n_pl,
            ret_pl,
            ret_CI,
            level,
            vec_index = post_fit_index
        )
    else:
        pl_res_post_fit = None

    # effects
    pl_res_post_eval = _gen_placebo_stats_from_diffs(
        diffs_post_eval_c,
        diffs_post_eval_t,
        max_n_pl,
        ret_pl,
        ret_CI,
        level,
        vec_index = post_eval_index
    )

    pl_res_post_eval_scaled = _gen_placebo_stats_from_diffs(
        diffs_post_eval_scaled_c,
        diffs_post_eval_scaled_t,
        max_n_pl,
        ret_pl,
        ret_CI,
        level,
        vec_index = post_eval_index
    )

    #reset to dataframes if possible
    if Y_df is not None:
        Y = Y_df
    if X_df is not None:
        X = X_df

    est_ret = SparseSCEstResults(
        Y, fits, unit_treatment_periods, unit_treatment_periods_idx, unit_treatment_periods_idx_fit, 
        T0, T1, pl_res_pre, pl_res_post_eval, pl_res_post_eval_scaled, max_n_pl, X, ind_CI, model_type, 
        T2, pl_res_post_fit
    )
    if treatment_unit_size is not None:
        setattr(est_ret, 'treatment_unit_size', treatment_unit_size)

    return est_ret

def get_sample_masks(unit_treatment_periods, treatment_period, T1):
    """
    Returns the sample mask for a particular treatment period

    :returns: control units mask, treated units mask, full sample mask (logical or of the previous two) 
    """
    is_fin_mask = np.isfinite(unit_treatment_periods)
    c_units_mask = np.logical_not(is_fin_mask)
    c_units_mask[is_fin_mask] = unit_treatment_periods[is_fin_mask]>(treatment_period+T1)
    #c_units_mask = np.logical_or(np.isnan(unit_treatment_periods),unit_treatment_periods>(treatment_period+T1))
    t_units_mask = (unit_treatment_periods == treatment_period)
    ct_units_mask = np.logical_or(c_units_mask, t_units_mask)
    return c_units_mask, t_units_mask, ct_units_mask

class SparseSCEstResults(object):
    """
    Holds estimation info
    """

    # pylint: disable=redefined-outer-name
    def __init__(self, outcomes, fits, unit_treatment_periods, unit_treatment_periods_idx, 
    unit_treatment_periods_idx_fit, T0, T1, pl_res_pre, pl_res_post, pl_res_post_scaled, 
    max_n_pl, covariates = None, ind_CI=None, model_type="retrospective", T2=None, pl_res_post_fit=None):
        """
        :param outcomes: Outcome for the whole sample
        :param fits: The fit() return objects
        :type fits: dictionary of period->SparseSCFit
        :param unit_treatment_periods: Vector or treatment periods for each unit
            (if a unit is never treated then use np.NaN if vector refers to time periods by numerical index
            and np.datetime64('NaT') if using DateTime to refer to time periods (and thne Y must be pd.DataFrame with columns in DateTime too))
        :param unit_treatment_periods_idx: the conversion of unit_treatment_periods to indexes (helpful if use had datetime index)
        :param unit_treatment_periods_idx_fit: the treatment period indexes passed to fit (helpful if prospective-based design)
        :param T0: Pre-history to match over
        :param T1: post-history to evaluate over
        :param pl_res_pre: Statistics for the average fit of the treated units
                in the pre-period (used for diagnostics)
        :type pl_res_pre: PlaceboResults
        :param pl_res_post: Statistics for the average treatment effect in the post-period
        :type pl_res_post: PlaceboResults
        :param pl_res_post_scaled: Statistics for the average scaled treatment
                effect (difference divided by pre-treatment RMS fit) in the
                post-period.
        :type pl_res_post_scaled: PlaceboResults
        :param max_n_pl: maximum number of of placebos effects used for inference
        :param covariates: Nxk matrix of full baseline covariates (or None)
        :param ind_CI: Confidence intervals for SC predictions at the unit
                level (not averaged over N1).  Used for graphing rather than
                treatment effect statistics
        :type ind_CI: dictionary of period->CI_int. Each CI_int is for the full sample (not necessarily T0+T1)
        :param model_type: Model type string
        :param T2: T2 (if prospective-type design)
        :param pl_res_post_fit: If prospective-type designs, the PlaceboResults for target period used for fit 
            (still before actual treatment)
        """
        self.Y = outcomes
        self.X = covariates
        self.fits = fits
        self.unit_treatment_periods = unit_treatment_periods
        self.unit_treatment_periods_idx = unit_treatment_periods_idx
        self.unit_treatment_periods_idx_fit = unit_treatment_periods_idx_fit
        self.T0 = T0
        self.T1 = T1
        self.T2 = T2
        self.pl_res_pre = pl_res_pre
        self.pl_res_post = pl_res_post
        self.pl_res_post_scaled = pl_res_post_scaled
        self.ind_CI = ind_CI
        self.model_type = model_type
        self.pl_res_post_fit = pl_res_post_fit
        self.Tpost = T1 if model_type=='retrospective' else T1+T2
        self.max_n_pl = max_n_pl
        self._using_dt_index = (unit_treatment_periods.dtype.kind=='M')

    @property
    def p_value(self):
        """
        p-value for the current model if relevant, else None.
        """
        return (
            self.pl_res_post.avg_joint_effect.p
            if self.pl_res_post.avg_joint_effect
            else None
        )

    @property
    def CI(self):
        """
        p-value for the current model if relevant, else None.
        """
        return self.pl_res_post.avg_joint_effect
    
    def _default_treatment_period(self, treatment_period=None):
        if treatment_period is None:
            t_periods = self.fits.keys()
            if len(t_periods) ==1:
                return list(t_periods)[0]
            else:
                raise ValueError("Need to pass in treatment_period when there are multiple")
        return treatment_period
    
    def get_tr_time_info(self, treatment_period):
        """ Returns treatment info:
            a) indexes for the time index (helpful if user used an np datetime index)
            b) treatment period used in the call to fit (helpful if model is a prospective type)
        :param treatment_period: treatment period in the user's view (could be TimeIndex)
        :returns: (reatment_period_idx, treatment_period_idx_fit, treatment_period_fit)
        """
        if self._using_dt_index:
            treatment_period_idx = _convert_dt_to_idx(treatment_period, self.Y.columns.values)
        else:
            treatment_period_idx = treatment_period
        treatment_period_idx_fit = treatment_period_idx if self.model_type=="retrospective" else treatment_period_idx - self.T1
        
        if self._using_dt_index:
            treatment_period_fit = self.Y.columns.values[treatment_period_idx_fit]
        else:
            treatment_period_fit = treatment_period_idx_fit
        return treatment_period_idx, treatment_period_idx_fit, treatment_period_fit

    def get_W(self, treatment_period=None):
        """
        Get W (np.ndarray 2D or pd.DataFrame depends on what was passed in) for one of the treatment periods
        """
        treatment_period = self._default_treatment_period(treatment_period)
        _, treatment_period_idx_fit, _ = self.get_tr_time_info(treatment_period)
        fit = self.fits[treatment_period]
        if isinstance(self.Y, pd.DataFrame):
            c_units_mask, _, ct_units_mask = get_sample_masks(self.unit_treatment_periods_idx_fit, treatment_period_idx_fit, self.Tpost)
            return pd.DataFrame(fit.sc_weights, index=self.Y.iloc[ct_units_mask,:].index, columns = self.Y.iloc[c_units_mask,:].index)
        else:
            return fit.sc_weights

    def get_V(self, treatment_period=None):
        """Returns V split across potential pre-treatment outcomes and baseline features
        """
        treatment_period = self._default_treatment_period(treatment_period)
        _, treatment_period_idx_fit, _ = self.get_tr_time_info(treatment_period)
        fit = self.fits[treatment_period]
        V = np.diag(fit.V)
        if fit.match_space is not None:
            if isinstance(fit.match_space_desc, np.ndarray):
                V = fit.match_space_desc
            else:
                return V

        if self.X is None:
            V_X, V_Y = None, V 
        else:
            X_k = self.X.shape[1]
            V_X, V_Y = V[:X_k], V[X_k:]

        if isinstance(self.X, pd.DataFrame):
            V_X = pd.Series(V_X, index=self.X.columns)
        if isinstance(self.Y, pd.DataFrame):
            V_Y = pd.Series(V_Y, index=self.Y.columns[(treatment_period_idx_fit-self.T0):treatment_period_idx_fit])

        return V_Y, V_X

    def get_sc(self, treatment_period=None):
        """Returns and NxT matrix of synthetic controls. For units not eligible (those previously treated or between treatment_period and treatment_period+T1)
        The results is left empty.
        """
        treatment_period = self._default_treatment_period(treatment_period)
        _, treatment_period_idx_fit, _ = self.get_tr_time_info(treatment_period)
        Y_sc = np.full(self.Y.shape, np.NaN)
        _, _, ct_units_mask = get_sample_masks(self.unit_treatment_periods_idx_fit, treatment_period_idx_fit, self.Tpost)
        if isinstance(self.Y, pd.DataFrame):
            Y_sc[ct_units_mask,:] = self.fits[treatment_period].predict(self.Y.iloc[ct_units_mask, :].values)
            Y_sc = pd.DataFrame(Y_sc, index = self.Y.index, columns=self.Y.columns)
        else:
            Y_sc[ct_units_mask,:] = self.fits[treatment_period].predict(self.Y[ct_units_mask, :])
        return Y_sc
    
    @property
    def fit(self):
        """Handy accessor if there is only one treatment-period
        """
        treatment_period = self._default_treatment_period()
        return self.fits[treatment_period]

    def __str__(self):
        """
        Parts that are omitted:
        * Diagnostics: joint_rms, effect_vec
        * Effect: joint_rms
        * Effect (Scaled): all
        """
        level_str = (
            "< " + str(1 - self.pl_res_pre.avg_joint_effect.ci.level)
            if self.pl_res_pre.avg_joint_effect.ci is not None
            else "close to 0"
        )

        return _SparseSCEstResults_template % (
            level_str,
            str(self.pl_res_pre.rms_joint_effect),
            str(self.pl_res_post.avg_joint_effect),
            str(self.pl_res_post.effect_vec),
        )

_SparseSCEstResults_template = """Pre-period fit diagnostic: Were the treated harder to match in the pre-period than the controls were.
Average difference in outcome for pre-period between treated and SC unit (concerning if p-value %s ): 
%s

(Investigate per-period match quality more using self.pl_res_pre.effect_vec)

Average Effect Estimation: %s

Effect Path Estimation:
 %s
 """

class _SparseSCPoolEstResults(object):
    # To do:
    # - Allow working with ests that have multiple diffs concatenated (store original and placebo sequence)
    def __init__(self, ests, max_n_pl=200, ret_pl=False, ret_CI=False, level=0.95):
        from SparseSC.utils.metrics_utils import _gen_placebo_stats_from_diffs
        #works if only 1 treated unit per-estimation
        #TODO: Get working with retrospective, and post_scaled, and wrap up nicer, use Est max_n_pl
        T0 = ests[0].pl_res_pre.effect_vec.placebos.shape[1]
        T1 = ests[0].pl_res_post.effect_vec.placebos.shape[1]
        diffs_pre_c = np.empty((0,T0))
        diffs_pre_t = np.empty((0,T0))
        diffs_post_eval_c = np.empty((0,T1))
        diffs_post_eval_t = np.empty((0,T1))
        for est in ests:
            diffs_pre_c_i = est.pl_res_pre.effect_vec.placebos
            diffs_pre_t_i = est.pl_res_pre.effect_vec.effect
            T0_i = diffs_pre_c_i.shape[1]
            if T0_i < T0:
                diffs_pre_c = diffs_pre_c[:,(T0-T0_i):]
                diffs_pre_t = diffs_pre_t[:,(T0-T0_i):]
                T0 = T0_i
            elif T0<T0_i:
                diffs_pre_c_i = diffs_pre_c_i[:,(T0_i-T0):]
                print(diffs_pre_t_i.shape)
                diffs_pre_t_i = diffs_pre_t_i[:,(T0_i-T0):]
            #print("init: " + str(diffs_pre_c.shape) + ". new:" + str(diffs_pre_c_i.shape))
            diffs_pre_c = np.vstack((diffs_pre_c, diffs_pre_c_i))
            diffs_pre_t = np.vstack((diffs_pre_t, diffs_pre_t_i[np.newaxis,:]))

            diffs_post_c_i = est.pl_res_post.effect_vec.placebos
            diffs_post_t_i = est.pl_res_post.effect_vec.effect
            T1_i = diffs_post_c_i.shape[1]
            if T1_i < T1:
                diffs_post_eval_c = diffs_post_eval_c[:,(T1-T1_i):]
                diffs_post_eval_t = diffs_post_eval_t[:,(T1-T1_i):]
                T1 = T1_i
            elif T1 < T1_i:
                diffs_post_c_i = diffs_post_c_i[:,(T1_i-T1):]
                diffs_post_t_i = diffs_post_t_i[:,(T1_i-T1):]
            diffs_post_eval_c = np.vstack((diffs_post_eval_c, diffs_post_c_i))
            diffs_post_eval_t = np.vstack((diffs_post_eval_t, diffs_post_t_i[np.newaxis,:]))

        self.pl_res_pre = _gen_placebo_stats_from_diffs(diffs_pre_c, diffs_pre_t, max_n_pl, ret_pl, ret_CI, level)
        self.pl_res_post_eval = _gen_placebo_stats_from_diffs(diffs_post_eval_c, diffs_post_eval_t, max_n_pl, ret_pl, ret_CI, level)
        
    def __str__(self):
        level_str = (
            "< " + str(1 - self.pl_res_pre.avg_joint_effect.ci.level)
            if self.pl_res_pre.avg_joint_effect.ci is not None
            else "close to 0"
        )
        return _SparseSCEstResults_template % (
            level_str,
            str(self.pl_res_pre.rms_joint_effect),
            str(self.pl_res_post_eval.avg_joint_effect),
            str(self.pl_res_post_eval.effect_vec),
        )
    
