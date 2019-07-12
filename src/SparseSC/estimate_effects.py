"""
Effect estimation routines
"""
import numpy as np
import pandas as pd
from .utils.metrics_utils import _gen_placebo_stats_from_diffs, CI_int
from .fit import fit
from .fit_fast import fit_fast


def estimate_effects(
    Y,
    unit_treatment_periods,
    T0=None,
    T1=None,
    X=None,
    max_n_pl=10000,
    ret_pl=False,
    ret_CI=False,
    level=0.95,
    fast = False,
    T_prosp_train = None,
    **kwargs
):
    r"""
    Determines statistical significance for average and individual effects

    :param Y: N x T matrix of outcomes
    :param unit_treatment_periods: N -vector of treatment periods (use value np.nan if never treated in sample)
    :param T0: pre-history length to match over. Default is pre-period for first treatment
    :param T1: post-history length to evaluate over. Default is post-period for last treatment
    :param X: N x K (or None)
    :param max_n_pl: The full number of placebos is choose(N0,N1). If N1=1
            then this is N0. This number grows quickly with N1 so we can
            set a maximum that we compute and this is drawn at random.
    :param ret_pl: Return the matrix of placebos (different from the SC of
            the controls when N1>1)
    :param ret_CI:
    :param level:
    :param kwargs: Additional parameters passed to fit()

    :returns: An instance of SparseSCEstResults with the fitted results
    :raises ValueError:  when invalid parameters are passed

    :Keyword Args: Passed on to fit()
    """
    Y_df = None
    if isinstance(Y, pd.DataFrame):
        Y_df = Y
        Y = Y.values
    N,T = Y.shape
    fin_t_periods = unit_treatment_periods[np.isfinite(unit_treatment_periods)].astype('int')
    if T0 is None:
        T0 = min(fin_t_periods[fin_t_periods>=1])
    if T1 is None:
        T1 = T-max(fin_t_periods[fin_t_periods<=(T-1)])

    #Set default parameters for fit()
    if 'model_type' in kwargs and kwargs['model_type'] != 'retrospective': 
        raise ValueError("parameter 'model_type' must be 'retrospective'" )
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
    if 'constrain' not in kwargs:
        kwargs['constrain'] = 'simplex'

    treatment_periods = np.unique(fin_t_periods[np.logical_and(fin_t_periods>=T0,fin_t_periods<=(T-T1))]) #sorts
    fits = {}
    if ret_CI:
        ind_CI = {}
    else:
        ind_CI = None
    diffs_pre_c = np.empty((0,T0))
    diffs_pre_t = np.empty((0,T0))
    diffs_post_c = np.empty((0,T1))
    diffs_post_t = np.empty((0,T1))
    diffs_post_scaled_c = np.empty((0,T1))
    diffs_post_scaled_t = np.empty((0,T1))

    for treatment_period in treatment_periods:
        #Get the local values
        c_units_mask_full, t_units_mask_full, ct_units_mask_full = get_sample_masks(unit_treatment_periods, treatment_period, T1)
        Y_local = Y[ct_units_mask_full,(treatment_period-T0):(treatment_period+T1)]
        if Y_df is not None:
            col_index = Y_df.columns[(treatment_period-T0):(treatment_period+T1)]
        else:
            col_index = None
        treated_units = t_units_mask_full[ct_units_mask_full].nonzero()[0]
        control_units = c_units_mask_full[ct_units_mask_full].nonzero()[0]

        Y_pre = Y_local[:,:T0]
        Y_post = Y_local[:,T0:]
        if X is None:
            X_and_Y_pre = Y_pre
        else:
            X_and_Y_pre = np.hstack((X[ct_units_mask_full,:], Y_pre))

        if not fast:
            fits[treatment_period] = fit(
                X=X_and_Y_pre,
                Y=Y_post,
                model_type="retrospective",
                treated_units=treated_units,
                **kwargs
            )
        else:
            fits[treatment_period] = fit_fast(
                X=X_and_Y_pre,
                Y=Y_post,
                model_type="retrospective",
                treated_units=treated_units,
                **kwargs
            )

            M = fits[treatment_period].match_space
            M_diffs_2 = np.square(M - fits[treatment_period].predict(M))
            #rmspe_M_unw = np.sqrt(np.mean(M_diff_2, axis=1))
            V_fit = np.diag(fits[treatment_period].V)
            V_fit_norm = V_fit / np.sum(V_fit)
            rmspe_M_w = np.sqrt(np.mean(M_diffs_2 * V_fit_norm, axis=1))

            rmspe_M_w_p = _gen_placebo_stats_from_diffs(rmspe_M_w[control_units,None], rmspe_M_w[treated_units,None], max_n_pl, False, True).rms_joint_effect.p
            setattr(fits[treatment_period], 'rmspe_M_w_p', rmspe_M_w_p)

        Y_sc = fits[treatment_period].predict(Y_local)
        diffs = Y_local - Y_sc
        diffs_pre_c = np.vstack((diffs_pre_c,diffs[control_units,:T0]))
        diffs_pre_t = np.vstack((diffs_pre_t,diffs[treated_units,:T0]))
        diffs_post_c = np.vstack((diffs_post_c,diffs[control_units,T0:]))
        diffs_post_t = np.vstack((diffs_post_t,diffs[treated_units,T0:]))
        rmspes_pre = np.sqrt(np.mean(np.square(diffs[:,:T0]), axis=1))
        diffs_post_scaled = np.diagflat(1/rmspes_pre).dot(diffs[:,T0:])
        diffs_post_scaled_c = np.vstack((diffs_post_scaled_c,diffs_post_scaled[control_units,:]))
        diffs_post_scaled_t = np.vstack((diffs_post_scaled_t,diffs_post_scaled[treated_units,:]))

        if ret_CI:
            Y_sc_full = fits[treatment_period].predict(Y[ct_units_mask_full,:])
            diffs_full = Y[ct_units_mask_full,:] - Y_sc_full
            ind_ci_vals = _gen_placebo_stats_from_diffs(diffs_full[control_units,:], np.zeros((1,diffs_full.shape[1])),
                                                                     max_n_pl, False, True, level, vec_index=col_index).effect_vec.ci
            ind_CI[treatment_period] = ind_ci_vals

    if len(treatment_periods) == 1 and Y_df is not None:
        pre_index = Y_df.columns[(treatment_period-T0):treatment_period]
        post_index = Y_df.columns[treatment_period:(treatment_period+T1)]
    else:
        pre_index, post_index = None, None
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

    pl_res_pre = _gen_placebo_stats_from_diffs(
        diffs_pre_c,
        diffs_pre_t,
        max_n_pl,
        ret_pl,
        ret_CI,
        level,
        vec_index = pre_index
    )

    # effects
    pl_res_post = _gen_placebo_stats_from_diffs(
        diffs_post_c,
        diffs_post_t,
        max_n_pl,
        ret_pl,
        ret_CI,
        level,
        vec_index = post_index
    )

    pl_res_post_scaled = _gen_placebo_stats_from_diffs(
        diffs_post_scaled_c,
        diffs_post_scaled_t,
        max_n_pl,
        ret_pl,
        ret_CI,
        level,
        vec_index = post_index
    )

    if Y_df is not None:
        Y = Y_df

    return SparseSCEstResults(
        Y, fits, unit_treatment_periods, T0, T1, pl_res_pre, pl_res_post, pl_res_post_scaled, X, ind_CI
    )

def get_sample_masks(unit_treatment_periods, treatment_period, T1):
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
    def __init__(self, Y, fits, unit_treatment_periods, T0, T1, pl_res_pre, pl_res_post, pl_res_post_scaled, X = None, ind_CI=None):
        """
        :param Y: Outcome for the whole sample
        :param fits: The fit() return objects
        :type fits: dictionary of period->SparseSCFit
        :param unit_treatment_periods: N -vector of treatment periods (use np.nan if never treated in sample)
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
        :param X: Nxk matrix of full baseline covariates (or None)
        :param ind_CI: Confidence intervals for SC predictions at the unit
                level (not averaged over N1).  Used for graphing rather than
                treatment effect statistics
        :type ind_CI: dictionary of period->CI_int. Each CI_int is for the full sample (not necessarily T0+T1)
        """
        self.Y = Y
        self.X = X
        self.fits = fits
        self.unit_treatment_periods = unit_treatment_periods
        self.T0 = T0
        self.T1 = T1
        self.pl_res_pre = pl_res_pre
        self.pl_res_post = pl_res_post
        self.pl_res_post_scaled = pl_res_post_scaled
        self.ind_CI = ind_CI

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

    def get_W(self, treatment_period=None):
        if treatment_period is None:
            t_periods = self.fits.keys()
            if len(t_periods) ==1:
                treatment_period = list(t_periods)[0]
            else:
                raise ValueError("Need to pass in treatment_period when there are multiple")
        fit = self.fits[treatment_period]
        if isinstance(self.Y, pd.DataFrame):
            c_units_mask, _, ct_units_mask = get_sample_masks(self.unit_treatment_periods, treatment_period, self.T1)
            return pd.DataFrame(fit.sc_weights, index=self.Y.iloc[ct_units_mask,:].index, columns = self.Y.iloc[c_units_mask,:].index)
        else:
            return fit.sc_weights

    def get_V(self, treatment_period=None):
        if treatment_period is None:
            t_periods = self.fits.keys()
            if len(t_periods) ==1:
                treatment_period = list(t_periods)[0]
            else:
                raise ValueError("Need to pass in treatment_period when there are multiple")
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
            V_Y = pd.Series(V_Y, index=self.Y.columns[(treatment_period-self.T0):treatment_period])

        return V_Y, V_X

    def get_sc(self, treatment_period=None):
        """Returns and NxT matrix of synthetic controls. For units not eligible (those previously treated or between treatment_period and treatment_period+T1)
        The results is left empty.
        """
        if treatment_period is None:
            t_periods = self.fits.keys()
            if len(t_periods) ==1:
                treatment_period = list(t_periods)[0]
            else:
                raise ValueError("Need to pass in treatment_period when there are multiple")
        Y_sc = np.full(self.Y.shape, np.NaN)
        _, _, ct_units_mask = get_sample_masks(self.unit_treatment_periods, treatment_period, self.T1)
        if isinstance(self.Y, pd.DataFrame):
            Y_sc[ct_units_mask,:] = self.fits[treatment_period].predict(self.Y.iloc[ct_units_mask, :].values)
            Y_sc = pd.DataFrame(Y_sc, index = self.Y.index, columns=self.Y.columns)
        else:
            Y_sc[ct_units_mask,:] = self.fits[treatment_period].predict(self.Y[ct_units_mask, :])
        return Y_sc
    
    @property
    def fit(self):
        t_periods = self.fits.keys()
        if len(t_periods) ==1:
            treatment_period = list(t_periods)[0]
            return(self.fits[treatment_period])
        else:
            raise ValueError("fit property ambiguous")

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
