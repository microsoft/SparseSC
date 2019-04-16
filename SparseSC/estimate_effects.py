"""
Effect estimation routines
"""
import numpy as np
from .utils.metrics_utils import _gen_placebo_stats_from_diffs, CI_int
from .fit import fit


def estimate_effects(
    Y_pre,
    Y_post,
    treated_units,
    X=None,
    max_n_pl=1000000,
    ret_pl=False,
    ret_CI=False,
    level=0.95,
    **kwargs
):
    r"""
    Determines statistical significance for average and individual effects

    :param Y_pre: N x T0 matrix
    :param Y_post: N x T1 matrix
    :param treated_units:
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

    :Keyword Args: Passed on to fit()
    """
    # TODO: Cleanup returning placebo distribution (incl pre?)
    N1 = len(treated_units)
    N = Y_pre.shape[0]
    # N0 = N - N1
    T0 = Y_pre.shape[1]
    T1 = Y_post.shape[1]
    T = T0 + T1
    Y = np.hstack((Y_pre, Y_post))
    control_units = list(set(range(N)) - set(treated_units))

    if X is None:
        X_and_Y_pre = Y_pre
    else:
        X_and_Y_pre = np.hstack((X, Y_pre))

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

    fit_res = fit(
        X=X_and_Y_pre,
        Y=Y_post,
        model_type="retrospective",
        treated_units=treated_units,
        **kwargs
    )
    Y_sc = fit_res.predict(Y)#[control_units, :]
    diffs = Y - Y_sc

    # diagnostics
    diffs_pre = diffs[:, :T0]
    pl_res_pre = _gen_placebo_stats_from_diffs(
        diffs_pre[control_units, :],
        diffs_pre[treated_units, :],
        max_n_pl,
        ret_pl,
        ret_CI,
        level,
    )

    # effects
    diffs_post = diffs[:, T0:]
    pl_res_post = _gen_placebo_stats_from_diffs(
        diffs_post[control_units, :],
        diffs_post[treated_units, :],
        max_n_pl,
        ret_pl,
        ret_CI,
        level,
    )

    rmspes_pre = np.sqrt(np.mean(np.square(diffs_pre), axis=1))
    diffs_post_scaled = np.diagflat(1 / rmspes_pre).dot(diffs_post)
    pl_res_post_scaled = _gen_placebo_stats_from_diffs(
        diffs_post_scaled[control_units, :],
        diffs_post_scaled[treated_units, :],
        max_n_pl,
        ret_pl,
        ret_CI,
        level,
    )

    if ret_CI:
        if N1 > 1:
            ind_CI = _gen_placebo_stats_from_diffs(
                diffs[control_units, :], np.zeros((1, T)), max_n_pl, False, True, level
            ).effect_vec.ci
        else:
            base = np.concatenate(
                (pl_res_pre.effect_vec.effect, pl_res_post.effect_vec.effect)
            )
            ci0 = np.concatenate(
                (pl_res_pre.effect_vec.ci.ci_low, pl_res_post.effect_vec.ci.ci_low)
            )
            ci1 = np.concatenate(
                (pl_res_pre.effect_vec.ci.ci_high, pl_res_post.effect_vec.ci.ci_high)
            )
            ind_CI = CI_int(ci0 - base, ci1 - base, level)
    else:
        ind_CI = None

    return SparseSCEstResults(
        fit_res, pl_res_pre, pl_res_post, pl_res_post_scaled, ind_CI
    )


class SparseSCEstResults(object):
    """
    Holds estimation info
    """

    # pylint: disable=redefined-outer-name
    def __init__(self, fit, pl_res_pre, pl_res_post, pl_res_post_scaled, ind_CI=None):
        """
        :param fit: The fit() return object
        :type fit: SparseSCFit
        :param pl_res_pre: Statistics for the average fit of the treated units
                in the pre-period (used for diagnostics)
        :type pl_res_pre: PlaceboResults
        :param pl_res_post: Statistics for the average treatment effect in the post-period
        :type pl_res_post: PlaceboResults
        :param pl_res_post_scaled: Statistics for the average scaled treatment
                effect (difference divided by pre-treatment RMS fit) in the
                post-period.
        :type pl_res_post_scaled: PlaceboResults
        :param ind_CI: Confidence intervals for SC predictions at the unit
                level (not averaged over N1).  Used for graphing rather than
                treatment effect statistics
        :type ind_CI: CI_int
        """
        self.fit = fit
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
            str(self.pl_res_pre.avg_joint_effect),
            str(self.pl_res_post.avg_joint_effect),
            str(self.pl_res_post.effect_vec),
        )

_SparseSCEstResults_template = """Pre-period fit diagnostic: Were we the treated harder to match in the pre-period than the controls were.
Average difference in outcome for pre-period between treated and SC unit (concerning if p-value %s ): 
%s

(Investigate per-period match quality more using self.pl_res_pre.effect_vec)

Average Effect Estimation: %s

Effect Path Estimation:
 %s
 """
