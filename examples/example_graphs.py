import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def raw_plots(Y, treated_units, control_units, T0):
    # Individual controls & treated
    if len(treated_units) > 1:
        lbl_t = "Treateds"
        lbl_mt = "Mean Treated"
    else:
        lbl_t = "Treated"
        lbl_mt = "Treated"
        
    istat = matplotlib.is_interactive()
    plt.ioff()
    raw_all_fig, raw_all_ax = plt.subplots(num="raw_all")
    raw_all_ax.axvline(x=T0, linestyle="--")
    raw_all_ax.plot(np.transpose(Y[control_units, :]), color="gray")
    raw_all_ax.plot(Y[control_units[0], :], color="gray", label="Controls")
    raw_all_ax.plot(np.transpose(Y[treated_units, :]), color="black")
    raw_all_ax.plot(Y[treated_units[0], :], color="black", label=lbl_t)
    raw_all_ax.set_xlabel("Time")
    raw_all_ax.set_ylabel("Outcome")
    raw_all_ax.legend(loc=1)
    # Means controls & treated
    raw_means_fig, raw_means_ax = plt.subplots(num="raw_means")
    raw_means_ax.axvline(x=T0, linestyle="--")
    raw_means_ax.plot(np.mean(Y[control_units, :], axis=0), color="gray", label="Mean Control")
    raw_means_ax.plot(np.mean(Y[treated_units, :], axis=0), color="black", label=lbl_mt)
    raw_means_ax.set_xlabel("Time")
    raw_means_ax.set_ylabel("Outcome")
    raw_means_ax.legend(loc=1)
    if istat:
        plt.ion()
    return [raw_all_fig, raw_means_fig]


#def ind_sc_plots(Y, Y_sc, T0, ind_ci=None):
def ind_sc_plots(est_ret, treatment_date, unit):
    Y = est_ret.Y[unit,:]
    Y_sc_full = est_ret.get_sc(treatment_date)
    Y_sc = Y_sc_full[unit,:]
    T0 = est_ret.T0
    if est_ret.ind_CI is not None:
        ind_ci = est_ret.ind_CI[treatment_date]
    else:
        ind_ci = None
    istat = matplotlib.is_interactive()
    plt.ioff()
    sc_raw_fig, sc_raw_ax = plt.subplots(num="sc_raw")
    if ind_ci is not None:
        sc_raw_ax.fill_between(
            range(len(Y_sc)),
            Y_sc + ind_ci.ci_low,
            Y_sc + ind_ci.ci_high,
            facecolor="gray",
            label="CI",
        )
    sc_raw_ax.axvline(x=T0, linestyle="--")
    sc_raw_ax.plot(Y, "bx-", label="Unit")
    sc_raw_ax.plot(Y_sc, "gx--", label="SC")
    sc_raw_ax.set_xlabel("Time")
    sc_raw_ax.set_ylabel("Outcome")
    sc_raw_ax.legend(loc=1)

    sc_diff_fig, sc_diff_ax = plt.subplots(num="sc_diff")
    diff = Y - Y_sc
    if ind_ci is not None:
        sc_diff_ax.fill_between(
            range(len(ind_ci.ci_low)),
            diff + ind_ci.ci_low,
            diff + ind_ci.ci_high,
            facecolor="gray",
            label="CI",
        )
    sc_diff_ax.axvline(x=T0, linestyle="--")
    sc_diff_ax.axhline(y=0, linestyle="--")
    sc_diff_ax.plot(diff, "kx--", label="Unit Diff")
    sc_diff_ax.set_xlabel("Time")
    sc_diff_ax.set_ylabel("Real-SC Outcome Difference")
    sc_diff_ax.legend(loc=1)
    if istat:
        plt.ion()
    return [sc_raw_fig, sc_diff_fig]


def te_plot(est_ret):
    import numpy as np

    T0 = len(est_ret.pl_res_pre.effect_vec.effect)
    effect_vec = np.concatenate(
        (est_ret.pl_res_pre.effect_vec.effect, est_ret.pl_res_post.effect_vec.effect)
    )
    if est_ret.pl_res_pre.effect_vec.ci is not None:
        ci0 = np.concatenate(
            (
                est_ret.pl_res_pre.effect_vec.ci.ci_low,
                est_ret.pl_res_post.effect_vec.ci.ci_low,
            )
        )
        ci1 = np.concatenate(
            (
                est_ret.pl_res_pre.effect_vec.ci.ci_high,
                est_ret.pl_res_post.effect_vec.ci.ci_high,
            )
        )
    istat = matplotlib.is_interactive()
    plt.ioff()
    te_fig, te_ax = plt.subplots(num="te")
    if est_ret.pl_res_pre.effect_vec.ci is not None:
        te_ax.fill_between(range(len(ci0)), ci0, ci1, facecolor="gray", label="CI")
    te_ax.plot(effect_vec, "kx--", label="Treated Diff")
    te_ax.axvline(x=T0, linestyle="--")
    te_ax.axhline(y=0, linestyle="--")
    te_ax.set_xlabel("Time")
    te_ax.set_ylabel("Real-SC Outcome Difference")
    te_ax.legend(loc=1)
    if istat:
        plt.ion()
    return [te_fig]


def diffs_plot(diffs, treated_units, control_units):
    if len(treated_units) > 1:
        lbl_t = "Treated Diffs"
    else:
        lbl_t = "Treated Diff"
    istat = matplotlib.is_interactive()
    plt.ioff()
    diffs_plt, diffs_plt_ax = plt.subplots(num="diffs_fig")
    diffs_plt_ax.axvline(x=T0, linestyle="--")
    diffs_plt_ax.plot(np.transpose(diffs[control_units, :]), alpha=0.5, color="gray")
    diffs_plt_ax.plot(diffs[control_units[0], :], alpha=0.5, color="gray", label="Control Diffs")
    diffs_plt_ax.plot(np.transpose(diffs[treated_units, :]), color="black")
    diffs_plt_ax.plot(diffs[treated_units[0], :], color="black", label=lbl_t)
    diffs_plt_ax.set_xlabel("Time")
    diffs_plt_ax.set_ylabel("Real-SC Outcome Difference")
    diffs_plt_ax.legend(loc=1)
    if istat:
        plt.ion()
    return diffs_plt
