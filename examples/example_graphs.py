import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


def raw_plots(Y, treated_units_idx, control_units_idx, treatment_period):
    # Individual controls & treated
    if len(treated_units_idx) > 1:
        lbl_t = "Treateds"
        lbl_mt = "Mean Treated"
    else:
        lbl_t = "Treated"
        lbl_mt = "Treated"
        
    istat = matplotlib.is_interactive()
    plt.ioff()
    raw_all_fig, raw_all_ax = plt.subplots(num="raw_all")
    if isinstance(Y, pd.DataFrame):
        raw_all_ax.plot(np.transpose(Y.iloc[control_units_idx, :]), color="gray")
        raw_all_ax.plot(Y.iloc[control_units_idx[0], :], color="gray", label="Controls")
        raw_all_ax.axvline(x=treatment_period, linestyle="--")
        raw_all_ax.plot(np.transpose(Y.iloc[treated_units_idx, :]), color="black")
        raw_all_ax.plot(Y.iloc[treated_units_idx[0], :], color="black", label=lbl_t)
    else:
        raw_all_ax.plot(np.transpose(Y[control_units_idx, :]), color="gray")
        raw_all_ax.plot(Y[control_units_idx[0], :], color="gray", label="Controls")
        raw_all_ax.axvline(x=treatment_period, linestyle="--")
        raw_all_ax.plot(np.transpose(Y[treated_units_idx, :]), color="black")
        raw_all_ax.plot(Y[treated_units_idx[0], :], color="black", label=lbl_t)
    raw_all_ax.set_xlabel("Time")
    raw_all_ax.set_ylabel("Outcome")
    raw_all_ax.legend(loc=1)

    # Means controls & treated
    raw_means_fig, raw_means_ax = plt.subplots(num="raw_means")
    raw_means_ax.axvline(x=treatment_period, linestyle="--")
    if isinstance(Y, pd.DataFrame):
        raw_means_ax.plot(np.mean(Y.iloc[control_units_idx, :], axis=0), color="gray", label="Mean Control")
        raw_means_ax.plot(np.mean(Y.iloc[treated_units_idx, :], axis=0), color="black", label=lbl_mt)
    else:
        raw_means_ax.plot(np.mean(Y[control_units_idx, :], axis=0), color="gray", label="Mean Control")
        raw_means_ax.plot(np.mean(Y[treated_units_idx, :], axis=0), color="black", label=lbl_mt)
    raw_means_ax.set_xlabel("Time")
    raw_means_ax.set_ylabel("Outcome")
    raw_means_ax.legend(loc=1)
    if istat:
        plt.ion()
    return [raw_all_fig, raw_means_fig]

def raw_all(Y, treated_units_idx, control_units_idx, treatment_period):
    # Individual controls & treated
    if len(treated_units_idx) > 1:
        lbl_t = "Treateds"
    else:
        lbl_t = "Treated"
        
    if isinstance(Y, pd.DataFrame):
        plt.plot(np.transpose(Y.iloc[control_units_idx, :]), color="gray")
        plt.plot(Y.iloc[control_units_idx[0], :], color="gray", label="Controls")
        plt.axvline(x=treatment_period, linestyle="--")
        plt.plot(np.transpose(Y.iloc[treated_units_idx, :]), color="black")
        plt.plot(Y.iloc[treated_units_idx[0], :], color="black", label=lbl_t)
    else:
        plt.plot(np.transpose(Y[control_units_idx, :]), color="gray")
        plt.plot(Y[control_units_idx[0], :], color="gray", label="Controls")
        plt.axvline(x=treatment_period, linestyle="--")
        plt.plot(np.transpose(Y[treated_units_idx, :]), color="black")
        plt.plot(Y[treated_units_idx[0], :], color="black", label=lbl_t)
    plt.xlabel("Time")
    plt.ylabel("Outcome")
    plt.legend(loc=1)



def raw_means(Y, treated_units_idx, control_units_idx, treatment_period):
    # Individual controls & treated
    if len(treated_units_idx) > 1:
        lbl_mt = "Mean Treated"
    else:
        lbl_mt = "Treated"
        
    plt.axvline(x=treatment_period, linestyle="--")
    if isinstance(Y, pd.DataFrame):
        plt.plot(np.mean(Y.iloc[control_units_idx, :], axis=0), color="gray", label="Mean Control")
        plt.plot(np.mean(Y.iloc[treated_units_idx, :], axis=0), color="black", label=lbl_mt)
    else:
        plt.plot(np.mean(Y[control_units_idx, :], axis=0), color="gray", label="Mean Control")
        plt.plot(np.mean(Y[treated_units_idx, :], axis=0), color="black", label=lbl_mt)
    plt.xlabel("Time")
    plt.ylabel("Outcome")
    plt.legend(loc=1)




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

def sc_diff(est_ret, treatment_date_idx, unit_idx, treatment_date):
    if isinstance(est_ret.Y, pd.DataFrame):
        Y_target = est_ret.Y.iloc[unit_idx,:]
        Y_target_sc = est_ret.get_sc(treatment_date_idx).iloc[unit_idx,:]
    else:
        Y_target = est_ret.Y[unit_idx,:]
        Y_target_sc = est_ret.get_sc(treatment_date_idx)[unit_idx,:]

    diff = Y_target - Y_target_sc
    if est_ret.ind_CI is not None:
        ind_ci = est_ret.ind_CI[treatment_date_idx]
        if isinstance(est_ret.Y, pd.DataFrame):
            fb_index = Y_target.index
        else:
            fb_index = range(len(ind_ci.ci_low))
        plt.fill_between(
            fb_index,
            diff + ind_ci.ci_low,
            diff + ind_ci.ci_high,
            facecolor="gray",
            label="CI",
        )
    plt.axhline(y=0, linestyle="--")
    plt.plot(diff, "kx--", label="Unit Diff")
    plt.axvline(x=treatment_date, linestyle="--")
    plt.xlabel("Time")
    plt.ylabel("Real-SC Outcome Difference")
    plt.legend(loc=1)

def sc_raw(est_ret, treatment_date_idx, unit_idx, treatment_date):
    if isinstance(est_ret.Y, pd.DataFrame):
        Y_target = est_ret.Y.iloc[unit_idx,:]
        Y_target_sc = est_ret.get_sc(treatment_date_idx).iloc[unit_idx,:]
    else:
        Y_target = est_ret.Y[unit_idx,:]
        Y_target_sc = est_ret.get_sc(treatment_date_idx)[unit_idx,:]

    if est_ret.ind_CI is not None:
        ind_ci = est_ret.ind_CI[treatment_date_idx]
        if isinstance(est_ret.Y, pd.DataFrame):
            fb_index = Y_target.index
        else:
            fb_index = range(len(Y_target_sc))
        plt.fill_between(
            fb_index,
            Y_target_sc + ind_ci.ci_low,
            Y_target_sc + ind_ci.ci_high,
            facecolor="gray",
            label="CI",
        )
    plt.axvline(x=treatment_date, linestyle="--")
    plt.plot(Y_target, "bx-", label="Unit")
    plt.plot(Y_target_sc, "gx--", label="SC")
    plt.xlabel("Time")
    plt.ylabel("Outcome")
    plt.legend(loc=1)


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

def te_plot2(est_ret, treatment_date):
    import numpy as np

    if isinstance(est_ret.pl_res_pre.effect_vec.effect, pd.Series):
        effect_vec = pd.concat((est_ret.pl_res_pre.effect_vec.effect, 
                                est_ret.pl_res_post.effect_vec.effect))
    else:
        effect_vec = np.concatenate((est_ret.pl_res_pre.effect_vec.effect, 
                                     est_ret.pl_res_post.effect_vec.effect))
    if est_ret.pl_res_pre.effect_vec.ci is not None:
        if isinstance(est_ret.pl_res_pre.effect_vec.ci.ci_low, pd.Series):
            ci0 = pd.concat((est_ret.pl_res_pre.effect_vec.ci.ci_low, 
                             est_ret.pl_res_post.effect_vec.ci.ci_low))
            ci1 = pd.concat((est_ret.pl_res_pre.effect_vec.ci.ci_high,
                             est_ret.pl_res_post.effect_vec.ci.ci_high))
            plt.fill_between(ci0.index, ci0, ci1, facecolor="gray", label="CI")
        else:
            ci0 = np.concatenate((est_ret.pl_res_pre.effect_vec.ci.ci_low, 
                                  est_ret.pl_res_post.effect_vec.ci.ci_low))
            ci1 = np.concatenate((est_ret.pl_res_pre.effect_vec.ci.ci_high,
                                  est_ret.pl_res_post.effect_vec.ci.ci_high))
            plt.fill_between(range(len(ci0)), ci0, ci1, facecolor="gray", label="CI")

    plt.plot(effect_vec, "kx--", label="Treated Diff")
    plt.axvline(x=treatment_date, linestyle="--")
    plt.axhline(y=0, linestyle="--")
    plt.xlabel("Time")
    plt.ylabel("Real-SC Outcome Difference")
    plt.legend(loc=1)


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

def diffs_plot2(diffs, treated_units_idx, control_units_idx, treatment_date):
    if len(treated_units_idx) > 1:
        lbl_t = "Treated Diffs"
    else:
        lbl_t = "Treated Diff"
    plt.axvline(x=treatment_date, linestyle="--")
    
    if isinstance(diffs, pd.DataFrame):
        plt.plot(diffs.iloc[control_units_idx, :].T, alpha=0.5, color="gray")
        plt.plot(diffs.iloc[control_units_idx[0], :], alpha=0.5, color="gray", label="Control Diffs")
        plt.plot(diffs.iloc[treated_units_idx, :].T, color="black")
        plt.plot(diffs.iloc[treated_units_idx[0], :], color="black", label=lbl_t)
    else:
        plt.plot(np.transpose(diffs[control_units_idx, :]), alpha=0.5, color="gray")
        plt.plot(diffs[control_units_idx[0], :], alpha=0.5, color="gray", label="Control Diffs")
        plt.plot(np.transpose(diffs[treated_units_idx, :]), color="black")
        plt.plot(diffs[treated_units_idx[0], :], color="black", label=lbl_t)
    plt.xlabel("Time")
    plt.ylabel("Real-SC Outcome Difference")
    plt.legend(loc=1)
