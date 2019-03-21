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

    raw_all = plt.figure("raw_all")
    plt.axvline(x=T0, linestyle="--")
    plt.plot(np.transpose(Y[control_units, :]), color="gray")
    plt.plot(Y[control_units[0], :], color="gray", label="Controls")
    plt.plot(np.transpose(Y[treated_units, :]), color="black")
    plt.plot(Y[treated_units[0], :], color="black", label=lbl_t)
    plt.xlabel("Time")
    plt.ylabel("Outcome")
    plt.legend(loc=1)
    # Means controls & treated
    raw_means = plt.figure("raw_means")
    plt.axvline(x=T0, linestyle="--")
    plt.plot(np.mean(Y[control_units, :], axis=0), color="gray", label="Mean Control")
    plt.plot(np.mean(Y[treated_units, :], axis=0), color="black", label=lbl_mt)
    plt.xlabel("Time")
    plt.ylabel("Outcome")
    plt.legend(loc=1)
    return [raw_all, raw_means]


def ind_sc_plots(Y, Y_sc, T0, ind_ci=None):
    sc_raw = plt.figure("sc_raw")
    if ind_ci is not None:
        plt.fill_between(
            range(len(Y_sc)),
            Y_sc + ind_ci.ci_low,
            Y_sc + ind_ci.ci_high,
            facecolor="gray",
            label="CI",
        )
    plt.axvline(x=T0, linestyle="--")
    plt.plot(Y, "bx-", label="Unit")
    plt.plot(Y_sc, "gx--", label="SC")
    plt.xlabel("Time")
    plt.ylabel("Outcome")
    plt.legend(loc=1)

    sc_diff = plt.figure("sc_diff")
    diff = Y - Y_sc
    if ind_ci is not None:
        plt.fill_between(
            range(len(ind_ci.ci_low)),
            diff + ind_ci.ci_low,
            diff + ind_ci.ci_high,
            facecolor="gray",
            label="CI",
        )
    plt.axvline(x=T0, linestyle="--")
    plt.axhline(y=0, linestyle="--")
    plt.plot(diff, "kx--", label="Unit Diff")
    plt.xlabel("Time")
    plt.ylabel("Real-SC Outcome Difference")
    plt.legend(loc=1)
    return [sc_raw, sc_diff]


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
    te = plt.figure("te")
    if est_ret.pl_res_pre.effect_vec.ci is not None:
        plt.fill_between(range(len(ci0)), ci0, ci1, facecolor="gray", label="CI")
    plt.plot(effect_vec, "kx--", label="Treated Diff")
    plt.axvline(x=T0, linestyle="--")
    plt.axhline(y=0, linestyle="--")
    plt.xlabel("Time")
    plt.ylabel("Real-SC Outcome Difference")
    plt.legend(loc=1)
    return [te]


def diffs_plot(diffs, treated_units, control_units):
    if len(treated_units) > 1:
        lbl_t = "Treated Diffs"
    else:
        lbl_t = "Treated Diff"
    diffs_plt = plt.figure("diffs_fig")
    plt.axvline(x=T0, linestyle="--")
    plt.plot(np.transpose(diffs[control_units, :]), alpha=0.5, color="gray")
    plt.plot(diffs[control_units[0], :], alpha=0.5, color="gray", label="Control Diffs")
    plt.plot(np.transpose(diffs[treated_units, :]), color="black")
    plt.plot(diffs[treated_units[0], :], color="black", label=lbl_t)
    plt.xlabel("Time")
    plt.ylabel("Real-SC Outcome Difference")
    plt.legend(loc=1)
    return diffs_plt
