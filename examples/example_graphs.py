"""Example graphing functions
"""
# To do:
# - Allow all functions to work with an estimation result object
# - Overlay (for selection) the variables we match on 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def raw(Y, treated_units_idx, control_units_idx, treatment_period):
    fig, ax = plt.subplots(num="raw")
    # Individual controls & treated
    if len(treated_units_idx) > 1:
        lbl_t = "Treateds"
        lbl_mt = "Mean Treated"
    else:
        lbl_t = "Treated"
        lbl_mt = "Treated"
        
    if isinstance(Y, pd.DataFrame):
        plt.plot(np.transpose(Y.iloc[control_units_idx, :]), color="lightgray")
        plt.plot(Y.iloc[control_units_idx[0], :], color="lightgray", label="Controls")
        plt.plot(np.mean(Y.iloc[control_units_idx, :], axis=0), "kx--", color="dimgray", label="Mean Control")
        plt.axvline(x=treatment_period, linestyle="--")
        plt.plot(np.transpose(Y.iloc[treated_units_idx, :]), color="black")
        plt.plot(Y.iloc[treated_units_idx[0], :], color="black", label=lbl_t)
        if len(treated_units_idx) > 1:
            plt.plot(np.mean(Y.iloc[treated_units_idx, :], axis=0), color="black", label=lbl_mt)
    else:
        plt.plot(np.transpose(Y[control_units_idx, :]), color="lightgray")
        plt.plot(Y[control_units_idx[0], :], color="lightgray", label="Controls")
        plt.plot(np.mean(Y[control_units_idx, :], axis=0), "kx--", color="dimgray", label="Mean Control")
        plt.axvline(x=treatment_period, linestyle="--")
        plt.plot(np.transpose(Y[treated_units_idx, :]), color="black")
        plt.plot(Y[treated_units_idx[0], :], color="black", label=lbl_t)
        if len(treated_units_idx) > 1:
            plt.plot(np.mean(Y[treated_units_idx, :], axis=0), "kx--", color="black", label=lbl_mt)
    plt.xlabel("Time")
    plt.ylabel("Outcome")
    plt.legend(loc=1)
    return fig, ax


def sc_diff(est_ret, treatment_date, unit_idx, treatment_date_fit=None):
    fig, ax = plt.subplots(num="sc_diff")
    if isinstance(est_ret.Y, pd.DataFrame):
        Y_target = est_ret.Y.iloc[unit_idx,:]
        Y_target_sc = est_ret.get_sc(treatment_date).iloc[unit_idx,:]
    else:
        Y_target = est_ret.Y[unit_idx,:]
        Y_target_sc = est_ret.get_sc(treatment_date)[unit_idx,:]

    diff = Y_target - Y_target_sc
    if est_ret.ind_CI is not None:
        ind_ci = est_ret.ind_CI[treatment_date]
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
    if treatment_date_fit is not None:
        plt.axvline(x=treatment_date, linestyle="--", label="Treatment")
        plt.axvline(x=treatment_date_fit, linestyle=":", label="End Fit Window")
    else:
        plt.axvline(x=treatment_date, linestyle="--")
    plt.xlabel("Time")
    plt.ylabel("Real-SC Outcome Difference")
    plt.legend(loc=1)
    return fig, ax

def sc_raw(est_ret, treatment_date, unit_idx, treatment_date_fit=None):
    fig, ax = plt.subplots(num="sc_raw")
    if isinstance(est_ret.Y, pd.DataFrame):
        Y_target = est_ret.Y.iloc[unit_idx,:]
        Y_target_sc = est_ret.get_sc(treatment_date).iloc[unit_idx,:]
    else:
        Y_target = est_ret.Y[unit_idx,:]
        Y_target_sc = est_ret.get_sc(treatment_date)[unit_idx,:]

    if est_ret.ind_CI is not None:
        ind_ci = est_ret.ind_CI[treatment_date]
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
    if treatment_date_fit is not None:
        plt.axvline(x=treatment_date, linestyle="--", label="Treatment")
        plt.axvline(x=treatment_date_fit, linestyle=":", label="End Fit Window")
    else:
        plt.axvline(x=treatment_date, linestyle="--")
    plt.plot(Y_target, "bx-", label="Unit")
    plt.plot(Y_target_sc, "gx--", label="SC")
    plt.xlabel("Time")
    plt.ylabel("Outcome")
    plt.legend(loc=1)
    return fig, ax


def te_plot(est_ret, treatment_date, treatment_date_fit=None):
    fig, ax = plt.subplots(num="te_plot")
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
    if treatment_date_fit is not None:
        plt.axvline(x=treatment_date, linestyle="--", label="Treatment")
        plt.axvline(x=treatment_date_fit, linestyle=":", label="End Fit Window")
    else:
        plt.axvline(x=treatment_date, linestyle="--")
    plt.axhline(y=0, linestyle="--")
    plt.xlabel("Time")
    plt.ylabel("Real-SC Outcome Difference")
    plt.legend(loc=1)
    return fig, ax

def diffs_plot(diffs, treated_units_idx, control_units_idx, treatment_date, est_ret=None, treatment_date_fit=None):
    #include est_ret for CI (usefull for combining Diff and TE graphs when 1 treated unit)
    fig, ax = plt.subplots(num="diffs_plot")
    if len(treated_units_idx) > 1:
        lbl_t = "Treated Diffs"
    else:
        lbl_t = "Treated Diff"
    if treatment_date_fit is not None:
        plt.axvline(x=treatment_date, linestyle="--", label="Treatment")
        plt.axvline(x=treatment_date_fit, linestyle=":", label="End Fit Window")
    else:
        plt.axvline(x=treatment_date, linestyle="--")
    
    if isinstance(diffs, pd.DataFrame):
        plt.plot(diffs.iloc[control_units_idx, :].T, alpha=0.5, color="lightgray")
        plt.plot(diffs.iloc[control_units_idx[0], :], alpha=0.5, color="lightgray", label="Control Diffs")

        if est_ret is not None:
            ci0 = pd.concat((est_ret.pl_res_pre.effect_vec.ci.ci_low, 
                                est_ret.pl_res_post.effect_vec.ci.ci_low))
            ci1 = pd.concat((est_ret.pl_res_pre.effect_vec.ci.ci_high,
                                est_ret.pl_res_post.effect_vec.ci.ci_high))
            plt.plot(ci0, color="dimgray")
            plt.plot(ci1, color="dimgray", label="CI")

        plt.plot(diffs.iloc[treated_units_idx, :].T, color="black")
        plt.plot(diffs.iloc[treated_units_idx[0], :], color="black", label=lbl_t)
    else:
        plt.plot(np.transpose(diffs[control_units_idx, :]), alpha=0.5, color="lightgray")
        plt.plot(diffs[control_units_idx[0], :], alpha=0.5, color="lightgray", label="Control Diffs")
        
        if est_ret is not None:
            ci0 = np.concatenate((est_ret.pl_res_pre.effect_vec.ci.ci_low, 
                                    est_ret.pl_res_post.effect_vec.ci.ci_low))
            ci1 = np.concatenate((est_ret.pl_res_pre.effect_vec.ci.ci_high,
                                    est_ret.pl_res_post.effect_vec.ci.ci_high))
            plt.plot(range(len(ci0)), ci0, color="dimgray")
            plt.plot(range(len(ci0)), ci1, color="dimgray", label="CI")

        
        plt.plot(np.transpose(diffs[treated_units_idx, :]), color="black")
        plt.plot(diffs[treated_units_idx[0], :], color="black", label=lbl_t)
    plt.xlabel("Time")
    plt.ylabel("Real-SC Outcome Difference")
    plt.legend(loc=1)
    return fig, ax
