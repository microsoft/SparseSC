import numpy as np
import itertools
import warnings
from collections import namedtuple

def simulation_eval(effects, CI_lowers, CI_uppers, true_effect=0):
    te_mse = np.mean(np.square((effects-true_effect)))
    cov = np.mean(np.logical_and(effects>=CI_lowers, effects <=CI_uppers).astype(int))
    ci_len = np.mean(CI_uppers-CI_lowers)
    return (te_mse, cov, ci_len)

EstResultCI = namedtuple('EstResults', 'effect p ci placebos')
PlaceboResults = namedtuple('PlaceboResults', 'effect_vec avg_joint_effect rms_joint_effect N_placebo')

def gen_placebo_stats_from_diffs(effect_vecs, control_effect_vecs, 
                                 max_n_pl = 1000000, ret_pl = False, ret_CI=False, level=0.95):
    """Generates placebo distribution to compare effects against. 
    For a single treated unit  this is just the control effects.
    If there are multiple treated units then the averaging process needs to be
    done to generate placebos also.
    Generates 2-sided p-values
  
    :param effect_vecs:
    :param control_effect_vecs:
    :param max_n_pl:
    :param ret_pl:
    :param ret_CI:
    :param level:
    """
    N1 = effect_vecs.shape[0]
    N0 = control_effect_vecs.shape[0]
    T1 = effect_vecs.shape[1]

    keep_pl = ret_pl or ret_CI

    #Get rest of the outcomes (already have effect_vecs)
    ##Get the RMSE joint effects 
    rms_joint_effects = np.sqrt(np.mean(np.square(effect_vecs), axis=1))
    control_rms_joint_effects = np.sqrt(np.mean(np.square(control_effect_vecs), axis=1))
    ##Get the avg joint effects
    avg_joint_effects = np.mean(effect_vecs, axis=1)
    control_avg_joint_effects = np.mean(control_effect_vecs, axis=1)

    #Compute the outcomes for treatment
    effect_vec = np.mean(effect_vecs, axis=0)
    rms_joint_effect = np.mean(rms_joint_effects)
    avg_joint_effect = np.mean(avg_joint_effects)

    
    def _ncr(n, r):
        #https://stackoverflow.com/questions/4941753/is-there-a-math-ncr-function-in-python
        import operator as op
        import functools
        r = min(r, n-r)
        numer = functools.reduce(op.mul, range(n, n-r, -1), 1) #from py2 xrange()
        denom = functools.reduce(op.mul, range(1, r+1), 1) #from py2 xrange()
        return numer//denom


    def _random_combination(iterable, r):
        "Random selection from itertools.combinations(iterable, r)"
        #https://stackoverflow.com/questions/22229796/choose-at-random-from-combinations
        import random
        pool = tuple(iterable)
        n = len(pool)
        indices = sorted(random.sample(range(n), r))
        return tuple(pool[i] for i in indices)

    def _repeatfunc(func, times=None, *args):
        #Repeat calls to func with specified arguments.
        #Example:  _repeatfunc(random.random)
        if times is None:
            return itertools.starmap(func, itertools.repeat(args))
        return itertools.starmap(func, itertools.repeat(args, times))

    n_pl = _ncr(N0, N1)
    if max_n_pl > 0 & n_pl > max_n_pl: #randomize
        comb_iter = itertools.combinations(range(N0), N1)
        comb_len = max_n_pl
    else:
        comb_iter = _repeatfunc(_random_combination, n_pl, range(N0), N1)
        comb_len = n_pl

    if keep_pl:
        placebo_effect_vecs = np.empty((comb_len,T1))
        placebo_avg_joint_effects = np.empty(comb_len)
        placebo_rms_joint_effects = np.empty(comb_len)
    else:
        placebo_effect_vecs = None
        placebo_avg_joint_effects = None
        placebo_rms_joint_effects = None
    vec_p = np.zeros((1,T1))
    rms_joint_p = 0
    avg_joint_p = 0

    for idx, comb in enumerate(comb_iter):
        placebo_effect_vec = np.mean(control_effect_vecs[comb,:], 0)
        placebo_rms_joint_effect = np.mean(control_rms_joint_effects[comb])
        placebo_avg_joint_effect = np.mean(control_avg_joint_effects[comb])
            
        #p1s += (effect_vec_sgn*placebo_effect_vec >= effect_vec_sgn*effect_vec)
        #p1s_std += (effect_vec_sgn*placebo_std_effect_vec >= effect_vec_sgn*std_effect_vec)
        vec_p += (abs(placebo_effect_vec) >= abs(effect_vec))
        rms_joint_p += (placebo_rms_joint_effect >= rms_joint_effect)
        avg_joint_p += (abs(placebo_avg_joint_effect) >= abs(avg_joint_effect))
        if keep_pl:
            placebo_effect_vecs[idx,:] = placebo_effect_vec
            placebo_avg_joint_effects[idx] = placebo_avg_joint_effect
            placebo_rms_joint_effects[idx] = placebo_rms_joint_effect

    def _pval_cal(npl_at_least_as_large, npl, incl_actual_in_set=True):
        """ADH10 incl_actual_in_set=True, CGNP13, ADH15 do not
        It depends on whether you (do|do not) you think the actual test is one of 
        the possible randomizations.
        p2s = 2*p1s #Ficher 2-sided p-vals (less common)
        """
        addition = int(incl_actual_in_set)
        return (npl_at_least_as_large + addition)/(npl + addition)

    vec_p = _pval_cal(vec_p, comb_len)
    rms_joint_p = _pval_cal(rms_joint_p, comb_len)
    avg_joint_p = _pval_cal(avg_joint_p, comb_len)
    
    if ret_CI:
        #CI - All hypothetical true effects (beta0) that would not be reject at the certain level
        # To test non-zero beta0, apply beta0 to get unexpected deviation beta_hat-beta0 and compare to permutation distribution
        # This means that we take the level-bounds of the permutation distribution then "flip it around beta_hat"
        # To make the math a bit nicer, I will reject a hypothesis if pval<=(1-level)
        assert level<1 and level>0; "Use a level in [0,1]"
        alpha = (1-level)
        p2min = 2/n_pl
        alpha_ind = max((1,round(alpha/p2min)))
        alpha = alpha_ind* p2min

        def _gen_CI(placebo_effects, alpha_ind, effect):
            npl = placebo_effects.shape[0]
            sorted_eff = np.sort(placebo_effects)
            low_avg_effect = sorted_eff[alpha_ind]
            high_avg_effect = sorted_eff[(npl+1)-alpha_ind]
            if np.sign(low_avg_effect)==np.sign(high_avg_effect):
                warnings.warn("CI doesn't containt effect. You might not have enough placebo effects.")
            return (effect - high_avg_effect, effect - low_avg_effect) 

        CI_vec = np.empty((2,T1))
        for t in range(T1):
            CI_vec[:,t] = _gen_CI(placebo_effect_vecs[:,t], alpha_ind, effect_vec[t])

        CI_avg = _gen_CI(placebo_avg_joint_effects, alpha_ind, avg_joint_effect)
        CI_rms = _gen_CI(placebo_rms_joint_effects, alpha_ind, rms_joint_effect)

    else:
        CI_vec = None
        CI_avg = None
        CI_rms = None

        
    ret_struct = PlaceboResults(EstResultCI(effect_vec, vec_p, CI_vec, placebo_effect_vecs), 
                                EstResultCI(avg_joint_effect, avg_joint_p, CI_avg, placebo_avg_joint_effects), 
                                EstResultCI(rms_joint_effect, rms_joint_p, CI_rms, placebo_rms_joint_effects),
                                comb_len)

        
    return ret_struct
