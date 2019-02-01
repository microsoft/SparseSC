import numpy as np

def simulation_eval(effects, CI_lowers, CI_uppers, true_effect=0):
    te_mse = np.mean(np.square((effects-true_effect)))
    cov = np.mean(np.logical_and(effects>=CI_lowers, effects <=CI_uppers).astype(int))
    ci_len = np.mean(CI_uppers-CI_lowers)
    return (te_mse, cov, ci_len)


def gen_placebo_stats_from_diffs(effect_vecs, control_effect_vecs, 
                                  tr_scalings = None, ct_scalings = None,
                                  max_n_pl = 1000000, ret_pl = False, ret_CI=False, level=0.95):
    """Generates placebo distribution to compare effects against. 
    For a single treated unit  this is just the control effects.
    If there are multiple treated units then the averaging process needs to be done to generate placebos also.
  
    
    :param tr_scalings: Usually the pre-treatment RMS prediction error
    :param ct_scalings: Usually the pre-treatment RMS prediction error
    :param max_n_pl:
    """
    N1 = effect_vecs.shape[0]
    N0 = control_effect_vecs.shape[0]
    T1 = effect_vecs.shape[1]
    #ret_p1s=False
    keep_pl = ret_pl or ret_CI

    #Get rest of the outcomes (already have effect_vecs)
    ##Get the RMSE joint effects 
    rms_joint_effects = np.sqrt(np.mean(np.square(effect_vecs), axis=1))
    control_rms_joint_effects = np.sqrt(np.mean(np.square(control_effect_vecs), axis=1))
    ##Get the avg joint effects
    avg_joint_effects = np.mean(effect_vecs, axis=1)
    control_avg_joint_effects = np.mean(control_effect_vecs, axis=1)
    if tr_scalings is not None and ct_scalings is not None:
        ## Standardized effect vecs
        std_effect_vecs = np.diagflat(1/tr_scalings).dot(effect_vecs)
        control_std_effect_vecs = np.diagflat(1/ ct_scalings).dot(control_effect_vecs)
        ##Get the standardized RMS joint effects
        rms_joint_std_effects = np.multiply((1 / tr_scalings), rms_joint_effects)
        control_rms_joint_std_effects = np.multiply((1/ ct_scalings), control_rms_joint_effects) 
        ##Get the standardized avg joint effects
        avg_joint_std_effects = np.multiply((1 / tr_scalings), avg_joint_effects)
        control_avg_joint_std_effects = np.multiply((1/ ct_scalings), control_avg_joint_effects) 

    #Compute the outcomes for treatment
    effect_vec = np.mean(effect_vecs, axis=0)
    rms_joint_effect = np.mean(rms_joint_effects)
    avg_joint_effect = np.mean(avg_joint_effects)
    if tr_scalings is not None and ct_scalings is not None:
        std_effect_vec = np.mean(std_effect_vecs, axis=0)
        rms_joint_std_effect = np.mean(rms_joint_std_effects)
        avg_joint_std_effect = np.mean(avg_joint_std_effects)

    
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
    if (max_n_pl > 0 & n_pl > max_n_pl): #randomize
        comb_iter = itertools.combinations(range(N0), N1)
        comb_len = max_n_pl
    else:
        comb_iter = _repeatfunc(_random_combination, n_pl, range(N0), N1)
        comb_len = n_pl
    placebo_effect_vecs = None
    if keep_pl:
        placebo_effect_vecs = np.empty((comb_len,T1))
        placebo_avg_joint_effects = np.empty(comb_len)
    #p1s = np.zero((1,T1))
    #p1s_std = np.zero((1,T1))
    #effect_vec_sgn = np.sign(effect_vec)
    p2s = np.zeros((1,T1))
    rms_joint_p = 0
    avg_joint_p = 0
    if tr_scalings is not None and ct_scalings is not None:
        p2s_std = np.zeros((1,T1))
        rms_joint_std_p = 0
        avg_joint_std_p = 0

    for idx, comb in enumerate(comb_iter):
        placebo_effect_vec = np.mean(control_effect_vecs[comb,:], 0)
        placebo_rms_joint_effect = np.mean(control_rms_joint_effects[comb,:])
        placebo_avg_joint_effect = np.mean(control_avg_joint_effects[comb,:])
        if tr_scalings is not None and ct_scalings is not None:
            placebo_std_effect_vec = np.mean(control_std_effect_vecs[comb,:], 0)
            placebo_rms_joint_std_effect = np.mean(control_rms_joint_std_effects[comb,:])
            placebo_avg_joint_std_effect = np.mean(control_avg_joint_std_effects[comb,:])
            
        #p1s += (effect_vec_sgn*placebo_effect_vec >= effect_vec_sgn*effect_vec)
        #p1s_std += (effect_vec_sgn*placebo_std_effect_vec >= effect_vec_sgn*std_effect_vec)
        p2s += (abs(placebo_effect_vec) >= abs(effect_vec))
        rms_joint_p += (placebo_rms_joint_effect >= rms_joint_effect)
        avg_joint_p += (abs(placebo_avg_joint_effect) >= abs(avg_joint_effect))
        if tr_scalings is not None and ct_scalings is not None:
            p2s_std += (abs(placebo_std_effect_vec) >= abs(std_effect_vec))
            rms_joint_std_p += (placebo_rms_joint_std_effect >= rms_joint_std_effect)
            avg_joint_std_p += (abs(placebo_avg_joint_std_effect) >= abs(avg_joint_std_effect))
        if keep_pl:
            placebo_effect_vecs[idx,:] = placebo_effect_vec
            placebo_avg_joint_effects[idx] = placebo_avg_joint_effect
    #p1s = p1s/comb_len
    #p1s_std = p1s_std/comb_len
    #p2s = 2*p1s #Ficher 2-sided p-vals (less common)
    p2s = p2s/comb_len
    rms_joint_p = rms_joint_p/comb_len
    avg_joint_p = avg_joint_p/comb_len
    if tr_scalings is not None and ct_scalings is not None:
        p2s_std = p2s_std/comb_len
        rms_joint_std_p = rms_joint_std_p/comb_len
        avg_joint_std_p = avg_joint_std_p/comb_len
    else:
        p2s_std = None; rms_joint_std_p = None; avg_joint_std_p = None
    
    if ret_CI:
        #CI - All hypothetical true effects (beta0) that would not be reject at the certain level
        # To test non-zero beta0, apply beta0 to get unexpected deviation beta_hat-beta0 and compare to permutation distribution
        # This means that we take the level-bounds of the permutation distribution then "flip it around beta_hat"
        # To make the math a bit nicer, I will reject a hypothesis if pval<=(1-level)
        assert level<1 & level>0; "Use a level in [0,1]"
        alpha = (1-level)
        p2min = 2/n_pl
        alpha_ind = max((1,round(alpha/p2min)))
        alpha = alpha_ind* p2min
        CI_vec = np.empty((2,T1))
        for t in range(T1):
            sorted_eff = np.sort(placebo_effect_vecs[:,t]) #TODO: check with Stata about sort order (here and below)
            low_effect = sorted_eff[alpha_ind]
            high_effect = sorted_eff[(comb_len+1)-alpha_ind]
            if np.sign(low_effect)==np.sign(high_effect):
                warnings.warn("CI doesn't containt effect. You might not have enough placebo effects.")
            CI_vec[:,t] = (effect_vec[t] - high_effect, effect_vec[t] - low_effect) 

        sorted_avg_eff = np.sort(placebo_avg_joint_effects)
        low_avg_effect = sorted_avg_eff[alpha_ind]
        high_avg_effect = sorted_avg_eff[(comb_len+1)-alpha_ind]
        if np.sign(low_avg_effect)==np.sign(high_avg_effect):
            warnings.warn("CI (avg) doesn't containt effect. You might not have enough placebo effects.")
        CI_avg = (avg_joint_effect - high_avg_effect, avg_joint_effect - low_avg_effect) 

    else:
        CI_vec = None
        CI_avg = None

    EstResultCI = namedtuple('EstResults', 'effect p ci')
    SparseSCEstResults = namedtuple('SparseSCEstResults', 'effect_vec_res effect_avg_res std_p rms_joint_p rms_joint_std_p N_placebo placebo_effect_vecs placebo_avg_joint_effects')
        
    ret_struct = SparseSCEstResults(EstResultCI(effect_vec, p2s, CI_vec), 
                                    EstResultCI(avg_joint_effect, avg_joint_p, CI_avg), 
                                    p2s_std, rms_joint_p, rms_joint_std_p, 
                                    comb_len, placebo_effect_vecs, placebo_avg_joint_effects)

        
    return ret_struct
