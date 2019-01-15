def simulation_eval(effects, CI_lowers, CI_uppers, true_effect=0):
    te_mse = np.mean(np.square((effects-true_effect)))
    cov = np.mean(np.logical_and(effects>=CI_lowers, effects <=CI_uppers).astype(int))
    ci_len = np.mean(CI_uppers-CI_lowers)
    return (te_mse, cov, ci_len)
