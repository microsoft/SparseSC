import matplotlib.pyplot as plt

def raw_plots(Y, treated_units, control_units, T0):
    if len(treated_units)>1:
        lbl_t = "Treateds"
        lbl_mt = "Mean Treated"
    else:
        lbl_t = "Treated"
        lbl_mt = "Treated"
        
    raw_all = plt.figure("raw_all")
    plt.plot(np.transpose(Y[control_units,:]), color='gray')
    plt.plot(Y[control_units[0],:], color='gray', label='Controls')
    plt.plot(np.transpose(Y[treated_units,:]), color='black',)
    plt.plot(Y[treated_units[0],:], color='black', label=lbl_t)
    plt.axvline(x=T0, linestyle='--')
    plt.xlabel("Time")
    plt.ylabel("Outcome")
    plt.legend(loc=1)
    #Means controls & treated
    raw_means = plt.figure("raw_means")
    plt.plot(np.mean(Y[control_units,:], axis=0), color='gray', label='Mean Control')
    plt.plot(np.mean(Y[treated_units,:], axis=0), color='black', label=lbl_mt)
    plt.axvline(x=T0, linestyle='--')
    plt.xlabel("Time")
    plt.ylabel("Outcome")
    plt.legend(loc=1)
    return([raw_all, raw_means])
    
def te_plots(Y, Y_sc, T0):
    t_raw = plt.figure("t_raw")
    plt.plot(Y, 'bx-', label='Treated')
    plt.plot(Y_sc, 'gx--', label='Treated SC')
    plt.axvline(x=T0, linestyle='--')
    plt.xlabel("Time")
    plt.ylabel("Outcome")
    plt.legend(loc=1)
    
    t_diff = plt.figure("t_diff")
    plt.plot(Y - Y_sc, 'kx--', label='Treated Diff')
    plt.axvline(x=T0, linestyle='--')
    plt.axhline(y=0, linestyle='--')
    plt.xlabel("Time")
    plt.ylabel("Real-SC Outcome Difference")
    plt.legend(loc=1)
    return([t_raw, t_diff])
    
def diffs_plot(diffs, treated_units, control_units):
    if len(treated_units)>1:
        lbl_t = "Treated Diffs"
    else:
        lbl_t = "Treated Diff"
    diffs_plt = plt.figure("diffs_fig")
    plt.plot(np.transpose(diffs[control_units,:]), alpha=.5, color='gray')
    plt.plot(diffs[control_units[0],:], alpha=.5, color='gray', label='Control Diffs')
    plt.plot(np.transpose(diffs[treated_units,:]), color='black')
    plt.plot(diffs[treated_units[0],:], color='black', label=lbl_t)
    plt.axvline(x=T0, linestyle='--')
    plt.xlabel("Time")
    plt.ylabel("Real-SC Outcome Difference")
    plt.legend(loc=1)
    return(diffs_plt)
