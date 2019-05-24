import unittest
import numpy as np

try:
    import SparseSC as SC
except ImportError:
    raise RuntimeError("SparseSC is not installed. use 'pip install -e .' from repo root to install in dev mode")
from SparseSC.fit import fit

#For the synth comparisons you will need rpy2 and the Synth package loaded in there.
# to do the latter:
# python
# from test.test_simulation import installRSynth
# installRSynth()

def installRSynth():
    from rpy2.robjects.packages import importr
    utils = importr('utils')
    utils.install_packages('Synth')

class Simulation(unittest.TestCase):

    @staticmethod
    def fitRSynth(Y_pre, treated_units, verbose=False):
        #See the Synth.PDF from the CRAN package
        import rpy2.robjects as ro #ro.r is the R instace
        assert len(treated_units)==1 #for now
        control_units = [u for u in range(Y_pre.shape[0]) if u not in treated_units]
        
        #np.matrix is not automatically converted so use ndarray
        if type(Y_pre).__name__=="matrix":
            Y_pre = Y_pre.A

        N1, N0 = len(treated_units), len(control_units)
        N = N1 + N0
        T0 = Y_pre.shape[1]
        
        class RSynth:
            def __init__(self, sc_weights, Vs, control_units):
                self.sc_weights = sc_weights #NxN0
                self.Vs = Vs #NxT0
                self.control_units = control_units

            def predict(self, mat_c):
                #mat_c: N0xk
                return np.dot(sc_weights, mat_c[control_units,:])


        #Automatically convert numpy arrays to R vector/matrix
        from rpy2.robjects import numpy2ri
        numpy2ri.activate()

        from rpy2.robjects.packages import importr
        try:
            Synth = importr('Synth')
        except:
            raise RuntimeError("Need the 'Synth' package loaded in the rpy2 R environment. Use test.test_simulation.installRSynth")
        
        sc_weights = np.zeros((N,N0))
        Vs = np.zeros((N,T0))
        
        X1=Y_pre[treated_units, :].T
        X0=Y_pre[control_units, :].T
        r_X1 = ro.r.matrix(X1, nrow=X1.shape[0], ncol=X1.shape[1])
        r_X0 = ro.r.matrix(X0, nrow=X0.shape[0], ncol=X0.shape[1])

        #Difficult to suppress Synth output. The following doesn't work
        #with capture() as out: #https://stackoverflow.com/questions/5136611/ #didn't suppress
        #r_synth_out = utils.capture_output(Synth.synth(X1=r_X1, X0=r_X0, Z1=r_X1, Z0=r_X0), file=tc) #API error
        if not verbose:
            r_tc = ro.r.textConnection("messages","w")
            ro.r.sink(r_tc)
        r_synth_out = Synth.synth(X1=r_X1, X0=r_X0, Z1=r_X1, Z0=r_X0)
        sc_weights[treated_units[0],:] = np.squeeze(np.array(r_synth_out[r_synth_out.names.index('solution.w')]))
        Vs[treated_units[0],:] = np.squeeze(np.array(r_synth_out[r_synth_out.names.index('solution.v')]))

        from tqdm import tqdm
        for p_c, p_ct in enumerate(tqdm(control_units)):
            donors_ct = control_units.copy()
            donors_ct.remove(p_ct)
            donors_c = list(range(N0))
            donors_c.remove(p_c)
            X1=Y_pre[[p_ct], :].T
            X0=Y_pre[donors_ct, :].T
            r_X1 = ro.r.matrix(X1, nrow=X1.shape[0], ncol=X1.shape[1])
            r_X0 = ro.r.matrix(X0, nrow=X0.shape[0], ncol=X0.shape[1])

            with capture() as out: #doesn't have quiet option
                r_synth_out = Synth.synth(X1=r_X1, X0=r_X0, Z1=r_X1, Z0=r_X0)
            sc_weights[p_ct,donors_c] = np.squeeze(np.array(r_synth_out[r_synth_out.names.index('solution.w')]))
            Vs[p_ct,:] = np.squeeze(np.array(r_synth_out[r_synth_out.names.index('solution.v')]))
            
        if not verbose:
            ro.r.sink()
            ro.r.close(r_tc)

        return RSynth(sc_weights, Vs, control_units)

    @staticmethod
    def dataFactorDGP_AA(N0s=[20], T0s=[10], S=1):
        import itertools
        from dgp.factor_model import factor_dgp
        
        N1,T1 = 1,1
        data_dict = {}
        for counter, (N0,T0) in enumerate(itertools.product(N0s, T0s)):
            data[(N0,T0)] = (N0,T0)
            for s in range(S):
                _, _, Y_pre_C, Y_pre_T, Y_post_C, Y_post_T, l_C, l_T = factor_dgp(N0, N1, T0, T1, K=0, R=0, F=1)
                Y_pre = np.vstack((Y_pre_T, Y_pre_C))
                Y_post = np.vstack((Y_post_T, Y_post_C))
                l = np.vstack((l_T, l_C))
                data_dict[(N0,T0)].append((Y_pre, Y_post, l))
        return(data_dict)

    @staticmethod
    def fitSparseSC_wrapper(Y_pre, Y_post, treated_units):
        return(fit(X=Y_pre, Y=Y_post, treated_units=treated_units, 
                              model_type="retrospective", constrain="simplex",
                              print_path = False, progress = False, verbose=0))

    @staticmethod
    def fitRSynth_wrapper(Y_pre, Y_post, treated_units):
        return(Simulation.fitRSynth(Y_pre, treated_units))
    
    @staticmethod
    def SCFactor_AA_runner(data_dict, sc_method):
        treated_units = [0]
        data_keys = data_dict.keys()
        results = np.zeros((len(data_keys), 2))
        for counter, (N0,T0) in enumerate(data_keys):
            comb_results = np.empty((S,2))
            print(counter)
            datasets = data_dict[(N0,T0)]
            for s in range(len(datasets)):
                Y_pre, Y_post, l = datasets[s]
                fit_res = sc_method(Y_pre, Y_post, treated_units)
                comb_results[s,0] = np.mean(np.square(Y_post-fit_res.predict(Y_post)))
                comb_results[s,1] = np.mean(np.square(l-fit_res.predict(l)))
            results[counter,:] = np.mean(comb_results, axis=1)
        return(results)

    def testFactorDGP_AA(self, N0s=[20], T0s=[10], S=1):
        import pickle
        import itertools
        
        treated_units = [0]

        N_combos = len(N0s)*len(T0s)
        data_dic = Simulation.dataFactorDGP_AA(N0s=[20], T0s=[10], S=1)
        
        data_pkl_file = 'sim_data.pkl'
        with open(data_pkl_file, 'wb') as output:
            pickle.dump(data_dic, output)
        #with open(data_pkl_file, 'rb') as input:
        #    data_dic = pickle.load(input)
        
        results_sc = Simulation.SCFactor_AA_runner(data_dict, Simulation.fitRSynth_wrapper)
        results_ssc = Simulation.SCFactor_AA_runner(data_dict, Simulation.fitSparseSC_wrapper)
        results = np.hstack((results_sc, results_ssc))

        print(results)
        res_pkl_file = 'sim_results.pkl'
        with open(res_pkl_file, 'wb') as output:
            pickle.dump(results, output)
        #with open(pkl_file, 'rb') as input:
        #    results = pickle.load(input)

    # Come up with examples targetting Synth's weaknesses (show it needs lots of data to get to where SparseSC can with little)
    # - Synth over fits by fitting a V for each or not penalizing V
    # Come up with example where 

    ##later) Can we match a longer set of factor loadings
    # Other Counterfactual prediction:
    ## a) Compare to SC (big N0, small T0, then SC; or many factors; should do bad) to basic time-series model


    
if __name__ == "__main__":
    import random

    random.seed(12345)
    np.random.seed(10101)

    s = Simulation()
    s.testFactorDGP_AA()
    # unittest.main()
