import unittest
import numpy as np

try:
    import SparseSC as SC
except ImportError:
    raise RuntimeError("SparseSC is not installed. use 'pip install -e .' from repo root to install in dev mode")
from SparseSC.fit import fit


class Simulation(unittest.TestCase):
    def testFactorDGP_AA(self, N0s=[100, 1000], T0s=[10], S=2):
        import pickle
        import itertools
        from dgp.factor_model import factor_dgp

        N1,T1 = 1,1
        treated_units = [0]

        results = np.empty((len(N0s)*len(T0s), 2))
        for counter, (N0,T0) in enumerate(itertools.product(N0s, T0s)):
            print(counter)
            comb_results = np.empty((S,2))
            for s in range(S):
                print(s)
                _, _, Y_pre_C, Y_pre_T, Y_post_C, Y_post_T, l_C, l_T = factor_dgp(N0, N1, T0, T1, K=0, R=0, F=1)
            
                Y_pre = np.vstack((Y_pre_T, Y_pre_C))
                Y_post = np.vstack((Y_post_T, Y_post_C))
                l = np.vstack((l_T, l_C))

                fit_res = fit(X=Y_pre, Y=Y_post, treated_units=treated_units, 
                              model_type="retrospective", constrain="simplex",
                              print_path = False, progress = False, verbose=0)
                
                Y_post_sc = fit_res.predict(Y_post)
                comb_results[s,0] = np.mean(np.square(Y_post-Y_post_sc))
                l_sc = fit_res.predict(l)
                comb_results[s,1] = np.mean(np.square(l-l_sc))
            results[counter,:] = np.mean(comb_results, axis=1)

        print(results)
        pkl_file = 'sim_results.pkl'
        with open(pkl_file, 'wb') as output:
            pickle.dump(results, output)

        #with open(pkl_file, 'rb') as input:
        #    results = pickle.load(input)

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
