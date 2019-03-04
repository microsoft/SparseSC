# --------------------------------------------------------------------------------
# Programmer: Jason Thorpe
# Date    1/11/2019 1:25:57 PM
# Purpose:   Implement round-robin fitting of Sparse Synthetic Controls Model for DGP based analysis
# Description:
#
# This is intended for use in Proofs of concepts where the underlying data
# model is known and the experiment is aimed at understing the extent to which
# the SparseSC model correctly and efficiently estimates the underlying data
# model
#
# This code therefor repeatedly splits the data into fitting and hold-out sets
# in a round-robin fassion, fits the covariate coefficients in the fitting set,
# applies the covariate weights estimsted in the fitting set to creat weights
# for individual units in the held-out set, and returns the fitted weights and
# synthetic controls for every unit.
#
# Usage:
#
#
#    import sys
#    import os
#    import numpy as np
#    repo_path = 'c:\path\to\the\SparseSC\git\repo'
#    sys.path.append(repo_path)
#    sys.path.append(os.path.join(repo_path,'examples'))
#    x = np.random.rand(100,20)
#    y = np.random.rand(100,8)
#    from fit_poc import fit_poc
#    weights, syntetic_y = fit_poc(x,y)
#
#
# --------------------------------------------------------------------------------

from sklearn.model_selection import KFold
import numpy as np
import SparseSC as SC

def fit_poc(X,Y,
            Lambda_min = 1e-6,
            Lambda_max = 1,
            grid_points = 20,
            grid = None,
            # fold tuning parameters: either a integer or list of test/train subsets such as the result of calling Kfold().split()
            outer_folds = 10,
            cv_folds = 10,
            gradient_folds = 10,
            random_state = 10101,
            ):

    if grid is None:
        grid = np.exp(np.linspace(np.log(Lambda_min),np.log(Lambda_max),grid_points))

    assert X.shape[0] == Y.shape[0]
    out_weights = np.zeros( (Y.shape[0],X.shape[0] ))
    out_predictions = np.zeros(Y.shape)

    try:
        iter(outer_folds)
    except TypeError: 
        outer_folds = KFold(outer_folds, shuffle=True, random_state = random_state).split(np.arange(Y.shape[0]))
    outer_folds = list(outer_folds)

    for i, (train,test) in enumerate(outer_folds):

        # --------------------------------------------------
        # Phase 0: Data wrangling
        # --------------------------------------------------

        Xtrain = X[train,:]
        Xtest = X[test,:]
        Ytrain = Y[train,:]
        Ytest = Y[test,:]

        # Get the L2 penalty guestimate:  very quick ( milliseconds )
        w_pen  = SC.L2_pen_guestimate(Xtrain) 

        # GET THE MAXIMUM v_penS: quick ~ ( seconds to tens of seconds )
        v_pen_max = SC.get_max_lambda(
                    Xtrain,
                    Ytrain,
                    w_pen = w_pen,
                    grad_splits=gradient_folds,
                    learning_rate = 0.2, # initial learning rate
                    verbose=1)

        # --------------------------------------------------
        # Phase 1: extract cross fold residual errors for each lambda
        # --------------------------------------------------

        # SCORES FOR EACH VALUE OF THE GRID: very slow ( minutes to hours )
        scores = SC.CV_score( X = Xtrain,
                              Y = Ytrain,
                              splits = cv_folds,
                              v_pen = grid * v_pen_max,
                              progress = True,
                              w_pen = w_pen,
                              grad_splits = gradient_folds) 

        # GET THE INDEX OF THE BEST SCORE
        best_i = np.argmin(scores)
        best_lambda = (grid * v_pen_max)[best_i]

        # --------------------------------------------------
        # Phase 2: extract V and weights: slow ( tens of seconds to minutes )
        # --------------------------------------------------

        best_V = SC.tensor(X = Xtrain, 
                           Y = Ytrain,
                           v_pen = best_lambda,
                           grad_splits = gradient_folds,
                           learning_rate = 0.2) 

        # GET THE BEST SET OF WEIGHTS
        out_of_sample_weights = SC.weights(Xtrain,
                                           Xtest,
                                           V = best_V,
                                           w_pen = w_pen)

        Y_SC_test = out_of_sample_weights.dot(Ytrain)

        # BUILD THE SYNTHETIC CONTROLS
        out_weights[test,:] = out_of_sample_weights
        out_predictions[test,:] = Y_SC_test

        # CALCULATE ERRORS AND R-SQUARED'S
        ct_prediction_error = Y_SC_test - Ytest
        null_model_error = Ytest - np.mean(Xtest)
        betternull_model_error = (Ytest.T - np.mean(Xtest,1)).T
        print("#--------------------------------------------------")
        print("OUTER FOLD %s OF %s: Group Mean R-squared: %0.3f%%; Individual Mean R-squared: %0.3f%%" % (
                i + 1,
                len(outer_folds) + 1,
                100*(1 - np.power(ct_prediction_error,2).sum()  / np.power(null_model_error,2).sum()) ,
                100*(1 - np.power(ct_prediction_error,2).sum()  /np.power(betternull_model_error,2).sum() )))
        print("#--------------------------------------------------")

    return out_weights, out_predictions
