# Main imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# "From" imports
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.linear_model import LassoCV, Lasso
from sklearn.metrics import confusion_matrix, mean_squared_error as mse
from sklearn.metrics import confusion_matrix as cm
##----------------------------------------------
class metrics():
    def __init__(self, point):
        self.point = point
        self.test_mse  = list()
        self.train_mse  = list()
        self.sens = list()
        self.spec = list()
        
    def add(self, y_test, y_train,pred_test, pred_train, coeffs_true, coeffs_pred):
        true_bool = (coeffs_true != 0)
        pred_bool = (coeffs_pred != 0)
        self.spec.append(specifity(true_bool, pred_bool))
        self.sens.append(sensitivity(true_bool, pred_bool))
        self.train_mse.append(mse(y_train, pred_train))
        self.test_mse.append(mse(y_test, pred_test))
       
##-----------------------------------------------
def main():
    fontsize = 10
    n_iter = 500 #number of iterations
    n_folds = 5
    params = {'p': [500],                      # Features
              'n': [500],           # Samples
       'sparsity': [0.85],  
            'SNR': [2],                        # Signal-to-noise
     'beta_scale': [5],                        # std of beta coeff
            'rng': [np.random.default_rng()]}
    #alpha_min = 1.7219
    #alpha_1lse = 1.2148
    param_grid = ParameterGrid(params)
    alphas = []
    # We choose a parameter permutation
    for point in param_grid:
        X,y,beta = simulate_data(**point)
        subsample_size = int(np.ceil(0.8*point['n']))
        coeff_counts_lse = np.zeros(point['p'])
        coeff_counts_min = np.zeros(point['p'])
        lasso = Lasso

        
        for iter in range(n_iter):
            subsample_indices = np.random.choice(point['n'], subsample_size, replace = False)
            X_, y_ = X[subsample_indices,:], y[subsample_indices]    
            lasso


            coeff_counts_lse[lasso_lse.coef_ != 0] += 1
            coeff_counts_min[lasso_min.coef_ != 0] += 1
        
        point.pop('rng', None)
        plt.figure()
        plt.subplot(131)
        plt.hist(coeff_counts_lse, bins = point['p'])  
        plt.ylabel("Occurences", fontsize = fontsize)
        plt.xlabel("Counts", fontsize = fontsize) 
        plt.legend([f"Lambda_lse ={lasso.alpha_}"], fontsize = fontsize)
        plt.xticks(fontsize = fontsize)
        plt.yticks(fontsize = fontsize)
        plt.title(point,fontsize = fontsize)

        plt.subplot(132)
        plt.hist(coeff_counts_lse, bins = point['p'])  
        plt.ylabel("Occurences", fontsize = fontsize)
        plt.xlabel("Count", fontsize = fontsize) 
        plt.legend([f"Lambda_lse = {alpha_lse}"], fontsize = fontsize)
        plt.xticks(fontsize = fontsize)
        plt.yticks(fontsize = fontsize)
        plt.title(point,fontsize = fontsize)

        plt.show()

        
def specifity(true_bool, pred_bool):
    cfm = cm(true_bool, pred_bool, labels = [False, True])

    tn, fp, fn, tp = cfm.ravel()
    spec = tn/(tn+fp)
    return spec

def sensitivity(true_bool, pred_bool):
    cfm = cm(true_bool, pred_bool, labels = [False, True])
    tn, fp, fn, tp = cfm.ravel()
    sens = tp/(tp+fn)
    return sens  
def Lasso_grid_treshold_repeat(param_grid, alphas, n_iter, thresholds, B):
    """Simulate data from parameter grid and run lasso over it

    Parameters
    ----------
    param_grid : A ParameterGrid object containing all combinations of n, p, sparsity etc. 
    threshold  : The histogram threshold
    B          : The number of times we simulate again

    """
    def specifity(true_bool, pred_bool):
    cfm = cm(true_bool, pred_bool, labels = [False, True])

    tn, fp, fn, tp = cfm.ravel()
    spec = tn/(tn+fp)
    return spec

    def sensitivity(true_bool, pred_bool):
        cfm = cm(true_bool, pred_bool, labels = [False, True])
        tn, fp, fn, tp = cfm.ravel()
        sens = tp/(tp+fn)
    return sens  

    fontsize = 15
    for point in param_grid:
        for threshold in thresholds:
            sensitivity = 0
            specificity = 0
            X,y,beta = simulate_data(**point)
            beta_bool = (beta != 0)
            subsample_size = int(np.ceil(0.8*point['n']))
            for alpha in alphas:
                for i in range(B)
                    coeff_counts = np.zeros(point['p'])
                    for iter in range(n_iter):
                        subsample_indices = np.random.choice(point['n'], subsample_size, replace = False)
                        X_, y_ = X[subsample_indices,:], y[subsample_indices]    
                        lasso = Lasso(alpha = alpha)
                        lasso.fit(X_, y_)
                        coeff_counts[lasso.coef_ != 0] += 1

                coeff_bool = coeff_counts/n_iter >= threshold
                cfm = confusion_matrix(beta_bool, coeff_bool)
                tn, fp, fn, tp = cfm.ravel()

                sensitivity += tp/(tp+fn)
                specificity += tn/(tn+fp)
                sensitivity = sensitivity/B
                specificity = specificity/B
                print(f"Thrsh: {threshold}, lambda: {alpha} | Sensitivity: {sensitivity}, Specificity: {specificity}")
        
    return             

def simulate_data(n, p, rng, *, sparsity=0.95, SNR=2.0, beta_scale=5.0):

    """Simulate data for Project 3, Part 1.

    Parameters
    ----------
    n : int
        Number of samples
    p : int
        Number of features
    rng : numpy.random.Generator
        Random number generator (e.g. from `numpy.random.default_rng`)
    sparsity : float in (0, 1)
        Percentage of zero elements in simulated regression coefficients
    SNR : positive float
        Signal-to-noise ratio (see explanation above)
    beta_scale : float
        Scaling for the coefficient to make sure they are large

    """
    X = rng.standard_normal(size=(n, p))
    
    q = int(np.ceil((1.0 - sparsity) * p))
    beta = np.zeros((p,), dtype=float)
    beta[:q] = beta_scale * rng.standard_normal(size=(q,))
    
    sigma = np.sqrt(np.sum(np.square(X @ beta)) / (n - 1)) / SNR

    y = X @ beta + sigma * rng.standard_normal(size=(n,))

    idx_col = rng.permutation(p)
    
    return X[:, idx_col], y, beta[idx_col]

if __name__ == "__main__":
    main()