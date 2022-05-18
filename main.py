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
    n_iter = 25 #number of iterations
    n_folds = 10
    params = {'p': [1000],                      # Features
              'n': [250, 500, 750],                     # Samples
       'sparsity': [0.95],  
            'SNR': [2],                        # Signal-to-noise
     'beta_scale': [5],                        # std of beta coeff
            'rng': [np.random.default_rng()]}

    param_grid = ParameterGrid(params)
    lasso_grid(param_grid, n_iter, n_folds)
    plt.show()
            
            
def lasso_grid(param_grid, n_iter = 50, n_folds = 10):
    # We choose a parameter permutation
    for point in param_grid:
        # We repeat our runs a number of times
        metrics_lse = metrics(point)
        metrics_min = metrics(point)
        X,y,beta = simulate_data(**point)

        for iter in range(n_iter):
            print(iter)
            test_size = int(np.ceil(0.8*point['n']))
            X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=test_size)
            lasso = LassoCV(cv = n_folds)
            lasso.fit(X_train,y_train)

            # Identify alpha_lse
            e = np.mean(lasso.mse_path_, axis=1)
            s = np.std(lasso.mse_path_, axis=1)
            idx_min_mean = np.argmin(e)
            idx_alpha = np.where(
            (e <= e[idx_min_mean] + s[idx_min_mean] / np.sqrt(n_folds)) &
            (e >= e[idx_min_mean])
                                )[0][0]
            alpha_lse = lasso.alphas_[idx_alpha]

            # Given the hyperparameter alpha and alpha_lse calculate mean square test- and training error
            # Test the overlap of chosen coeffs with actual coeffs - i.e sensitivity and specificity of coeff. choices
            lasso_min = Lasso(alpha = lasso.alpha_)
            lasso_lse = Lasso(alpha = alpha_lse)
            lasso_min.fit(X_train, y_train)
            lasso_lse.fit(X_train, y_train)
            metrics_lse.add(y_test, y_train, lasso_lse.predict(X_test), lasso_lse.predict(X_train), beta, lasso_lse.coef_)
            metrics_min.add(y_test, y_train, lasso_min.predict(X_test), lasso_min.predict(X_train), beta, lasso_min.coef_)
            
        # Given this point we scatterplot the sensitivity and specificity
        point.pop('rng',None)
        plt.figure()
        plt.scatter(metrics_lse.spec, metrics_lse.sens, color = 'r')   
        plt.scatter(metrics_min.spec, metrics_min.sens, color = 'y')
        plt.title(point) 
        plt.legend(["1se model","Min model"])
        plt.ylabel("Sensitivity") 
        plt.xlabel("Specificity")
        # Test specificity and sensitivity against n and sparsity  
    return
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