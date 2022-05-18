# Main imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

# "From" imports
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import roc_curve


##-----------------------------------------------
def main():

    n_iter = 5 #number of iterations
    n_folds = 5
    p = 1000
    sparsity = 0.95
    n = [200, 500, 750]
    SNR = 2
    beta_scale = 5
    rng = np.random.default_rng()
    
    train_1se_list = []
    test_1se_list = []
    train_min_list = []
    test_min_list = []
    
    FPR_1se_list = []
    TPR_1se_list = []
    FPR_min_list = []
    TPR_min_list = []
    
    for n_val in n:
        train_1se, test_1se, y_1se, y_1se_pred = MSE_lambda(n_val, p, rng, '1se', n_iter, sparsity, SNR, beta_scale, n_folds)
        train_min, test_min, y_min, y_min_pred = MSE_lambda(n_val, p, rng, 'min', n_iter, sparsity, SNR, beta_scale, n_folds)
        
        FPR, TPR, thresh = roc_curve(y_1se, y_1se_pred)
        FPR_1se_list.append(FPR)
        TPR_1se_list.append(TPR)
        
        FPR, TPR, thresh = roc_curve(y_min, y_min_pred)
        FPR_min_list.append(FPR)
        TPR_min_list.append(TPR)
        
        train_1se_list.append(train_1se)
        test_1se_list.append(test_1se)
        train_min_list.append(train_min)
        test_min_list.append(test_min)
        
        
    MSE_plot(n, 'n', train_min_list, test_min_list, train_1se_list, test_1se_list)
    ROC_plot(n, 'n', FPR_min_list, TPR_min_list, FPR_1se_list, TPR_1se_list)

def ROC_plot(x_vals, x_str, FPR_min_list, TPR_min_list, FPR_1se_list, TPR_1se_list):
    
    for i, val in enumerate(x_vals):
        plt.plot(FPR_min_list[i], TPR_min_list[i], label = f'lambda_min: {x_str}={val}')
        plt.plot(FPR_1se_list[i], TPR_1se_list[i], label = f'lambda_1se: {x_str}={val}')
    
    plt.ylabel('Sensitivity')
    plt.xlabel('1-Specificity')
    plt.title('ROC curve')
    plt.legend()
    plt.savefig(f'data/ROC_diff_{x_str}')
    plt.close()
  
def MSE_plot(x_vals, x_str, train_min, test_min, train_1se, test_1se):
    
    #Means and standard deviations from tuples
    y_train_1se = np.array([idx[0] for idx in train_1se])
    y_test_1se  = np.array([idx[0] for idx in test_1se])
    y_train_min = np.array([idx[0] for idx in train_min])
    y_test_min = np.array([idx[0] for idx in test_min])
    
    s_train_1se = np.array([idx[1] for idx in train_1se])
    s_test_1se  = np.array([idx[1] for idx in test_1se])
    s_train_min = np.array([idx[1] for idx in train_min])
    s_test_min = np.array([idx[1] for idx in test_min])
    
    #Plot MSE vs different n / sparsity
    plt.plot(x_vals, y_train_1se, label = 'Train 1se')
    plt.fill_between(x_vals, np.add(y_train_1se, s_train_1se), np.subtract(y_train_1se, s_train_1se), facecolor='blue', alpha=0.3)
    
    plt.plot(x_vals, y_test_1se, label = 'Test 1se')
    plt.fill_between(x_vals, np.add(y_test_1se, s_test_1se), np.subtract(y_test_1se, s_test_1se), facecolor='green', alpha=0.3)
    
    plt.plot(x_vals, y_train_min, label = 'Train min')
    plt.fill_between(x_vals, np.add(y_train_min, s_train_min), np.subtract(y_train_min, s_train_min), facecolor='yellow', alpha=0.3)
    
    plt.plot(x_vals, y_test_min, label = 'Test min')
    plt.fill_between(x_vals, np.add(y_test_min, s_test_min), np.subtract(y_test_min, s_test_min), facecolor='red', alpha=0.3)
    
    plt.title(f'MSE for different {x_str}')
    plt.xlabel(f'{x_str}')
    plt.ylabel("MSE")
    plt.legend()
    plt.savefig(f'data/MSE_diff_{x_str}')
    plt.close()

def MSE_lambda(n, p, rng, lbda, n_iter, sparsity, SNR, beta_scale, n_folds):
    mse_1se = []
    mse_min = []
    for iter in range(n_iter):
        X, y, beta = simulate_data(n, p, rng, sparsity=sparsity, SNR=SNR, beta_scale=beta_scale)
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, train_size=0.8)
        lasso = LassoCV(cv = n_folds, n_jobs = -1, selection = 'random')
        lasso.fit(X_train, y_train)
        
        if lbda == '1se':
            cv_mean = np.mean(lasso.mse_path_, axis=1)
            cv_std = np.std(lasso.mse_path_, axis=1)
            idx_min_mean = np.argmin(cv_mean)
            idx_alpha = np.where(
            (cv_mean <= cv_mean[idx_min_mean] + cv_std[idx_min_mean] / np.sqrt(n_folds)) &
            (cv_mean >= cv_mean[idx_min_mean])
                        )[0][0]
            alpha_1se = lasso.alphas_[idx_alpha]
            lasso_1se = linear_model.Lasso(alpha=alpha_1se)
            lasso_1se.fit(X_train, y_train)
            mse_1se.append((mse(y_train, lasso_1se.predict(X_train)), mse(y_test, lasso_1se.predict(X_test))))
            
        elif lbda == 'min':
            lasso_min = linear_model.Lasso(alpha=lasso.alpha_)
            lasso_min.fit(X_train, y_train)
            mse_min.append((mse(y_train, lasso_min.predict(X_train)), mse(y_test, lasso_min.predict(X_test))))
        else:
            return print('Wrong input')
    
    if lbda == '1se':
        train_1se = (np.mean([idx[0] for idx in mse_1se]), np.std([idx[0] for idx in mse_1se]))
        test_1se = (np.mean([idx[1] for idx in mse_1se]), np.std([idx[1] for idx in mse_1se]))
        return train_1se, test_1se, beta != 0, 1*(lasso_1se.coef_ != 0)
    elif lbda == 'min':
        train_min = (np.mean([idx[0] for idx in mse_min]), np.std([idx[0] for idx in mse_min]))
        test_min = (np.mean([idx[1] for idx in mse_min]), np.std([idx[1] for idx in mse_min]))
        return train_min, test_min, beta != 0, 1*(lasso_min.coef_ != 0)

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