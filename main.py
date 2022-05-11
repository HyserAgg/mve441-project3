# Main imports
import numpy as np
import pandas as pd


# "From" imports
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error as mse


##-----------------------------------------------
def main():
    n_iter = 5 #number of iterations
    n_folds = 10
    params = {'p': [1000],                     # Features
              'n': [200, 500, 750],            # Samples
       'sparsity': [0.75, 0.9, 0.95, 0.99],  
            'SNR': [2],                        # Signal-to-noise
     'beta_scale': [5],                        # std of beta coeff
            'rng': [np.random.default_rng()]}

    param_grid = ParameterGrid(params)
    # We choose a parameter permutation
    for point in param_grid:
        # We repeat our runs a number of times
        for iter in range(n_iter):
            X,y,beta = simulate_data(**point)
            test_size = int(np.ceil(0.8*point['n']))
            X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=test_size)
            lasso = LassoCV(cv = n_folds, n_jobs = -1, selection = 'random')
            lasso.fit(X_train,y_train)

            cv_mean = np.mean(lasso.mse_path_, axis=1)
            cv_std = np.std(lasso.mse_path_, axis=1)
            idx_min_mean = np.argmin(cv_mean)
            idx_alpha = np.where(
            (cv_mean <= cv_mean[idx_min_mean] + cv_std[idx_min_mean] / np.sqrt(n_folds)) &
            (cv_mean >= cv_mean[idx_min_mean])
                                )[0][0]
            alpha_1se = lasso.alphas_[idx_alpha]
            

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

def save_simulated_data(n_iter, p, n, sparsity, SNR, beta_scale, rng):
    # Save simulated dataframes in h5 file.
    # A dataframe has the following structure:
    # > P features (p)
    # > N samples (n)
    # > N responses (y)
    # > P beta coefficients (b)
    # |(n_1, p_1), ..  (n_1, p_P), (n_1, y_1)|
    # |     :               :           :    |            
    # |(n_N, p_1)  ..  (n_N, p_P), (p_P, y_N)|
    # |(n_1, b_1)  ..  (n_N, b_P),    NaN    |
    # 
    # Select dataframe of interest through key:
    # N: Number of samples
    # P: Number of features 
    # sparsity: decimal number [0.00, 1.00]
    # i: Iteration number
    # key='n_{N}_p_{P}_sparsity_{100*sparsity}_iter_{i}'
    
    for n_i in range(len(n)):
        for spar_i in range(len(sparsity)):
            for iter_i in range(n_iter):
                X, y, beta = simulate_data(n[n_i], p, rng, sparsity=sparsity[spar_i], SNR=SNR, beta_scale=beta_scale)
                df_beta = pd.DataFrame(beta, columns=['beta'])
                df_beta = df_beta.transpose()
                df = pd.DataFrame(X)
                df['response'] = y
                df = pd.concat([df, df_beta]) 
                df.to_hdf('data/data.h5', f'n_{n[n_i]}_p_{p}_sparsity_{int(100*sparsity[spar_i])}_iter_{iter_i}', mode='a')
                print(f'Iteration: {(len(n)+1)*(len(sparsity)+1)*n_i+(len(sparsity)+1)*(spar_i)+iter_i}')
                    
if __name__ == "__main__":
    main()