import numpy as np
import pandas as pd

def main():
    n_iter = 5 #number of iterations
    p = 1000 #features
    n = [200, 500, 750] #samples
    sparsity = [0.75, 0.9, 0.95, 0.99]
    SNR = 2
    beta_scale = 5
    rng = np.random.default_rng()
    #save_simulated_data(n_iter, p, n, sparsity=sparsity, SNR=SNR, beta_scale=beta_scale, rng=rng)
    
    #df_key = f'n_{200}_p_{p}_sparsity_{75}_iter_{1}'
    #read = pd.read_hdf("data/data.h5", key=df_key)

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

    Returns
    -------
    X : `n x p` numpy.array
        Matrix of features
    y : `n` numpy.array
        Vector of responses
    beta : `p` numpy.array
        Vector of regression coefficients
    """
    X = rng.standard_normal(size=(n, p))
    
    q = int(np.ceil((1.0 - sparsity) * p))
    beta = np.zeros((p,), dtype=float)
    beta[:q] = beta_scale * rng.standard_normal(size=(q,))
    
    sigma = np.sqrt(np.sum(np.square(X @ beta)) / (n - 1)) / SNR

    y = X @ beta + sigma * rng.standard_normal(size=(n,))

    # Shuffle columns so that non-zero features appear
    # not simply in the first (1 - sparsity) * p columns
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