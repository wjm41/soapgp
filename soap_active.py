"""
Script for performing an active learning experiment with a GP on fragment/fingerprint molecular features
"""

import os
import warnings
import argparse

import gpflow
import numpy as np
import pandas as pd
from gpflow.utilities import print_summary
from rdkit.Chem import Descriptors, MolFromSmiles, AllChem
import tensorflow as tf
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from helper import parse_dataset, transform_data, scaffold_split
from acquisition_function import gp_var

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Adjust accordingly for your own file system
FREESOLV_PATH = 'data/FreeSolv/FreeSolv.csv'
CATS_PATH = 'data/CatS/CatS.csv'
LIPO_PATH = 'data/lipo/lipo.csv'
ESOL_PATH = 'data/esol/esol.csv'
DLS_PATH = 'data/DLS/DLS-100.csv'
BRADLEY_PATH = 'data/bradley/bradley.csv'
MALARIA_PATH = 'data/Malaria/Malaria.csv'

PATHS = {'FreeSolv': FREESOLV_PATH, 'esol': ESOL_PATH, 'lipo': LIPO_PATH, 'DLS': DLS_PATH, 'CatS':CATS_PATH, 'bradley':BRADLEY_PATH, 'Malaria':MALARIA_PATH}

TASK_NAME = 'FreeSolv'  # Change dataset. Options: ['ESOL', 'FreeSolv', 'QM9', 'CEP', 'CatS', 'Melt', 'Malaria'

def initial_data_split(x, smiles_list, y, seed, split, test_size=0.2, init_size=0.25):

    """
    Splits data by initial/train/test
    :param x: input features
    :param y: output targets
    :param test_size: size of test set expressed as fraction of total data - 20% (Yao's paper)
    :param init_size: size of initial training subset as fraction of training set - 25% (Yao's paper)
    :return: X_init, y_init, X_holdout, y_holdout, X_test, y_test
    """
    if split=='random':
       X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed)
    
       X_holdout, X_init, y_holdout, y_init = train_test_split(X_train, y_train, test_size=init_size, random_state=seed)
    elif split=='scaffold':
       train_ind, test_ind = scaffold_split(smiles_list, seed=seed)
       smiles_train = np.array(smiles_list)[train_ind]
       X_train, X_test = x[train_ind], x[test_ind]
       y_train, y_test = y[train_ind], y[test_ind] 
    
       holdout_ind, init_ind = scaffold_split(smiles_train.tolist(), sizes=(1-init_size, init_size), seed=seed)
       X_holdout, X_init = X_train[holdout_ind], X_train[init_ind]
       y_holdout, y_init = y_train[holdout_ind], y_train[init_ind]
    y_init = y_init.reshape(-1, 1)
    y_holdout = y_holdout.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    return X_init, y_init, X_holdout, y_holdout, X_test, y_test


def suggest_sample(A, model, samp, n_samp):

    if samp == 'var': #sample using predicted variance
        scores = gp_var(A, model)
        #print(scores.shape)
        indices = np.argpartition(scores,-n_samp)[-n_samp:]
    if samp == 'rand': #random sampling
        indices = np.random.choice(len(A), size=n_samp)
    return indices


def main(task, samp, split, seed, n_iter):
    global rem_mat, rem_diag
    warnings.filterwarnings('ignore')
 
    print(task+' active learning with SOAP-GP, '+samp+' sampling, '+split+' splitting, seed = '+str(seed))


    print('\nGenerating features...')
    if task in PATHS:
        smiles_list, y  = parse_dataset(task, PATHS[task]) #NEED TO FIX MALARIA
        if task=='Malaria':
           df = pd.read_csv(PATHS[task], header=0)
           pred_val = 'XC50_3D7 (microM)'
           X = np.where(((df[pred_val]!='ND') & (df[pred_val]!='<')) & (df[pred_val].notnull()))[0]
           X=X.reshape(-1,1)
        else:
           X = np.arange(len(smiles_list)).reshape(-1,1)
    else:
        raise Exception('Must provide dataset')

    n_samp  = round(len(X)*0.8*0.025)

    X_init, y_init, X_holdout, y_holdout, X_test, y_test = initial_data_split(X, smiles_list, y, seed, split)

    rmse_list = []

    rem_mat = np.load('/rds-d2/user/wjm41/hpc-work/kernels/dscribe/'+task+'_dscribe_kernel_rematch.npy')

    max_rem = rem_mat.max()
    rem_diag = tf.constant(np.diag(rem_mat),dtype=tf.float64)
    rem_mat = tf.constant(rem_mat,dtype=tf.float64)
    from gpflow.utilities import positive
    class Matern32_rem(gpflow.kernels.Kernel):
            def __init__(self):
                    super().__init__(active_dims=[0])
                    self.var = gpflow.Parameter(1.0, transform=positive())
                    self.mag = gpflow.Parameter(1.0, transform=positive())

            def K(self, X, X2=None, presliced=None):
                    global rem_mat
                    if X2 is None:
                            X2=X
                    A = tf.cast(X,tf.int32)
                    A = tf.reshape(A,[-1])
                    A2 = tf.reshape(X2,[-1])
                    A2 = tf.cast(A2,tf.int32)
                    K_mat = tf.gather(rem_mat, A, axis=0)
                    K_mat = tf.gather(K_mat, A2, axis=1)
                    z = tf.math.sqrt(6*(max_rem-K_mat))*self.var
                    K_final = self.mag*(1+z)*tf.math.exp(-z)
                    return K_final

            def K_diag(self, X, presliced=None):
                    global rem_diag
                    A=tf.cast(X,tf.int32)
                    K_diag = tf.gather_nd(rem_diag, A)
                    z = tf.math.sqrt(6*(max_rem-K_diag))*self.var
                    return self.mag*(1+z)*tf.math.exp(-z)


    m=None

    def objective_closure():
        return - m.log_marginal_likelihood()
    opt = gpflow.optimizers.Scipy()

    # Active learning loop

    for i in range(n_iter + 1):
        y_init_scaled, y_test_scaled, y_scaler = transform_data(y_init, y_test)

        X_init_tf = tf.convert_to_tensor(X_init, dtype = tf.float64)
        X_test_tf = tf.convert_to_tensor(X_test, dtype = tf.float64) 

        k = Matern32_rem()+gpflow.kernels.White(0.1)
        m = gpflow.models.GPR(data=(X_init_tf, y_init_scaled), kernel=k, mean_function=None, noise_variance=1)

        opt_logs = opt.minimize(objective_closure, m.trainable_variables, options=dict(maxiter=10000))

        y_pred, _ = m.predict_f(X_test_tf)
        y_pred = y_scaler.inverse_transform(y_pred)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse_list.append(rmse)
        r2 = r2_score(y_test, y_pred)
        print('\nIteration '+str(i)+' RMSE = ' + str(rmse))
        print('Iteration '+str(i)+' R2 = ' + str(r2))
        print('model training size = '+ str(len(X_init)))

        X_holdout_tf = tf.convert_to_tensor(X_holdout, dtype=tf.float64)
        y_init_scaled, y_holdout_scaled, y_scaler = transform_data( y_init, y_holdout)

        #Find sample indices and update init and holdouting training sets
        sample_indices = suggest_sample(X_holdout_tf,  m, samp, n_samp)
        X_init = np.vstack((X_init, X_holdout[sample_indices]))
        y_init = np.vstack((y_init, y_holdout[sample_indices]))
        X_holdout = np.delete(X_holdout, sample_indices, axis=0)
        y_holdout = np.delete(y_holdout, sample_indices, axis=0)

    # Saves rmse vs num acquisitions into a 'results' folder

    np.save('results/soap_'+task+'_samp_'+samp+'_split_'+split+'_seed_'+str(seed)+'.npy', rmse_list)


if __name__ == "__main__":
     parser = argparse.ArgumentParser()
     parser.add_argument('-task', type=str, default=TASK_NAME, 
                         help='Dataset on which to train the GP')
     parser.add_argument('-samp', type=str, default='var', 
                         help='Method used to select new samples')
     parser.add_argument('-split', type=str, default='random', 
                         help='Method used to split dataset, choices = random, scaffold')
     parser.add_argument('-seed', type=int, default=42, 
                         help='Random seed used for initial train/test split')
     parser.add_argument('-n_iter', type=int, default=10, 
                         help='Number of active learning iterations (2.5% drawn each iteration)')
     args = parser.parse_args()    

     main(args.task, args.samp, args.split, args.seed, args.n_iter)
