"""
Script for training a GP on either fragment or fingerprint representations of molecules.
"""

import os
import warnings
import argparse

import gpflow
import numpy as np
import tensorflow as tf
from gpflow.utilities import print_summary
from rdkit.Chem import MolFromSmiles, AllChem
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error

from helper import parse_dataset, transform_data, scaffold_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Adjust accordingly for your own file system
FREESOLV_PATH = 'data/FreeSolv/FreeSolv.csv'
CATS_PATH = 'data/CatS/CatS.csv'
LIPO_PATH = 'data/lipo/lipo.csv'
ESOL_PATH = 'data/esol/esol.csv'
DLS_PATH = 'data/dls/dls.csv'
BRADLEY_PATH = 'data/bradley/bradley.csv'
MALARIA_PATH = 'data/Malaria/Malaria.csv'

PATHS = {'FreeSolv': FREESOLV_PATH, 'esol': ESOL_PATH, 'lipo': LIPO_PATH, 'dls': DLS_PATH, 'CatS':CATS_PATH, 'bradley':BRADLEY_PATH, 'Malaria':MALARIA_PATH}

TASK_NAME = 'FreeSolv'  # Change dataset. Options: ['ESOL', 'FreeSolv', 'QM9', 'CEP', 'CatS', 'Melt', 'Malaria']

def main(task, split, n_runs, n_fold):
    global rem_mat, rem_diag, max_rem

    warnings.filterwarnings('ignore')

    print('\nTraining E3FP-GP on '+task+' dataset')
    print('\nGenerating features...')

    if task in PATHS:
        smiles_list, y  = parse_dataset(task, PATHS[task]) #NEED TO FIX MALARIA
        X = np.arange(len(smiles_list)).reshape(-1,1)
    else:
        raise Exception('Must provide dataset')
    
    rem_mat = np.load('kernels/'+task+'_e3fp.npy')
    #rem_mat = np.load('/rds-d2/user/wjm41/hpc-work/kernels/'+data_name+'_'+method_name+'_kernel_rematch.npy')

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
                    z = tf.math.sqrt(3*K_mat)*self.var
                    K_final = self.mag*(1+z)*tf.math.exp(-z)
                    return K_final
    
            def K_diag(self, X, presliced=None):
                    global rem_diag
                    A=tf.cast(X,tf.int32)
                    K_diag = tf.gather_nd(rem_diag, A)
                    z = tf.math.sqrt(3*K_diag)*self.var
                    return self.mag*(1+z)*tf.math.exp(-z)
    m = None

    def objective_closure():
        return -m.log_marginal_likelihood()

    r2_list = []
    rmse_list = []
    logP_list = []
    print('\nBeginning training loop...')
  
    j=0 
    for i in range(n_runs):
        if split=='random':
           kf = KFold(n_splits=n_fold, random_state=i, shuffle=True)
           split_list = kf.split(X)
        elif split=='scaffold':
            train_ind, test_ind = scaffold_split(smiles_list, seed=i)
            split_list = [(train_ind, test_ind)]
        for train_ind, test_ind in split_list:
            X_train, X_test = X[train_ind], X[test_ind]
            y_train, y_test = y[train_ind], y[test_ind]
            y_train, y_test, y_scaler = transform_data(y_train, y_test)
                
            X_train = tf.convert_to_tensor(X_train, dtype = tf.float64)
            X_test = tf.convert_to_tensor(X_test, dtype = tf.float64)        

            k = Matern32_rem()+gpflow.kernels.White(0.1)
            m = gpflow.models.GPR( data=(X_train, y_train), kernel=k)
        
            opt = gpflow.optimizers.Scipy()
        
            opt_logs = opt.minimize(objective_closure, m.trainable_variables, options=dict(maxiter=10000))
        
            #print_summary(m)
        
            #mean and variance GP prediction
            y_pred, y_var = m.predict_f(X_test)
            y_pred = y_scaler.inverse_transform(y_pred)
            y_test = y_scaler.inverse_transform(y_test)
            y_var = y_scaler.var_ * y_var
            score = r2_score(y_test, y_pred)
        
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            logP = -m.log_likelihood()       
        
            print("\nR^2: {:.3f}".format(score))
            print("RMSE: {:.3f}".format(rmse))
            print("-ve logP: {:.3f}".format(logP))
        
            r2_list.append(score)
            rmse_list.append(rmse)
            logP_list.append(logP)
        
            np.savetxt('results/e3fp_'+task+'_split_'+split+'_run_'+str(j)+'_ypred.txt', y_pred)
            np.savetxt('results/e3fp_'+task+'_split_'+split+'_run_'+str(j)+'_ytest.txt', y_test)
            np.savetxt('results/e3fp_'+task+'_split_'+split+'_run_'+str(j)+'_ystd.txt', np.sqrt(y_var))
            j+=1

    r2_list = np.array(r2_list)
    rmse_list = np.array(rmse_list)
    logP_list = np.array(logP_list)
    print("\nmean R^2: {:.4f} +- {:.4f}".format(np.mean(r2_list), np.std(r2_list)/np.sqrt(len(r2_list))))
    print("mean RMSE: {:.4f} +- {:.4f}".format(np.mean(rmse_list), np.std(rmse_list)/np.sqrt(len(rmse_list))))
    print("mean -ve logP: {:.4f} +- {:.4f}\n".format(np.mean(logP_list), np.std(logP_list)/np.sqrt(len(logP_list))))


if __name__ == "__main__":

     parser = argparse.ArgumentParser()
     parser.add_argument('-task', type=str, default=TASK_NAME, 
                         help='Dataset on which to train the GP')
     parser.add_argument('-split', type=str, default='random',
                         help='Train/Test splitting method. Possible choices: random/scaffold')
     parser.add_argument('-nruns', type=int, default=3, 
                         help='number of runs for train/test split.')
     parser.add_argument('-nfold', type=int, default=5, 
                         help='number of folds in K-fold cross-validation. Only for random splitting')
     args = parser.parse_args()
    
     main(args.task, args.split, args.nruns, args.nfold)
