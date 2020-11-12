"""
Script for training a GP on either fragment or fingerprint representations of molecules.
"""

import os
import warnings
import argparse

import gpflow
import pandas as pd
import numpy as np
import tensorflow as tf
from gpflow.utilities import print_summary, positive
from rdkit.Chem import MolFromSmiles, AllChem
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score

from helper import scaffold_split
from parse_data import parse_dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Matern32_rem(gpflow.kernels.Kernel):
        def __init__(self, rem_mat):
                super().__init__(active_dims=[0])
                self.var = gpflow.Parameter(1.0, transform=positive())
                self.mag = gpflow.Parameter(1.0, transform=positive())
                self.rem_mat = tf.constant(rem_mat,dtype=tf.float64)
                self.max_rem = rem_mat.max()

        def K(self, X, X2=None, presliced=None):
                if X2 is None:
                        X2=X
                A = tf.cast(X,tf.int32)
                A = tf.reshape(A,[-1])
                A2 = tf.reshape(X2,[-1])
                A2 = tf.cast(A2,tf.int32)
                K_mat = tf.gather(self.rem_mat, A, axis=0)
                K_mat = tf.gather(K_mat, A2, axis=1)
                z = tf.math.sqrt(6*(self.max_rem-K_mat))*self.var
                K_final = self.mag*(1+z)*tf.math.exp(-z)
                return K_final

        def K_diag(self, X, presliced=None):
                return self.mag*tf.reshape(tf.ones_like(X),-1)

def train_soapgp(X_train, y_train, rem_mat, log=False):

    k = Matern32_rem(rem_mat)+gpflow.kernels.White(0.1)
    m = gpflow.models.GPR( data=(X_train, y_train), kernel=k)
    
    opt = gpflow.optimizers.Scipy()

    def objective_func():
        return -m.log_marginal_likelihood()
    
    opt_logs = opt.minimize(objective_func, m.trainable_variables, options=dict(maxiter=10000))
    
    if log:
        print_summary(m)
    return m

def main(args):

    warnings.filterwarnings('ignore')

    print('\nTraining SOAP-GP on '+task+' dataset')
    print('\nGenerating features...')
    
    if args.task=='IC50':
       PATHS[args.task] = PATHS[args.task]+args.suffix+'.can'
       print('Subtask: '+args.suffix)
    smiles_list, y  = parse_dataset(args.task)
    X = np.arange(len(smiles_list)).reshape(-1,1)
    
    rem_mat = np.load('kernels/'+task+'_soap.npy') # adjust this accordingly TODO 

    r2_list = []
    rmse_list = []
    logP_list = []
    print('\nBeginning training loop...')
  
    j=0 
    for i in range(args.n_runs):
        if split=='random':
           kf = KFold(n_splits=n_fold, random_state=i, shuffle=True)
           split_list = kf.split(X)
        elif split=='scaffold':
            train_ind, test_ind = scaffold_split(smiles_list, seed=i)
            split_list = [train_ind, test_ind]
        for train_ind, test_ind in split_list:
            X_train, X_test = X[train_ind], X[test_ind]
            y_train, y_test = y[train_ind], y[test_ind]

            X_train = tf.convert_to_tensor(X_train, dtype = tf.float64)
            X_test = tf.convert_to_tensor(X_test, dtype = tf.float64)        

            m = train_soapgp(X_train, y_train, rem_mat)
  
            #mean and variance GP prediction
            y_pred, y_var = m.predict_f(X_test)

            score = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
            print("\nR^2: {:.3f}".format(score))
            print("RMSE: {:.3f}".format(rmse))
        
            r2_list.append(score)
            rmse_list.append(rmse)
        
            #np.savetxt('results/soapgp_'+task+'_split_'+split+'_run_'+str(j)+'_ypred.txt', y_pred)
            #np.savetxt('results/soapgp_'+task+'_split_'+split+'_run_'+str(j)+'_ytest.txt', y_test)
            #np.savetxt('results/soapgp_'+task+'_split_'+split+'_run_'+str(j)+'_ystd.txt', np.sqrt(y_var))
            j+=1

    print("\nmean R^2: {:.4f} +- {:.4f}".format(np.mean(r2_list), np.std(r2_list)/np.sqrt(len(r2_list))))
    print("mean RMSE: {:.4f} +- {:.4f}".format(np.mean(rmse_list), np.std(rmse_list)/np.sqrt(len(rmse_list))))


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
    
     main(args)
