"""
Script for training a GP on either fragment or fingerprint representations of molecules.
"""
import time
import os
import warnings
import argparse
from tqdm import tqdm
import gpflow
import pandas as pd
import numpy as np
import scipy
import tensorflow as tf
from gpflow.utilities import print_summary
from gpflow.utilities import positive
from rdkit.Chem import MolFromSmiles, AllChem
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score, auc, precision_recall_curve
from sklearn.preprocessing import normalize

from dscribe.descriptors import SOAP
from dscribe.kernels import REMatchKernel, AverageKernel

from helper import parse_dataset, transform_data, scaffold_split, read_xyz, split_by_lengths

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Adjust accordingly for your own file system
FREESOLV_PATH = 'data/FreeSolv/FreeSolv.csv'
CATS_PATH = 'data/CatS/CatS.csv'
LIPO_PATH = 'data/lipo/lipo.csv'
ESOL_PATH = 'data/esol/esol.csv'
DLS_PATH = 'data/dls/dls.csv'
SARS_PATH = 'data/sars/sars.csv'
BRADLEY_PATH = 'data/bradley/bradley.csv'
MALARIA_PATH = 'data/Malaria/Malaria.can'
CHEMBL5118_PATH = 'data/CHEMBL5118.csv'
CHEMBL3927_PATH = 'data/CHEMBL3927.csv'
BENDER_PATH = 'data/Bender/'
#COVID_PATH = 'data/covid/covid.csv'
COVID_PATH = 'DATA/COVID_NEW/COVID_NEW.CSV'
extra_PATH = 'data/extra_mols/extra_mols.csv'

PATHS = {'FreeSolv': FREESOLV_PATH, 'esol': ESOL_PATH, 'lipo': LIPO_PATH, 'nusolv': DLS_PATH, 'CatS':CATS_PATH, 'bradley':BRADLEY_PATH, 'Malaria':MALARIA_PATH, 'CHEMBL5118': CHEMBL5118_PATH, 'CHEMBL3927' : CHEMBL3927_PATH, 'CHEMBL5118_typical': CHEMBL5118_PATH, 'CHEMBL3927_typical' : CHEMBL3927_PATH, 'Bender': BENDER_PATH, 'sars':SARS_PATH, 'SARS': SARS_PATH, 'covid_new':COVID_PATH, 'extra_mols':extra_PATH}
TASK_NAME = 'FreeSolv'  # Change dataset. Options: ['ESOL', 'FreeSolv', 'QM9', 'CEP', 'CatS', 'Melt', 'Malaria']


def main(task, split, n_runs, n_fold, suffix):
    warnings.filterwarnings('ignore')

    print('\nTraining SOAP-GP on '+task+' dataset')
    print('\nGenerating features...')

    if task in PATHS:
        if task=='Bender':
           PATHS[task] = PATHS[task]+suffix+'.can'
           print('Subtask: '+suffix)
        smiles_list, y  = parse_dataset(task, PATHS[task]) #NEED TO FIX MALARIA
    else:
        raise Exception('Must provide dataset')

    #y = np.where(y<5, 0, y)
    #y = np.where(y>=5, 1, y)

    mols, num_list, atom_list, species = read_xyz('data/'+task+"/"+task+".xyz")    

    X = np.arange(len(mols)).reshape(-1,1)

    # Setting up the SOAP descriptor
    rcut_small = 3.0
    sigma_small = 0.2
    rcut_large = 6.0
    sigma_large = 0.4
    
    small_soap = SOAP(
        species=species,
        periodic=False,
        rcut=rcut_small,
        nmax=12,
        lmax=8,
        sigma = sigma_small,
        sparse=True
    )
    
    large_soap = SOAP(
        species=species,
        periodic=False,
        rcut=rcut_large,
        nmax=12,
        lmax=8,
        sigma = sigma_large,
        sparse=True
    )
    re = REMatchKernel(metric="polynomial", degree=3, gamma=1, coef0=0, alpha=0.5, threshold=1e-6, normalize_kernel=True)
    
    class SOAP_kern(gpflow.kernels.Kernel):
            def __init__(self):
                    super().__init__(active_dims=[0])
                    self.var = gpflow.Parameter(10.0, transform=positive())
                    self.mag = gpflow.Parameter(1.0, transform=positive())
    
            def K(self, X, X2=None, presliced=None):
                    A = X.numpy().flatten().astype(int)
                    X_soap = scipy.sparse.load_npz('/rds-d2/user/wjm41/hpc-work/soap/covid_'+str(A[0])+'.npz')
                    X_list = [num_list[A[0]]]
                    for i in A[1:]:
                        X_list.append(num_list[i])
                        X_soap = scipy.sparse.vstack([X_soap, scipy.sparse.load_npz('/rds-d2/user/wjm41/hpc-work/soap/covid_'+str(i)+'.npz')])
                    X_soap = normalize(X_soap, copy=False)                    
                    X_soap =  split_by_lengths(X_soap, X_list)

                    if X2 is None:
                        #t1 = time.time()
                        K_mat = re.create(X_soap)
                        #t2 = time.time()
                        #print('Time taken to calculate kernel = {:.1f}s'.format(t2-t1))
                        max_rem = K_mat.max()
                        z = tf.math.sqrt(6*(max_rem-tf.constant(K_mat, dtype=tf.float64)))*self.var
                        K_final = self.mag*(1+z)*tf.math.exp(-z)
                        return K_final
                    else:        
                        A2 = X2.numpy().flatten().astype(int)
                        X2_soap = scipy.sparse.load_npz('/rds-d2/user/wjm41/hpc-work/soap/covid_'+str(A2[0])+'.npz')
                        X2_list = [num_list[A2[0]]]
                        for i in A2[1:]:
                            X2_list.append(num_list[i])
                            X2_soap = scipy.sparse.vstack([X2_soap, scipy.sparse.load_npz('/rds-d2/user/wjm41/hpc-work/soap/covid_'+str(i)+'.npz')])
                        X2_soap = normalize(X2_soap, copy=False)                    
                        X2_soap =  split_by_lengths(X2_soap, X2_list)

                        #t3 = time.time()
                        K_mat = re.create(X_soap, X2_soap)
                        #t4 = time.time()
                        #print('Time taken to calculate kernel = {:.1f}s'.format(t4-t3))
                        max_rem = K_mat.max()
                        z = tf.math.sqrt(6*(max_rem-tf.constant(K_mat, dtype=tf.float64)))*self.var
                        K_final = self.mag*(1+z)*tf.math.exp(-z)
                        return K_final
    
            def K_diag(self, X, presliced=None):
                    return self.mag*tf.reshape(tf.ones_like(X),-1)
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
           #kf = KFold(n_splits=n_fold, random_state=i, shuffle=True)
           #split_list = kf.split(X)
            train_ind, test_ind = train_test_split(X.flatten())
            split_list = [[train_ind, test_ind]]
        elif split=='scaffold':
            train_ind, test_ind = scaffold_split(smiles_list, seed=i)
            split_list = [[train_ind, test_ind]]
        for train_ind, test_ind in split_list:
            X_train, X_test = X[train_ind], X[test_ind]
            y_train, y_test = y[train_ind].reshape(-1,1), y[test_ind].reshape(-1,1)

            #train_inds = [x!=4.0 for x in y_train.flatten()]
            #X_train = X_train[train_inds]
            #y_train = y_train[train_inds]

            #test_inds = [x!=4.0 for x in y_test.flatten()]
            #X_test = X_test[test_inds]
            #y_test = y_test[test_inds]

            X_train = tf.convert_to_tensor(X_train, dtype = tf.float64)
            X_test = tf.convert_to_tensor(X_test, dtype = tf.float64)        

            k = SOAP_kern()+gpflow.kernels.White(0.1)
            #m = gpflow.models.VGP( data=(X_train, y_train), kernel=k, likelihood=gpflow.likelihoods.Bernoulli())
            m = gpflow.models.GPR( data=(X_train, y_train), kernel=k)
        
            opt = gpflow.optimizers.Scipy()
        
            opt_logs = opt.minimize(objective_closure, m.trainable_variables, options=dict(maxiter=10000))
        
            #print_summary(m)
        
            #mean and variance GP prediction
            y_pred, y_var = m.predict_f(X_test)
            y_pred = y_pred.numpy()
            #y_pred = y_scaler.inverse_transform(y_pred)
            #y_test = y_scaler.inverse_transform(y_test)
            #y_var = y_scaler.var_ * y_var
            #score = roc_auc_score(y_test,y_pred)
            #precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
            #rmse = auc(recall, precision)
            score = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            #logP = -m.log_likelihood()       
        
            #print("\nROC-AUC: {:.3f}".format(score))
            #print("PRC-AUC: {:.3f}".format(rmse))
            print("\nR^2: {:.3f}".format(score))
            print("RMSE: {:.3f}".format(rmse))
        
            r2_list.append(score)
            rmse_list.append(rmse)
        
            j+=1

    r2_list = np.array(r2_list)
    #rmse_list = np.array(rmse_list)
    #print("\nmean ROC-AUC: {:.3f} +- {:.3f}".format(np.mean(r2_list), np.std(r2_list)/np.sqrt(len(r2_list))))
    #print("mean PRC-AUC: {:.3f} +- {:.3f}".format(np.mean(rmse_list), np.std(rmse_list)/np.sqrt(len(rmse_list))))
    print("\nmean R^2: {:.3f}\pm{:.3f}".format(np.mean(r2_list), np.std(r2_list)/np.sqrt(len(r2_list))))
    print("mean RMSE: {:.3f}\pm{:.3f}".format(np.mean(rmse_list), np.std(rmse_list)/np.sqrt(len(rmse_list))))


if __name__ == "__main__":

     parser = argparse.ArgumentParser()
     parser.add_argument('-suffix', type=str, default='A2a', 
                         help='Data suffix for Bender dataset')
     parser.add_argument('-task', type=str, default=TASK_NAME, 
                         help='Dataset on which to train the GP')
     parser.add_argument('-split', type=str, default='random',
                         help='Train/Test splitting method. Possible choices: random/scaffold')
     parser.add_argument('-nruns', type=int, default=3, 
                         help='number of runs for train/test split.')
     parser.add_argument('-nfold', type=int, default=5, 
                         help='number of folds in K-fold cross-validation. Only for random splitting')
     args = parser.parse_args()
    
     main(args.task, args.split, args.nruns, args.nfold, args.suffix)
