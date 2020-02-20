"""
Script for training a GP on either fragment or fingerprint representations of molecules.
"""

import os
import warnings
import argparse

import gpflow
from gpflow.utilities import print_summary
import numpy as np
from rdkit.Chem import MolFromSmiles, AllChem
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from data_utils import parse_dataset, transform_data, scaffold_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Adjust accordingly for your own file system

FREESOLV_PATH = '~/ml_physics/GraphChem/data/orig/FreeSolv_SAMPL.csv'
ESOL_PATH = '~/ml_physics/GraphChem/data/orig/ESOL_delaney-processed.csv'
CATS_PATH = '~/ml_physics/GraphChem/data/orig/CatS.csv'
MELTING_PATH = '~/ml_physics/GraphChem/data/orig/Meltingpoint.csv'
MALARIA_PATH = '~/ml_physics/GraphChem/data/orig/Malaria.csv'
QM9_PATH = '~/ml_physics/GraphChem/data/orig/qm9.csv'
CEP_PATH = '~/ml_physics/GraphChem/data/orig/CEP_pce.csv'

PATHS = {'FreeSolv': FREESOLV_PATH, 'esol': ESOL_PATH, 'lipo': QM9_PATH, 'DLS': DLS_PATH, 'CatS':CATS_PATH, 'bradley':BRADLEY_PATH, 'Malaria':MALARIA_PATH}

TASK_NAME = 'FreeSolv'  # Change dataset. Options: ['ESOL', 'FreeSolv', 'QM9', 'CEP', 'CatS', 'Melt', 'Malaria']


def main(task, feat, split, seed):

    warnings.filterwarnings('ignore')

    use_frag = True
    if feat != 'fragments':
        use_frag = False

    print('\nTraining GP on '+task+' dataset with '+feat+' features ('+str(n_com)+' components)')
    print('\nGenerating features...')

    if task in PATHS:
        X, y, smiles_list = parse_dataset(task, PATHS[task], use_frag)
    else:
        raise Exception('Must provide dataset')

    if feat == 'fingerprints':

        rdkit_mols = [MolFromSmiles(smiles) for smiles in X]
        X = [AllChem.GetMorganFingerprintAsBitVect(mol, 2) for mol in rdkit_mols]
        X = np.asarray(X)
    #else:
    #    raise Exception('Invalid features chosen')

    m = None

    def objective_closure():
        return -m.log_marginal_likelihood()

    r2_list = []
    rmse_list = []
    logP_list = []
    print('\nBeginning training loop...')
    j = 0

    for i in range(5, 10):
        if split=='random':
           X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
        elif split=='scaffold':
           train_ind, test_ind = scaffold_split(smiles_list, seed=i)
           X_train, X_test = X[train_ind], X[test_ind]
           y_train, y_test = y[train_ind], y[test_ind]
        X_train, y_train, X_test, y_test, y_scaler = transform_data(X_train, y_train, X_test, y_test, n_com)

        k = gpflow.kernels.RBF(lengthscale=np.ones(n_com))
        m = gpflow.models.GPR( data=(X_train, y_train), kernel=k, noise_variance=1)

        opt = gpflow.optimizers.Scipy()

        opt_logs = opt.minimize(objective_closure, m.trainable_variables, options=dict(maxiter=100))

        print_summary(m)

        #mean and variance GP prediction
        y_pred, y_var = m.predict_f(X_test)
        y_pred = y_scaler.inverse_transform(y_pred)
        y_test = y_scaler.inverse_transform(y_test)
        score = r2_score(y_test, y_pred)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        logP = -m.log_likelihood()       
 
        print("\nR^2: {:.3f}".format(score))
        print("RMSE: {:.3f}".format(rmse))
        print("-ve logP: {:.3f}".format(logP))

        r2_list.append(score)
        rmse_list.append(rmse)
        logP_list.append(logP)

        np.savetxt('results/'+task+'_seed_'+str(j)+'_ypred_'+feat+'.txt', y_pred)
        np.savetxt('results/'+task+'_seed_'+str(j)+'_ytest.txt', y_test)
        np.savetxt('results/'+task+'_seed_'+str(j)+'_ystd_'+feat+'.txt', np.sqrt(y_var))

        j += 1

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
     parser.add_argument('-seed', type=int, default=0, 
                         help='random seed used for train/test split.')
     args = parser.parse_args()
    
     main(args.task, args.feat, args.split, args.seed)
