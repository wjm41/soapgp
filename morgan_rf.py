"""
Script for training a GP on either fragment or fingerprint representations of molecules.
"""

import os
import warnings
import argparse

import pandas as pd
import numpy as np
from rdkit import DataStructs
from rdkit.Chem import MolFromSmiles, AllChem
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from helper import parse_dataset, transform_data, scaffold_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Adjust accordingly for your own file system
FREESOLV_PATH = 'data/FreeSolv/FreeSolv.csv'
CATS_PATH = 'data/CatS/CatS.csv'
LIPO_PATH = 'data/lipo/lipo.csv'
ESOL_PATH = 'data/esol/esol.csv'
dls_PATH = 'data/dls/dls.csv'
BRADLEY_PATH = 'data/bradley/bradley.csv'
MALARIA_PATH = 'data/Malaria/Malaria.csv'
CHEMBL5118_PATH = 'data/CHEMBL5118.csv'
CHEMBL3927_PATH = 'data/CHEMBL3927.csv'

PATHS = {'FreeSolv': FREESOLV_PATH, 'esol': ESOL_PATH, 'lipo': LIPO_PATH, 'dls': dls_PATH, 'CatS':CATS_PATH, 'bradley':BRADLEY_PATH, 'Malaria':MALARIA_PATH, 'CHEMBL5118': CHEMBL5118_PATH, 'CHEMBL3927' : CHEMBL3927_PATH, 'CHEMBL5118_typical': CHEMBL5118_PATH, 'CHEMBL3927_typical' : CHEMBL3927_PATH}

TASK_NAME = 'FreeSolv'  # Change dataset. Options: ['ESOL', 'FreeSolv', 'QM9', 'CEP', 'CatS', 'Melt', 'Malaria']

def generate_fingerprints(smile):    
    mol = MolFromSmiles(smile)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=1024)
    array = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, array)
    #print(array)
    return array

def fit_forest(X,y):
    params = {'n_estimators': [100, 1000, 10000], 'max_depth': [1, 2, 3], 'min_samples_split': [2, 4]}
    #params = {'n_estimators': [1000], 'max_depth': [1, 2]}
    search = GridSearchCV(RandomForestRegressor(), params, cv=5)
    model  = search.fit(X, y).best_estimator_
    return model
    #model = RandomForestRegressor(n_estimators=100)
    #return model.fit(X,y)

def main(task, split, n_runs, n_fold):
    global rem_mat, rem_diag, max_rem

    warnings.filterwarnings('ignore')

    print('\nTraining SOAP-GP on '+task+' dataset')
    print('\nGenerating features...')

    if task in PATHS:
        smiles_list, y  = parse_dataset(task, PATHS[task]) #NEED TO FIX MALARIA
        X = np.arange(len(smiles_list)).reshape(-1,1)
        
    else:
        raise Exception('Must provide dataset')
    
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
            split_list = [train_ind, test_ind]
        for train_ind, test_ind in split_list:
            y_train, y_test = y[train_ind], y[test_ind]
 
            smiles_df = pd.DataFrame(smiles_list, columns=['smiles'])
            train_smiles = smiles_df.iloc[train_ind]['smiles'].to_list()
            test_smiles = smiles_df.iloc[test_ind]['smiles'].to_list()

            X_train = np.asarray([generate_fingerprints(s) for s in train_smiles])      
            X_test = np.asarray([generate_fingerprints(s) for s in test_smiles])      
            y_train, y_test, y_scaler = transform_data(y_train, y_test)

            m = fit_forest(X_train, y_train)
       
            #mean and variance GP prediction
            y_pred = m.predict(X_test)
            y_pred = y_scaler.inverse_transform(y_pred)
            y_test = y_scaler.inverse_transform(y_test)
            
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

    r2_list = np.array(r2_list)
    rmse_list = np.array(rmse_list)
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
    
     main(args.task, args.split, args.nruns, args.nfold)
