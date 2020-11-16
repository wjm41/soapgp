"""
Script for training a Random Forest model on fingerprint representations of molecules.
"""


import os
import warnings
import argparse

import pandas as pd
import numpy as np
from rdkit.Chem import MolFromSmiles, AllChem
from rdkit import DataStructs
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from helper import scaffold_split
from parse_data import parse_dataset


def generate_fingerprints(smile):
    mol = MolFromSmiles(smile)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=1024)
    array = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, array)
    #print(array)
    return array

def fit_forest(X,y):
    # params = {'n_estimators': [100, 1000, 10000], 'max_depth': [1, 2, 3], 'min_samples_split': [2, 4]}
    # search = GridSearchCV(RandomForestRegressor(), params, cv=5)
    # model  = search.fit(X, y).best_estimator_
    model = RandomForestRegressor(n_estimators=100)
    return model.fit(X,y)

def main(args):
    warnings.filterwarnings('ignore')

    print('\nTraining ECFP-RF on ' + args.task + ' dataset')
    print('\nGenerating features...')

    if args.task=='IC50':
        print('Subtask: {}'.format(args.subtask))

    smiles_list, y = parse_dataset(args.task, subtask=args.subtask)
    X = np.arange(len(smiles_list)).reshape(-1,1) # array of data indices

    r2_list = []
    rmse_list = []

    print('\nBeginning training loop...')

    j = 0
    for i in range(args.n_runs):
        if args.split == 'random':
            kf = KFold(n_splits=args.n_folds, random_state=i, shuffle=True)
            split_list = kf.split(X)
        elif args.split == 'scaffold':
            train_ind, test_ind = scaffold_split(smiles_list, seed=i)
            split_list = [train_ind, test_ind]
        for train_ind, test_ind in split_list:
            y_train, y_test = y[train_ind], y[test_ind]

            smiles_df = pd.DataFrame(smiles_list, columns=['smiles'])
            train_smiles = smiles_df.iloc[train_ind]['smiles'].to_list()
            test_smiles = smiles_df.iloc[test_ind]['smiles'].to_list()

            X_train = np.asarray([generate_fingerprints(s) for s in train_smiles])
            X_test = np.asarray([generate_fingerprints(s) for s in test_smiles])

            m = fit_forest(X_train, y_train)

            y_pred = m.predict(X_test)

            score = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            print("\nR^2: {:.3f}".format(score))
            print("RMSE: {:.3f}".format(rmse))

            r2_list.append(score)
            rmse_list.append(rmse)

            # np.savetxt('results/ecfp_'+task+'_split_'+split+'_run_'+str(j)+'_ypred.txt', y_pred)
            # np.savetxt('results/ecfp_'+task+'_split_'+split+'_run_'+str(j)+'_ytest.txt', y_test)
            # np.savetxt('results/ecfp_'+task+'_split_'+split+'_run_'+str(j)+'_ystd.txt', np.sqrt(y_var))
            j += 1

    print("\nmean R^2: {:.4f} +- {:.4f}".format(np.mean(r2_list), np.std(r2_list) / np.sqrt(len(r2_list))))
    print("mean RMSE: {:.4f} +- {:.4f}".format(np.mean(rmse_list), np.std(rmse_list) / np.sqrt(len(rmse_list))))

if __name__ == "__main__":

     parser = argparse.ArgumentParser()
     parser.add_argument('-task', type=str, default='IC50',
                         help='Dataset on which to train ECFP-RF')
     parser.add_argument('-subtask', type=str, default='A2a',
                         help='Dataset on which to train ECFP-RF')
     parser.add_argument('-split', type=str, default='random',
                         help='Train/Test splitting method. Possible choices: random/scaffold')
     parser.add_argument('-n_runs', type=int, default=3,
                         help='number of runs for train/test split.')
     parser.add_argument('-n_folds', type=int, default=5,
                         help='number of folds in K-fold cross-validation. Only for random splitting')
     args = parser.parse_args()
    
     main(args)
