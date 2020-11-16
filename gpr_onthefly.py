"""
Script for training a SOAP GP that calculates the SOAP kernel on-the-fly. More practical (memory efficient) for
training on a small dataset and screening a large library as opposed to benchmarking on datasets.
"""

import time
import os
import warnings
import argparse

import gpflow
import scipy
import numpy as np
import tensorflow as tf
from gpflow.utilities import print_summary
from gpflow.utilities import positive
from rdkit.Chem import MolFromSmiles, AllChem
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import normalize

from dscribe.descriptors import SOAP
from dscribe.kernels import REMatchKernel, AverageKernel

from helper import scaffold_split
from parse_data import parse_dataset
from helper import split_by_lengths

class SOAP_onthefly(gpflow.kernels.Kernel):
    """
    A kernel class that dynamically calculates the SOAP kernel on-the-fly by loading SOAP descriptors from
    args.soap_path by index and calculating the SOAP kernel.
    """
    def __init__(self, args):
        super().__init__(active_dims=[0])
        self.var = gpflow.Parameter(10.0, transform=positive())
        self.mag = gpflow.Parameter(1.0, transform=positive())
        self.args = args
        self.re = REMatchKernel(metric="polynomial", degree=3, gamma=1, coef0=0, alpha=0.5, threshold=1e-6, normalize_kernel=True)

    def K(self, X, X2=None, presliced=None):
        A = X.numpy().flatten().astype(int)
        X_soap = scipy.sparse.load_npz(args.soap_path + args.task + '_soap_' + str(A[0]) + '.npz')
        X_list = [X_soap.get_shape()[0]]
        for i in A[1:]:
            X_next = scipy.sparse.load_npz(args.soap_path + args.task + '_soap_' + str(i) + '.npz')
            X_list.append(X_next.get_shape()[0])
            X_soap = scipy.sparse.vstack([X_soap, X_next])
        X_soap = normalize(X_soap, copy=False)
        X_soap = split_by_lengths(X_soap, X_list)

        if X2 is None:
            # t1 = time.time()
            K_mat = self.re.create(X_soap)
            # t2 = time.time()
            # print('Time taken to calculate kernel = {:.1f}s'.format(t2-t1))

            max_rem = K_mat.max()
            z = tf.math.sqrt(6 * (max_rem - tf.constant(K_mat, dtype=tf.float64))) * self.var
            K_final = self.mag * (1 + z) * tf.math.exp(-z)
            return K_final
        else:
            A2 = X2.numpy().flatten().astype(int)
            X2_soap = scipy.sparse.load_npz(args.soap_path + args.task + '_soap_' + str(A2[0]) + '.npz')
            X2_list = [X2_soap.get_shape()[0]]
            for i in A2[1:]:
                X_next = scipy.sparse.load_npz(args.soap_path + args.task + '_soap_' + str(i) + '.npz')
                X2_list.append(X_next.get_shape()[0])
                X2_soap = scipy.sparse.vstack([X2_soap, X_next])
            X2_soap = normalize(X2_soap, copy=False)
            X2_soap = split_by_lengths(X2_soap, X2_list)

            # t3 = time.time()
            K_mat = self.re.create(X_soap, X2_soap)
            # t4 = time.time()
            # print('Time taken to calculate kernel = {:.1f}s'.format(t4-t3))

            max_rem = K_mat.max()
            z = tf.math.sqrt(6 * (max_rem - tf.constant(K_mat, dtype=tf.float64))) * self.var # Matern v=3/2 kernel
            K_final = self.mag * (1 + z) * tf.math.exp(-z)
            return K_final

    def K_diag(self, X, presliced=None):
        return self.mag * tf.reshape(tf.ones_like(X), -1) # diagonal of ones * self.mag

def train_soapgp_onthefly(X_train, y_train, args, log=False):
    """
    Initialises the kernel and GP model, then calls on the scipy L-BFGS optimizer to minimise the training loss.
    Call with log=True to print the final values of the kernel parameters for save/load purposes.
    """
    k = SOAP_onthefly(args) + gpflow.kernels.White(0.1)
    m = gpflow.models.GPR(data=(X_train, y_train), kernel=k)

    opt = gpflow.optimizers.Scipy()

    # needs compile=False so tensorflow remains in eager mode, allowing calls to X.numpy()
    opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=10000), compile=False)

    if log:
        print_summary(m)
    return m


def main(args):
    warnings.filterwarnings('ignore')

    print('\nTraining SOAP-GP on ' + task + ' dataset')
    print('\nGenerating features...')

    smiles_list, y  = parse_dataset(args.task, subtask=args.subtask)
    X = np.arange(len(smiles_list)).reshape(-1,1)

    r2_list = []
    rmse_list = []

    print('\nBeginning training loop...')

    j = 0
    for i in range(args.n_runs):
        if split == 'random':
            kf = KFold(n_splits=n_fold, random_state=i, shuffle=True)
            split_list = kf.split(X)

        elif split == 'scaffold':
            train_ind, test_ind = scaffold_split(smiles_list, seed=i)
            split_list = [train_ind, test_ind]

        for train_ind, test_ind in split_list:
            X_train, X_test = X[train_ind], X[test_ind]
            y_train, y_test = y[train_ind], y[test_ind]

            X_train = tf.convert_to_tensor(X_train, dtype=tf.float64)
            X_test = tf.convert_to_tensor(X_test, dtype=tf.float64)

            m = train_soapgp_onthefly(X_train, y_train, args)

            # mean and variance GP prediction
            y_pred, y_var = m.predict_f(X_test)

            score = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            print("\nR^2: {:.3f}".format(score))
            print("RMSE: {:.3f}".format(rmse))

            r2_list.append(score)
            rmse_list.append(rmse)

            # np.savetxt('results/soapgp_'+task+'_split_'+split+'_run_'+str(j)+'_ypred.txt', y_pred)
            # np.savetxt('results/soapgp_'+task+'_split_'+split+'_run_'+str(j)+'_ytest.txt', y_test)
            # np.savetxt('results/soapgp_'+task+'_split_'+split+'_run_'+str(j)+'_ystd.txt', np.sqrt(y_var))
            j += 1

    print("\nmean R^2: {:.4f} +- {:.4f}".format(np.mean(r2_list), np.std(r2_list) / np.sqrt(len(r2_list))))
    print("mean RMSE: {:.4f} +- {:.4f}".format(np.mean(rmse_list), np.std(rmse_list) / np.sqrt(len(rmse_list))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', type=str, default=TASK_NAME,
                        help='Dataset on which to train SOAP-GP')
    parser.add_argument('-subtask', type=str, default='A2a',
                        help='For IC50, data subset to train SOAP-GP')
    parser.add_argument('-soap_path', type=str, default='soap/',
                        help='Path to directory containing saved SOAP descriptors')
    parser.add_argument('-split', type=str, default='random',
                        help='Train/Test splitting method. Possible choices: random/scaffold')
    parser.add_argument('-nruns', type=int, default=3,
                        help='number of runs for train/test split.')
    parser.add_argument('-nfold', type=int, default=5,
                        help='number of folds in K-fold cross-validation. Only for random splitting')
    args = parser.parse_args()

    main(args)
