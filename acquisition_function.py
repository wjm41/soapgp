"""
Module containing acquisition functions for active learning
"""

import numpy as np
from scipy.stats import norm

def cnp_var(x_context, y_context, x_test, model):
    """ 
    Suggests samples by maximising the predicted uncertainty (no exploitation, full exploration)

    :param X: input data
    :param model: CNP model
    :return: y_var
    """
    _, _, predict_test_var = model.predict(x_context, y_context, x_test)
    return np.squeeze(predict_test_var.data.numpy())

def gp_var(X, model):
    """ 
    Suggests samples by maximising the predicted uncertainty (no exploitation, full exploration)

    :param X: input data
    :param model: GPFlow model
    :return: y_var
    """
    _, y_var = model.predict_f(X)
    return y_var.numpy().flatten()


def gp_ei(X_test, y_train, model, xi=0.01):
    """
    Suggests samples by maximising the predicted expected improvement
    Balance exploitation & exploration via parameter xi

    :param X_test: input features of points to be sampled
    :param y_train: target values from observed samples
    :param model: GPFlow model
    :param xi: Controls balance of explore/exploit
    :return: EI scores
    """
    y_pred, y_var = model.predict_y(X_test)
    y_best = np.amax(y_train) # Best sample so far
    y_std = np.sqrt(y_var)
    
    with np.errstate(divide='warn'):
         imp = y_pred - y_best - xi
         Z =  imp/y_std
         ei = imp*norm.cdf(Z) + y_std*norm.pdf(Z)
         #ei[y_var<1e-8] = 0.0
    return ei.flatten()


def gp_greed(X, model):
    """
    Suggests samples by maximising the predicted y (full exploitation, no exploration)
    :param X: input data
    :param model: GPFlow model
    :return: X_samples
    """
    y_pred, _ = model.predict_y(X)
    return y_pred.numpy().flatten()
