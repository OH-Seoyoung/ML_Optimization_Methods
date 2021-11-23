import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.io
import math
import sklearn
import sklearn.datasets

def init_rmsp(parameters) :

    L = len(parameters) // 2 # number of layers
    s = {}  # the exponentially weighted average of the squared gradient

    for l in range(L):
        s["dW" + str(l+1)] = np.zeros((parameters["W" + str(l+1)].shape[0], parameters["W" + str(l+1)].shape[1]))
        s["db" + str(l+1)] = np.zeros((parameters["b" + str(l+1)].shape[0], parameters["b" + str(l+1)].shape[1]))
    
    return s

def update_params_rmsp(parameters, grads, s, learning_rate, beta, epsilon):

    L = len(parameters) // 2                
    s_corrected = {}   # second moment estimate
    
    for l in range(L):
        
        # Moving average of the squared gradients
        s["dW" + str(l+1)] = beta*s["dW" + str(l+1)] + (1 - beta)*np.square(grads['dW' + str(l+1)])
        s["db" + str(l+1)] = beta*s["db" + str(l+1)] + (1 - beta)*np.square(grads['db' + str(l+1)])

        # Update parameters
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*grads['dW' + str(l+1)]/(np.sqrt(s["dW" + str(l+1)])+epsilon)
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*grads['db' + str(l+1)]/(np.sqrt(s["db" + str(l+1)])+epsilon)

    return parameters, s