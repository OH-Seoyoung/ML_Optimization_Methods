import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.io
import math
import sklearn
import sklearn.datasets

def init_mome(parameters):
    
    L = len(parameters) // 2 # number of layers
    v = {}
    
    # Initialize velocity
    for l in range(L):
        v["dW" + str(l+1)] = np.zeros((parameters["W" + str(l+1)].shape[0], parameters["W" + str(l+1)].shape[1]))
        v["db" + str(l+1)] = np.zeros((parameters["b" + str(l+1)].shape[0], parameters["b" + str(l+1)].shape[1]))
        
    return v

def update_params_mome(parameters, grads, v, beta, learning_rate):

    L = len(parameters) // 2 # number of layers in the neural networks
    
    # Momentum update
    for l in range(L):
        # velocities
        v["dW" + str(l+1)] = beta*v["dW" + str(l+1)] + (1 - beta)*grads['dW' + str(l+1)]
        v["db" + str(l+1)] = beta*v["db" + str(l+1)] + (1 - beta)*grads['db' + str(l+1)]
        
        # update params
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*v["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*v["db" + str(l+1)]
        
    return parameters, v