import numpy as np


def identity(x):
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-4.9*x))

def relu(x):
    return np.max(np.hstack((np.zeros_like(x),x)).reshape(-1,2),axis=1)

def tanh(x):
    return np.tanh(x)


ACTIVATIONS = [identity, tanh, tanh]
