"""
Arrumar um nome melhor para esse classe
E documentar essa classe
"""

import numpy as np

def ativacao(X,w,b):
    #(w·x)-b
    return np.dot(X, w) - b

#ainda terminar
def kernel_linear(X,Y):
    y_ = np.where(Y <= 0, -1, 1)
    n_samples, n_features = X.shape

    w = np.zeros(n_features)
    b = 0

    return [w,b]


def kernel_rbf():
    pass

def kernel_poli():
    pass

#se for o caso, colocar outros kernels