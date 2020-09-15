"""
Arrumar um nome melhor para esse classe
E documentar essa classe
"""

from types import CodeType
import numpy as np
import random
import math

# kernels


def linear_kernel(x1, x2):
    return np.dot(x1, x2)


def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p


def gaussian_kernel(x, y, sigma:float=5.0):
    return np.exp(-np.linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

# não sei como chamar


def shuffle_date_base(data_base: list) -> list:
    """Função para embaralhar a base em para uma nova variável 

    Args:
        data_base (list): Base de dados

    Returns:
        list: Base embaralhada
    """
    aux = data_base
    random.shuffle(aux)
    return aux


def split_Ys_from_data_base(data_base: list, colum_Y: int) -> tuple:
    """Função para separa a base de dados em {Características, classes} 

    Args:
        data_base (list): Base de dados completa [Características + classes]
        colum_Y (int): Colina onde estão as classes

    Returns:
        tuple: Retorna uma Tupla (X: Características, Y:classes)
    """
    X = []
    Y = []

    for i in data_base:
        aux = []
        for e, j in enumerate(i):
            if e == colum_Y:
                Y.append(int(j))
                continue
            aux.append(j)
        X.append(aux)

    return X, Y


def split_in_training_and_test(X: np.ndarray, Y: np.ndarray, size_training_db: int) -> tuple:
    """Separa a os X's e Y's segunto a porcentagem size_training_db

    Args:
        X ([type]): [description]
        Y ([type]): [description]
        size_training_db (int): [description]

    Returns:
        tuple: [description]
    """
    aux = math.ceil(len(X)*(size_training_db/100))
    X_training = X[:aux]
    Y_training = Y[:aux]
    X_test = X[aux:len(X)]
    Y_test = Y[aux:len(X)]
    return X_training, Y_training, X_test, Y_test


def accuracy(y: list, yi: list) -> float:
    ac = 0
    for i in range(len(y)):
        if y[i] == yi[i]:
            ac += 1
    # porcentagem da acuracia
    return (ac*100)/len(y)
###
