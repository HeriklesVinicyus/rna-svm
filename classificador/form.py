"""
Arrumar um nome melhor para esse classe
E documentar essa classe
"""

from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
import numpy as np
import random
import math


def find_alpha(X, Y, C, K) -> np.ndarray:
    # Setting solver parameters (change default to decrease tolerance)
    cvxopt_solvers.options['show_progress'] = False
    cvxopt_solvers.options['abstol'] = 1e-10
    cvxopt_solvers.options['reltol'] = 1e-10
    cvxopt_solvers.options['feastol'] = 1e-10

    if C == None:
        m, n = X.shape
        H = _H(X, Y, K)

        # Converting into cvxopt format
        P = cvxopt_matrix(H)
        q = cvxopt_matrix(-np.ones((m, 1)))
        G = cvxopt_matrix(-np.eye(m))
        h = cvxopt_matrix(np.zeros(m))
        A = cvxopt_matrix(Y, (1, m), 'd')
        b = cvxopt_matrix(np.zeros(1))

        # Run solver
        sol = cvxopt_solvers.qp(P, q, G, h, A, b)
        alphas = np.array(sol['x'])
    else:
        C = 10
        m, n = X.shape
        H = _H(X, Y, K)

        # Converting into cvxopt format - as previously
        P = cvxopt_matrix(H)
        q = cvxopt_matrix(-np.ones((m, 1)))
        G = cvxopt_matrix(np.vstack((np.eye(m)*-1, np.eye(m))))
        h = cvxopt_matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
        A = cvxopt_matrix(Y, (1, m), 'd')
        b = cvxopt_matrix(np.zeros(1))

        # Run solver
        sol = cvxopt_solvers.qp(P, q, G, h, A, b)
        alphas = np.array(sol['x'])
    return alphas


# não sei como chamar esse bloco
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


def accuracy(y: np.ndarray, yi: np.ndarray) -> float:

    # Validar que os vetores são arrays' numpy
    aux_y = y
    aux_yi = yi
    if type(y) != np.ndarray:
        aux_y = np.array(y)
    if type(yi) != np.ndarray:
        aux_yi = np.array(yi)

    ac = 0
    for i in range(len(y)):
        if aux_y[i] == aux_yi[i]:
            ac += 1
    # porcentagem da acuracia
    return (ac*100)/len(y)
###

###


def _select_kernel(X, y, k='linear', n_l_p=3):
    if k == 'gaus':
        return kernel_gaussian(X, y, sigma=n_l_p)
    elif k == 'poli':
        return kernel_polinomial(X, y, p=n_l_p)
    return kernel_linear_t(X, y)


def kernel_linear_t(x1, x2):
    return np.dot(x1, x2)


def kernel_polinomial(xi, xj, p=3):
    return (1 + np.dot(xi, xj)) ** p


def kernel_gaussian(xi, xj, sigma=5.0):
    return np.exp(-np.linalg.norm(xi-xj)**2 / (2 * (sigma ** 2)))


def _H(X, y, k):
    n, m = X.shape
    aux_H = [[0 for x in range(n)] for y in range(n)]
    for i in range(len(aux_H)):
        for j in range(len(aux_H[i])):
            t = y[i]*y[j]*_select_kernel(X[i], X[j], k)
            aux_H[i][j] = t
    aux_H = np.array(aux_H)
    return aux_H
###
