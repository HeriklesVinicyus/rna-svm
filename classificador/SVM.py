"""
    Colocar uma descrição
    """

from os import times
import numpy as np
import cvxopt
import cvxopt.solvers

import classificador.form as form
from classificador.form import linear_kernel


class svm:
    def __init__(self, kernel: str = 'linear', C: float = None, non_linear_parametro: float = 3) -> None:
        """[summary]

        Args:
            kernel (str, optional): linear, poli, gaus. Defaults to 'linear'.
            C (float, optional): [description]. Defaults to 1.0.
            non_linear_parametro (float, optional): [description]. Defaults to 3.
        """

        self.tipo = kernel
        self.C = C
        self.w = None
        self.b = None

        # polinomial
        if (kernel == 'poli'):
            self.kernel = form.polynomial_kernel  # função
            self.grau = int(non_linear_parametro)
        # gaussiano
        elif (kernel == 'gaus'):
            self.kernel = form.gaussian_kernel  # função
            self.sigma = non_linear_parametro
        # linear
        else:
            self.kernel = form.linear_kernel  # função

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Compute the Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.kernel(X[i], X[j])
        # construct P, q, A, b, G, h matrices for CVXOPT
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples), 'd')
        b = cvxopt.matrix(0.0)
        if self.C is None:      # hard-margin SVM
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:              # soft-margin SVM
            G = cvxopt.matrix(
                np.vstack((np.diag(np.ones(n_samples) * -1), np.identity(n_samples))))
            h = cvxopt.matrix(
                np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C)))
        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        # Lagrange multipliers
        a = np.ravel(solution['x'])
        self.alphas = a
        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5  # some small threshold
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        print('############',a,'\n\n',len(a), len(self.a))
        self.sv = X[sv]
        self.sv_y = y[sv]

        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n], sv])
        self.b /= len(self.a)

        # Weight vector
        
        if self.tipo == 'linear':
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None

        
    def project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b   
        elif(self.tipo == 'gaus'):
            s = 0
            for i in range(len(self.a)):
                s += self.a[i] * self.sv_y[i] * self.kernel(self.sv[i],X,self.sigma)
            return s + self.b

        elif(self.tipo == 'poli'):
            s = 0
            for i in range(len(self.a)):
                s += self.a[i] * self.sv_y[i] * self.kernel(self.sv[i],X,self.grau)
            return s + self.b
            


    def predict(self, X):
        return np.sign(self.project(X))
