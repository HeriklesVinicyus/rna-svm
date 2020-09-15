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

    def fit(self, X, Y) -> None:
        self.X = X
        self.Y = Y

        n_samples, n_features = self.X.shape

        # Gram matrix
        # P = X^T X
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                if(self.tipo=='poli'):
                    K[i, j] = self.kernel(self.X[i], self.X[j], p=self.grau)
                if(self.tipo=='gaus'):
                    K[i, j] = self.kernel(self.X[i], self.X[j], sigma=self.sigma)
                elif(self.tipo=='linear'):
                    K[i, j] = self.kernel(self.X[i], self.X[j])
                
        P = cvxopt.matrix(np.outer(Y, Y) * K)
        # q = -1 (1xN)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        # A = y^T 
        A = cvxopt.matrix(Y, (1, n_samples), 'd')

        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            print('###############')
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))
        
        #print(len(P), len(q), len(G), len(h), len(A), len(b))
        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        a = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = Y[sv]
        #print("%d support vectors out of %d points" % (len(self.a), n_samples))

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
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel(X[i], sv)
                y_predict[i] = s
            return y_predict + self.b

    def predict(self, X):
        return np.sign(self.project(X))