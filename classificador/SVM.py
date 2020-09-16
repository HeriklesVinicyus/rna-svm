
from os import times
import numpy as np
import cvxopt
import cvxopt.solvers

import classificador.form as form
#from classificador.form import linear_kernel


class svm:
    def __init__(self, kernel: str = 'linear', C: float = None, non_linear_parametro: float = 3) -> None:
        """[summary]

        Args:
            kernel (str, optional): linear, poli, gaus. Defaults to 'linear'.
            C (float, optional): [description]. Defaults to 1.0.
            non_linear_parametro (float, optional): [description]. Defaults to 3.
        """

        self._tipo = kernel
        self.kernel = kernel
        self.C = C
        self.w = None
        self.b = None

    def fit(self, X, y):
        # linear
        alphas = form.find_alpha(X, y, self.C)
        S = (alphas > 1e-4).flatten()

        self.w = self._find_W(X,y,alphas)

        self.b = self._find_b(X, y, alphas, S)

    def predict(self, X):
        if self._tipo == 'linear':
            print('predict ', (X.shape), (self.w.shape), self.b)
            return np.sign(np.dot(X, self.w) + self.b)
        elif(self._tipo == 'gaus'):
            return []

    # metodos auxiliares
    def _select_kernel(self):
        if self.kernel == 'gaus':
            pass
        elif self.kernel == 'poli':
            pass
        else:  # caso não tenha, sera encarado como linear
            pass

    def _find_b(self, X, y, alphas, S):
        _aux_b = 0
        _count_S = 0
        for s, i in enumerate(S):
            if i:
                aux = 0
                for m, j in enumerate(S):
                    if j:
                        aux += alphas[m]*y[m]*np.dot(X[m], X[s])
                _aux_b += (y[s]-aux)
                _count_S += 1

        return (1/_count_S)*_aux_b

    def _find_W(self, X, y, alphas):
        m,c = X.shape
        w = np.zeros(c)
        for n in range(len(alphas)):
            w += alphas[n] * y[n] * X[n]
        return w
    ###
