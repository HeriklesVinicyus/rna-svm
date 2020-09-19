import numpy as np
import classificador.form as form
from classificador.form import kernel_tangente_hiperbolica


class svm:
    def __init__(self, kernel: str = 'linear', C: float = None, non_linear_parametro: float = 3, a:float = 0.6, const: float = 2) -> None:
        """[summary]

        Args:
            kernel (str, optional): linear, poli, gaus, lrbf, tanh, exp. Defaults to 'linear'.
            C (float, optional): [description]. Defaults to None.
            non_linear_parametro (float, optional): . Defaults to 3.
            a (float, optional): [description]. Defaults to 0.6.
            const (float, optional): [description]. Defaults to 2.
        """

        self.kernel = kernel
        self.C = C
        self.w = None
        self.b = None
        self.n_l_p = non_linear_parametro
        if kernel == 'tanh':
            self.a = a
            self.const = const

    def fit(self, X, y):
        if(self.kernel == 'linear'):
            self.alphas = form.find_alpha(X, y, self.C, self.kernel)
            self.S = (self.alphas > 1e-4).flatten()
            self._X = X
            self.w = self._find_W_linear(X, y, self.alphas)
            self.b = self._find_b_linear(X, y, self.alphas)
        else:
            self.alphas = form.find_alpha(X, y, self.C, self.kernel)
            self.S = (self.alphas > 1e-4).flatten()
            self._X = X
            self.w = self._find_W_non_linear(X, y, self.alphas,self.S)
            self.b = self._find_b_non_linear(X, y, self.alphas)

    def predict(self, X):
        if (self.kernel != 'linear' and self.kernel != 'teste' and self.kernel != 'test'):
            '''
            aux = 0
            for e, i in enumerate(self.S):
                if i:
                    aux += self.alphas[e]*self._K(X,self._X[e])
            return np.sign(aux+self.b)
            '''

        return np.sign(np.dot(X, self.w) + self.b)

    # metodos auxiliares
    def _find_b_linear(self, X, y, alphas):
        _aux_b = 0
        _count_S = 0
        for s, i in enumerate(self.S):
            if i:
                aux = 0
                for m, j in enumerate(self.S):
                    if j:
                        aux += alphas[m]*y[m]*np.dot(X[m], X[s])
                _aux_b += (y[s]-aux)
                _count_S += 1
        return (1/_count_S)*_aux_b

    def _find_W_linear(self, X, y, alphas):
        m, c = X.shape
        w = np.zeros(c)
        for n in range(len(alphas)):
            w += alphas[n] * y[n] * X[n]
        return w
        
    def _find_b_non_linear(self, X, y, alphas):
        _aux_b = 0
        _count_S = 0
        for s, i in enumerate(self.S):
            if i:
                aux = 0
                for m, j in enumerate(self.S):
                    if j:
                        aux += alphas[m]*y[m]*np.dot(X[m], X[s])
                _aux_b += (y[s]-aux)
                _count_S += 1
        return (1/_count_S)*_aux_b

    def _find_W_non_linear(self, X, y, alphas, S):
        m, c = X.shape
        w = np.zeros(c)
        for e,n in enumerate(S):
            if m:
                w += alphas[e] * y[e] * X[e]
        return w

    def _K(self, x1, x2):
        return form._select_kernel(x1, x2, self.kernel, self.n_l_p)
    ###
