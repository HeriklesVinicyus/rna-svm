import numpy as np
import classificador.form as form
from classificador.form import kernel_tangente_hiperbolica


class svm:
    def __init__(self, kernel: str = 'linear', C: float = None, non_linear_parametro: float = 3, a: float = 0.6) -> None:
        """
        Não documentado

        Args:
            kernel (str, optional): linear, poli (para polinomial), gaus(Gaussiano), lrbf(laplace rbf), tanh (Tangente hiperbolica), exp(Exponencial), mqi (Multiquadratica inversa), qr (Quadratico racional), mq (Multiquadratica), log (Logarítmica). Defaults to 'linear'.
            C (float, optional): [description]. Defaults to None.
            non_linear_parametro (float, optional): . Defaults to 3.
            a (float, optional): Alpha para a Tangente hiperbolica. Defaults to 0.6.
            const (float, optional): [description]. Defaults to 2.
        """

        self.kernel = kernel
        self.C = C
        self.w = None
        self.b = None
        self.n_l_p = non_linear_parametro
        if kernel == 'tanh':
            self.a = a

    def fit(self, X: np.array, y: np.array) -> None:
        """[summary]

        Args:
            X (np.array): [description]
            y (np.array): [description]
        """
        self.alphas = form.find_alpha(X, y, self.C, self.kernel)
        self.S = (self.alphas > 1e-4).flatten()
        self._X = X
        self.w = self._find_W(X, y, self.alphas)
        self.b = self._find_b(X, y, self.alphas)
        print('SVM com kernel {} ajustado'.format(self.kernel))

    def predict(self, X: np.array) -> int:
        """[summary]

        Args:
            X (np.array): [description]

        Returns:
            int: [description]
        """
        return int(np.sign(np.dot(X, self.w) + self.b))

    # metodos auxiliares
    def _find_b(self, X, y, alphas):
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

    def _find_W(self, X, y, alphas):
        w = np.zeros(X.shape[1])
        for e, n in enumerate(self.S):
            if n:
                w += alphas[e] * y[e] * X[e]
        return w

    def _K(self, x1, x2):
        return form._select_kernel(x1, x2, self.kernel, self.n_l_p, a=self.a)
    ###
