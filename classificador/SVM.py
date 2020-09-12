"""
    Colocar uma descrição
    """

import numpy as np
import classificador.form as form

class svm:
    def __init__(self, X, Y, tipo: str = 'linear', kernel: str = 'linear', C: float = 1.0, grau: int = 1) -> None:
        """[summary]

        Args:
            X ([type]): Vetor/matrix de características
            Y ([type]): [description]
            tipo (str, optional): 'linear' ou 'non-linear'. Defaults to 'linear'.
            kernel (str, optional): [description]. Defaults to 'linear'.
            C (float, optional): [description]. Defaults to 1.0.
            grau (int, optional): [description]. Defaults to 1.
        """
        self.X = X
        self.Y = Y
        self.tipo = tipo
        self.C = C
        self.w = None
        self.b = None

        if(self.tipo == 'non-linear'):
            self.kernel = kernel  # 'linear', 'poli', 'rbf'
            # polinomial
            if(self.kernel == 'poli'):
                self.grau = grau

            # não completo para non-lineares

    def fit(self) -> None:
        self.w = np.zeros(len(self.X))
        if(self.tipo == 'linear'):
            self.w, self.b = form.kernel_linear(self.X,self.Y)

    def predict(self, X) -> int:
        """[Função para classificar exemplos que não estão na base de treinamento]

        Args:
            X ([Numeric]): [Vetor de características]

        Returns:
            [int]: [qual a classe que X pertence]
        """

        return form.ativacao(X, self.w, self.b)
