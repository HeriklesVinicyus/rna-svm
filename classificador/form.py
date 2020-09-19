"""
Arrumar um nome melhor para esse classe
E documentar essa classe
"""

from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
import numpy as np
import random
import math


def find_alpha(X, Y, C, K, n_l_p=3) -> np.ndarray:
    # Setting solver parameters (change default to decrease tolerance)
    cvxopt_solvers.options['show_progress'] = False
    cvxopt_solvers.options['abstol'] = 1e-10
    cvxopt_solvers.options['reltol'] = 1e-10
    cvxopt_solvers.options['feastol'] = 1e-10

    m, n = X.shape
    H = _H(X, Y, K, n_l_p)

    P = cvxopt_matrix(H)
    q = cvxopt_matrix(-np.ones((m, 1)))
    A = cvxopt_matrix(Y, (1, m), 'd')
    b = cvxopt_matrix(np.zeros(1))

    if C == None:
        # Converting into cvxopt format
        G = cvxopt_matrix(-np.eye(m))
        h = cvxopt_matrix(np.zeros(m))
    else:
        # Converting into cvxopt format - as previously
        G = cvxopt_matrix(np.vstack((np.eye(m)*-1, np.eye(m))))
        h = cvxopt_matrix(np.hstack((np.zeros(m), np.ones(m) * C)))

    # Run solver
    sol = cvxopt_solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])
    return alphas


#
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
def _H(X, y, k, n_l_p):
    n, m = X.shape
    aux_H = [[0 for x in range(n)] for y in range(n)]
    for i in range(len(aux_H)):
        for j in range(len(aux_H[i])):
            t = y[i]*y[j]*_select_kernel(X[i], X[j], k, n_l_p)
            aux_H[i][j] = t
    aux_H = np.array(aux_H)
    return aux_H


def _select_kernel(x1, x2, k='linear', n_l_p=3, a=0.5,):
    if k == 'gaus':
        return kernel_gaussian(x1, x2, sigma=n_l_p)
    elif k == 'poli':
        return kernel_polinomial(x1, x2, p=n_l_p)
    elif k == 'lrbf':
        return kernel_laplace_rbf(x1, x2, sigma=n_l_p)
    elif k == 'exp':
        return kernel_exponencial(x1, x2, sigma=n_l_p)
    elif k == 'tanh':
        return kernel_tangente_hiperbolica(x1, x2, a, n_l_p)
    elif k == 'qr':
        return kernel_quadratico_racional(x1, x2, const=n_l_p)
    elif k == 'mq':
        return kernel_multiquadrico(x1, x2, const=n_l_p)
    elif k == 'mqi':
        return kernel_multiquadrico_inverso(x1, x2, const=n_l_p)
    elif k == 'log':
        return kernel_logaritmico(x1, x2, d=n_l_p)

    return kernel_linear_t(x1, x2)


def kernel_linear_t(x1: np.array, x2: np.array) -> float:
    """
    O kernel Linear é a função de kernel mais simples. É dado pelo produto interno <x1, x2>.

    Args:
        x1 (np.array): Vetor
        x2 (np.array): Vetor com mesma dimensão do x1

    Returns:
        float: x1 · x2
    """
    return float(np.dot(x1, x2))


def kernel_gaussian(x1: np.array, x2: np.array, sigma: float = 5.0) -> float:
    """O kernel gaussiano é um exemplo de kernel de função de base radial.
    sigma desempenha um papel importante no desempenho do kernel e deve ser
    cuidadosamente ajustado para o problema em questão. Se superestimado,
    o exponencial se comportará quase linearmente e a projeção em dimensões
    superiores começará a perder seu poder não linear. Por outro lado,
    se subestimada, a função carecerá de regularização e o limite de
    decisão será altamente sensível ao ruído nos dados de treinamento.

    Args:
        x1 (np.array): Vetor
        x2 (np.array): Vetor com mesma dimensão do x1
        sigma (float, optional): σ é o parâmetro de espalhamento que desempenha o mesmo papel que o desvio padrão na função de densidade normal. Defaults to 5.0.

    Returns:
        float: exp{–||x1−x2||^2 / 2σ^2}
    """
    return float(np.exp(-np.linalg.norm(x1-x2)**2 / (2 * (sigma ** 2))))


def kernel_laplace_rbf(x1: np.array, x2: np.array, sigma=5.0) -> float:
    """O Kernel Laplace é completamente equivalente ao kernel exponencial,
    exceto por ser menos sensível a mudanças no parâmetro sigma.
    Sendo equivalente, é também um kernel de função de base radial.

    Args:
        x1 (np.array): Vetor
        x2 (np.array): Vetor com mesma dimensão do x1
        sigma (float, optional): σ é o parâmetro de espalhamento que desempenha o mesmo papel que o desvio padrão na função de densidade normal. Defaults to 5.0.

    Returns:
        float: exp{–||x1−x2|| / σ}
    """
    return float(np.exp(-np.linalg.norm(x1-x2) / sigma))


def kernel_exponencial(x1: np.array, x2: np.array, sigma=5.0) -> float:
    """
    O kernel exponencial está intimamente relacionado ao kernel gaussiano,
    com apenas o quadrado da norma deixado de fora. É também um kernel de função de base radial.

    Args:
        x1 (np.array): Vetor
        x2 (np.array): Vetor com mesma dimensão do x1
        sigma (float, optional): σ é o parâmetro de espalhamento que desempenha o mesmo papel que o desvio padrão na função de densidade normal. Defaults to 5.0.

    Returns:
        float: exp{–||x1−x2|| / 2σ^2}
    """
    return np.exp(-np.linalg.norm(x1-x2) / (2 * (sigma ** 2)))


def kernel_tangente_hiperbolica(x1: np.array, x2: np.array, a: float = 0.5, const: float = 2) -> float:
    """Hyperbolic Tangent (Sigmoid) Kernel: k(x,x′)=;
    O kernel Tangente Hiperbólico também é conhecido como Sigmoid Kernel e como Multilayer Perceptron kernel.
    O núcleo sigmóide vem do campo de redes neurais, onde a função sigmóide bipolar é frequentemente usada
    como uma função de ativação para neurônios artificiais.
    Args:
        x1 (np.array): Vetor
        x2 (np.array): Vetor com mesma dimensão do x1
        a (float): inclinação alfa; Um valor comum para alfa é 1 / N, onde N é a dimensão dos dados.
        const (float): constante de interceptação
    Returns:
        float: tanh(a * x⋅x′ + c)

    note: É interessante notar que um modelo SVM usando uma função de kernel sigmóide é equivalente a uma rede neural perceptron de duas camadas.

    """
    return float(np.tanh(a*np.dot(x1, x2)+const))


def kernel_multiquadrico_inverso(x1: np.array, x2: np.array, const: float = 2) -> float:
    """
    Tal como acontece com o kernel gaussiano, resulta em uma matriz de kernel com classificação
    completa e, portanto, forma um espaço de recursos de dimensão infinita.

    Args:
        x1 (np.array): Vetor
        x2 (np.array): Vetor com mesma dimensão do x1
        const (float): constante de interceptação

    Returns:
        float: 1-((||x1−x2||^2) + const)^(1/2)
    """
    return 1/kernel_multiquadrico(x1, x2, const)



#todos abaixo estão apresentando o erro ValueError: Rank(A) < p or Rank([P; A; G]) < n. Investicar posteriomento


def kernel_logaritmico(x1: np.array, x2: np.array, d: int = 3) -> float:
    """
    O kernel Logarítmico parece ser particularmente interessante para imagens,
     mas só é definido positivamente condicionalmente.

    Args:
        x1 (np.array): Vetor
        x2 (np.array): Vetor com mesma dimensão do x1
        d (float): Dimasão. Defaults to 3.

    Returns:
        float: -log((||x1−x2||^2) + 1)
    """
    _norm = np.linalg.norm(x1-x2)
    return float(-np.log(_norm**d)+1)


def kernel_multiquadrico(x1: np.array, x2: np.array, const: float = 2) -> float:# não está pengando quando usado só mais quando usado na inversa funciona bem
    """
    O kernel Multiquadrico pode ser usado nas mesmas situações que o kernel QR. É também um exemplo de kernel definido não positivo.

    Args:
        x1 (np.array): Vetor
        x2 (np.array): Vetor com mesma dimensão do x1
        const (float): constante de interceptação

    Returns:
        float: ((||x1−x2||^2) + const)^(1/2)
    """
    return float(np.sqrt((np.linalg.norm(x1-x2)**2)+const))


def kernel_quadratico_racional(x1: np.array, x2: np.array, const: float = 2) -> float:
    """
    O kernel QR é menos computacionalmente intensivo do que o kernel Gaussiano
    e pode ser usado como uma alternativa quando o uso do Gaussiano se tornar muito caro.

    Args:
        x1 (np.array): Vetor
        x2 (np.array): Vetor com mesma dimensão do x1
        const (float): constante de interceptação

    Returns:
        float: 1 – (||x1−x2||^2) / (||x1−x2||^2) + const 
    """
    return float(1-(np.linalg.norm(x1-x2)**2)/(np.linalg.norm(x1-x2)**2)+const)


def kernel_polinomial(x1: np.array, x2: np.array, p: int = 3):
    # não está funcionando
    aux_p = int(p)
    return float(float(np.dot(x1, x2)+1) ** aux_p)
###
