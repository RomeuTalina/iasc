import random

import numpy as np

from lib.funcao_ativacao import FuncaoAtivacao

# O neurónio é a componente principal de uma rede neuronal.

# Cada neurónio recebe um vetor de entradas (d, 1), e possui um vetor de pesos (d, 1).

# Estes pesos são adaptados durante o processo de aprendizagem, e são o que permite
# à rede neuronal "racionalizar" sobre as entradas, sendo que cada entrada é multiplicada
# pelo seu peso correspondente, no neurónio.

# As entradas correspondem a uma representação numérica do domínio do problema, permitindo
# que os neurónios as processem.

# h iguala o somatório dos produtos das entradas pelos pesos, mais um pendor ou "bias" (b).

# h = w1 * x1 + w2 * x2 + ... wn * xn

# A função de ativação transforma h para obter a saída (y), propagada para os neurónios da camada seguinte.

# y = sigma(h)

# As funções de ativação estudadas foram as de Degrau, Sigmóide e Tangente Hiperbólica.

# Os neurónios estão organizados em camadas, cada um recebendo as saídas de todos os neurónios
# da camada anterior.
class Neuronio:

    def __init__(self, d: int, sigma: FuncaoAtivacao):
        self.d: int = d # A dimensão do vetor de entrada
        self.sigma: FuncaoAtivacao = sigma # A função de ativação

        self.w: list[float] = [random.uniform(-1, 1) for _ in range(d)] # Pesos aleatórios entre −1 e 1
        self.b: float = random.uniform(-1, 1) # Pendor aleatório para o neurónio
        self.h: float = 0 # O somatório dos produtos das entradas pelos pesos
        self.y: float = 0 # A saída da função de ativação do neurónio
        self.dy: float = 0 # Derivada da saída

        self.delta_w: list[float] = [0 for _ in range(d)] # Vetor de variação dos pesos
        self.delta_b: float = 0 # Variação do pendor

    # Calcula o valor de h (produto interno das entradas pelos pesos, somando depois o pendor),
    # passando esse valor à função de ativação (sigma) para determinar a saída.
    # Este valor de saída será depois propagado para todos os neurónios da camada seguinte.
    def propagar(self, x) -> float:
        self.h = np.dot(self.w, x) + self.b # Cálculo de h
        self.y  = self.sigma.ativacao(self.h) # Cálculo da saída
        self.dy = self.sigma.derivada(self.h) # Cálculo da derivada
        return self.y

    # Calcula e adapta os pesos e pendores do neurónio conforme a sua componente de erro.
    #
    # O neurónio calcula o gradiente, e determina o ajuste a fazer aos seus pesos
    # de acordo com a seguinte fórmula:

        # -alfa * y' * delta . y

        # alfa : dimensão do passo
        # y' : derivada da saída
        # y : saída do neurónio
        # delta : componente de erro

    # O passo é dado na direção oposta ao máximo do gradiente, e a dimensão de passo não deve
    # ser nem muito grande nem muito pequena. Caso contrário, os pesos podem nunca encontrar o
    # ponto de menor erro, ou fixar-se num mínimo local.

    # Por sua vez, o ajuste ao pendor é feito de acordo com a fórmula

        # -alfa * y' * delta

    # Este processo é repetido muitas vezes, até que o erro seja minimizado, ou o número máximo de
    # épocas (repetições) seja atingido.
    def adaptar(self, comp_err: float, y_anterior: list[float], alfa: float, beta: float):
        momento_w: list[float] = [beta * d_w for d_w in self.delta_w]

        self.delta_w = np.dot((-alfa * self.dy * comp_err), y_anterior) + momento_w

        self.w = [w + dw for w, dw in zip(self.w, self.delta_w)]

        momento__b = beta * self.delta_b

        self.delta_b = (-alfa * self.dy * comp_err) + momento__b

        self.b = self.b + self.delta_b
