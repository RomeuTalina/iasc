import numpy as np

from lib.camada_densa import CamadaDensa, Neuronio
from lib.camada_entrada import CamadaEntrada
from lib.funcao_ativacao import FuncaoAtivacao

# A rede neuronal é composta por camadas de neurónios, cada um destes
# propagando a sua saída para todos os neurónios da camada seguinte.

# A primeira camada é a camada de entrada, cujos neurónios não fazem
# qualquer processamento das entradas, e apenas as propagam para todos
# os neurónios seguintes.

# As camadas seguintes são camadas densas, cujos neurónios processam as
# saídas da camada anterior, conforme os seus pesos, adicionando o pendor, e passando
# o valor obtido pela função de ativação, que determina a saída, antes de ser propagada.

# Após as entradas serem processadas por todas as camadas, obtemos a previsão final da rede neuronal.
class RedeNeuronal:

    def __init__(self, forma: tuple, sigma: FuncaoAtivacao):
        self.forma: tuple[int, int, int] = forma
        self.sigma: FuncaoAtivacao = sigma

        self.camadas: list = []
        self.N = len(self.forma)
        self.d_s1 = self.forma[0]
        camada1: CamadaEntrada = CamadaEntrada(self.d_s1)
        self.camadas.append(camada1)

        for n in range(1, self.N):
            d_e = self.forma[n-1]
            d_s = self.forma[n]
            camada = CamadaDensa(d_e, d_s, self.sigma)
            self.camadas.append(camada)

        self.lista_erros: list[float] = []

    # Passa a entrada por todas as camadas da rede, para obter a saída final.
    def propagar(self, x: list[float]) -> list[float]:
        y: list[float] = x
        for camada in self.camadas:
            y = camada.propagar(y)
        return y

    # Fornece um conjunto de vetores de entrada à rede, que são propagados
    # por todas as camadas, obtendo a saída final.
    def prever(self, X: list[list[float]]):
        Y: list[list[float]] = [self.propagar(x) for x in X]
        return Y

    # Calcula o erro na saída da rede neuronal, para cada entrada que recebeu
    def delta_saida(self, yN: list[float], target: list[float]) -> list[float]:
        return [yN[k] - target[k] for k in range(len(target))]

    # Envia o erro pelas camadas da rede, para que estas possam adaptar os seus pesos e pendores.
    #
    # Ao receber a componente de erro, cada neurónio calcula o ajuste dos pesos e pendores proporcionalmente
    # à sua componente de erro, e alfa (tamanho do passo). Cada "passo" é dado na direção oposta ao gradiente.
    #
    # A componente de erro de cada camada pode ser obtida através da componente de erro da camada seguinte,
    # pelo que precisamos de propagar a componente de erro total da rede (e última camada) por todas as camadas,
    # uma de cada vez. Assim, todas as camadas se adaptam a partir do erro total da rede.

    # alfa -> taxa de aprendizagem
    # beta -> fator de momento
    def retropropagar(self, v_erro_rede: list[float], alfa: float, beta: float):
      v_erro: list[float] = v_erro_rede
      for n in range(self.N-1, 0, -1):
        y_camada_anterior: list[float] = self.camadas[n-1].y
        dimensao_camada_atual: int = self.camadas[n].d_s # nº de saídas da camada atual
        neur: list[Neuronio] = self.camadas[n].neuronios # neurónios da camada atual

        # cálculo da componente de erro da camada anterior
        if n > 1:
            dimensao_camada_anterior: int = self.camadas[n-1].d_s # nº de saídas da camada para que vamos propagar a componente de erro
            v_erro_anterior: list[float] = [
                np.sum([neur[j].w[i] * v_erro[j] * neur[j].dy for j in range(dimensao_camada_atual)])
                for i in range(dimensao_camada_anterior)
            ]
            v_erro = v_erro_anterior # atualizamos a componente de erro, para que a camada anterior use o valor correto

        self.camadas[n].adaptar(v_erro, y_camada_anterior, alfa, beta)


    # Propaga as entradas, e depois adapta os seus valores através de retropropagação.
    def adaptar(self, x: list[float], y: list[float], alfa: float, beta: float) -> float:
        yN = self.propagar(x)
        deltaN: list[float] = self.delta_saida(yN, y)
        self.retropropagar(deltaN, alfa, beta)
        K: int = len(deltaN)
        erro: float = 1/K * np.sum([deltaN[k]**2 for k in range(K)])
        return erro

    # Treina a rede neuronal, efetuando cada passo do processo:
    #   - Propaga as entradas por todas as camadas.
    #   - Calcula o erro na saída, para cada entrada do vetor de entrada.
    #   - Efetua a retropropagação.
    #   - Repete um dado número de vezes (n_epocas).
    #
    #   - Caso a rede tenha erro suficientemente pequeno em qualquer ponto, interrompe o processo.
    def treinar(self, X: list[list[float]], Y: list[list[float]], n_epocas: int, erro_max: float, alfa: float, beta: float):
        for _ in range(n_epocas):
            erro: float = 0
            for x, y in zip(X, Y):
                erro_x: float = self.adaptar(x, y, alfa, beta) # erro relativo à entrada x
                erro = max(erro, erro_x)

            self.lista_erros.append(erro)

            if erro <= erro_max:
                break
