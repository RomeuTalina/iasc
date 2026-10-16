import numpy as np

from lib.funcao_ativacao import FuncaoAtivacao


# Implementação da função de tangente hiperbólica, que serve de função de ativação.
class TangenteHiperbolica(FuncaoAtivacao):

    def ativacao(self, h: float) -> float:
        return np.tanh(h)

    def derivada(self, h: float) -> float:
        return 1 - np.tanh(h)**2
