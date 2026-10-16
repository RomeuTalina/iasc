from lib.camada import Camada


# A camada de entrada é a primeira camada da rede neuronal.
# Esta apenas garante que todas as entradas são propagadas para todos os neurónios
# da camada densa.
class CamadaEntrada(Camada):

    def __init__(self, d_s: int):
        self.y: list[float] = [0 for _ in range(d_s)]

    # A camada de entrada retorna imediatamente as entradas, sem qualquer alteração,
    # para que estas sejam propagadas para todos os neurónios da camada seguinte.
    def propagar(self, x: list[float]) -> list[float]:
        self.y = x
        return self.y
