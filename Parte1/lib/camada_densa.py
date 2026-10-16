from lib.camada import Camada
from lib.funcao_ativacao import FuncaoAtivacao
from lib.neuronio import Neuronio

# Na camada densa, ao contrário da camada de entrada, os neurónios efetuam
# processamento das suas entradas, multiplicando-as pelos pesos correspondentes e
# propagando a sua saída para a próxima camada.

# Cada camada recebe as saídas dos neurónios da camada anterior, permitindo à rede
# efetuar um raciocínio mais complexo sobre as entradas:

# Tendo apenas uma camada na rede neuronal, esta apenas consegue separar os dados conforme
# um conjunto de características. Contudo, ao propagar a saída de cada neurónio para os neurónios seguintes,
# a rede consegue abstrair estas características, permitindo-lhe efetuar raciocícios muito mais complexos.
class CamadaDensa(Camada):

    def __init__(self, d_e: int, d_s: int, sigma: FuncaoAtivacao):
        self.d_e: int = d_e
        self.d_s: int = d_s
        self.neuronios: list[Neuronio] = [Neuronio(d_e, sigma) for _ in range(d_s)]

    # Retorna o valor atual das saídas de todos os neurónios da camada.
    @property
    def y(self) -> list[float]:
        return [neuronio.y for neuronio in self.neuronios]

    # Retorna as saídas de todos os neurónios, após o processamento.
    def propagar(self, x: list[float]) -> list[float]:
        y: list[float] = [neuronio.propagar(x) for neuronio in self.neuronios]
        return y

    # Adapta todos os seus neurónios.
    def adaptar(self, comps_erro: list[float], x: list[float], alfa: float, beta: float):
        for j in range(self.d_s):
            self.neuronios[j].adaptar(comps_erro[j], x, alfa, beta)
