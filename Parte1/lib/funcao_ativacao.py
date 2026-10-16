from abc import ABC, abstractmethod

# Interface que define uma função de ativação para um neurónio.
#
# Permite calcular a saída e a derivada da mesma, a partir de h.

# Vários tipos de funçao podem ser usados como função de ativação.
# Contudo, as suas características são importantes.

# Três tipos de função são destacados:
#
# - Função Degrau: Apenas retorna 0 ou 1, não havendo
# possibilidade de aprendizagem.
#
# - Função Sigmóide: Retorna valores entre 0 e 1, permitindo que exista
# incerteza, e que a rede aprenda.
#
# - Função de Tangente Hiperbólica: Semelhante à sigmóide, mas retorna valores entre -1 e 1,
# criando uma maior polarização da incerteza, e incentivando a aprendizagem.
class FuncaoAtivacao(ABC):

    @abstractmethod
    def ativacao(self, h: float) -> float:
        pass

    @abstractmethod
    def derivada(self, h: float) -> float:
        pass
