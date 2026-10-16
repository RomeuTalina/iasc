from abc import ABC, abstractmethod


# Classe abstrata para as camadas da rede neuronal.
class Camada(ABC):

    @abstractmethod
    def propagar(self, x:list[float]) -> list[float]:
        pass
