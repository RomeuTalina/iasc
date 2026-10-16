import numpy as np
import sklearn.datasets as dt
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from lib import RedeNeuronal
from lib.tanh import TangenteHiperbolica

tanh = TangenteHiperbolica()

rede_neuronal = RedeNeuronal((30, 15, 1), tanh)

# Dados: 569 amostras, 30 características, classe 0 (maligno) ou 1 (benigno)
dados = dt.load_breast_cancer()

# Separação em treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(
    dados.data, dados.target, test_size=0.25, random_state=0, stratify=dados.target
)

# Normalização (ajustada apenas com o conjunto de treino)
scaler = StandardScaler().fit(X_treino)
input = scaler.transform(X_treino).tolist()
input_teste = scaler.transform(X_teste).tolist()

target = [[float(y)] for y in y_treino]


def taxa_acertos(previsoes, y_real):
    previsoes = np.array(previsoes).ravel()
    return np.mean((previsoes > 0.5) == (y_real == 1))


output1 = rede_neuronal.prever(input_teste)

print(
    f"""
    Primeiro Output (acertos no teste): {taxa_acertos(output1, y_teste):.3f};
    """
)

rede_neuronal.treinar(input, target, n_epocas=500, erro_max=0.002, alfa=0.001, beta=0.1)

output2 = rede_neuronal.prever(input_teste)

plt.plot(rede_neuronal.lista_erros)
plt.xlabel("Época")
plt.ylabel("Erro máximo")
plt.title("Erro por época")
plt.yscale("log")
plt.savefig("./tests/erro_cancro.png")

print(
    f"""
    Segundo Output (acertos no teste): {taxa_acertos(output2, y_teste):.3f}
    """
)
