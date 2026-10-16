import sklearn.datasets as dt
from matplotlib import pyplot as plt

from lib import RedeNeuronal
from lib.tanh import TangenteHiperbolica

tanh = TangenteHiperbolica()

rede_neuronal = RedeNeuronal((9, 3, 1), tanh)

input = [
    [
        1.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        1.0, 0.0, 0.0
    ],
    [
        0.0, 1.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 1.0, 0.0
    ],
    [
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    ],
    [
        1.0, 1.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 1.0, 1.0
    ],
    [
        0.0, 0.0, 0.0,
        1.0, 1.0, 1.0,
        0.0, 0.0, 0.0
    ],
    [
        0.0, 0.0, 1.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0
    ]
]

target = [
    [1.0],
    [1.0],
    [0.0],
    [0.0],
    [1.0],
    [0.0]
]

output1 = rede_neuronal.prever(input)

print(
    f"""
    Primeiro Output: {output1};
    """
)

rede_neuronal.treinar(input, target, n_epocas=500, erro_max=0.002, alfa=0.1, beta=0.1)

output2 = rede_neuronal.prever(input)

plt.plot(rede_neuronal.lista_erros)
plt.xlabel("Época")
plt.ylabel("Erro máximo")
plt.title("Erro por época")
plt.yscale("log")
plt.savefig("./tests/erro.png", dpi=150)

print(
    f"""
    Segundo Output: {output2}
    """
)
