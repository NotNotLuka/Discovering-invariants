import numpy as np


def freefall(gs, N=1000):
    data = np.zeros((len(gs), N, 2))
    T = 2
    for i in range(len(gs)):
        g = gs[i]
        t = T + np.random.randn(N) * T
        h = 0.5 * g * np.power(t, 2)
        data[i, :, :] = np.array([h, t]).T

    return data
