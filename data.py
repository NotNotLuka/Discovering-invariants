import numpy as np


def freefall(gs, N=1000):
    data = np.zeros((len(gs), N, 2))
    T = 2
    for i in range(len(gs)):
        g = gs[i]
        t = np.random.rand(N) * T + 1
        h = 0.5 * g * np.power(t, 2)
        data[i, :, :] = np.array([h, t]).T

    return data


def freefall_g(N=1000):
    A = 1e-1
    H = 4
    H_noise = A * np.random.rand(N)
    T = 2
    T_noise = A * np.random.rand(N)

    h = np.random.rand(N) * H + A
    t = np.random.rand(N) * T + A + 1

    g = 2 * h / np.power(t, 2)

    return np.array([h + H_noise, t + T_noise]).T, g
