import numpy as np


def euler_method(func, step_size, min_range=0, max_range=0.1, y_0=100):
    x = np.arange(min_range, max_range + step_size, step_size)
    y = np.zeros(len(x))

    y[0] = y_0
    for index in range(1, len(x)):
        y[index] = y[index - 1] + step_size * func(y[index - 1])
    return x, y
