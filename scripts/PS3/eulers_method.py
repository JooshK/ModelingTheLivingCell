from typing import Callable


def euler_method(func: Callable, step_size, t_0, y_0, n):
    """
    Runs euler method to approximate the solution for a differential equation
    :param func: the function dy/dt (depends only on y)
    :param step_size: Step size to take
    :param t_0: Initial value of t
    :param y_0: Initial value of y
    :param n: Number of step sizes
    :return: List of time steps and corresponding y values
    """

    y = [y_0]
    t = [t_0]
    for i in range(n):
        y_next = step_size * func(y[i])+y[i]
        y.append(y_next)
        t.append(t[i] + step_size)
    return t, y