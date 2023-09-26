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
        dy_dt = func(y[i])
        y_next = y[i] + (step_size * dy_dt)
        t_next = t[i] + step_size
        print('%.4f\t%.4f\t%0.4f\t%.4f' % (t[i], y[i], dy_dt, y_next))
        y.append(y_next)
        t.append(t_next)
    return t, y