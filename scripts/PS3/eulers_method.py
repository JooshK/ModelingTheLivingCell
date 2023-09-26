from typing import Callable
import numpy as np


def euler_method(func: Callable, step_size, t_0, t_f, y_0, n):
    """
    Runs euler method to approximate the solution for a differential equation
    :param t_f: Final t value
    :param func: the function dy/dt
    :param step_size: Step size to take
    :param t_0: Initial value of t
    :param y_0: Initial value of y
    :param n: Number of step sizes
    :return: List of time steps and corresponding y values
    """

