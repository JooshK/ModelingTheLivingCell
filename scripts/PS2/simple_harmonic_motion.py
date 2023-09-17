import matplotlib.pyplot as plt
import numpy as np


def position(t, x_0, w):
    return x_0 * np.cos(w*t) + x_0


def velocity(t, x_0, w):
    return -1 * x_0 * w * np.sin(w*t)


def force(t, x_0, w, m):
    return -1 * m * x_0 * w**2 * np.cos(w*t)


def plot_oscillator(range, w, x_0, m, save_file=False, save_location=None):
    time = np.linspace(0, range, 500)
    positions = []
    forces = []
    velocities = []
    for t in time:
        positions.append(position(t, x_0, w))
        velocities.append(velocity(t, x_0, w))
        forces.append(force(t, x_0, w, m))

    fig, axes = plt.subplots()
    axes.set_xlabel("time (s)")
    axes.plot(time, positions, label='x(t)')
    axes.plot(time, velocities, label='v(t)')
    axes.plot(time, forces, label='f(t)')
    axes.set_title("Position, Velocity, and Force \n Over Time for the Simple Bond Oscillation")
    axes.legend()

    if save_file and save_location is not None:
        plt.savefig(save_location)

    return axes
