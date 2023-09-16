import matplotlib.pyplot as plt
import gaussian_dist
import numpy as np


def diffusional_spreading(out_file: str):
    """
    Plots 10 gaussian distributions at varying std. from 0.1 to 2 with a mean of zero
    :param out_file: The file to save the graph to
    """
    x = np.linspace(-6, 6, 100)
    mean = 0
    std = np.linspace(0.1, 2, 10)  # 10 equally spaced values from 0.1 to 2

    for s in std:  # loop over the standard devs and plot without saving
        gaussian_dist.plot_gaussian(mean, s, save=False)

    plt.title("Gaussian Distributions With Varying Std.Dev.")
    plt.savefig(out_file)


if __name__ == "__main__":
    diffusional_spreading("../../data/diffusional_spreading.png")