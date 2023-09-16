import math

import matplotlib.pyplot as plt
import numpy as np


def calculate_gaussian(mean: float, standard_dev: float, x) -> float:
    """
    Calculates a standard normal distribution
    :param mean: the mean of distribution
    :param standard_dev: the standard deviation
    :param x: the value to evaluate at
    :return: the probability density for the normal distribution at that point
    """
    p = 1 / (standard_dev * math.sqrt(2 * math.pi))
    return p * math.exp(-1 * (x - mean) ** 2 / (2 * standard_dev))


def plot_gaussian(range, mean: float, standard_dev: float, out_file=None, save=False):
    """
    Plots the gaussian distribution with a given mean and std on the interval -6 to 6 with 100 data points
    :param range: The range of x values to plot for
    :param save: If True, save the image to a file
    :param out_file: the name of the file to save
    :param mean: The mean
    :param standard_dev: The standard deviation
    """
    x_values = np.linspace(-range, range, 100)  # set 100 equally spaced values
    y_values = []

    for x in x_values:  # call the calculateGaussian function over the x values
        p = calculate_gaussian(mean, standard_dev, x)
        y_values.append(p)

    # normalize the y values and check
    norm_y_values = [y / sum(y_values) for y in y_values]
    print("Check for normalization", sum(norm_y_values))

    # plot the gaussian distribution and save to a .png file
    plt.rcParams.update({'font.size': 12})
    plt.plot(x_values, norm_y_values, color='black', linewidth=2)
    plt.xlabel("x")
    plt.ylabel("p(x)")
    plt.title("Gaussian Distribution with Mean " + str(mean) + " and Standard Deviation " + str(standard_dev))
    if save:
        plt.savefig(out_file)


if __name__ == "__main__":
    plot_gaussian(6, 0, 1, out_file="../../data/gaussian.png", save=True)
