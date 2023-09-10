import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from numpy import random
from PS1.gaussian_dist import calculateGaussian


def sampleGaussian(n, mean, std):
    """
    Samples from a normal probability distribution with seed given by the date
    :param n: The number of samples
    :param mean: Mean of the distribution
    :param std: Std. of the distribution
    :return: A list of samples
    """
    seed = datetime.now()  # get the current date and time
    seed = abs(hash(seed)) % (10 ** 8)  # has the seed into an integer
    random.seed(seed)  # set the numpy seed

    samples = random.normal(mean, std, size=n)
    return samples


def plotSampleHist(n, mean, std, range, save=False, out_file=None):
    """
    Plots histogram of random normal samples compared to the continuous distribution.
    Divided into 30 bins
    :param n: Number of samples
    :param mean: Mean of the dist
    :param std: Std. of the dist
    :param range: Range to plot for
    :param save: If the file should be saved
    :param out_file: Where to save the file
    """
    samples = sampleGaussian(n, mean, std)
    count, bins, ignored = plt.hist(samples, 30, density=True)  # generates the normalized histogram
    xs = np.linspace(-range, range, 100)  # set the x-axis
    ys = []

    # continuous distribution
    for x in xs:
        ys.append(calculateGaussian(mean, std, x))

    # plotting code
    plt.plot(xs, ys)
    plt.xlabel("x")
    plt.ylabel("p(x)")
    plt.title("Sampled vs. Continuous Normal Distribution")

    if save:
        plt.savefig(out_file)
