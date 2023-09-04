import random
import matplotlib.pyplot as plt


def random_trajectory(iterations: int) -> list:
    """
    Starting at zero, generates a random trajectory in steps from -1 to 1
    :param iterations: The number of iterations to run for
    :return: The trajectory generated
    """
    trajectory = []  # store the trajectory
    value = 0  # the initial value is set to zero
    trajectory.append(value)

    # loop over the iterations and generate a random step in [-1, 1], then update the value
    for i in range(iterations):
        step = random.uniform(-1, 1)
        value += step
        trajectory.append(value)

    return trajectory


def monte_carlo_trajectory(iterations: int) -> list:
    """
    Starting at zero, generates a random trajectory in random steps from -1 to 1, if the step is less than 0.5 then
    the same value is kept
    :param iterations: The number of iterations to run for
    :return: The trajectory generated
    """
    trajectory = []
    value = 0
    trajectory.append(value)

    for i in range(iterations):
        step = random.uniform(-1, 1)
        if step > 0.5:
            value += step
        trajectory.append(value)

    return trajectory


def plot_trajectory(iterations: int, file: str):
    """
    Plot a random trajectory
    :param iterations: The number of iterations to generate the trajectory
    :param file: The output file to save the graph
    """
    trajectory = random_trajectory(iterations)  # generate the trajectory

    # plot on a separate figure
    fig1 = plt.figure()
    plt.plot(range(iterations), trajectory)
    plt.xlabel("Iteration Number")
    plt.ylabel("Value")
    plt.title("Random Trajectory from Uniform Sampling Between -1 and 1")
    plt.savefig(file)


def plot_monte_carlo_trajectory(iterations: int, file: str):
    """
    Plot a monte Carlo trajectory
    :param iterations: The number of iterations to generate the trajectory
    :param file: The output file to save the graph
    """
    trajectory = monte_carlo_trajectory(iterations)

    fig2 = plt.figure()
    plt.plot(range(iterations), trajectory)
    plt.xlabel("Iteration Number")
    plt.ylabel("Value")
    plt.title("Monte Carlo Trajectory from Uniform Sampling Between -1 and 1")
    plt.savefig(file)


if __name__ == "__main__":
    plot_trajectory(1000, "../../data/random_trajectory.png")
    plot_monte_carlo_trajectory(1000, "../../data/monte_carlo_trajectory.png")
