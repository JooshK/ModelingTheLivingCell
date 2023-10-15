import pandas as pd
import numpy as np


def squared_distance(x1, x2, length):
    """
    Squared distance between two vectors with periodic boundary conditions
    :param x1: Vector 1
    :param x2: Vector 2
    :param length: Box dimensions
    :return: Squared distance scalar
    """
    delta_x = x1 - x2
    delta_x -= length * np.round(delta_x/length)
    return delta_x


def generate_distance_matrix(coordinates: pd.DataFrame, length):
    """
    Generates a matrix of pairwise distances given a matrix of coordinates with PBC
    :param coordinates: Matrix of coordinates
    :param length:
    :return: Distance matrix
    """
    coordinates = coordinates.to_numpy()
    distances = []
    for i, particle_1 in enumerate(coordinates):
        for j in range(i + 1, len(coordinates)):
            particle_2 = coordinates[j]
            distance = np.linalg.norm(squared_distance(particle_1, particle_2, length))
            distances.append(distance)
    return distances


def lennard_jones_potential(distance, e, sigma):
    """
    Calculates the Lennard Jones potential given the distance between two molecules
    """
    return 4 * e * ((sigma / distance) ** 12 - (sigma / distance) ** 6)


def calculate_configuration_potential(coordinates, e, sigma, length):
    """
    Given a configuration of molecules, calculates the lennard jones potential energy for the system.
    :param coordinates: Coordinates of a configuration of molecules
    :param e: Lennard Jones energy
    :param sigma: Size of molecules
    :param length: Length for PBC
    :return: potential energy in Joules
    """
    distances = generate_distance_matrix(coordinates, length)
    potentials = [lennard_jones_potential(dist, e, sigma) for dist in distances]
    return sum(potentials)


def potential_move(coordinates: pd.DataFrame, delta):
    """
    Generates a perturbation on the position of a random molecule using Uniform Random Numbers and adds it to a
    coordinate matrix
    :param coordinates: System coordinates
    :param delta: Size of URN (-delta/2 to delta/2)
    :return: The updated coordinate matrix
    """
    p = coordinates.sample()
    new_p = p.copy()
    for c in new_p:
        new_p[c] += delta * (np.random.random() - 0.5)
    coordinates.iloc[p.index] = new_p
    return coordinates


def acceptance_boltzmann(e1, e2, kt):
    """
    Boltzmann distribution for probability of transitioning from e1 to e2
    :param e1: Original energy
    :param e2: New energy
    :param kt: Temperature parameter beta
    :return: probability of transition
    """
    return min(1, np.exp((-1/kt) * (e2 - e1)))


def monte_carlo(iterations, coordinates, delta, e, sigma, length, kt=1):
    """
    Markov chain Monte Carlo simulation starting from one set of coordinates to find average potential energy.
    :param iterations: Number of iterations to run for
    :param coordinates: Initial coordinate system of molecules
    :param delta: Size of jumps to make in potential changes
    :param e: LJ Potential energy
    :param sigma: Particle Size
    :param length: Length for periodic boundary conditions
    :param kt: Temperature Parameter
    :return: Average potential energy
    """
    V = np.zeros(iterations)
    coordinate_list = []

    for i in range(iterations):
        if i > 0:
            old_coordinates = coordinate_list[i - 1]
            old_potential = V[i - 1]

            new_coordinates = potential_move(old_coordinates, delta)
            new_potential = calculate_configuration_potential(new_coordinates, e, sigma, length)

            acc_p = acceptance_boltzmann(old_potential, new_potential, kt)
            if np.random.random() < acc_p:
                coordinate_list.append(new_coordinates)
                V[i] = new_potential
            else:
                coordinate_list.append(old_coordinates)
                V[i] = old_potential
        else:
            V[i] = calculate_configuration_potential(coordinates, e, sigma, length)
            coordinate_list.append(coordinates)

    return np.mean(V)
