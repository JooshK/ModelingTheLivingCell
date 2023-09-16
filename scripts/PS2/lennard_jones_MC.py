import numpy as np
from scipy.spatial.distance import pdist


def calculate_distance_matrix(file, length):
    """
    Given a file of particle coordinates calculates the distance matrix subject to periodic boundary conditions
    :param length: Length of box for PBCs
    :param file: File of particle coords
    """
    coordinates = np.loadtxt(file)
    n = len(coordinates)
    distances = np.empty((n, n))

    for i, particle_1 in enumerate(coordinates):
        for j, particle_2 in enumerate(coordinates):
            dist = 0
            for component in range(len(particle_2)):
                dist = abs(particle_2[component] - particle_1[component])  # calculate the distance per component
                dist = abs(dist - length * round(dist/length))  # enforce PBC
            distances[i, j] = dist

    return distances


def lennard_jones_potential(dist, e, sigma):
    return 4 * e * ((sigma / dist) ** 12 - (sigma / dist) ** 6)


def calculate_lj_potential(file, e, sigma, length):
    distances = calculate_distance_matrix(file, length)
    potentials = []
    for particle in distances:
        for dist in particle:
            if dist != 0:
                potentials.append(lennard_jones_potential(dist, e, sigma))
    return potentials


print(calculate_distance_matrix("../../data/init_crds_boxl_3.5-2.dat", 3.5))
