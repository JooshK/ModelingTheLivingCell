import pandas as pd
import numpy as np


def squared_distance(x1, x2, length):
    delta_x = x1 - x2
    delta_x -= length * np.round(delta_x/length)
    return delta_x


def generate_distance_matrix(coordinates, length):
    distances = []
    for i, particle_1 in enumerate(coordinates):
        for j in range(i + 1, len(coordinates)):
            particle_2 = coordinates[j]
            distance = np.linalg.norm(squared_distance(particle_1, particle_2, length))
            distances.append(distance)
    return distances


def lennard_jones_potential(distance, e, sigma):
    return 4 * e * ((sigma / distance) ** 12 - (sigma / distance) ** 6)


def calculate_configuration_potential(coordinates, e, sigma, length):
    distances = generate_distance_matrix(coordinates, length)
    potentials = [lennard_jones_potential(dist, e, sigma) for dist in distances]
    return sum(potentials)