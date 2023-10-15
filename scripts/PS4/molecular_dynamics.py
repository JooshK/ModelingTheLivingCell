import pandas as pd
import numpy as np

rng = np.random.default_rng()
k_b = 1.38


def lennard_jones_potential(distance, epsilon, sigma):
    """
    :returns LJ potential with parameters epsilon, sigma
    """
    return 4 * epsilon * ((sigma ** 12 / distance ** 6) - (sigma ** 6 / distance ** 3))


def calculate_force(delta, r2_ij, epsilon, sigma):
    """
    Calculates the LJ force on a particle in direction delta from another particle
    :param delta: The direction to calculate the force in
    :param r2_ij: Square distance
    :param epsilon: LJ parameter energy
    :param sigma: LJ parameter size
    :return: Pairwise force between two particles
    """
    return delta * (((48 * epsilon) / r2_ij) * ((sigma ** 12 / r2_ij ** 6) - (0.5 * sigma ** 6 / r2_ij ** 3)))


def calculate_configuration_force(coordinates, epsilon, sigma, box_length):
    """
    Given some configuration of particles, calculates the LJ potential/forces pairwise
    :param coordinates: Positions of the particles
    :param epsilon: LJ parameter energy
    :param sigma: LJ parameter size
    :param box_length: Box length to enforce periodic boundary conditions
    :return: The configuration potential, matrix of force on each particle
    """
    potential = 0
    forces = np.zeros((len(coordinates), 3))

    for i, particle_1 in enumerate(coordinates):
        for j in range(i + 1, len(coordinates)):
            particle_2 = coordinates[j]

            # Calculates the delta in each direction and enforce PBC
            delta_x = particle_1[0] - particle_2[0]
            delta_y = particle_1[1] - particle_2[1]
            delta_z = particle_1[2] - particle_2[2]

            delta_x -= box_length * np.round(delta_x / box_length)
            delta_y -= box_length * np.round(delta_y / box_length)
            delta_z -= box_length * np.round(delta_z / box_length)

            r2_ij = delta_x ** 2 + delta_y ** 2 + delta_z ** 2  # the squared distance

            potential_ij = lennard_jones_potential(r2_ij, epsilon, sigma)  # LJ potential
            f_ij_x = calculate_force(delta_x, r2_ij, epsilon, sigma)  # force in each direction
            f_ij_y = calculate_force(delta_y, r2_ij, epsilon, sigma)
            f_ij_z = calculate_force(delta_z, r2_ij, epsilon, sigma)
            f_ij = [f_ij_x, f_ij_y, f_ij_z]

            forces[i, :] = f_ij + forces[i, :]  # Uses N3L to calculate the force on j as - force on i
            forces[j, :] = forces[j, :] - f_ij

            potential += potential_ij

    return potential, forces


class MolecularDynamics:
    """
    Represents an instance of a molecular dynamics simulation
    """
    def __init__(self, sigma, epsilon, box_length, m, T, n, dt, positions, iterations):
        self.epsilon = epsilon
        self.sigma = sigma
        self.m = m
        self.T = T
        self.n = n
        self.positions = positions
        self.box_length = box_length
        self.dt = dt

        self.kinetic_energies = np.zeros(iterations)
        self.temperatures = np.zeros(iterations)
        self.potentials = np.zeros(iterations)
        self.energies = np.zeros(iterations)
        self.forces = np.zeros(iterations)

        self.dof = 3 * len(self.positions) - 3
        self.v = rng.standard_normal((self.n, 3)) * np.sqrt(self.T)
        self.px = np.sum(self.m * self.v, axis=0)
        self.v -= self.px / self.n

        self.temp = sum(sum([v ** 2 / self.dof for v in self.v]))
        self.k = self.dof * 0.5 * k_b * self.temp

        self.kinetic_energies[0] = self.k
        self.temperatures[0] = self.temp

        self.potential0, self.force = calculate_configuration_force(self.positions, self.epsilon, self.sigma, 3.5)
        self.potentials[0] = self.potential0
        self.energies[0] = self.potential0 + self.k

    def verlet_update_position(self):
        position_update = []
        for r_i, v_i, f_i in zip(self.positions, self.v, self.force):
            component_update = []
            for i in range(3):
                x_i_dt = r_i[i] + self.dt * v_i[i] + ((self.dt ** 2) / (2 * self.m)) * f_i[i]
                component_update.append(x_i_dt)
            position_update.append(component_update)
        self.positions = np.array(position_update)

    def verlet_update_velocity(self):
        velocity_update = []
        for v_i, f_i in zip(self.positions, self.force):
            component_update = []
            for i in range(3):
                v_i_dt = v_i + (self.dt/(2*self.m))*(f_i[i])
                component_update.append(v_i_dt)
            velocity_update.append(component_update)
        self.v = np.array(velocity_update)

