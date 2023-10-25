import numpy as np

rng = np.random.default_rng(1234)


def lennard_jones_potential(distance, epsilon, sigma):
    """
    returns LJ potential with parameters epsilon, sigma.
    """
    return 4 * epsilon * ((sigma ** 12 / distance ** 12) - (sigma ** 6 / distance ** 6))


def calculate_scalar_force(r, epsilon, sigma):
    """
    Calculates the scalar LJ force on a particle from another particle
    :param r: distance
    :param epsilon: LJ parameter energy
    :param sigma: LJ parameter size
    :return: Pairwise force between two particles
    """
    return 48 * epsilon * ((sigma ** 12 / r ** 13) - 0.5 * (sigma ** 6 / r ** 7))


def calculate_distance(r1, r2, box_length):
    """
    Calculates the absolute distance using PBC
    :param r1: Position 1
    :param r2: Position 2
    :param box_length: Length of PBC box
    """
    dr = r1 - r2
    dr -= box_length * np.round(dr / box_length)
    return dr


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

    for i, particle_1 in enumerate(coordinates):  # loop over all the particles
        for j in range(i + 1, len(coordinates)):
            particle_2 = coordinates[j]

            # calculates the Lennard Jones potential contribution
            dist = calculate_distance(particle_1, particle_2, box_length)
            r = np.linalg.norm(dist)
            potential += lennard_jones_potential(r, epsilon, sigma)

            # calculates the vector force
            scalar_force = calculate_scalar_force(r, epsilon, sigma)
            force = scalar_force * dist / r  # force in the radial direction, see
            # https://math.stackexchange.com/questions/1742524/numerical-force-due-to-lennard-jones-potential

            # Uses N3L to update the force matrix
            forces[i, :] += force
            forces[j, :] -= force

    return potential, forces


# These functions implement the verlet update for positions and velocity
def verlet_position_update(position, velocity, force, mass, time_step):
    return position + (time_step * velocity) + ((time_step ** 2) / (2 * mass)) * force


def verlet_velocity_update(velocity, force_1, force_2, mass, dt):
    return velocity + (dt / 2 * mass) * (force_1 + force_2)


# These functions update the position or velocity matrix in place
def verlet_propagation_position(positions, velocities, forces, m, dt):
    positions[:] = verlet_position_update(positions, velocities, forces, m, dt)


def verlet_propagation_velocity(velocities, force_1, force_2, m, dt):
    velocities[:] = verlet_velocity_update(velocities, force_1, force_2, m, dt)


def calculate_kinetic_energy(velocities, mass):
    """
    Calculate the kinetic energy given a velocity matrix
    """
    kinetic_energy = 0
    for v_i in velocities:
        v_i_squared = np.sum(v_i ** 2)
        kinetic_energy_i = 0.5 * mass * v_i_squared

        kinetic_energy += kinetic_energy_i
    return kinetic_energy


def maxwell_boltzmann(temp, n, m, kb=1):
    """
    Generates an N X 3 matrix of numbers distributed according to the Maxwell Boltzmann dist
    :param temp: Temp of system
    :param n: Number of particles
    :param m: Mass of each particle
    :param kb: Value for kb (default 1)
    :return: np.ndarray of n x 3 elements.
    """
    velocity_matrix = np.sqrt(kb * temp / m) * rng.standard_normal((n, 3))
    average_velocity = np.sum(velocity_matrix) / n
    velocity_matrix -= average_velocity  # zero the mean velocity

    return velocity_matrix


def run(positions, T, m, dt, epsilon, sigma, box_length, iterations):
    """
    Executes the Molecular Dynamics simulation
    :param positions: Position matrix to start with
    :param T: Temperature of the system
    :param m: Mass of each particle
    :param dt: Time step to use
    :param epsilon: LJ param energy
    :param sigma: LJ param size
    :param box_length: Box length to enforce PBC
    :param iterations: Number of iterations to run for
    :return: The potential, kinetic, and total energy, the final force, velocity and positions
    """
    n = len(positions)
    velocities = maxwell_boltzmann(T, n, m)  # initialize velocities using the Maxwell-Boltzmann dist
    potential_record = []
    kinetic_record = []
    energies = []

    # Set up initial values and append to appropriate lists

    potential, forces = calculate_configuration_force(positions, epsilon, sigma, box_length)
    potential_record.append(potential)

    kinetic_energy = calculate_kinetic_energy(velocities, m)
    kinetic_record.append(kinetic_energy)
    total_energy = potential + kinetic_energy
    energies.append(total_energy)

    for i in range(1, iterations):  # main iteration loop
        verlet_propagation_position(positions, velocities, forces, m, dt)  # update the position

        old_forces = np.copy(forces)  # store the old force for velocity verlet
        potential, forces = calculate_configuration_force(positions, epsilon, sigma, box_length)  # potential
        potential_record.append(potential)

        verlet_propagation_velocity(velocities, old_forces, forces, m, dt)  # velocity using updated forces
        kinetic_energy = calculate_kinetic_energy(velocities, m)
        kinetic_record.append(kinetic_energy)

        total_energy = potential + kinetic_energy
        energies.append(total_energy)

    return potential_record, kinetic_record, energies, forces, velocities, positions
