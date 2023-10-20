import pandas as pd
import numpy as np

rng = np.random.default_rng(123)
k_b = 1.38


def lennard_jones_potential(distance, epsilon, sigma):
    """
    returns LJ potential with parameters epsilon, sigma
    """
    return 4 * epsilon * ((sigma ** 12 / distance ** 6) - (sigma ** 6 / distance ** 3))


def calculate_scalar_force(r2_ij, epsilon, sigma):
    """
    Calculates the scalar LJ force on a particle from another particle
    :param r2_ij: Square distance
    :param epsilon: LJ parameter energy
    :param sigma: LJ parameter size
    :return: Pairwise force between two particles
    """
    r = np.linalg.norm(r2_ij)
    return 48 * epsilon * ((sigma**12/r**13) - 0.5 * (sigma**6/r**7))


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

            r = np.sqrt(r2_ij)
            f_scalar = calculate_scalar_force(r2_ij, epsilon, sigma)
            f_vector = f_scalar * dr / r

            forces[i] = forces[i] + f_ij  # Uses N3L to calculate the force on j as - force on i
            forces[j] = forces[j] + f_ij

            potential += potential_ij

    return potential, forces


def verlet_position_update(position, velocity, force, mass, time_step):
    return position + time_step * velocity + ((time_step ** 2) / 2 * mass) * force


def verlet_velocity_update(force, mass, time_step):
    return (time_step/2*mass)*force


def verlet_propagation_position(positions, velocities, forces, m, dt):
    updated_positions = np.zeros((len(positions), 3))
    for particle_i in range(len(positions)):
        new_position = verlet_position_update(positions[particle_i, :], velocities[particle_i, :],
                                              forces[particle_i, :], m, dt)
        updated_positions[particle_i] = new_position

    return updated_positions


def verlet_propagation_velocity(velocities, forces, m, dt):
    updated_velocities = np.zeros((len(velocities), 3))
    for particle_i in range(len(velocities)):
        v_adj = verlet_velocity_update(forces[particle_i, :], m, dt)
        v_new = v_adj + velocities[0, :]
        updated_velocities[particle_i] = v_new

    return updated_velocities


def calculate_kinetic_energy(velocities, mass):
    kinetic_energy = 0
    for v_i in velocities:
        squared_speed = v_i[0] ** 2 + v_i[1] ** 2 + v_i[2] ** 2
        kinetic_energy_i = 0.5 * mass * squared_speed

        kinetic_energy += kinetic_energy_i
    return kinetic_energy


def run(initial_positions, T, m, dt, epsilon, sigma, box_length, iterations):
    dof = 3 * len(initial_positions) - 3

    # initialize the velocity and shift according to total momentum
    initial_velocities = rng.standard_normal((len(initial_positions), 3)) * np.sqrt(T)
    px = np.sum(m * initial_velocities, axis=0)
    initial_velocities -= px / len(initial_positions)

    # initialize the potential and the forces
    initial_potential, initial_forces = calculate_configuration_force(initial_positions, epsilon, sigma, box_length)

    # calculate energies
    initial_ke = calculate_kinetic_energy(initial_velocities, m)
    total_energy = initial_ke + initial_potential

    velocities = [initial_velocities]
    potentials = [initial_potential]
    kinetic_energies = [initial_ke]
    configurations = [initial_positions]
    forces = [initial_forces]

    print("Initial total energy", total_energy)

    # simulation loop
    for i in range(iterations):
        current_position = configurations[i]
        current_velocities = velocities[i]
        current_forces = forces[i]

        position_dt = verlet_propagation_position(current_position, current_velocities, current_forces, m, dt)
        velocity_dt_1 = verlet_propagation_velocity(current_velocities, current_forces, m, dt)

        new_potential, new_forces = calculate_configuration_force(position_dt, epsilon, sigma, box_length)
        velocity_dt = verlet_propagation_velocity(velocity_dt_1, new_forces, m, dt)

        new_kinetic_energy = calculate_kinetic_energy(velocity_dt, m)
        total_energy = new_kinetic_energy + new_potential

        configurations.append(position_dt)
        velocities.append(velocity_dt)
        potentials.append(new_potential)
        forces.append(new_forces)
        kinetic_energies.append(new_kinetic_energy)

        if i % 100 == 0:
            print("Final total energy", total_energy)
            print("Configuration", configurations[i])
            print("Velocities", velocities[i])
            print("Potential", potentials[i])
            print("Forces", forces[i])
            print("Kinetic Energy", kinetic_energies[i])

    return configurations, velocities, potentials, forces, kinetic_energies
