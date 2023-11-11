import numpy as np
from molecular_dynamics import lennard_jones_potential, calculate_scalar_force
from molecular_dynamics import maxwell_boltzmann, calculate_kinetic_energy

rng = np.random.default_rng()


def harmonic_bond_potential(r, k, l):
    return 0.5 * k * (abs(r) - l) ** 2


def harmonic_bond_force(r, k, l):
    return (-k * r * (abs(r) - l)) / abs(r)


def langevian_force(friction, m, kT, v, dt):
    stochastic_force = rng.normal(0, np.sqrt((2 * kT * m * friction) / dt))
    return stochastic_force - friction * m * v


def verlet_position_update(coordinates, velocities, forces, m, dt):
    new_coordinates = np.zeros((len(coordinates), 3))
    for i in range(len(coordinates)):
        new_coordinates[i, :] = coordinates[i, :] + dt * velocities[i, :] + (1 / (2 * m)) * (dt ** 2) * forces[i, :]
    return new_coordinates


def verlet_velocity_update(friction, velocities, m, force, f_c, R, dt):
    new_velocities = np.zeros((len(velocities), 3))
    for i in range(len(velocities)):
        new_velocities[i, :] = (1 / (1 + dt * friction * 0.5) *
                                (velocities[i, :] + (dt / (2 * m)) * force[i, :] + (dt / (2 * m)) * (f_c[i, :] + R[i, :])))
    return new_velocities


def calculate_configuration_force(coordinates, residue, k, l):
    potential = 0
    forces = np.zeros((len(coordinates), 3))

    for i, particle_1 in enumerate(coordinates):  # loop over all the particles
        for j in range(i + 1, len(coordinates)):
            particle_2 = coordinates[j]

            residue_1 = residue[i]
            residue_2 = residue[j]

            dist = particle_1 - particle_2  # no PBC for Langevian protein
            r = np.linalg.norm(dist)
            # calculates the Lennard Jones potential contribution for nb residue
            if j != i + 1:
                if residue_1 == 'H' and residue_2 == 'H':
                    potential += lennard_jones_potential(r, epsilon=1, sigma=1)
                    scalar_force = calculate_scalar_force(r, epsilon=1, sigma=1)
                else:
                    potential += lennard_jones_potential(r, epsilon=2 / 3, sigma=1)
                    scalar_force = calculate_scalar_force(r, epsilon=2 / 3, sigma=1)
            else:  # bonded potential
                potential += harmonic_bond_potential(r, k, l)
                scalar_force = harmonic_bond_force(r, k, l)

            # calculates the vector force
            force = scalar_force * dist / r  # force in the radial direction, see
            # https://math.stackexchange.com/questions/1742524/numerical-force-due-to-lennard-jones-potential

            # Uses N3L to update the force matrix
            forces[i, :] += force
            forces[j, :] -= force

    return potential, forces


def random_term(friction, velocities, kT, m, dt):
    return np.random.normal(0, np.sqrt(2 * kT * m * friction / dt), (len(velocities), 3))


def calculate_stochastic_force(friction, velocities, kT, m, R, dt):
    return R - friction * m * velocities


def run(coordinates, m, residues, k, l, friction, kT, dt, iterations):
    n = len(coordinates)
    velocities = maxwell_boltzmann(kT, n, m)  # initialize velocities using the Maxwell-Boltzmann dist
    potential_record = []
    kinetic_record = []
    energies = []
    temperatures = []

    potential, f_c = calculate_configuration_force(coordinates, residues, k, l)
    R = random_term(friction, velocities, kT, m, dt)
    total_forces = f_c + calculate_stochastic_force(friction, velocities, kT, m, R, dt)
    potential_record.append(potential)

    kinetic_energy, T = calculate_kinetic_energy(velocities, m, 3 * n - 3)
    kinetic_record.append(kinetic_energy)
    temperatures.append(T)
    total_energy = potential + kinetic_energy
    energies.append(total_energy)
    current_state = coordinates

    for i in range(iterations):
        new_position = verlet_position_update(current_state, velocities, total_forces, m, dt)  # update the position

        potential, f_c_new = calculate_configuration_force(new_position, residues, k, l)
        R = random_term(friction, velocities, kT, m, dt)
        potential_record.append(potential)

        new_velocities = verlet_velocity_update(friction, velocities, m, total_forces, f_c_new, R, dt)

        total_forces = f_c_new + calculate_stochastic_force(friction, new_velocities, kT, m, R, dt)

        kinetic_energy, T = calculate_kinetic_energy(new_velocities, m, 3 * n - 3)
        kinetic_record.append(kinetic_energy)
        temperatures.append(T)

        total_energy = potential + kinetic_energy
        energies.append(total_energy)

        current_state = new_position
        velocities = new_velocities

        if i % 1000 == 0:
            print(potential, kinetic_energy, total_energy)

    return potential_record, kinetic_record, energies, temperatures, total_forces, velocities, current_state
