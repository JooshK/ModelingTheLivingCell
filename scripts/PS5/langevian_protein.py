import numpy as np
from PS4.molecular_dynamics import *


def harmonic_bond_potential(r, k, l):
    return 0.5*k*(abs(r) - l)**2


def harmonic_bond_force(r, k, l):
    return (-k * r * (abs(r) - l))/abs(r)


def langevian_force(friction, m, kT, v, dt):
    stochastic_force = rng.normal(0, np.sqrt((2*kT*m*friction)/dt))
    return stochastic_force - friction*m*v


def verlet_velocity_update(friction, v, m, f, f_c, R, dt):
    return 1/(1 + dt*friction*0.5) * (v + (dt/(2*m))*f + (dt/(2*m))*(f_c + R))


def verlet_propagation_velocity(friction, velocities, force, force_c, R, m, dt):
    velocities[:] = verlet_velocity_update(friction, velocities, m, force, force_c, R, dt)


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
                    potential += lennard_jones_potential(r, epsilon=2/3, sigma=1)
                    scalar_force = calculate_scalar_force(r, epsilon=2/3, sigma=1)
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


def calculate_stochastic_force(friction, velocities, kT, m, R, dt):
    forces = np.zeros((len(velocities), 3))

    for i, vel in enumerate(velocities):
        scalar_force = R - friction*m*vel
        forces[i, :] = scalar_force

    return forces


def run(coordinates, m, residues, k, l, friction, kT, dt, iterations):
    n = len(coordinates)
    velocities = maxwell_boltzmann(kT, n, m)  # initialize velocities using the Maxwell-Boltzmann dist
    potential_record = []
    kinetic_record = []
    energies = []
    temperatures = []

    potential, f_c = calculate_configuration_force(coordinates, residues, k, l)
    R = rng.normal(0, np.sqrt(2 * kT * m * friction / dt))
    total_forces = f_c + calculate_stochastic_force(friction, velocities, kT, m, R, dt)
    potential_record.append(potential)

    kinetic_energy, T = calculate_kinetic_energy(velocities, m, 3 * n - 3)
    kinetic_record.append(kinetic_energy)
    temperatures.append(T)
    total_energy = potential + kinetic_energy
    energies.append(total_energy)

    for i in range(iterations):
        print(coordinates)
        verlet_propagation_position(coordinates, velocities, total_forces, m, dt)  # update the position

        old_total_force = np.copy(total_forces)
        potential, f_c = calculate_configuration_force(coordinates, residues, k, l)
        potential_record.append(potential)

        R = rng.normal(0, np.sqrt(2*kT*m*friction/dt))
        verlet_propagation_velocity(friction, velocities, old_total_force, f_c, R, m, dt)

        total_forces = f_c + calculate_stochastic_force(friction, velocities, kT, m, R, dt)

        kinetic_energy, T = calculate_kinetic_energy(velocities, m, 3 * n - 3)
        kinetic_record.append(kinetic_energy)
        temperatures.append(T)

        total_energy = potential + kinetic_energy
        energies.append(total_energy)

    return potential_record, kinetic_record, energies, temperatures, total_forces, velocities, coordinates



















