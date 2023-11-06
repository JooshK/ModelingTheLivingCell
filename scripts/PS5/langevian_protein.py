import numpy as np
from PS4.molecular_dynamics import *


def harmonic_bond_potential(r, k, l):
    return 0.5*k*(abs(r) - l)**2


def langevian_force(friction, m, kT, v, dt):
    stochastic_force = rng.normal(0, np.sqrt((2*kT*m*friction)/dt))
    return stochastic_force - friction*m*v


def verlet_velocity_update(friction, v, m, f, f_c, R, dt):
    return 1/(1 + dt*friction*0.5) * (v + (dt/(2*m))*f + (dt/(2*m))*(f_c + R))


def verlet_propagation_velocity(friction, velocities, force, force_c, R, m, dt):
    velocities[:] = verlet_velocity_update(friction, velocities, m, force, force_c, R, dt)


def calculate_configuration_force(coordinates, residue, epsilon, sigma, friction, m, k, l, kT, v, dt):
    potential = 0
    forces = np.zeros(len(coordinates))

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
                    potential += lennard_jones_potential(r, epsilon, sigma)
            else:  # bonded potential
                potential += harmonic_bond_potential(r, k, l)

            # calculates the vector force
            scalar_force = calculate_scalar_force(r, epsilon, sigma) + langevian_force(friction, m, kT, v, dt)
            force = scalar_force * dist / r  # force in the radial direction, see
            # https://math.stackexchange.com/questions/1742524/numerical-force-due-to-lennard-jones-potential

            # Uses N3L to update the force matrix
            forces[i, :] += force
            forces[j, :] -= force

    return potential, forces