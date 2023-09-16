import numpy as np
import random


def lennard_jones_potential(dist, e, sigma):
    """
    Calculates the lennard jones potential for a given distance r_ij
    :param dist: The distance
    :param e: energy
    :param sigma: 'particle size'
    :return: Lennard Jones potential energy
    """
    return 4 * e * ((sigma / dist) ** 12 - (sigma / dist) ** 6)


def calculate_lj_potential(coordinates, e, sigma, length):
    """
    Given a file of coordinates calculates the LJ potential for the collection of particles with periodic boundary conditions
    :param coordinates: Matrix of particle coordinates
    :param e: Energy
    :param sigma: Particle Size
    :param length: Length of box for PBC
    :return: LJ potential for particle configurations
    """

    potential = 0
    for i in range(len(coordinates)):
        for j in range(i+1, len(coordinates)):
            r = coordinates[i] - coordinates[j]  # calculate pairwise distances
            r -= length * np.round(r/length)  # Periodic Boundary
            r = np.linalg.norm(r)

            potential += lennard_jones_potential(r, e, sigma)
    return potential


def acceptance_probability(old_energy, new_energy, kbt):
    """
    The probability of accepting a transition to a new energy according to the
    Boltzmann distribution
    :param old_energy: Energy of x
    :param new_energy: New state energy
    :param kbt: temperature scale (defaults to 1)
    :return: Acceptance probability
    """
    return min(1, np.exp((new_energy-old_energy)*-1/kbt))


def monte_carlo(file, iterations, e, sigma, length, delta, kbt=1):
    """
    Runs the Markov Chain Monte Carlo simulation with the given parameters with Periodic Boundary Conditions
    :param file: File containing initial coordinate data
    :param iterations: Number of iterations to run the simulation
    :param e: Energy (LJ Potential parameter)
    :param sigma: Particle size (LJ Potential parameter)
    :param length: Length of box
    :param delta: Step size to make
    :param kbt: The units of energy size, default to 1
    :return: The average potential energy of the system
    """
    v_lj = np.zeros(iterations)  # pre-allocate the array for optimization
    coordinates = np.loadtxt(file)
    p_acc_total = np.zeros(iterations)

    for i in range(iterations):
        # randomly select a particle and get its state
        selection = random.randint(0, len(coordinates)-1)
        initial_state = coordinates[selection].copy()
        if i > 0:  # only re calculate the energy after the first time (the LJ calculation is expensive)
            initial_energy = v_lj[i - 1]
        else:
            initial_energy = calculate_lj_potential(coordinates, e, sigma, length)

        # perform some potential move and find the new energy
        for c in range(len(initial_state)):
            coordinates[selection][c] += delta * (random.random() - 0.5)
        new_state_energy = calculate_lj_potential(coordinates, e, sigma, length)

        # get the acceptance probability
        p_acc = acceptance_probability(initial_energy, new_state_energy, kbt)
        p_acc_total[i] = p_acc

        if random.random() < p_acc:  # if we accept the state change then keep the changed coordinates
            v_lj[i] = new_state_energy
        else:
            coordinates[selection] = initial_state  # rejected state change, revert to old coordinates
            v_lj[i] = initial_energy

    print("Average acceptance probability:", p_acc_total.mean())
    return sum(v_lj)/len(v_lj)


