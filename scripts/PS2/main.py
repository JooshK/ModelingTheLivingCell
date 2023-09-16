import numpy as np

from gaussian_sampling import plot_sample_hist
from residue_sampling import *
from lennard_jones_MC import monte_carlo, calculate_lj_potential
import matplotlib.pyplot as plt


def response1():
    plot_sample_hist(2000, 2, 3, 10, save=True, out_file="../../data/normalHist.png")


def response2():
    uniform_sequence = ''.join(uniform_sampling(300))  # generate the sequence of 300 uniform amino acids
    print(uniform_sequence)

    # calculate the frequencies of each amino acid in the above sequence
    freqs = {}
    for amino_acid in amino_acid_keys.values():
        freqs[amino_acid] = uniform_sequence.count(amino_acid) / 300
    print("Frequency of AA In Uniform Distribution")
    print(freqs)

    print("\n")

    # sampling from the uniprot frequency data
    actualSequence = ''.join(swiss_prot_sampling(300))
    print(actualSequence)

    freqs = {}
    for amino_acid in amino_acid_scale_values.keys():
        freqs[amino_acid] = actualSequence.count(amino_acid) / 300
    print("Frequency of AA Sampled from SwissProtKB Database")
    print(freqs)

    fig, ax = plt.subplots()
    ax.bar(amino_acid_scale_values.keys(), [a / 100 for a in amino_acid_scale_values.values()])
    plt.title("Amino Acid Frequency in SwissProt Database")
    plt.xlabel("Amino Acid")
    plt.ylabel("Sequence Frequency (%)")
    plt.savefig("../../data/swissAAFreq.png")


def response3():
    coordinate_data = r'../../data/init_crds_boxl_3.5-2.dat'
    coords = np.loadtxt(coordinate_data)
    e = 0.25
    sigma = 1
    l = 3.5
    delta = 0.2
    iterations = 100000

    print("Initial Lennard Jones potential energy")
    print(calculate_lj_potential(coords, e, sigma, l))

    print("MCMC Simulation with kt = 1: ")
    print("Average potential energy: ",monte_carlo(coordinate_data, iterations, e, sigma, l, delta), "Joules")

    print("kt = 0.2")
    print("Average potential energy: ", monte_carlo(coordinate_data, iterations, e, sigma, l, delta, 0.2), "Joules")

    print("kt = 2")
    print("Average potential energy: ", monte_carlo(coordinate_data, iterations, e, sigma, l, delta, 2), "Joules")

    print("Box length l = 3.5, kt = 1: ")
    print("Average potential energy: ", monte_carlo(coordinate_data, iterations, e, sigma, l, delta), "Joules")


if __name__ == "__main__":
    # response1()
    # response2()
    response3()
