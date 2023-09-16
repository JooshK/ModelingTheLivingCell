from gaussian_sampling import plot_sample_hist
from residue_sampling import *
import matplotlib.pyplot as plt


def response_1():
    plot_sample_hist(2000, 2, 3, 10, save=True, out_file="../../data/normalHist.png")


def response_2():
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


if __name__ == "__main__":
    response_1()
    response_2()
