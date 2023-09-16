import numpy as np
import random

# dictionary of integers to amino acid one-letter codes
amino_acid_keys = {
    1: "A",
    2: "R",
    3: "N",
    4: "D",
    5: "C",
    6: "Q",
    7: "E",
    8: "G",
    9: "H",
    10: "I",
    11: "L",
    12: "K",
    13: "M",
    14: "F",
    15: "P",
    16: "S",
    17: "T",
    18: "W",
    19: "Y",
    20: "V"
}

# percent composition in the Swiss prot KB database
amino_acid_scale_values = {
    "A":  8.25,
    "R":  5.53,
    "N":  4.06,
    "D":  5.45,
    "C":  1.37,
    "Q":  3.93,
    "E":  6.75,
    "G":  7.07,
    "H":  2.27,
    "I":  5.96,
    "L":  9.66,
    "K":  5.84,
    "M":  2.42,
    "F":  3.86,
    "P":  4.70,
    "S":  6.56,
    "T":  5.34,
    "W":  1.08,
    "Y":  2.92,
    "V":  6.87
}


def uniform_sampling(length):
    samples = []
    sequence = []

    for i in range(length):
        samples.append(np.random.randint(1, 21))  # uniform sampling of one of the 20 amino acids

    for sample in samples:
        sequence.append(amino_acid_keys[sample])  # convert to one-letter codes

    return sequence


def swiss_prot_sampling(length):
    amino_acids = list(amino_acid_scale_values.keys())
    probs = list(amino_acid_scale_values.values())

    return random.choices(amino_acids, probs, k=length)
