import numpy as np
from scipy.spatial.distance import euclidean


def structure(beads):
    """
    :param beads: An array of [x, y, z] coordinates for each bead
    :return: The structure object of specified distances
    """
    structure = np.array([
        euclidean(beads[0], beads[9]),
        euclidean(beads[0], beads[3]),
        euclidean(beads[0], beads[4]),
        euclidean(beads[1], beads[5]),
        euclidean(beads[3], beads[6]),
        euclidean(beads[4], beads[9]),
        euclidean(beads[4], beads[8]),
    ])
    return structure


def evaluate_structures(x, y, z):
    """
    Given arrays of x, y, z data for each bead vs time, returns the structures vs time array.
    """
    structures = []

    for i in range(len(x)):
        beads = []

        for j in range(10):
            bead_j = np.array([x[i][j], y[i][j], z[i][j]])
            beads.append(bead_j)

        structure_t = structure(beads)
        structures.append(structure_t)

    return np.array(structures)


def transition_matrix(clusters, step_dist):
    transition_mat = np.zeros((6, 6))

    for i, cluster in enumerate(clusters):
        idx1 = cluster

        try:
            idx2 = clusters[i + step_dist]
        except IndexError:
            break

        transition_mat[idx1][idx2] += 1  # increment the count of that transition

    row_sums = transition_mat.sum(axis=1)
    normalized_matrix = transition_mat / row_sums[:, np.newaxis]

    return normalized_matrix


def state_prob(clusters):
    prob_mat = np.zeros(6)

    for cluster in clusters:
        prob_mat[cluster] += 1

    normalized_matrix = prob_mat / prob_mat.sum()

    return normalized_matrix
