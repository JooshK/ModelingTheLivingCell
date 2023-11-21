import numpy as np
from sklearn.metrics import pairwise_distances_argmin

rng = np.random.default_rng()


def k_means(X, k):
    i = rng.permutation(X.shape[0])[:k]  # randomly generate k clusters
    centers = X[i]

    while True:
        cluster = pairwise_distances_argmin(X, centers)  # assign cluster based on smallest pairwise distance
        new_centers = np.array([X[cluster == i].mean(0) for i in range(k)])  # new cluster based on mean

        if np.all(new_centers == centers):  # if the centers stop updating
            break

        centers = new_centers
    return centers, cluster