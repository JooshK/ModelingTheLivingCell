import numpy as np


def calculate_propensity(state, pm):
    """
    order - geneA, mRNA_A, A, geneA_bound, geneR, mRNA_R, geneR_bound, R, C
    requires a class pm of kinetic parameters
    :return: Returns the propensity vector
    """
    propensities = np.zeros(16)
    geneA = state[0]
    mRNA_A = state[1]
    A = state[2]
    geneA_bound = state[3]
    geneR = state[4]
    mRNA_R = state[5]
    geneR_bound = state[6]
    R = state[7]
    C = state[8]

    propensities[0] = pm.k1 * geneA
    propensities[1] = pm.k2a * geneA * A
    propensities[2] = pm.k2b * geneA_bound
    propensities[3] = pm.k3 * geneA_bound
    propensities[4] = pm.k4 * geneR
    propensities[5] = pm.k5a * geneR * A
    propensities[6] = pm.k5b * geneR_bound
    propensities[7] = pm.k6 * geneR_bound
    propensities[8] = pm.k7 * mRNA_A
    propensities[9] = pm.k8 * mRNA_R
    propensities[10] = pm.k9 * A * R
    propensities[11] = pm.k10 * C
    propensities[12] = pm.k11 * A
    propensities[13] = pm.k12 * R
    propensities[14] = pm.k13 * mRNA_A
    propensities[15] = pm.k14 * mRNA_R

    return propensities


def run(max_time, pm):
    time = 0
    state = [pm.geneA_init, 0, 0, 0, pm.geneR_init, 0, 0, 0, 0]
    # order -    geneA, mRNA_A, A, geneA_bound, geneR, mRNA_R, geneR_bound, R, C

    reaction_1 = [0, 1, 0, 0, 0, 0, 0, 0, 0]
    reaction_2a = [-1, 0, -1, 1, 0, 0, 0, 0, 0]
    reaction_2b = [1, 0, 1, -1, 0, 0, 0, 0, 0]
    reaction_3 = [0, 1, 0, 0, 0, 0, 0, 0, 0]
    reaction_4 = [0, 0, 0, 0, 0, 1, 0, 0, 0]
    reaction_5a = [0, 0, -1, 0, -1, 0, 1, 0, 0]
    reaction_5b = [0, 0, 1, 0, 1, 0, -1, 0, 0]
    reaction_6 = [0, 0, 0, 0, 0, 1, 0, 0, 0]
    reaction_7 = [0, 0, 1, 0, 0, 0, 0, 0, 0]
    reaction_8 = [0, 0, 0, 0, 0, 0, 0, 1, 0]
    reaction_9 = [0, 0, -1, 0, 0, 0, 0, -1, 1]
    reaction_10 = [0, 0, 0, 0, 0, 0, 0, 1, -1]
    reaction_11 = [0, 0, -1, 0, 0, 0, 0, 0, 0]
    reaction_12 = [0, 0, 0, 0, 0, 0, 0, -1, 0]
    reaction_13 = [0, -1, 0, 0, 0, 0, 0, 0, 0]
    reaction_14 = [0, 0, 0, 0, 0, -1, 0, 0, 0]

    reaction_matrix = np.array([reaction_1, reaction_2a, reaction_2b, reaction_3, reaction_4,
                                reaction_5a, reaction_5b, reaction_6, reaction_7, reaction_8,
                                reaction_9, reaction_10, reaction_11, reaction_12, reaction_13, reaction_14])

    state_record = [np.copy(state)]
    time_record = [time]
    index = np.array(range(0, 16))

    while time < max_time:

        propensities = calculate_propensity(state, pm)

        r_tot = sum(propensities)
        tau = - (1 / r_tot) * np.log(np.random.rand())

        idx = np.random.choice(index, p=propensities/r_tot)
        dy = reaction_matrix[idx]
        state += reaction_matrix[idx]
        time += tau

        state_record.append(np.copy(state))
        time_record.append(time)
    return state_record, time_record
