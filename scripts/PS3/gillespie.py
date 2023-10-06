import numpy as np


class GillespieSimulation:
    """
    Runs the Gillespie Algo for the stochastic calculation of reaction kinetics.
    Given some initial state, reaction description, channels and propensity functions, returns the matrix of sampled
    time and states.
    """
    def __init__(self, c, initial_state, reactions, m, h):
        self.c = c
        self.initial_state = initial_state
        self.reactions = reactions
        self.m = m
        self.h = h
        self.a0 = 0
        self.t = 0

    def run(self, n):
        states = [self.initial_state]
        t = [0]

        for i in range(n):
            x_i = states[i][0]
            y_i = states[i][1]
            a = []

            for rxn in range(self.m):
                a_i = self.h[rxn](x_i, y_i)*self.c[rxn]
                a.append(a_i)

            r_1 = np.random.random()
            r_2 = np.random.random()
            a0 = sum(a)

            tau = (1/a0)*np.log(1/r_1)
            mu = 0
            N = r_2*a0 - a[mu]
            while N > 0:
                mu += 1
                N -= a[mu]
            x = x_i + self.reactions[mu][0]
            y = y_i + self.reactions[mu][1]
            t.append(t[i] + tau)
            states.append([x, y])
        return np.array(t), np.array(states)
