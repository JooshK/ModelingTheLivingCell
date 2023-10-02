import numpy as np
import random


def reaction_combinations(state):
    h0 = 1
    h1 = state[0]*state[1]
    h2 = state[0]
    h3 = state[0]
    return [h0, h1, h2, h3]


class GillespieSimulation:
    def __init__(self, c, initial_state, reactions):
        self.c = c
        self.initial_state = initial_state
        self.reactions = reactions
        self.a = []
        self.a0 = 0
        self.t = 0

    def generate_propensity(self, state):
        for i, c_i in enumerate(self.c):
            h = reaction_combinations(state)
            a_i = h[i]*c_i
            self.a.append(a_i)
        self.a0 = sum(self.a)

    def sample_reaction(self, states):
        r1 = random.random()
        r2 = random.random()
        self.generate_propensity(states)

        tau = 1 / self.a0 * np.log(1 / r1)
        mu = 0
        n = r2*self.a0 - self.a[mu]
        while n > 0:
            mu += 1
            n -= self.a[mu]

        return tau, mu

    def run(self, n):
        state = [self.initial_state]
        for i in range(n):
            print(state)
            tau, mu = self.sample_reaction(state[i])

            self.t += tau
            r_mu = self.reactions[mu]
            new_state = []
            for species, coefficient in enumerate(r_mu):
                new_state.append(state[i][species] + coefficient)
            state.append(new_state)

        return state


