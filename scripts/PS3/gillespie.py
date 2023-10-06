import numpy as np
import random


def reaction_combinations(state):
    h0 = 1
    h1 = state[0]*state[1]
    h2 = state[0]
    h3 = state[0]
    return [h0, h1, h2, h3]


class GillespieSimulation:
    def update_propensity(self, propensities, state, c):
        """
        Updates propensity matrix
        :param propensities: matrix of props
        :param state: Current State
        :param c: Parameters
        """
        x, y = state
        c0, c1, c2, c3 = c

        propensities[0] = 1*c0
        propensities[1] = x*y*c1
        propensities[2] = x*c2
        propensities[3] = x*c3


    def mu(self, probs):
        r2 = np.random.rand()
        i = 0
        cdf = 0.0

        while cdf < r2:
            cdf += probs[i]
            i += 1
        return i - 1

    def gillespie_update(self, propensity):
        a0 = sum(propensity)
        tau = np.random.exponential(1.0/a0)  # equivalent but faster version of ln(URN)
        reaction_probabilities = propensity/a0

        mu = self.mu(reaction_probabilities)
        return mu, tau

    def run(self, reactions, initial_propensity, initial_state, time, c):
        states = np.empty((len(time), reactions.shape[1]), dtype=int)

        time_index = 0
        i = 0
        t = time[0]
        state = initial_state.copy()
        states[0, :] = state
        propensity = initial_propensity

        while i < len(time):

            while t < time[time_index]:
                reaction, tau = self.gillespie_update(propensity)
                state += reactions[reaction, :]
                t += tau
                self.update_propensity(propensity, state, c)

            i = np.searchsorted(time > t, True)
            states[time_index:min(i, len(time))] = state
            time_index = i
        return states




