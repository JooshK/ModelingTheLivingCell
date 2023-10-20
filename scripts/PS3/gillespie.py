import random
import numpy as np


class Gillespie:
    def __init__(self, initial_state, reaction_matrix, propensity_func, max_time, M, start_t=0.0):
        self.start_time = start_t
        self.max_time = max_time
        self.propensity_func = propensity_func
        self.state = initial_state
        self.reaction_matrix = reaction_matrix
        self.M = M

        self.index = np.array([i for i in range(M)])
        self.x = [self.state[0]]
        self.y = [self.state[1]]
        self.time_record = [start_t]
        self.propensity_matrix = np.zeros(M)

    def run(self):
        time = self.start_time
        while time < self.max_time:
            for i in range(self.M):
                self.propensity_matrix[i] = self.propensity_func[i](self.state[0], self.state[1])

            r_tot = sum(self.propensity_matrix)
            tau = - (1 / r_tot) * np.log(np.random.rand())

            self.state += self.reaction_matrix[np.random.choice(self.index, p=self.propensity_matrix/r_tot)]
            time += tau

            self.x.append(self.state[0])
            self.y.append(self.state[1])
            self.time_record.append(time)
        return self.x, self.y, self.time_record