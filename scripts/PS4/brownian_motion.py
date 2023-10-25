import numpy as np


class BrownianMotion:
    l = 1 / (np.tan(70 / 180 * np.pi)) ** 2
    rng = np.random.default_rng(123)

    def __init__(self, D, dt, kT, pos, iterations):
        self.iterations = iterations
        self.dt = dt
        self.D = D
        self.kT = kT

        self.x = pos[0]
        self.y = pos[1]

        self.potential = self.bi_stable_V()
        self.trajectory = [pos]

    def bi_stable_V(self):
        return 5 * (self.x ** 2 - 1) ** 2 + 10 * (self.y ** 2) * self.l

    def dV_dx(self):
        return 20 * self.x * (self.x ** 2 - 1)

    def dV_dy(self):
        return 20 * self.l * self.y

    def update_position_x(self):
        return self.x - (self.dt * self.D / self.kT) * self.dV_dx() + \
            np.sqrt(2 * self.D * self.dt) * self.rng.standard_normal()

    def update_position_y(self):
        return self.y - (self.dt * self.D / self.kT) * self.dV_dy() + \
            np.sqrt(2 * self.D * self.dt) * self.rng.standard_normal()

    def run(self):
        for i in range(self.iterations):
            self.x = self.update_position_x()
            self.y = self.update_position_y()

            self.trajectory.append([self.x, self.y])
            self.potential = self.bi_stable_V()

        return self.trajectory