
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class Updateable():
    def __init__(self, position: np.ndarray, dt=1.0):
        self.dt = dt
        self.reset(position)
    def update(self):
        raise NotImplementedError
    def snapshot(self, event: str):
        self.history['time'].append(self.time)
        self.history['position'].append(self.position)
        self.history['velocity'].append(self.velocity)
        self.history['event'].append(event)
    def reset(self, position: np.ndarray):
        self.time = 0.0
        self.position = position
        self.velocity = np.zeros(2)
        self.history = {"position": [], "velocity": [], "event": [], "time": []}

class Fixation(Updateable):
    def __init__(self, position: np.ndarray, target: np.ndarray, eye_drift: float, eye_tremor: float, dur_threshold: float, dur_drift: float, dt=1.0):
        super().__init__(dt = dt)
        self.position = position
        self.target = target

        self.eye_drift = eye_drift
        self.eye_tremor = eye_tremor
        self.dur_threshold = dur_threshold
        self.dur_drift = dur_drift

    def __call__(self):
        accumulator = 0.0
        while accumulator < self.threshold:
            accumulator = accumulator + np.random.normal(loc=self.drift_rate) * self.dt
            self.update()
            self.snapshot("fixation")

    def update(self):
        drift = self.eye_drift * (self.target - self.position)
        tremor = np.random.normal(size=2)

        self.velocity = (drift + tremor) * self.dt
        self.position = (self.position) + self.velocity

class Saccade(Updateable):
    def __init__(
            self, 
            position: np.ndarray, 
            target: np.ndarray, 
            alpha: float, 
            beta: float,
            cn: float, 
            sdn: float, 
            dt: float = 1.0
            ):
        
        self.reset(position)
        self.target_position = target
        self.diff = target - position
        self.target = np.linalg.norm(self.diff)
        self.direction = self.diff / self.target

        self.alpha = alpha
        self.beta = beta
        self.sdn = sdn
        self.cn = cn

        self.time = 0.0
        self.t = 0.0
        self.dt = dt

    def __call__(self):
        t, a, b = self.plan()

        self.run(t, a, b)

        return t, a, b

    def run(self, t, a, b):
        #while self.t < t:
        while np.dot(self.direction, self.velocity) >= 0:
            self.update(a, b) 
    
    def plan(self):
        t = self.optim()
        a, b = self.solve_ab(t)

        return t, a, b

    def burst(self, t, a):
        return a * np.exp(-self.alpha * t)
    
    def update(self, a, b):
        t = self.t + self.dt

        burst = self.burst(t, a)
        drag = self.beta * self.velocity

        cn = np.random.normal(size=2, scale=self.cn)
        sdn_a = np.random.normal(size=2, scale=(self.sdn * burst))
        sdn_b = np.random.normal(size=2, scale=(self.sdn * b))


        acc = (burst - b - drag + cn + sdn_a + sdn_b) * self.dt

        self.velocity = self.velocity + acc * self.direction
        self.position = self.position + self.velocity * self.dt
        self.t = t
        self.time = t

        self.snapshot(event="saccade")  

    def velocity_expectation(self, t, a, b):
        return (a / (self.beta - self.alpha)) * (np.exp(-self.alpha * t) - np.exp(-self.beta * t)) - \
                    (b / self.beta) * (1 - np.exp(-self.beta * t))
    
    def velocity_variance(self, t, a, b):
        cn = self.cn**2 * t
        sdn_a = a**2 * (1 - np.exp(-2*self.alpha*t)) / (2 * self.alpha)
        sdn_b = (b*self.sdn)**2 * t

        return cn + sdn_a + sdn_b
    
    def velocity_peak(self, a, b):
        ratio = (a * self.beta + self.alpha * b - b * self.beta) / (a * self.alpha)
        t = -np.log(ratio) / (self.alpha - self.beta)
        v = self.velocity_expectation(t, a, b)

        return t, v
    
    def position_expectation(self, t, a, b):
        return (-a / (self.alpha * (self.beta - self.alpha)) * np.exp(-self.alpha * t) +
                     a / (self.beta * (self.beta - self.alpha)) * np.exp(-self.beta * t) -
                     b / self.beta * t -
                     b / self.beta**2 * np.exp(-self.beta * t) +
                     a / (self.alpha * (self.beta - self.alpha)) -
                     a / (self.beta * (self.beta - self.alpha)) +
                     b / self.beta**2)
    
    def position_variance(self, t, a, b):
        cn = self.cn ** 2 * t**3 / 3
        cn = self.cn ** 2 * t

        bracket = t**2 - t / self.alpha + (1 - np.exp(-2*self.alpha*t))/(2*self.alpha**2)
        sdn_a = a**2 * bracket / (2 * self.alpha)
        sdn_a = (a + self.sdn)**2 * t**3 / 3
        
        sdn_b = (b * self.sdn)**2 * t**3 / 3

        sdn = sdn_a + sdn_b

        return cn + sdn, cn, sdn

    def solve_a(self, t):
        numerator = self.alpha * self.beta * self.target * (-self.alpha * np.exp(self.beta * t) + self.alpha + self.beta * np.exp(self.beta * t) - self.beta) * np.exp(self.alpha * t)
        denominator = (
            self.alpha * self.beta * t * np.exp(self.alpha * t) - self.alpha * self.beta * t * np.exp(self.beta * t) +
            self.alpha * np.exp(self.alpha * t) + self.alpha * np.exp(self.beta * t) -
            self.alpha * np.exp(t * (self.alpha + self.beta)) - self.alpha -
            self.beta * np.exp(self.alpha * t) - self.beta * np.exp(self.beta * t) +
            self.beta * np.exp(t * (self.alpha + self.beta)) + self.beta
        )
        return numerator / denominator

    def solve_b(self, t, a):
        numerator = a * self.beta * (np.exp(-self.alpha * t) - np.exp(-self.beta * t))
        denominator = (self.beta - self.alpha) * (1 - np.exp(-self.beta * t))
        return numerator / denominator
    
    def solve_ab(self, t):
        a = self.solve_a(t)
        b = self.solve_b(t, a)

        return a, b
    
    def solve_position_variance(self, t):
        a, b = self.solve_ab(t)

        return self.position_variance(t, a, b)[0]
    
    def optim(self):
        result = minimize(
            fun=self.solve_position_variance, 
            x0=[20.0],
            bounds=[(1, None)])
        
        return result.x[0]
    

class Eye(Updateable):
    def __init__(self, position: np.ndarray, fixation_params: dict, saccade_params: dict, dt = 1):
        super().__init__(dt = dt)
        self.position = position

        self.display_min = np.zeros(2)
        self.display_max = np.array([1024,  768])

        self.fixation_params = fixation_params
        self.saccade_params = saccade_params


if __name__ == "__main__":
    saccade = Saccade(position=np.zeros(2), target=np.array([150, 70]), alpha=0.1, beta=0.05, sdn=0.5, cn=0.5)
    t, a, b = saccade()

    position = np.array(saccade.history['position'])
    time = np.array(saccade.history['time'])
    plt.plot(time, position)
    plt.show()