
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from bayesflow.simulation import Prior, Simulator, GenerativeModel


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
        super().__init__(position=position, dt = dt)
        self.target = target

        self.eye_drift = eye_drift
        self.eye_tremor = eye_tremor
        self.dur_threshold = dur_threshold
        self.dur_drift = dur_drift

    def __call__(self):
        accumulator = 0.0
        while accumulator < self.dur_threshold:
            accumulator = accumulator + np.random.normal(loc=self.dur_drift) * self.dt
            self.update()
            self.snapshot("fixation")

    def update(self):
        drift = self.eye_drift * (self.target - self.position)
        tremor = np.random.normal(size=2)

        self.velocity = (drift + tremor) * self.dt
        self.position = (self.position) + self.velocity
        self.time = self.time + self.dt

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
        #drag = np.zeros(2)

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
    def __init__(self, fixation_params: dict, saccade_params: dict, dt = 1):

        self.display_min = np.zeros(2)
        self.display_max = np.array([1024,  768])

        self.position = np.random.uniform(low = self.display_min, high = self.display_max)
        self.history = {"position": []}
        self.fixation_params = fixation_params
        self.saccade_params = saccade_params
        self.dt = dt
        self.time = 0.0

    def __call__(self, max_time=10_000):
        target = self.position
        while self.time <= max_time:
            print("new round, time", self.time)
            fixation = Fixation(position=self.position, target=target, dt=self.dt, **self.fixation_params)
            fixation()
            self.position = fixation.position
            print("fixation done")
            target = np.random.uniform(low = self.display_min, high = self.display_max)
            saccade = Saccade(position=self.position, target=target, dt=self.dt, **self.saccade_params)
            saccade()
            self.position = saccade.position
            print("saccade done")
            self.time = self.time + fixation.time + saccade.time
            self.snapshot(fixation, saccade)


    def snapshot(self, fixation: Fixation, saccade: Saccade):
        fp = fixation.history["position"]
        self.history['position'].append(np.array(fp))
        sp = saccade.history["position"]
        self.history['position'].append(np.array(sp))




if __name__ == "__main__":
    fixation_params = {"eye_drift": 0.1, "eye_tremor": 25.0, "dur_threshold": 1000, "dur_drift": 3}
    saccade_params = {"alpha": 0.2, "beta": 0.01, "cn": 0.1, "sdn": 0.2}


    # saccade = Saccade(position=np.array([348.0685267, 729.03394094]), target=np.array([727.58391991, 470.95124244]), **saccade_params)
    # t, a, b = saccade()

    # position = np.array(saccade.history['position'])
    # time = np.array(saccade.history['time'])
    # plt.plot(time, position)
    # plt.show()


    eye = Eye(fixation_params=fixation_params, saccade_params=saccade_params)
    eye(max_time=5_000)
    position = np.vstack(eye.history['position'])
    print(position.shape)

    plt.plot(position)
    plt.show()



def prior_fun():
    drift = np.random.lognormal(mean=0.05, sigma=1)
    tremor = np.random.lognormal(mean=5, sigma=1)
    threshold = np.random.lognormal(mean=100, sigma=5)
    drift_rate = np.random.lognormal(mean=2, sigma=0.5)

    alpha = np.random.gamma(shape=5, scale=0.1)
    beta  = np.random.gamma(shape=3, scale=0.003)
    cn    = np.random.gamma(shape=2, scale=0.2)
    sdn   = np.random.gamma(shape=2, scale=0.2)


    pars = np.r_[drift, tremor, threshold, drift_rate, alpha, beta, cn, sdn]

    return pars


prior = Prior(prior_fun=prior_fun)

def simulator_fun(theta):
    drift, tremor, threshold, drift_rate, alpha, beta, cn, sdn = theta

    eye = Eye(
        fixation_params = {"eye_drift": drift, "eye_tremor": tremor, "dur_threshold": threshold, "dur_drift": drift_rate},
        saccade_params  = {"alpha": alpha, "beta": beta, "cn": cn, "sdn": sdn}
    )

    eye(max_time=10_000)

    position = np.vstack(eye.history['position'])

    return position


simulator = Simulator(simulator_fun=simulator_fun)


model = GenerativeModel(prior=prior, simulator=simulator, skip_test=True)