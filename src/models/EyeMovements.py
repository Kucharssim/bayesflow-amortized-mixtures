
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from bayesflow.simulation import Prior, Simulator, GenerativeModel
from tensorflow import one_hot
from copy import deepcopy

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
    def __init__(self, position: np.ndarray, target: np.ndarray, drift: float, tremor: float, threshold: float, drift_rate: float, dt=1.0):
        super().__init__(position=position, dt = dt)
        self.target = target

        self.drift = drift
        self.tremor = tremor
        self.threshold = threshold
        self.drift_rate = drift_rate

    def __call__(self):
        accumulator = 0.0
        while accumulator < self.threshold:
            accumulator = accumulator + np.random.normal(loc=self.drift_rate) * self.dt
            self.update()
            self.snapshot(0)

    def update(self):
        drift = self.drift * (self.target - self.position)
        tremor = np.random.normal(size=2)

        self.velocity = (drift + tremor)
        self.position = (self.position) + self.velocity * self.dt
        self.time = self.time + self.dt

class Saccade(Updateable):
    def __init__(
            self, 
            position: np.ndarray, 
            target: np.ndarray, 
            alpha: float, 
            cn: float, 
            sdn: float, 
            dt: float = 1.0
            ):
        
        self.reset(position)
        self.target = target
        self.direction = target - position
        self.distance = np.linalg.norm(self.direction)
        self.direction = self.direction / self.distance

        self.alpha = alpha
        self.sdn = sdn
        self.cn = cn

        self.time = 0.0
        self.dt = dt

    def __call__(self):
        t, a, b = self.plan()

        self.run(t, a, b)

        return t, a, b

    def run(self, t, a, b):
        while np.dot(self.direction, self.velocity) >= 0:
            self.update(a, b) 
    
    def plan(self):
        # determine optimal saccade duration
        t = self.optim()
        # determine optimal agonist and antagonist forces
        a, b = self.solve_ab(t)

        return t, a, b
    
    def update(self, a, b):
        t = self.time + self.dt
        self.time = t

        burst = a * np.exp(-self.alpha * t)

        cn = np.random.normal(size=2, scale=self.cn)
        sdn_a = np.random.normal(size=2, scale=(self.sdn * burst))
        sdn_b = np.random.normal(size=2, scale=(self.sdn * b))
            
        acc = (burst - b + cn + sdn_a + sdn_b) * self.dt

        self.velocity = self.velocity + acc * self.direction
        self.position = self.position + self.velocity * self.dt

        self.snapshot(event=1)  

    def velocity_expectation(self, t, a, b):
        return a * (1 - np.exp(-self.alpha*t)) / self.alpha - b * t
    
    def velocity_peak(self, a, b):
        t = -(np.log(b) - np.log(a)) / self.alpha
        v = self.velocity_expectation(t, a, b)

        return t, v
    
    def position_expectation(self, t, a, b):
        return a * (np.exp(-self.alpha*t) + self.alpha*t - 1) / self.alpha**2 - b * t**2 / 2
    
    def position_variance(self, t, a, b):
        cn = (self.cn * t) ** 2
        sdn_a = self.sdn ** 2 * (a * np.exp(-self.alpha*t) + self.alpha*t - 1) / self.alpha**2
        sdn_b = (b * self.sdn * t) ** 2
        sdn = sdn_a + sdn_b

        return cn + sdn, cn, sdn

    def solve_a(self, t):
        num = self.distance
        expt = np.exp(-self.alpha*t)
        den1 = (expt + self.alpha*t - 1) / self.alpha**2
        den2 = -(t * (1-expt)) / (2*self.alpha)
        den = den1 + den2
        return num/den

    def solve_b(self, t, a):
        num = a * (1 - np.exp(-self.alpha * t))
        den = self.alpha * t
        return num/den
    
    def solve_ab(self, t):
        a = self.solve_a(t)
        b = self.solve_b(t, a)

        return a, b
    
    def saccade_duration_loss(self, t):
        t=t[0]
        a, b = self.solve_ab(t)

        return self.position_variance(t, a, b)[0]
    
    def optim(self):
        result = minimize(
            fun=self.saccade_duration_loss, 
            x0=[20.0],
            bounds=[(1, None)])
        
        return result.x[0]
    

class Eye(Updateable):
    def __init__(self, fixation_params: dict, saccade_params: dict, dt = 1, max_time=5_000):

        self.display_min = np.zeros(2)
        self.display_max = np.array([1024,  768])

        self.position = np.random.uniform(low = self.display_min, high = self.display_max)
        self.history = {"position": [], "event": []}
        self.fixation_params = fixation_params
        self.saccade_params = saccade_params
        self.dt = dt
        self.time = 0.0
        self.max_time = max_time

    def __call__(self):
        target = self.position
        while self.time <= self.max_time:
            fixation = Fixation(position=self.position, target=target, dt=self.dt, **self.fixation_params)
            fixation()
            self.position = fixation.position

            target = np.random.uniform(low = self.display_min, high = self.display_max)
            saccade = Saccade(position=self.position, target=target, dt=self.dt, **self.saccade_params)
            saccade()
            self.position = saccade.position
            self.time = self.time + fixation.time + saccade.time
            self.snapshot(fixation, saccade)
        
        position = self.history['position']
        position = np.vstack(position)

        event = self.history['event']
        event = np.concatenate(event)
        return position, event


    def snapshot(self, fixation: Fixation, saccade: Saccade):
        fp = fixation.history["position"]
        self.history['position'].append(np.array(fp))
        fe = fixation.history['event']
        self.history['event'].append(np.array(fe))
        sp = saccade.history["position"]
        self.history['position'].append(np.array(sp))
        se = saccade.history['event']
        self.history['event'].append(np.array(se))




if __name__ == "__main__":
    fixation_params = {"drift": 0.01, "tremor": 25.0, "threshold": 1000, "drift_rate": 3}
    saccade_params = {"alpha": 0.1, "cn": 0.01, "sdn": 0.01}


    eye = Eye(fixation_params=fixation_params, saccade_params=saccade_params, dt=1.0)
    position, event = eye()

    print(position.shape)
    print(event.shape)

    plt.plot(position)
    plt.show()



def prior_fun():
    drift = np.random.gamma(shape=5, scale=0.1)
    tremor = np.random.gamma(shape=5, scale=5)
    threshold = np.random.gamma(shape=20, scale=2.5)
    drift_rate = np.random.gamma(shape=15, scale=0.01)

    alpha = np.random.gamma(shape=3, scale=0.03)
    cn    = np.random.gamma(shape=2, scale=0.2)
    sdn   = np.random.gamma(shape=2, scale=0.2)


    pars = np.r_[drift, tremor, threshold, drift_rate, alpha, cn, sdn]

    return pars

prior = Prior(prior_fun=prior_fun)

def unconstrain_parameters(theta):
    theta = np.copy(theta)
    return np.log(theta)

def constrain_parameters(theta):
    theta = np.copy(theta)
    return np.exp(theta)

def simulator_fun(theta):
    drift, tremor, threshold, drift_rate, alpha, cn, sdn = theta

    eye = Eye(
        fixation_params = {"drift": drift, "tremor": tremor, "threshold": threshold, "drift_rate": drift_rate},
        saccade_params  = {"alpha": alpha, "cn": cn, "sdn": sdn}
    )

    position, event = eye()

    position = position[0:5000:2]
    event = event[0:5000:2]

    return np.column_stack((position, event))


simulator = Simulator(simulator_fun=simulator_fun)

model = GenerativeModel(prior=prior, simulator=simulator, skip_test=True)

def configurator_posterior(input_dict):
    parameters = unconstrain_parameters(input_dict["prior_draws"]).astype(np.float32)
    position = input_dict["sim_data"][...,:2].astype(np.float32)

    return {
        "parameters": parameters,
        "summary_conditions": position
    }

def configurator_mixture(input_dict, posterior_dict=None):
    if posterior_dict is None:
        output = configurator_posterior(input_dict)
    else:
        output = deepcopy(posterior_dict)

    output["parameters"] = np.expand_dims(output["parameters"], axis=1)

    latents = input_dict["sim_data"][...,-1]
    latents = one_hot(latents, 2)
    latents = np.expand_dims(latents, axis=1)

    output["latents"] = latents

    return output
    

def configurator(input_dict):
    posterior_dict = configurator_posterior(input_dict)
    return {
        "posterior_inputs": posterior_dict,
        "mixture_inputs": configurator_mixture(input_dict, posterior_dict)
    }
