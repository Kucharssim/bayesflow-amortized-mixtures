
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import levy_stable


class Eye():
    def __init__(
        self, position: np.ndarray = np.zeros(2), 
        dt=0.001, drift=500, tremor=500, 
        threshold=2, drift_rate=6,
        display_min: np.ndarray = np.zeros(2),
        display_max: np.ndarray = np.array([1024,  768])
        ):
        self.position = position
        self.velocity = np.zeros(2)
        self.dt = dt

        self.drift = drift
        self.tremor = tremor
        self.threshold = threshold
        self.drift_rate = drift_rate

        self.display_min = np.zeros(2)
        self.display_max = np.array([1024,  768])

        self.history = {"position": [], "velocity": [], "event": []}

    def __str__(self):
        return "Eye with position: {}, velocity: {}".format(self.position, self.velocity)

    def saccade(self):
        target = np.random.uniform(self.display_min, self.display_max)

        accumulator = 0.0
        while accumulator < self.threshold:
            accumulator = accumulator + np.random.normal(loc=self.drift_rate * self.dt, scale = np.sqrt(self.dt))
            self.fixation_update(target)
            self.snapshot("saccade")

    def saccade_update(self, target: np.ndarray):
        pass

    def fixation(self, target: np.ndarray):
        accumulator = 0.0
        while accumulator < self.threshold:
            accumulator = accumulator + np.random.normal(loc=self.drift_rate * self.dt, scale = np.sqrt(self.dt))
            self.fixation_update(target)
            self.snapshot("fixation")
        
    def fixation_update(self, target: np.ndarray):
        drift = self.drift * (target - self.position)
        tremor = np.random.normal(size=2, scale=1)

        self.velocity = (drift + tremor) * self.dt
        self.position = self.position + self.velocity

    def snapshot(self, event: str):
        self.history['position'].append(self.position)
        self.history['velocity'].append(self.velocity)
        self.history['event'].append(event)




eye=Eye()
print(eye)
target = np.array([0, 0])
eye.fixation(target)
print(eye)

positions = np.array(eye.history['position'])
velocities = np.array(eye.history['velocity'])
fig,axs = plt.subplots(nrows=3)
axs[0].plot(positions[:,0], positions[:,1], alpha=0.7)
axs[1].plot(positions[:,0])
axs[1].plot(positions[:,1])
axs[2].plot(velocities[:,0])
axs[2].plot(velocities[:,1])
plt.show()