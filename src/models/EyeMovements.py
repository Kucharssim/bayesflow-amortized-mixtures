
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



# Example usage
if __name__ == "__main__" and False:
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




import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Parameters
target = 50.0  # Target position
sigma_sdn = 0.01  # signal dependent noise coefficient
sigma_cn = 0.01 # constant noise coefficient
alpha = 1.0   # Exponential decay rate
lam = 5.0 # weight of speed preference

dt = 0.001

def burst(t, A, alpha):
    return A * np.exp(-alpha * t)

def acc(t, A, alpha, beta):
    return burst(t, A, alpha) - beta

def vel(t, A, alpha, beta):
    return burst(t, A, alpha) / (-alpha) - beta * t + A / alpha

def t_vel_zero(A, alpha, beta):
    t0 = A / (beta + A)
    t1 = t0 + 1
    while np.abs(t1 - t0) > 0.001:
        t2 = (A/alpha) / beta * (1 - np.exp(-alpha*t0))
        t0, t1 = t1, t2
    return t1

def pos(t, A, alpha, beta):
    return  burst(t, A, alpha) / (alpha ** 2) - t**2 * beta / 2 + A / alpha * t - A / (alpha**2)

def var_pos_sdn(t, A, alpha, sigma):
    first_term = sigma ** 2 * A ** 2 / (2 * alpha)
    bracket = t**2 - t/alpha + (1 - np.exp(-2*alpha*t))/(2*alpha**2)

    return first_term * bracket

def var_pos_cn(t, beta, sigma):
    return sigma**2 * beta**2 * t**3 / 3

def var_pos(t, A, alpha, beta, sigma_sdn, sigma_cn):
    return var_pos_sdn(t, A, alpha, sigma_sdn) + var_pos_cn(t, beta, sigma_cn)

def std_pos(t, A, alpha, beta, sigma_sdn, sigma_cn):
    return np.sqrt(var_pos(t, A, alpha, beta, sigma_sdn, sigma_cn))

def bias_pos(t, A, alpha, beta, target):
    x = pos(t, A, alpha, beta)

    return (target - x)**2

def loss(params):
    A, beta = params
    #A = params
    T = t_vel_zero(A, alpha, beta)
    b = bias_pos(T, A, alpha, beta, target)
    v = var_pos(T, A, alpha, beta, sigma_sdn, sigma_cn)

    return b + v + lam * T

from scipy.optimize import minimize

initial = [15, 2]

result = minimize(loss, initial, bounds=[(0.01, 1000), (0.1, 100)], method="L-BFGS-B")
A, beta = result.x
T = t_vel_zero(A, alpha, beta)
# result = minimize(loss, initial, bounds=[(0.01, 1000)])
# A = result.x[0]

print("A: ", A, "beta: ", beta, "T:", T)
print("bias:", bias_pos(T, A, alpha, beta, target))
print("var:", var_pos(T, A, alpha, beta, sigma_sdn, sigma_cn))
print("lambda * T:", lam * T)


positions=[]
velocities=[]


time = [i * dt for i in range(int(T / dt))]

for repetition in range(100):
    position = [0.0]
    velocity = [0.0]
    for t in time:
        F = A * np.exp(-alpha * t)
        nF = np.random.normal(scale=np.sqrt((F * sigma_sdn)**2 * dt))
        nB = np.random.normal(scale=np.sqrt((beta * sigma_cn)**2 * dt))
        a = F - beta
        v = velocity[-1] + a * dt + nF + nB
        velocity.append(v)
        position.append(position[-1] + v * dt)
    positions.append(position)
    velocities.append(velocity)

positions = np.array(positions)
velocities = np.array(velocities)
var_positions = np.var(positions, axis=0)

fig, axs=plt.subplots(nrows=3)
axs[0].plot(positions.transpose(), c="gray", alpha=0.1)
axs[0].plot(pos(np.array(time), A, alpha, beta))
axs[0].axhline(target)
axs[1].plot(velocities.transpose(), c="gray", alpha=0.1)
axs[1].plot(vel(np.array(time), A, alpha, beta))
axs[2].plot(var_positions)
axs[2].plot(var_pos(np.array(time), A, alpha, beta, sigma_sdn, sigma_cn))

plt.show()

# As = np.linspace(1, 100, 21)
# betas = np.linspace(0.1, 10, 21)

# ts = np.zeros((21, 21))
# bs = np.zeros((21, 21))
# vs = np.zeros((21, 21))

# for i, A in enumerate(As):
#     for j, beta in enumerate(betas):
#         Ti = t_vel_zero(A, alpha, beta)
#         bi = bias_pos(T, A, alpha, beta, target)
#         vi = var_pos(T, A, alpha, beta, sigma_sdn, sigma_cn)

#         ts[i,j] = Ti
#         bs[i,j] = bi
#         vs[i,j] = vi

# As, betas = np.meshgrid(As, betas)

# fig,ax = plt.subplots(subplot_kw={"projection": "3d"})

# ax.plot_surface(As, betas, bs+vs+ts)

# # plt.plot(ts,np.array(bs))
# # plt.plot(ts,np.array(vs))
# # plt.plot(ts,np.array(bs) + np.array(vs))

# plt.show()