import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist


class Saccade():
    def __init__(self, position, target, eta, gamma, alpha, beta, sigma):
        self.eta = eta
        self.gamma = gamma 
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.position = position
        self.amplitude = np.abs(target - position)
        self.direction = 1 if target > position else -1
        self._duration = None
        self._max_velocity = None

    def __call__(self, dt=2):
        t = 0.0
        velocity = 0.0
        position = self.position
        positions = []
        times = []
        W = 0.0
        while not (velocity < 0 and t >= self.max_velocity_time):
            W += np.random.normal(scale=self.sigma) * np.sqrt(dt)
            velocity = self.velocity_expectation(t) * dt + W
            t += dt
            position = position + velocity
            positions.append(position)
            times.append(t)

        positions = np.array(positions)
        times = np.array(times)
        return positions, times

    def position_expectation(self, t):
        x = t / self.duration
        return self.amplitude * beta_dist.cdf(x, self.alpha, self.beta)
    
    def velocity_expectation(self, t):
        x = t / self.duration
        return self.amplitude / self.duration * beta_dist.pdf(x, self.alpha, self.beta)
    
    @property
    def main_sequence(self,):
        return self.eta * (1 - np.exp(-self.amplitude / self.gamma))
    
    @property
    def max_velocity_time_proportion(self):
        return (self.alpha - 1) / (self.alpha + self.beta - 2)
    
    @property
    def max_velocity_time(self):
        return self.duration * self.max_velocity_time_proportion

    @property
    def max_velocity(self):
        t = self.max_velocity_time
        return self.velocity_expectation(t)
    
    @property
    def duration(self):
        term1 = self.amplitude / self.main_sequence
        term2 = beta_dist.pdf(self.max_velocity_time_proportion, self.alpha, self.beta)
        return term1 * term2

if __name__ == "__main__":
    # Parameters
    A = 300   # Saccadic amplitude
    eta = 20  # Peak velocity
    gamma = 100   # Main sequence scaling factor
    alpha = 2.5 # shape of velocity profile
    beta = 5 # shape of velocity profile
    sigma = 0.1 # noise intensity

    s = Saccade(position=0, target=A, eta=eta, gamma=gamma, alpha=alpha, beta=beta, sigma=sigma)

    pos, tim = s()

    # # Time vector
    t = np.linspace(0, s.duration, 1000)

    # Plotting the results
    fig, axs = plt.subplots(2, 1)

    # Plot position
    axs[0].plot(tim, pos, label="Noisy")
    axs[0].plot(t, s.position_expectation(t), label="Expectation")
    axs[0].set_title("Saccadic Position")
    axs[0].set_xlabel("Time (ms)")
    axs[0].set_ylabel("Position (pix)")
    axs[0].grid(True)
    axs[0].legend()

    # Plot velocity
    axs[1].plot(t, s.velocity_expectation(t), label="Saccadic Velocity", color="orange")
    axs[1].set_title("Saccadic Velocity")
    axs[1].set_xlabel("Time (ms)")
    axs[1].set_ylabel("Velocity (pix/ms)")
    axs[1].grid(True)
    axs[1].legend()
    plt.tight_layout()


    fig, axs = plt.subplots(3, 1)

    As = [(i + 1)*10 for i in range(60)]
    Ts = []
    Vs = []


    for A in As:
        s = Saccade(position=0, target=A, eta=eta, gamma=gamma, alpha=alpha, beta=beta, sigma=sigma)
        Ts.append(s.duration)
        Vs.append(s.max_velocity)

    Ts = np.array(Ts)
    Vs = np.array(Vs)

    axs[0].plot(As, Ts)
    axs[0].set_xlabel("Amplitude (pix)")
    axs[0].set_ylabel("Duration (ms)")
    axs[0].grid(True)
    axs[1].plot(As, Vs)
    axs[1].set_xlabel("Amplitude (pix)")
    axs[1].set_ylabel("Peak velocity (pix/ms)")
    axs[1].grid(True)

    As = [(i+1) * 100 for i in range(6)]

    for A in As:
        s = Saccade(position=0, target=A, eta=eta, gamma=gamma, alpha=alpha, beta=beta, sigma=sigma)
        t = np.linspace(0, s.duration, num=101)
        v = s.velocity_expectation(t)
        axs[2].plot(t, v, label=A)
    axs[2].set_xlabel("Time (ms)")
    axs[2].set_ylabel("Velocity (pix/ms)")
    axs[2].grid(True)
    axs[2].legend()

    plt.tight_layout()
    plt.show()

