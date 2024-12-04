import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class Saccade():
    def __init__(
        self,
        target: float = 1.0,
        alpha: float = 0.01,
        beta: float = 0.005,
        cn: float = 0.001,
        sdn: float = 0.001,
        dt: float = 1
    ):
        self.target = target
        self.alpha = alpha
        self.beta = beta
        self.sdn = sdn
        self.cn = cn
        self.dt = dt

        self.position = 0.0
        self.velocity = 0.0
        self.t = 0.0

        self.history = {"position": [], "velocity": [], "event": [], "time": []}

    def __call__(self):
        t, a, b = self.plan()

        self.run(t, a, b)

        return t, a, b

    def run(self, t, a, b):
        while self.t < t:
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

        cn = np.random.normal(scale=self.cn)
        sdn_a = np.random.normal(scale=(self.sdn * burst))
        sdn_b = np.random.normal(scale=(self.sdn * b))


        acc = (burst - b - drag + cn + sdn_a + sdn_b) * self.dt

        self.velocity = self.velocity + acc
        self.position = self.position + self.velocity * self.dt
        self.t = t

        self.snapshot()  

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
        

    def snapshot(self, event: str = "saccade"):
        self.history['position'].append(self.position)
        self.history['velocity'].append(self.velocity)
        self.history['time'].append(self.t)
        self.history['event'].append(event)


if __name__ == "__main__":
    alpha = 0.1
    beta = 0.06
    sdn=0.1
    cn=1.0
    amplitude = 50
    sac = Saccade(target=amplitude, dt = 0.1, alpha=alpha, beta=beta, sdn=sdn, cn=cn)
    durations = range(5, 150)
    cns = []
    sdns = []
    vars = []
    for t in durations:
        a, b = sac.solve_ab(t)
        p = sac.position_expectation(t, a, b)
        cs, c, s = sac.position_variance(t, a, b)
        cns.append(c)
        sdns.append(s)
        vars.append(cs)

    plt.plot(durations, cns, label="CN")
    plt.plot(durations, sdns, label="SDN")
    plt.plot(durations, vars, label="Total")
    plt.legend()
    plt.show()

    t=55
    sac = Saccade(target=amplitude, dt = 0.1, alpha=alpha, beta=beta,  sdn=sdn, cn=cn)
    a, b = sac.solve_ab(t)
    sac.run(t, a, b)
    time = np.array(sac.history['time'])

    fig, axs=plt.subplots(nrows=2)
    axs[0].plot(time, sac.history['position'], label='sim')
    axs[0].plot(time, sac.position_expectation(time, a, b), label='est')
    axs[1].plot(time, sac.history['velocity'])
    axs[1].plot(time, sac.velocity_expectation(time, a, b), label='est')
    axs[0].legend()
    plt.show()

    amplitudes = [20 + i * 20 for i in range(10)]
    durations = []
    peak_velocity = []
    fig, axs=plt.subplots(nrows=2, ncols=2)
    axs[0,0].set_xlabel("Amplitude")
    axs[0,0].set_ylabel("Duration (ms)")
    axs[1,0].set_xlabel("Amplitude")
    axs[1,0].set_ylabel("Peak velocity")
    axs[0,1].set_xlabel("Time/Duration (ms)")
    axs[0,1].set_ylabel("Amplitude")
    axs[1,1].set_xlabel("Time/Duration (ms)")
    axs[1,1].set_ylabel("Velocity")
    for i, amplitude in enumerate(amplitudes):
        s = Saccade(target = amplitude, dt = 0.1, alpha=alpha, beta=beta, sdn=sdn, cn=sdn)
        t, a, b = s.plan()
        time = np.linspace(0, t, num=101)
        p = s.position_expectation(time, a, b)
        axs[0, 1].plot(time, p)

        v = s.velocity_expectation(time, a, b)
        axs[1, 1].plot(time, v)

        t_peak, v_peak = s.velocity_peak(a, b)

        axs[1, 1].scatter(t_peak, v_peak)
        
        durations.append(t)
        peak_velocity.append(v_peak)

    axs[0,0].plot(amplitudes, durations)
    axs[1,0].plot(amplitudes, peak_velocity)

    fig.tight_layout()

    plt.show()
