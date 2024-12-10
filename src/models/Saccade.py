import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class Saccade():
    def __init__(
        self,
        target: float = 1.0,
        alpha: float = 100,
        beta: float = 0.1,
        cn: float = 0.01,
        sdn: float = 0.01,
        dt: float = 1
    ):
        self.target = target
        self.distance = self.target
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
        t, a, b, c = self.plan()

        self.run(t, a, b, c)

        return t, a, b, c

    def run(self, t, a, b, c):
        while self.t < t:
            self.update(a, b, c) 
    
    def plan(self):
        t = self.optim()
        a, b, c = self.solve_abc(t)

        return t, a, b, c

    def update(self, a, b, c):
        t = self.t + self.dt

        burst = a * np.exp(-c * t)

        cn = np.random.normal(scale=self.cn)
        sdn_a = np.random.normal(scale=(self.sdn * burst))
        sdn_b = np.random.normal(scale=(self.sdn * b))


        acc = (burst - b + sdn_a + sdn_b + cn) * self.dt

        self.velocity = self.velocity + acc
        self.position = self.position + self.velocity * self.dt
        self.t = t

        self.snapshot()  
    
    def peak_velocity_constraint(self):
        return self.alpha * np.log(1+self.beta * self.distance)

    def velocity_expectation(self, t, a, b, c):
        return a * (1 - np.exp(-c*t)) / c - b * t
    
    def velocity_variance(self, t, a, b, c):
        cn = self.cn**2 * t
        sdn_a = a**2 * (1 - np.exp(-2*c*t)) / (2 * c)
        sdn_b = (b*self.sdn)**2 * t

        return cn + sdn_a + sdn_b
    
    def velocity_peak(self, a, b, c):
        t = -(np.log(b) - np.log(a)) / c
        v = self.velocity_expectation(t, a, b, c)

        return t, v
    
    def position_expectation(self, t, a, b, c):
        return a * (np.exp(-c*t) + c*t - 1) / c**2 - b * t**2 / 2
    
    def position_variance(self, t, a, b, c):
        cn = (self.cn * t) ** 2 / 2
        sdn_a = self.sdn ** 2 * (np.exp(-c*t) + c*t - 1) / c**2
        sdn_b = (b * self.sdn * t) ** 2
        sdn = sdn_a + sdn_b

        return cn + sdn, cn, sdn

    def solve_a(self, t, c):
        num = self.distance
        expt = np.exp(-c*t)
        den1 = (expt + c*t - 1) / c**2
        den2 = -(t * (1-expt)) / (2*c)
        den = den1 + den2
        return np.max([num/den, -1000])

    def solve_b(self, t, a, c):
        num = a * (1 - np.exp(-c * t))
        den = c * t
        return np.max([num/den, -1000])

    def solve_c(self, a, b):
        ln = np.log(b) - np.log(a)
        return (a - a*ln - b*ln) / self.peak_velocity_constraint()
    
    def solve_ab(self, t, c):
        a = self.solve_a(t, c)
        b = self.solve_b(t, a, c)

        return a, b
    
    def solve_abc(self, t):
        c = 0.1

        for i in range(50):
            a, b = self.solve_ab(t, c)
            c = self.solve_c(a, b)

        a, b = self.solve_ab(t, c)
        
        return a, b, c

    
    def solve_position_variance(self, t):
        t=t[0]
        a, b, c = self.solve_abc(t)

        return self.position_variance(t, a, b, c)[0]
    
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
    alpha = 100
    beta = 2
    sdn=0.1
    cn=0.01
    amplitude = 100
    sac = Saccade(target=amplitude, dt = 0.1, alpha=alpha, beta=beta, sdn=sdn, cn=cn)
    durations = range(5, 150)
    cns = []
    sdns = []
    vars = []
    for t in durations:
        a, b, c = sac.solve_abc(t)
        p = sac.position_expectation(t, a, b, c)
        cs, c, s = sac.position_variance(t, a, b, c)
        cns.append(c)
        sdns.append(s)
        vars.append(cs)

    plt.plot(durations, cns, label="CN")
    plt.plot(durations, sdns, label="SDN")
    plt.plot(durations, vars, label="Total")
    plt.legend()
    plt.show()

    t=35
    sac = Saccade(target=amplitude, dt = 0.1, alpha=alpha, beta=beta,  sdn=sdn, cn=cn)
    a, b, c = sac.solve_abc(t)
    sac.run(t, a, b, c)
    time = np.array(sac.history['time'])

    fig, axs=plt.subplots(nrows=2)
    axs[0].plot(time, sac.history['position'], label='sim')
    axs[0].plot(time, sac.position_expectation(time, a, b, c), label='est')
    axs[1].plot(time, sac.history['velocity'])
    axs[1].plot(time, sac.velocity_expectation(time, a, b, c), label='est')
    axs[0].legend()
    plt.show()

    amplitudes = [20 + i * 20 for i in range(10)]
    durations = []
    peak_velocity = []
    pv = []
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
        pv.append(s.alpha * np.log(1 + s.beta * amplitude))
        t, a, b, c = s.plan()
        time = np.linspace(0, t, num=101)
        p = s.position_expectation(time, a, b, c)
        axs[0, 1].plot(time, p)

        v = s.velocity_expectation(time, a, b, c)
        axs[1, 1].plot(time, v)

        t_peak, v_peak = s.velocity_peak(a, b, c)

        axs[1, 1].scatter(t_peak, v_peak)
        
        durations.append(t)
        peak_velocity.append(v_peak)

    axs[0,0].plot(amplitudes, durations)
    axs[1,0].plot(amplitudes, peak_velocity)
    axs[1,0].plot(amplitudes, pv)

    fig.tight_layout()

    plt.show()
