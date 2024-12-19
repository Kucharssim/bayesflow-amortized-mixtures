import bayesflow as bf
import numpy as np
from scipy.stats import beta as beta_dist
from tensorflow import one_hot
from copy import deepcopy


def saccade_velocity_expectation(t, amplitude, duration, alpha, beta):
    x = t / duration
    return amplitude / duration * beta_dist.pdf(x, alpha, beta)

def simulator_fn(theta, dt=2, max_time=5_000):
    drift, tremor, threshold, drift_rate, eta, gamma, alpha, beta, sigma = theta

    num_obs = max_time // dt
    positions = np.zeros((num_obs, 2))
    events = np.zeros((num_obs, 1))
    time = 0.0

    position = np.random.uniform(low=np.zeros(2), high=np.array([1024,  768]), size=2)
    target = position
    event = 0
    event_start_time = time
    i = 0

    # variables for fixations
    evidence = 0.0

    # constants for saccade generation
    mode_beta = (alpha - 1) / (alpha + beta - 2)
    duration_scaling = beta_dist.pdf(mode_beta, alpha, beta)

    while time < max_time:
        positions[i,:] = position 
        events[i,0] = event

        if event == 0: # fixation
            evidence += np.random.normal(loc=drift_rate) * np.sqrt(dt)
            position = position + drift * (target - position) * dt + np.random.normal(size=2, scale=tremor) * np.sqrt(dt)
            if evidence > threshold:
                event = 1
                event_start_time = time
                target = np.random.uniform(low=np.zeros(2), high=np.array([1024,  768]), size=2)
                direction = target - position
                amplitude = np.linalg.norm(direction)
                direction = direction / amplitude
                w = np.zeros(2)
                main_sequence = eta * (1 - np.exp(-amplitude / gamma))
                duration_expectation = duration_scaling * amplitude / main_sequence
        else: # saccade
            event_time = time - event_start_time
            w = w - 0.1 * w * dt + np.random.normal(size=2, scale=sigma) * np.sqrt(dt)
            v = saccade_velocity_expectation(event_time, amplitude, duration_expectation, alpha, beta) * dt + w
            position = position + v * direction

            if np.dot(direction, v) < 0 or event_time > 2 * duration_expectation:
                event = 0
                event_start_time = time
                evidence = 0.0


        i += 1
        time += dt

    return np.concatenate([positions, events], axis=-1)

simulator = bf.simulation.Simulator(simulator_fun=simulator_fn)


def prior_fn():
    drift = np.random.gamma(shape=5, scale=0.05) # change from scale=0.01 to scale=0.05
    tremor = np.random.gamma(shape=2, scale=0.5) # change from gamma(5, 0.5) to gamma(2, 0.5) 
    threshold = np.random.gamma(shape=20, scale=2.5)
    drift_rate = np.random.gamma(shape=15, scale=0.01)

    eta = np.random.gamma(shape=5, scale=4)
    gamma = np.random.gamma(shape=5, scale=20)
    alpha = np.random.gamma(shape=2, scale=1) + 1
    beta = np.random.gamma(shape=2, scale=2) + 1
    sigma = np.random.gamma(shape=5, scale=0.1)

    return np.r_[drift, tremor, threshold, drift_rate, eta, gamma, alpha, beta, sigma]

prior = bf.simulation.Prior(prior_fun=prior_fn, param_names=["drift", "tremor", "threshold", "drift_rate", "eta", "gamma", "alpha", "beta", "sigma"])

model = bf.simulation.GenerativeModel(prior=prior, simulator=simulator, skip_test=True)

def unconstrain_parameters(theta):
    theta = np.copy(theta)
    return np.log(theta)

def constrain_parameters(theta):
    theta = np.copy(theta)
    return np.exp(theta)


def configurator_posterior(input_dict):
    parameters = unconstrain_parameters(input_dict["prior_draws"][...,:4]).astype(np.float32)
    position = input_dict["sim_data"][...,:2]
    position = np.clip(position, a_min=np.zeros(2), a_max=np.array([1024,  768]))

    batch_size, num_timesteps, data_dim = position.shape
    time_encoding = np.linspace(0, 1, num_timesteps)
    time_encoding_batched = np.tile(time_encoding[np.newaxis, :, np.newaxis], (batch_size, 1, 1))

    return {
        "parameters": parameters,
        "summary_conditions": np.concatenate((position, time_encoding_batched), axis=-1).astype(np.float32)
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



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    #theta = [0.2, 1, 50, 0.15, 20, 100, 2.5, 5, 0.5]
    #data = simulator_fn(theta)
    data = model(1)['sim_data'][0]
    plt.plot(data[:,0])
    plt.plot(data[:,1])
    plt.plot(data[:,2] * 100)
    plt.show()

