import numpy as np

from bayesflow.simulation import Prior, Simulator, ContextGenerator, GenerativeModel
from scipy.special import logit, expit
from tensorflow import one_hot, expand_dims

def prior_fun():
    a11 = np.random.beta(2, 2)
    a22 = np.random.beta(2, 2)

    loc = [-1,1]
    mu = np.random.normal(loc=loc)
    while mu[1] < mu[0]: # order constraint
        mu = np.random.normal(loc=loc)

    return np.r_[a11, a22, mu[0], mu[1]]

prior = Prior(prior_fun=prior_fun, param_names={
    "constrained": [r"$\alpha_{11}$", r"$\alpha_{22}$", r"$\mu_{1}$", r"$\mu_{2}$"],
    "unconstrained": [r"$logit(\alpha_{11})$", r"$logit(\alpha_{22})$", r"$\mu_1$", r"$\log(\mu_2 - \mu_1)$"]
    })

def simulator_fun(theta):
    a11, a22 = theta[:2]
    mu = theta[2:]

    a = [[a11, 1-a11], [1-a22, a22]]

    state = np.random.choice(a=[0, 1], p=[0.5, 0.5])
    states = []
    trials = []
    observations = []

    for _ in range(100):
        n = np.random.randint(low=2, high=6)
        y = np.random.normal(loc=mu[state], size=n)
        y = np.pad(y, (0, 5-n), 'constant', constant_values=-10)
        states.append(state)
        trials.append([1 if i < n else 0 for i in range(5)])
        observations.append(y)

        state = np.random.choice(a=[0,1], p=a[state])
        

    return np.c_[observations, trials, states]

simulator = Simulator(simulator_fun=simulator_fun)

model = GenerativeModel(prior=prior, simulator=simulator)


def constrain_parameters(parameters):
    parameters = np.array(parameters)
    parameters[..., 0] = expit(parameters[..., 0])
    parameters[..., 1] = expit(parameters[..., 1])
    parameters[..., 3] = parameters[..., 2] + np.exp(parameters[..., 3])

    return parameters

def unconstrain_parameters(parameters):
    parameters = np.array(parameters)
    parameters[..., 0] = logit(parameters[..., 0])
    parameters[..., 1] = logit(parameters[..., 1])
    parameters[..., 3] = np.log(parameters[..., 3] - parameters[..., 2])

    return parameters

def configurator_posterior(input_dict):
    parameters = unconstrain_parameters(input_dict["prior_draws"]).astype(np.float32)

    observables=input_dict["sim_data"][...,:-1].astype(np.float32)
    observables=np.reshape(observables, observables.shape[:-1] + (2, 5)).swapaxes(-1, -2)

    return {
        "parameters": parameters,
        "summary_conditions": observables
    }

def configurator_mixture(input_dict, output=None):
    if output is None:
        output=configurator_posterior(input_dict)
    else:
        output=output.copy()

    output["parameters"] = expand_dims(output["parameters"], axis=1)

    latents=input_dict["sim_data"][...,-1]
    latents=one_hot(latents, 2)
    latents=expand_dims(latents, axis=1)

    output["latents"] = latents

    return output

def configurator(input_dict):
    posterior_inputs = configurator_posterior(input_dict)
    return {
        "posterior_inputs": posterior_inputs,
        "mixture_inputs": configurator_mixture(input_dict, posterior_inputs)
    }