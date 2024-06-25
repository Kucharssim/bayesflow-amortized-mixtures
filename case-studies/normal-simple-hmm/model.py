import numpy as np

from bayesflow.simulation import Prior, Simulator, GenerativeModel
from scipy.special import logit, expit
from tensorflow import one_hot

def prior_fun():
    a11 = np.random.beta(2, 2)
    a22 = np.random.beta(2, 2)

    
    mu = np.random.normal(loc=[-1.5, 1.5])
    while mu[1] < mu[0]: # order constraint
        mu = np.random.normal(loc=[-1.5, 1.5])

    # transform parameters into unconstrained real space
    return np.r_[logit(a11), logit(a22), mu[0], np.log(mu[1] - mu[0])]

prior = Prior(prior_fun=prior_fun, param_names=[r"$logit(\alpha_{11})$", r"$logit(\alpha_{22})$", r"$\mu_1$", r"$\log(\mu_2 - \mu_1)$"])


def simulator_fun(theta):
    a = [
        [expit(theta[0]), 1-expit(theta[0])],
        [1-expit(theta[1]), expit(theta[1])]
    ]
    mu = [theta[2], np.exp(theta[3]) + theta[2]]

    state = np.random.choice(a=[0, 1], p=[0.5, 0.5])
    states = []
    observations = []

    for _ in range(100):
        y = np.random.normal(loc=mu[state])
        states.append(state)
        observations.append(y)

        state = np.random.choice(a=[0,1], p=a[state])
        

    return np.c_[observations, states]


simulator = Simulator(simulator_fun=simulator_fun)

model = GenerativeModel(prior=prior, simulator=simulator)


def constrain_parameters(parameters):
    parameters = np.array(parameters)
    parameters[..., 0] = expit(parameters[..., 0])
    parameters[..., 1] = expit(parameters[..., 1])
    parameters[..., 3] = parameters[..., 2] + np.exp(parameters[..., 3])

    return parameters

constrained_parameter_names = [r"$\alpha_{11}$", r"$\alpha_{22}$", r"$\mu_1$", r"$\mu_2$"]

def configurator(input_dict):
    output_dict = {
        "parameters": input_dict["prior_draws"].astype(np.float32),
        "observables": input_dict["sim_data"][...,:-1].astype(np.float32),
        "latents": one_hot(input_dict["sim_data"][...,-1], 2)
    }

    return output_dict



def posterior_configurator(input_dict):
    output_dict = configurator(input_dict)
    output_dict["summary_conditions"] = output_dict["observables"]

    return output_dict