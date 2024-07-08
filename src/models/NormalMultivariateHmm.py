import numpy as np

from bayesflow.simulation import Prior, Simulator, GenerativeModel
from scipy.special import logit, expit
from tensorflow import one_hot, expand_dims, reverse

def prior_fun():
    a11 = np.random.beta(2, 2)
    a22 = np.random.beta(2, 2)

    
    mu = np.random.normal(loc=[-0.5, -0.5])
    while mu[1] < mu[0]: # order constraint
        mu = np.random.normal(loc=[-0.5, 0.5])

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
    trials = []
    observations = []


    for _ in range(100):
        n = np.random.randint(low=1, high=6)
        y = np.random.normal(loc=mu[state], size=n)
        y = np.pad(y, (0, 5-n), 'constant', constant_values=-10)
        states.append(state)
        trials.append([1 if i < n else 0 for i in range(5)])
        observations.append(y)

        state = np.random.choice(a=[0,1], p=a[state])
        

    return np.c_[observations, trials, states]


simulator = Simulator(simulator_fun=simulator_fun)

model = GenerativeModel(prior=prior, simulator=simulator)


latex_model = r"""
\begin{equation}
\begin{aligned}
\alpha_k & \sim \text{Dirichlet}\Big((2, 2)\Big) & k \in \{1, 2\} \\
(\mu_1, \mu_2) & \sim \text{Normal}\Big((-1.5, 1.5), \mathbb{I} \Big)_{\mu_1 < \mu_2} \\
z_{1} & \sim \text{Categorical}\Big((0.5, 0.5)\Big) \\
z_{t} & \sim \text{Categorical}(\alpha_{z_{t-1}}) & t \in \{ 2, \dots, \text{T} \}\\
y_{t} & \sim \text{Normal}(\mu_{z_{t}}, 1) & t \in \{ 1, \dots, \text{T} \}
\end{aligned}
\end{equation}
"""


def constrain_parameters(parameters):
    parameters = np.array(parameters)
    parameters[..., 0] = expit(parameters[..., 0])
    parameters[..., 1] = expit(parameters[..., 1])
    parameters[..., 3] = parameters[..., 2] + np.exp(parameters[..., 3])

    return parameters

constrained_parameter_names = [r"$\alpha_{11}$", r"$\alpha_{22}$", r"$\mu_1$", r"$\mu_2$"]


def configurator_posterior(input_dict):
    parameters = input_dict["prior_draws"].astype(np.float32)

    observables=input_dict["sim_data"][...,:-1].astype(np.float32)
    observables=np.reshape(observables, observables.shape[:-1] + (2, 5)).swapaxes(-1, -2)

    return {
        "parameters": parameters,
        "summary_conditions": observables
    }

def configurator_mixture(input_dict):
    output = configurator_posterior(input_dict)

    output["parameters"] = expand_dims(output["parameters"], axis=1)

    latents=input_dict["sim_data"][...,-1]
    latents=one_hot(latents, 2)
    latents=expand_dims(latents, axis=1)

    output["latents"] = latents

    return output


def configurator(input_dict):
    return {
        "posterior_inputs": configurator_posterior(input_dict),
        "mixture_inputs": configurator_mixture(input_dict)
    }