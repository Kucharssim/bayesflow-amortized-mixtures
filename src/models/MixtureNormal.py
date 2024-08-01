import numpy as np

from bayesflow.simulation import Prior, ContextGenerator, Simulator, GenerativeModel
from tensorflow import one_hot, expand_dims
from ..constraints import simplex, ordered

def ordered_normal(loc):
    output = np.random.normal(loc)
    while np.any(np.sort(output) != output):
        output = np.random.normal(loc)
    return np.array(output)

def unconstrain_parameters(x):
    x = np.copy(x)
    pi = simplex.unconstrain(x[..., :3], axis=-1)
    mu  = ordered.unconstrain(x[..., 3:], axis=-1)

    return np.concatenate([pi, mu], axis=-1)

def constrain_parameters(x):
    x = np.copy(x)
    pi = simplex.constrain(x[..., :2], axis=-1)
    mu  = ordered.constrain(x[..., 2:], axis=-1)

    return np.concatenate([pi, mu], axis=-1)

def prior_fun():
    pi = np.random.dirichlet([2, 2, 2])
    mu = ordered_normal([-2.0, 0.0, 2.0])

    pars = np.r_[pi, mu]

    # transform parameters into unconstrained real space
    #return unconstrain_parameters(pars)

    return pars

param_names = {
    "constrained": [r"$\pi_1$", r"$\pi_2$", r"$\pi_3$", r"$\mu_1$", r"$\mu_2$", r"$\mu_3$"],
    "unconstrained": [r"$\text{logit}(\pi_1)$", r"$\text{logit}\left(\frac{\pi_2 - \pi_1}{1-\pi_1}\right)$", r"$\log(\mu_1)$", r"$\log(\mu_2-\mu_1)$", r"$\log(\mu_3-\mu_2)$"]
}

prior = Prior(prior_fun=prior_fun, param_names=param_names)

constrained_parameter_names = [r"$\pi_1$", r"$\pi_2$", r"$\pi_3$", r"$\mu_1$", r"$\mu_2$", r"$\mu_3$"]

def context_fun():
    n_obs = np.random.randint(low=100, high=301)
    n_rep = np.random.randint(low=1, high=6)

    return np.r_[n_obs, n_rep]

context = ContextGenerator(non_batchable_context_fun=context_fun)


def simulator_fun(theta, context):
    batch_size = theta.shape[0]
    #theta = constrain_parameters(theta)
    pi = theta[...,:3]
    mu = theta[...,3:]

    n_obs = context[...,0]
    n_rep = context[...,1]

    # generate latent mixture indicators `z`
    # z = np.random.choice(range(3), size=(batch_size, n_obs), replace=True, p=pi)
    eta = np.random.random_sample((batch_size, n_obs, 1))
    cumulative_pi = np.cumsum(pi, axis=-1)
    cumulative_pi = np.expand_dims(cumulative_pi, axis=1)
    cumulative_pi = np.tile(cumulative_pi, reps=[1, n_obs, 1])

    z = (eta < cumulative_pi).argmax(axis=-1)

    # generate data `y`
    mus = mu[np.arange(batch_size)[:, None], z]
    mus = np.expand_dims(mus, axis=-1)
    mus = np.tile(mus, reps=[1, 1, n_rep])
    y = np.random.normal(loc=mus, scale=1)
    z = np.expand_dims(z, axis=-1)

    return np.concatenate([y, z], axis=-1)

simulator = Simulator(batch_simulator_fun=simulator_fun, context_generator=context)

model = GenerativeModel(prior=prior, simulator=simulator)

def modelFixedContext(n_obs, n_rep):
    def cnt():
        return np.r_[n_obs, n_rep]
    con = ContextGenerator(non_batchable_context_fun=cnt)

    sim = Simulator(batch_simulator_fun=simulator_fun, context_generator=con)

    return GenerativeModel(prior=prior, simulator=sim)


def configurator_posterior(input_dict):
    parameters = input_dict["prior_draws"]

    y = input_dict["sim_data"][...,:-1].astype(np.float32)
    y = np.expand_dims(y, axis=-1)

    context = input_dict["sim_non_batchable_context"]
    context = np.expand_dims(context, axis=0)
    context = np.tile(context, reps = (parameters.shape[0], 1))
    context = context.astype(np.float32)

    return {
        "parameters": unconstrain_parameters(parameters).astype(np.float32),
        "summary_conditions": y,
        "direct_conditions": context
    }

def configurator_mixture(input_dict, output=None):
    if output is None:
        output=configurator_posterior(input_dict)
    else:
        output=output.copy()

    output["parameters"] = expand_dims(output["parameters"], axis=1)

    latents=input_dict["sim_data"][...,-1]
    latents=one_hot(latents, 3)
    latents=expand_dims(latents, axis=1)

    output["latents"] = latents

    return output


def configurator(input_dict):
    post = configurator_posterior(input_dict)
    return {
        "posterior_inputs": post,
        "mixture_inputs": configurator_mixture(input_dict, post)
    }


def generate_fixed_dataset(n_obs, n_rep, mu):
    n_total = np.sum(n_obs)
    y = []

    for i, n in enumerate(n_obs):
        x = np.random.standard_normal(size=(n, n_rep))
        x = (x - np.mean(x)) / np.std(x)
        x = mu[i] + x
        y.append(x)

    y = np.concatenate(y, axis=0)
    
    latents = np.zeros((n_total, 1))

    y = np.concatenate([y, latents], axis=-1)


    output = {
        "prior_draws": np.zeros((1, 5)),
        "sim_data": np.expand_dims(y, axis=0),
        "sim_non_batchable_context": np.array([n_total, n_rep]),
    }

    return configurator(output)