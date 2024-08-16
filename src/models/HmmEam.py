import numpy as np

from bayesflow.simulation import Prior, Simulator, GenerativeModel
from scipy.special import logit, expit
from tensorflow import one_hot, expand_dims, concat
from scipy.stats import norm
from numba import njit

def truncated_normal(loc, scale, lower, upper, size):
    p1 = norm.cdf(lower, loc, scale)
    p2 = norm.cdf(upper, loc, scale)
    u = np.random.uniform(p1, p2, size)
    x = (scale * norm.ppf(u)) + loc
    return x

def positive_normal(loc, scale, size=1):
    return truncated_normal(loc, scale, lower=0, upper=np.inf, size=size)

@njit
def wald(alpha, nu, size=1):
    mu = alpha/nu
    lam = alpha**2
    zeta = np.random.standard_normal(size)
    zeta_sq = zeta**2
    x = mu + (mu**2*zeta_sq)/(2*lam) - mu/(2*lam)*np.sqrt(4*mu*lam*zeta_sq + mu**2*zeta_sq**2)
    z = np.random.uniform(0, 1, size=size)
    y = np.zeros(size)

    y[np.where(z <= mu / (mu + x))] = x
    y[np.where(z > mu / (mu + x))] = mu**2/x
    
    return y

def unconstrain_parameters(x, rts, axis=-1):
    x = np.copy(x)
    def unc(x):
        y = np.copy(x)
        y[0] = logit(y[0]) # rho_11
        y[1] = logit(y[1]) # rho_22
        y[2] = np.log(y[2]) # alpha_1
        y[3] = np.log(y[3] - x[2]) # alpha_2_diff
        y[4] = np.log(y[4]) # nu_1
        y[5] = np.log(y[5]) # nu_21
        y[6] = np.log(y[6] - x[5]) # nu_22_diff
        y[7] = logit(y[7]) # tau / min(rt)
        return y

    min_rts = np.min(rts, axis=-1)

    x[...,7] = x[...,7] / min_rts 
    return np.apply_along_axis(unc, axis=axis, arr=x)

def constrain_parameters(x, rts, axis=-1):
    x = np.copy(x)
    def con(x):
        y = np.copy(x)
        y[0] = expit(y[0]) # rho_11
        y[1] = expit(y[1]) # rho_22
        y[2] = np.exp(y[2]) # alpha_1
        y[3] = np.exp(y[3]) + y[2] # alpha_2
        y[4] = np.exp(y[4]) # nu_1
        y[5] = np.exp(y[5]) # nu_21
        y[6] = np.exp(y[6]) + y[5] # nu_22
        y[7] = expit(y[7]) # tau / min(rt)
        return y
    
    min_rts = np.min(rts, axis=-1)

    y = np.apply_along_axis(con, axis=axis, arr=x)
    y[...,7] = y[...,7] * min_rts # tau
    return y


def prior_fun():
    rho_11 = np.random.beta(10, 4)
    rho_22 = np.random.beta(10, 4)

    alpha_1 = positive_normal(0.5, 0.3) # guessing
    alpha_2_diff = positive_normal(1.5, 0.5)
    alpha_2 = alpha_1 + alpha_2_diff # controlled

    nu_1 = positive_normal(5.5, 1.0) # guessing

    nu_21 = positive_normal(2.5, 0.5) # controlled-incorrect
    nu_22_diff = positive_normal(2.5, 1.0) 
    nu_22 = nu_21 + nu_22_diff # controlled-correct

    tau = np.random.exponential(scale=0.2)

    pars = np.r_[rho_11, rho_22, alpha_1, alpha_2, nu_1, nu_21, nu_22, tau]

    # transform parameters into unconstrained real space
    return pars

prior = Prior(
    prior_fun=prior_fun, 
    param_names={
        "unconstrained": [r"$logit(\rho_{11})$", r"$logit(\rho_{22})$", 
                        r"$\log(\alpha_1)$", r"$\log(\alpha_2-\alpha_1)$",
                        r"$\log(\nu_1)$", r"$\log(\nu_{21})$", r"$\log(\nu_{22}-\nu_{21})$", 
                        r"$logit(\tau / min(rt))$"],
        "constrained": [r"$\rho_{11}$", r"$\rho_{22}$", 
                 r"$\alpha_1$", r"$\alpha_2$",
                 r"$\nu_1$", r"$\nu_{21}$", r"$\nu_{22}$", 
                 r"$\tau$"]
    })

def simulator_fun(theta):
    (rho_11, rho_22, alpha_1, alpha_2, nu_1, nu_21, nu_22, tau) = theta

    transition_matrix = [[rho_11, 1-rho_11], [1-rho_22, rho_22]]

    state = np.random.choice(a=[0, 1], p=[0.5, 0.5])

    n_obs = 400
    states = np.zeros(n_obs)
    rts = np.zeros(n_obs)
    responses = np.zeros(n_obs)
    for i in range(n_obs):
        if state == 0:
            rt = wald(alpha_1, nu_1)[0] + tau
            res = np.random.choice(a=[0,1], p=[0.5, 0.5])
        else:
            pass_time_1 = wald(alpha_2, nu_21)[0]
            pass_time_2 = wald(alpha_2, nu_22)[0]

            if pass_time_1 < pass_time_2:
                rt = pass_time_1 + tau
                res = 0
            else:
                rt = pass_time_2 + tau
                res = 1

        states[i] = state
        rts[i] = rt
        responses[i] = res

        state = np.random.choice(a=[0,1], p=transition_matrix[state])
        
    return np.c_[rts, responses, states]

simulator = Simulator(simulator_fun=simulator_fun)

model = GenerativeModel(prior=prior, simulator=simulator)


def configurator_posterior(input_dict):
    rts=input_dict["sim_data"][...,:1].astype(np.float32)
    responses=one_hot(input_dict["sim_data"][...,1], 2)

    parameters = input_dict["prior_draws"]
    parameters = unconstrain_parameters(parameters, rts[...,0]).astype(np.float32)

    return {
        "parameters": parameters,
        "summary_conditions": concat([rts, responses], axis=-1)
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