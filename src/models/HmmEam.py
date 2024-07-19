import numpy as np

from bayesflow.simulation import Prior, Simulator, GenerativeModel
from scipy.special import logit, expit
from tensorflow import one_hot, expand_dims, concat
from scipy.stats import norm

def truncated_normal(loc, scale, lower, upper, size):
    p1 = norm.cdf(lower, loc, scale)
    p2 = norm.cdf(upper, loc, scale)
    u = np.random.uniform(p1, p2, size)
    x = (scale * norm.ppf(u)) + loc
    return x

def wald(alpha, nu, size=1):
    mu = alpha/nu
    lam = alpha**2
    zeta = np.random.standard_normal(size)
    zeta_sq = zeta**2
    x = mu + (mu**2*zeta_sq)/(2*lam) - mu/(2*lam)*np.sqrt(4*mu*lam*zeta_sq + mu**2*zeta_sq**2);
    z = np.random.uniform(0, 1, size=size);
    y = np.zeros(size)

    y[np.where(z <= mu / (mu + x))] = x
    y[np.where(z > mu / (mu + x))] = mu**2/x
    
    return y

def unconstrain_parameters(x, axis=-1):
    def unc(x):
        y = np.copy(x)
        y[0] = logit(y[0])
        y[1] = logit(y[1])
        y[2] = np.log(y[2])
        y[3] = np.log(y[3] - x[2])
        y[4] = np.log(y[4])
        y[5] = np.log(y[5])
        y[6] = np.log(y[6] - x[5])
        y[7] = np.log(y[7])
        return y

    return np.apply_along_axis(unc, axis=axis, arr=x)

def constrain_parameters(x, axis=-1):
    def con(x):
        y = np.copy(x)
        y[0] = expit(y[0])
        y[1] = expit(y[1])
        y[2] = np.exp(y[2])
        y[3] = np.exp(y[3]) + y[2]
        y[4] = np.exp(y[4])
        y[5] = np.exp(y[5])
        y[6] = np.exp(y[6]) + y[5]
        y[7] = np.exp(y[7])
        return y

    return np.apply_along_axis(con, axis=axis, arr=x)


def prior_fun():
    r11 = np.random.beta(2, 2)
    r22 = np.random.beta(2, 2)

    a1 = truncated_normal(1.0, 0.5, 0.0, np.inf, 1) # guessing
    a2 = a1 + truncated_normal(0.5, 0.5, 0.0, np.inf, 1) # controlled

    v1 = truncated_normal(2.0, 0.5, 0.0, np.inf, 1) # guessing

    v21 = truncated_normal(0.5, 0.5, 0.0, np.inf, 1) # controlled-incorrect
    v22 = v21 + truncated_normal(1.0, 0.5, 0.0, np.inf, 1) # controlled-correct

    tau = np.random.exponential(scale=0.2)

    pars = np.r_[r11, r22, a1, a2, v1, v21, v22, tau]

    # transform parameters into unconstrained real space
    return unconstrain_parameters(pars)

prior = Prior(
    prior_fun=prior_fun, 
    param_names=[r"$logit(\rho_{11})$", r"$logit(\rho_{22})$", 
                 r"$\log(\alpha_1)$", r"$\log(\alpha_2-\alpha1)$",
                 r"$\log(\nu_1)$", r"$\log(\nu_{21})$", r"$\log(\nu_{22})$", 
                 r"$\log(\tau)$"])

constrained_parameter_names = [r"$\rho_{11}$", r"$\rho_{22}$", 
                 r"$\alpha_1$", r"$\alpha_2$",
                 r"$\nu_1$", r"$\nu_{21}$", r"$\nu_{22}$", 
                 r"$\tau$"]

def simulator_fun(theta):
    (r11, r22, a1, a2, v1, v21, v22, tau) = constrain_parameters(theta)

    transition_matrix = [[r11, 1-r11], [1-r22, r22]]

    state = np.random.choice(a=[0, 1], p=[0.5, 0.5])
    states = []
    rts = []
    responses = []


    for _ in range(200):
        if state == 0:
            rt = wald(a1, v1)[0] + tau
            res = np.random.choice(a=[0,1], p=[0.5, 0.5])
        else:
            passage_times = [wald(a2, v21), wald(a2, v22)]
            rt = np.min(passage_times) + tau
            res = np.argmin(passage_times)
        
        states.append(state)
        rts.append(rt)
        responses.append(res)

        state = np.random.choice(a=[0,1], p=transition_matrix[state])
        
    return np.c_[rts, responses, states]

simulator = Simulator(simulator_fun=simulator_fun)

model = GenerativeModel(prior=prior, simulator=simulator)




def configurator_posterior(input_dict):
    parameters = input_dict["prior_draws"].astype(np.float32)

    rts=input_dict["sim_data"][...,:1].astype(np.float32)
    responses=one_hot(input_dict["sim_data"][...,1], 2)

    return {
        "parameters": parameters,
        "summary_conditions": expand_dims(concat([rts, responses], axis=-1), axis=-2)
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