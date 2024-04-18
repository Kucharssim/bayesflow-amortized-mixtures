import numpy as np


def prior(dirichlet_shape=[1,1], scale=1.0, dim=1, ordered=True):
    num_states = len(dirichlet_shape)
    class_probs = np.random.dirichlet(dirichlet_shape)
    class_means = np.random.normal(size=(num_states, dim), scale=1.0)
    if ordered:
        class_means[:,0] = np.sort(class_means[:,0])
    return {
        'class_probabilities': class_probs,
        'class_parameters': class_means
    }

def simulator(theta, n):
    pi, mu = theta.values()
    z = np.random.choice(a=len(pi), size=n, replace=True, p=pi)
    mu = mu[z,:]
    y = np.random.normal(loc=mu, size=mu.shape)
    return {
        "latents": z,
        "observables": y
    }

def context():
    return np.random(50, 1001)



def configurator(input_dict, num_states):
    class_probabilities = np.array([a['class_probabilities'] for a in input_dict['prior_draws']])
    class_parameters = np.array([a['class_parameters'] for a in input_dict['prior_draws']])
    new_shape = input_dict['prior_draws'].shape[0], np.prod(class_parameters.shape[1:])
    class_parameters = class_parameters.reshape(new_shape)

    parameters = np.concatenate([class_probabilities, class_parameters])

    #parameters = np.array([np.concatenate([a['class_parameters'], a['class_probabilities']]) for a in input_dict['prior_draws']]).astype(np.float32)
    latents = [a['latents'] for a in input_dict['sim_data']]
    latents = tf.one_hot(np.array(latents), num_states)

    observables = np.array([a['observables'] for a in input_dict['sim_data']])[..., np.newaxis]

    return {
        "parameters": parameters,
        "latents": latents,
        "observables": observables
    }


# context = bf.simulation.ContextGenerator(non_batchable_context_fun=lambda: np.random.randint(50, 1000))
# model = bf.simulation.GenerativeModel(
#     prior=bf.simulation.Prior(prior_fun=normal_mixture.prior), 
#     simulator=bf.simulation.Simulator(simulator_fun=normal_mixture.simulator, context_generator=context)
#     )