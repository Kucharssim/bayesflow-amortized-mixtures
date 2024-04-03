import tensorflow as tf
import numpy as np
from bayesflow.amortizers import *
from bayesflow.simulation import Prior, Simulator

def prior(prior_fun):
    def pr():
        prior = prior_fun()
        return prior
    
    return Prior(prior_fun=pr)

def simulator(simulator_fun):
    def sf(theta, *args, **kwargs):
        sim = simulator_fun(theta, *args, **kwargs)
        return sim

    return Simulator(simulator_fun=sf)


class AmortizedMixture(tf.keras.Model, AmortizedTarget):
    def __init__(
            self,
            inference_net,
            summary_net=None,
            loss_fun=None,
            **kwargs
    ):
        tf.keras.Model.__init__(self, **kwargs)

        self.inference_net = inference_net
        self.summary_net = summary_net
        self.loss = self._determine_loss(loss_fun)
        self.num_states = inference_net.num_states


    def call(self, input_dict, **kwargs):
        pass



class AmortizedPosteriorLikelihoodMixture(tf.keras.Model, AmortizedTarget):
    def __init__(
        self,
        amortized_posterior=None,
        amortized_likelihood_components=None,
        **kwargs
        ):
        tf.keras.Model.__init__(self, **kwargs)

        self.amortized_posterior = amortized_posterior
        self.amortized_likelihood_components = amortized_likelihood_components
    
    def call(self, input_dict, **kwargs):

        post_out = self.amortized_posterior(input_dict["posterior_inputs"], **kwargs)
        liks_out = [a(y, **kwargs) for a, y in zip(self.amortized_likelihood_components, input_dict[DEFAULT_KEYS["likelihood_inputs"]])]

        return post_out, liks_out
    
    def compute_loss(self, input_dict, **kwargs):
        loss_post = self.amortized_posterior.compute_loss(input_dict[DEFAULT_KEYS["posterior_inputs"]], **kwargs)
        loss_liks = [a.compute_loss(y) for a, y in zip(self.amortized_likelihood_components, input_dict[DEFAULT_KEYS["likelihood_inputs"]])]
        return {"Post.Loss": loss_post, "Liks.loss": sum(loss_liks)}
    def loss(self, a, y):
        loss = []
        for yy in y:
            loss.append(a.compute_loss(yy))
        return loss
    def log_likelihood(self, input_dict, to_numpy=True, **kwargs):
        pass
    def log_posterior(self, input_dict, to_numpy=True, **kwargs):
        pass
    def log_prob(self, input_dict, to_numpy=True, **kwargs):
        log_post = self.log_posterior(input_dict, to_numpy=to_numpy, **kwargs)
        log_liks = self.log_likelihood(input_dict, to_numpy=to_numpy, **kwargs)
        out_dict = {"log_posterior": log_post, "log_likelihood": log_liks}
        return out_dict
    def sample_data(self, input, n_samples, to_numpy=True, **kwargs):
        if input.get(DEFAULT_KEYS["likelihood_inputs"]) is not None:
            input = input[DEFAULT_KEYS["likelihood_inputs"]]
        
        #output = [a.sample(y, n_samples, to_numpy=to_numpy, **kwargs) for a, y in zip(self.amortized_likelihood_components, input)]
        output = [y for a, y in zip(self.amortized_likelihood_components, input)]
        return output
    def sample_parameters(self, input_dict, n_samples, to_numpy=True, **kwargs):
        if input_dict.get(DEFAULT_KEYS["posterior_inputs"]) is not None:
            return self.amortized_posterior.sample(
                input_dict[DEFAULT_KEYS["posterior_inputs"]], n_samples, to_numpy=to_numpy, **kwargs
            )
        return self.amortized_posterior.sample(input_dict, n_samples, to_numpy=to_numpy, **kwargs)
    def sample(self, input_dict, n_post_samples, n_lik_samples, to_numpy=True, **kwargs):
        """Identical to calling `sample_parameters()` and `sample_data()` separately.

        Returns
        -------
        out_dict : dict with keys `posterior_samples` and `likelihood_samples` corresponding
        to the `n_samples` from the approximate posterior and likelihood, respectively
        """

        post_samples = self.sample_parameters(input_dict, n_post_samples, to_numpy=to_numpy, **kwargs)
        lik_samples = self.sample_data(input_dict, n_lik_samples, to_numpy=to_numpy, **kwargs)
        out_dict = {"posterior_samples": post_samples, "likelihood_samples": lik_samples}
        return out_dict

class DefaultJointMixtureConfigurator:
    def __init__(self, num_states = 2, default_float_type=np.float32):
        self.num_states = num_states
        self.default_float_type = default_float_type

    def __call__(self, forward_dict):
        pi = {}
        pi['parameters'] = self.extract_parameters(forward_dict).astype(self.default_float_type)
        pi['summary_conditions'] = self.extract_observables(forward_dict).astype(self.default_float_type)

        li = self.extract_likelihood_input(forward_dict)

        return {'posterior_inputs': pi, 'likelihood_inputs': li}
    @staticmethod
    def extract_parameters(forward_dict):
        prior_draws = forward_dict['prior_draws']
        parameters = []
        for b in range(prior_draws.shape[0]):
            pars = []
            for v in prior_draws[b].values():
                pars = pars + v.tolist()
            parameters.append(pars)
        return np.array(parameters)
    @staticmethod
    def extract_observables(forward_dict):
        sim_data = forward_dict['sim_data']
        observables = [sim['observables'] for sim in sim_data]
        return np.array(observables)
    @staticmethod
    def extract_latents(forward_dict):
        sim_data = forward_dict['sim_data']
        latents = [sim['latents'] for sim in sim_data]
        return np.array(latents)
    def extract_likelihood_input(self, forward_dict):
        observables = self.extract_observables(forward_dict)
        latents     = self.extract_latents(forward_dict)
        conditions  = forward_dict['prior_draws']

        likelihood_input = []
        for c in range(self.num_classes):
            obs = []
            con = []
            for b in range(observables.shape[0]):
                for i in range(latents.shape[1]):
                    if latents[b, i] == c:
                        obs.append(np.atleast_1d(observables[b,i,]))
                par = conditions[b]['class_parameters'][c]
                con.append(np.atleast_1d(par))

            likelihood_input.append({
                'observables': obs,#np.array(obs).astype(self.default_float_type),
                'conditions': con#np.array(con).astype(self.default_float_type)
            })
        return likelihood_input