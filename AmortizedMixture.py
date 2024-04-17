import tensorflow as tf
import numpy as np
from bayesflow.amortizers import *
from bayesflow.simulation import Prior, Simulator
from bayesflow.helper_networks import MCDropout
from bayesflow.losses import log_loss


def _ensure_tensor(tensor):
    if not tf.is_tensor(tensor):
        return tf.convert_to_tensor(tensor)
    
    return tensor
def merge_first_two_dims(tensor):
    tensor = _ensure_tensor(tensor)
    shape = tensor.shape.as_list()
    shape[0] *= shape[1]
    shape.pop(1)
    return tf.reshape(tensor, shape)

def split_first_two_dims(tensor, dim_0, dim_1):
    tensor = _ensure_tensor(tensor)
    shape = tensor.shape.as_list()
    new_shape = [dim_0] + [dim_1] + shape[1:]
    return tf.reshape(tensor, new_shape)

def expand_tensor(tensor, dim_1):
    tensor = _ensure_tensor(tensor)
    shape = tensor.shape.as_list()
    tensor = tf.expand_dims(tensor, axis=1)
    
    multiples = [1, dim_1]
    rest = [1 for _ in shape[1:]]
    tensor = tf.tile(tensor, multiples=multiples + rest)

    return tensor

class IndependentClassificator(tf.keras.Model):
    """
    Outputs the distribution p(s | y, θ) = ∏ p(s_i | y_i, θ)
    """
    def __init__(
        self,
        num_outputs,
        dense_args=dict(units=64, activation="relu"),
        num_dense=3,
        dropout=True,
        mc_dropout=False,
        dropout_prob=0.05,
        output_activation=tf.nn.softmax,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Sequential model with optional (MC) Dropout
        self.net = tf.keras.Sequential()
        for _ in range(num_dense):
            self.net.add(tf.keras.layers.Dense(**dense_args))
            if mc_dropout:
                self.net.add(MCDropout(dropout_prob))
            elif dropout:
                self.net.add(tf.keras.layers.Dropout(dropout_prob))
            else:
                pass
        self.output_layer = tf.keras.layers.Dense(num_outputs)
        self.output_activation = output_activation
        self.num_outputs = num_outputs

    def call(self, observables, conditions, return_probs=True, **kwargs):
        """
        Parameters
        ----------
        observables : tf.Tensor of shape (batch_size, num_obs, ...)
        conditions : tf.Tensor of shape (batch_size, ...)

        Returns
        -------
        out: tf.Tensor of shape (batch_size, num_obs, num_outputs)
        """
        
        batch_size, num_obs, _ = observables.shape
        long_observables = merge_first_two_dims(observables)
        long_conditions = merge_first_two_dims(expand_tensor(conditions, num_obs))
        cond = tf.concat([long_observables, long_conditions], axis = -1)

        rep = self.net(cond)
        out = self.output_layer(rep, **kwargs)
        if return_probs:
            out = self.output_activation(out)
        
        out = split_first_two_dims(out, batch_size, num_obs)

        return out

class AmortizedMixture(tf.keras.Model, AmortizedTarget):
    """
    Infers p(s | y) = ∫ p(s, θ | y) dθ
    """
    def __init__(
            self,
            inference_net,
            local_summary_net=None,
            global_summary_net=None,
            loss_fun=log_loss,
            **kwargs
    ):
        tf.keras.Model.__init__(self, **kwargs)

        self.inference_net = inference_net
        self.local_summary_net = local_summary_net
        self.global_summary_net = global_summary_net
        self.loss = loss_fun

        self.is_conditional = global_summary_net is None


    def call(self, input_dict, return_summary=False, **kwargs):
        observables = input_dict.get("observables")
        if self.local_summary_net is not None:
            observables = self.local_summary_net(observables)

        if self.is_conditional:
            conditions = input_dict.get("parameters")
            assert conditions is not None
        else:
            conditions = self.global_summary_net(observables, **kwargs)

        out = self.inference_net(observables, conditions)

        if not return_summary:
            return out
        return out, conditions
    
    def posterior_probs(self, input_dict, **kwargs):
        out = self(input_dict, return_summary=False, **kwargs)
        return out

    def compute_loss(self, input_dict, **kwargs):
        preds = self.posterior_probs(input_dict, **kwargs)
        loss = self.loss(input_dict.get("latents"), preds)
        return tf.reduce_mean(loss)
    
    def sample(self, input_dict, n_samples, **kwargs):
        probs = self.posterior_probs(input_dict, **kwargs)
        return probs
    
    def log_prob(self):
        pass


class AmortizedMixturePosterior(tf.keras.Model, AmortizedTarget):
    """
    Infers p(θ, s | y) = p(s | θ, y) x p(θ | y)
    """
    def __init__(self, amortized_mixture, amortized_posterior, **kwargs):
        tf.keras.Model.__init__(self, **kwargs)
        self.amortized_mixture   = amortized_mixture
        self.amortized_posterior = amortized_posterior

    def call(self, input_dict, **kwargs):
        mix_out = self.amortized_mixture(input_dict, **kwargs)
        pos_out = self.amortized_mixture(input_dict, **kwargs)
        return pos_out, mix_out
    
    def compute_loss(self, input_dict, **kwargs):
        mix_loss = self.amortized_mixture.compute_loss(input_dict["mixture_inputs"], **kwargs)
        pos_out = self.amortized_posterior.compute_loss(input_dict["posterior_inputs"], **kwargs)
        return {"Mix.Loss": mix_loss, "Post.Loss": pos_out}

    def sample(self, input_dict, n_samples, to_numpy=True, **kwargs):
        post_samples = self.amortized_posterior.sample(input_dict['posterior_inputs'], n_samples=n_samples, to_numpy=False, **kwargs)


        if len(post_samples.shape) == 2: # because posterior drops first dim if batches==1
            post_samples = tf.expand_dims(post_samples, axis=0)

        n_batches = post_samples.shape[0]
        mixture_inputs = input_dict["mixture_inputs"]
        mixture_inputs['parameters'] = merge_first_two_dims(post_samples)
        mixture_inputs['observables'] = merge_first_two_dims(expand_tensor(mixture_inputs['observables'], n_samples))
        posterior_probs = self.amortized_mixture.posterior_probs(mixture_inputs, **kwargs)
        posterior_probs = split_first_two_dims(posterior_probs, n_batches, n_samples)
        return post_samples, posterior_probs
    
    def log_prob(self):
        pass

class AmortizedPosteriorMixture(tf.keras.Model, AmortizedTarget):
    """
    Infers p(θ, s | y) = p(θ | s, y) x p(s | y)
    """
    def __init__(self, amortized_mixture, amortized_posterior, **kwargs):
        tf.keras.Model.__init__(self, **kwargs)
        self.amortized_mixture   = amortized_mixture
        self.amortized_posterior = amortized_posterior

    def sample(self, input_dict, n_samples, to_numpy=True, **kwargs):
        mix_samples = self.amortized_mixture.sample(input_dict['mixture_inputs'], n_samples=n_samples, **kwargs)
        return input_dict['posterior_inputs']['summary_conditions']

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