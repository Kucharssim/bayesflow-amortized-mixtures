import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, TimeDistributed
from tensorflow.keras.models import Sequential
from bayesflow.amortizers import AmortizedTarget
from bayesflow.helper_networks import MultiConv1D
from bayesflow.default_settings import DEFAULT_SETTING_MULTI_CONV
from bayesflow.losses import log_loss
from numpy import expand_dims, zeros, array
from tensorflow.keras.losses import CategoricalCrossentropy 
    

class Reverse(tf.keras.layers.Layer):
    def __init__(self, axis, **kwargs):
        super(Reverse, self).__init__(**kwargs)
        self.axis = axis

    def call(self, input):
        return tf.reverse(input, axis=[self.axis])
    

def Backward(net, axis=1):
    return Sequential([
        Reverse(axis),
        net,
        Reverse(axis)
    ])
    
class Shift2D(tf.keras.layers.Layer):
    def __init__(self, by=1):
        super(Shift2D, self).__init__()
        self.padding  = tf.keras.layers.ZeroPadding2D(((0, 0),(0, by)))
        self.cropping = tf.keras.layers.Cropping2D(((0, 0),(by, 0)))
    def __call__(self, input):
        output = self.padding(input)
        output = self.cropping(output)
        return output


class Classifier(tf.keras.Model):
    def __init__(self, n_classes, n_units):
        super(Classifier, self).__init__()
        net = Sequential()
        for unit in n_units:
            net.add(Dense(unit))
        net.add(Dense(n_classes))
        net = TimeDistributed(net) # over n_observations
        self.net = net

    def call(self, inputs):
        return self.net(inputs)


class AmortizedMixture(tf.keras.Model, AmortizedTarget):
    def __init__(self, inference_net, local_summary_net=None, loss=CategoricalCrossentropy(from_logits=True)):
        super(AmortizedMixture, self).__init__()

        self.inference_net=inference_net
        self.local_summary_net=local_summary_net
        self.loss=loss

    def __call__(self, input_dict):
        """
        observables - (batch_size, n_observations, n_features)
        parameters - (batch_size, n_samples, n_conditions)

        output - (batch_size, n_samples, n_observations, n_classes)
        """ 

        input = self._concat_conditions(input_dict)
        output = self.inference_net(input)
        return output
    
    def _calculate_summaries(self, input_dict):
        output = input_dict.get("summary_conditions")

        if self.local_summary_net:
            output = self.local_summary_net(output)

        return output
    
    def _concat_conditions(self, input_dict):
        summaries = self._calculate_summaries(input_dict) # (batch_size, n_observations, n_units)
        parameters = input_dict.get("parameters") # (bacth_size, n_samples, n_parameters)
        conditions = input_dict.get("direct_conditions") # (batch_size, n_conditions)

        output = []

        summaries = tf.expand_dims(summaries, 1)
        summaries = tf.tile(summaries, [1, tf.shape(parameters)[1], 1, 1])
        output.append(summaries)

        parameters = tf.expand_dims(parameters, 2)
        parameters = tf.tile(parameters, [1, 1, tf.shape(summaries)[2], 1])
        output.append(parameters)

        if conditions is not None:
            conditions = tf.expand_dims(conditions, 1)
            conditions = tf.expand_dims(conditions, 1)
            conditions = tf.tile(conditions, [1, tf.shape(parameters)[1], tf.shape(summaries)[2], 1])
            output.append(conditions)

        # (batch_size, n_samples, n_observations, n_units + n_parameters + n_conditions)
        output = tf.concat(output, axis=-1)

        return output

    def compute_loss(self, input_dict, **kwargs):
        logits = self(input_dict)
        loss = self.loss(input_dict.get("latents"), logits)
        return loss
    
    def log_prob(self, input_dict):
        logits = self(input_dict)
        probs = tf.nn.softmax(logits, axis=-1)
        return tf.math.log(probs)

    def sample(self, input_dict, return_logits=False):
        logits = self(input_dict)
        if return_logits:
            output = logits
        else:
            output = tf.nn.softmax(logits, axis=-1)

        return output


class AmortizedSmoothing(tf.keras.Model, AmortizedTarget):
    def __init__(self, forward_net, backward_net, local_summary_net=None, loss=CategoricalCrossentropy(from_logits=True)):
        super(AmortizedSmoothing, self).__init__()

        self.forward_net = forward_net
        self.backward_net = Backward(backward_net)
        self.local_summary_net = local_summary_net
        self.loss = loss
        self.shift = Shift2D()

    def __call__(self, input_dict, shift=False):
        """
        observables - (batch_size, n_observations, n_features)
        parameters - (batch_size, n_samples, n_conditions)

        output - (batch_size, n_samples, n_observations, n_classes)
        """ 

        input = self._concat_conditions(input_dict)
        input, shape = self._to_long(input)
        
        forward  = self.forward_net(input)
        backward = self.backward_net(input)

        forward  = self._to_wide(forward,  tf.concat([shape, [tf.shape(forward)[-1]]], axis=0))
        backward = self._to_wide(backward, tf.concat([shape, [tf.shape(backward)[-1]]], axis=0))

        if shift:
           backward = self.shift(backward)

        return forward, backward, forward + backward
    
    def _calculate_summaries(self, input_dict):
        output = input_dict["summary_conditions"]

        if self.local_summary_net:
            output = self.local_summary_net(output)

        return output
    
    def _concat_conditions(self, input_dict):
        summaries = self._calculate_summaries(input_dict) # (batch_size, n_observations, n_units)
        parameters = input_dict.get("parameters") # (bacth_size, n_samples, n_parameters)
        conditions = input_dict.get("direct_conditions") # (batch_size, n_conditions)

        output = []

        summaries = tf.expand_dims(summaries, 1)
        summaries = tf.tile(summaries, [1, tf.shape(parameters)[1], 1, 1])
        output.append(summaries)

        parameters = tf.expand_dims(parameters, 2)
        parameters = tf.tile(parameters, [1, 1, tf.shape(summaries)[2], 1])
        output.append(parameters)

        if conditions is not None:
            conditions = tf.expand_dims(conditions, 1)
            conditions = tf.expand_dims(conditions, 1)
            conditions = tf.tile(conditions, [1, tf.shape(parameters)[1], tf.shape(summaries)[2], 1])
            output.append(conditions)

        # (batch_size, n_samples, n_observations, n_units + n_parameters + n_conditions)
        output = tf.concat(output, axis=-1)

        return output
    
    def _to_long(self, input):
        # input: (batch_size, n_samples, n_observations, n_units + n_parameters + n_conditions)
        # output: (batch_size * n_samples, n_observations, n_units + n_parameters + n_conditions)

        batch_size = tf.shape(input)[0]
        n_samples = tf.shape(input)[1]
        n_observations = tf.shape(input)[2]
        n_features = tf.shape(input)[3]

        output = tf.reshape(input, (batch_size * n_samples, n_observations, n_features))

        return output, tf.stack([batch_size, n_samples, n_observations])

    def _to_wide(self, input, new_shape):
        # inverse of _to_long
        output = tf.reshape(input, new_shape)
        return output

    def compute_loss(self, input_dict, **kwargs):
        f, b, _ = self(input_dict)

        latents = input_dict["latents"]
        f = self.loss(latents, f)
        b = self.loss(latents, b)
        return {"forward.loss": f, "backward.loss:": b}
    
    def log_prob(self, input_dict):
        pass

    def sample(self, input_dict, return_logits=False, shift=True):
        logits = self(input_dict, shift=shift)
        if return_logits:
            output = logits
        else:
            output = tf.nn.softmax(logits, axis=-1)

        return output
    
class AmortizedPosteriorMixture(tf.keras.Model, AmortizedTarget):
    def __init__(self, amortized_posterior, amortized_mixture):
        super(AmortizedPosteriorMixture, self).__init__()

        self.amortized_posterior=amortized_posterior
        self.amortized_mixture=amortized_mixture
    
    def __call__(self, input_dict, **kwargs):
        posterior = self.amortized_posterior(input_dict["posterior_inputs"], **kwargs)
        mixture = self.amortized_mixture(input_dict["mixture_inputs"], **kwargs)

        return posterior, mixture
    
    def compute_loss(self, input_dict, **kwargs):
        posterior = self.amortized_posterior.compute_loss(input_dict["posterior_inputs"], **kwargs)
        mixture = self.amortized_mixture.compute_loss(input_dict["mixture_inputs"], **kwargs)

        if isinstance(mixture, dict):
            loss = mixture
            loss['posterior.loss'] = posterior
        else:
            loss = {"posterior.loss": posterior, "mixture.loss": mixture}
        
        return loss
    
    def log_prob(self, **kwargs):
        pass
    
    def sample(self, input_dict, n_samples, to_numpy=True, **kwargs):
        posterior = self.amortized_posterior.sample(input_dict["posterior_inputs"], n_samples, to_numpy=False, **kwargs)

        if len(posterior.shape) == 2: # because posterior drops first dim if batches==1
            posterior = tf.expand_dims(posterior, axis=0)

        input_dict["mixture_inputs"]["parameters"] = posterior
        mixture = self.amortized_mixture.sample(input_dict["mixture_inputs"], **kwargs)

        if to_numpy:
            posterior = np.array(posterior)
            mixture = np.array(mixture)

        return posterior, mixture
