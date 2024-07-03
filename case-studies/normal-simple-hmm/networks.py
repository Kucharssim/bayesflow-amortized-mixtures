import tensorflow as tf
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

class Backward(tf.keras.Model):
    def __init__(self, net, axis, **kwargs):
        super(Backward, self).__init__(**kwargs)

        self.net = Sequential([
            Reverse(axis),
            net,
            Reverse(axis)
        ])

    def __call__(self, input):
        return self.net(input)
    
class Shift(tf.keras.layers.Layer):
    def __init__(self, by=1):
        super(Shift, self).__init__()
        self.padding  = tf.keras.layers.ZeroPadding1D((0, by))
        self.cropping = tf.keras.layers.Cropping1D((by,0))
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
        net = TimeDistributed(net)
        self.net = net

    def call(self, inputs):
        return self.net(inputs)


class AmortizedMixture(tf.keras.Model, AmortizedTarget):
    def __init__(self, inference_net, local_summary_net=None, loss=CategoricalCrossentropy(from_logits=True)):
        super(AmortizedMixture, self).__init__()

        self.inference_net=TimeDistributed(inference_net)
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
        output = input_dict["summary_conditions"]

        if self.local_summary_net:
            output = self.local_summary_net(output)

        return output
    
    def _concat_conditions(self, input_dict):
        summaries = self._calculate_summaries(input_dict) # (batch_size, n_observations, n_features)
        conditions = input_dict["direct_conditions"] # (batch_size, n_samples, n_conditions)

        summaries = tf.expand_dims(summaries, 1)
        summaries = tf.tile(summaries, [1, tf.shape(conditions)[1], 1, 1])

        conditions = tf.expand_dims(conditions, 2)
        # (batch_size, n_samples, n_observations, n_features + n_conditions)
        conditions = tf.tile(conditions, [1, 1, tf.shape(summaries)[2], 1])

        output = tf.concat([summaries, conditions], axis=-1)

        return output

    def compute_loss(self, input_dict, **kwargs):
        logits = self(input_dict)
        loss = self.loss(input_dict["latents"], logits)
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
    def __init__(self, forward, backward, loss=CategoricalCrossentropy(from_logits=True)):
        super(AmortizedSmoothing, self).__init__()

        self.forward = forward
        self.backward = backward
        self.loss = loss
        self.shift = TimeDistributed(Shift(by=1))
    def __call__(self, input_dict, shift=False):
        f=self.forward(input_dict)
        b=self.backward(input_dict)

        if shift:
            b = self.shift(b)

        return f, b, f+b
    def compute_loss(self, input_dict, **kwargs):
        f, b, _ = self(input_dict)

        latents = input_dict["latents"]
        f = self.loss(latents, f)
        b = self.loss(latents, b)
        #fb = self.loss(latents, fb)
        return {"forward.loss": f, "backward.loss:": b}#, "smoothing.loss:": fb}
    
    def log_prob(self, input_dict):
        logits = self(input_dict)
        probs = tf.nn.softmax(logits, axis=-1)
        return tf.math.log(probs)
    def sample(self, input_dict, return_logits=False, shift=True):
        logits = self(input_dict, shift=shift)
        if return_logits:
            output = logits
        else:
            output = tf.nn.softmax(logits, axis=-1)

        return output