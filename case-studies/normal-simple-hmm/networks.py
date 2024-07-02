import tensorflow as tf
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, TimeDistributed
from tensorflow.keras.models import Sequential
from bayesflow.amortizers import AmortizedTarget
from bayesflow.helper_networks import MultiConv1D
from bayesflow.default_settings import DEFAULT_SETTING_MULTI_CONV
from bayesflow.losses import log_loss
from numpy import expand_dims, zeros, array
from tensorflow.keras.losses import CategoricalCrossentropy 


class DependentMixture(tf.keras.Model):
    def __init__(self, n_classes, convolutional, bidirectional):
        super().__init__()

        self.n_classes = n_classes
        self.convolutional=convolutional
        
        if convolutional:
            self.convolution = Sequential(
                [MultiConv1D(DEFAULT_SETTING_MULTI_CONV) for _ in range(8)]
                )
        
        lstm = LSTM(128, return_sequences=True)
        self.lstm = Bidirectional(lstm) if bidirectional else lstm

        dense = Sequential([Dense(64), Dense(32), Dense(n_classes)])
        self.dense = TimeDistributed(dense)

    def call(self, observables, conditions):

        conditions = tf.expand_dims(conditions, 1)
        conditions = tf.tile(conditions, [1, tf.shape(observables)[1], 1])

        output = tf.concat([observables, conditions], axis=-1)

        if self.convolutional:
            output = self.convolution(output)

        output = self.lstm(output)
        output = self.dense(output)

        return tf.nn.softmax(output, axis=-1)
    
class Smoothing(DependentMixture):
    def __init__(self, n_classes, convolutional):
        super().__init__(n_classes=n_classes, convolutional=convolutional, bidirectional=True)

class Filtering(DependentMixture):
    def __init__(self, n_classes, convolutional):
        super().__init__(n_classes=n_classes, convolutional=convolutional, bidirectional=False)


class MixtureAmortizer(tf.keras.Model, AmortizedTarget):
    def __init__(self, inference_net):
        tf.keras.Model.__init__(self)
        self.inference_net = inference_net
        self.loss = CategoricalCrossentropy()#log_loss
    
    def compute_loss(self, input_dict, **kwargs):
        probs = self.inference_net(input_dict['observables'], input_dict['parameters'])
        loss = self.loss(input_dict['latents'], probs)
        return loss
    
    def log_prob(self):
        pass

    def sample(self, observables, parameters):
        n_samples = parameters.shape[0]
        n_timestamps = observables.shape[1]
        probs = zeros((n_samples, n_timestamps, self.inference_net.n_classes))

        for s in range(n_samples):
            probs[s] = array(self.inference_net(observables, expand_dims(parameters[s], 0)))

        return probs
    

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
            output = tf.nn.softmax(logits)

        return output


class AmortizedSmoothing(tf.keras.Model, AmortizedTarget):
    def __init__(self, forward, backward):
        super(AmortizedSmoothing, self).__init__()

        self.forward = forward
        self.backward = backward
    def __call__(self, forward_dict, backward_dict):
        f=self.forward(forward_dict)
        f=f-tf.reduce_mean(f)
        b=self.backward(backward_dict)
        b=b-tf.reduce_mean(f)
        return f+b
    def compute_loss(self, forward_dict, backward_dict):
        f=self.forward(forward_dict)
        b=self.backward(backward_dict)
        fb = f+b
        return f, b, fb
    
    def log_prob(self, input_dict):
        logits = self(input_dict)
        probs = tf.nn.softmax(logits, axis=-1)
        return tf.math.log(probs)
    def sample(self, forward_dict, backward_dict, return_logits=False):
        logits = self(forward_dict, backward_dict)
        if return_logits:
            output = logits
        else:
            output = tf.nn.softmax(logits)

        return output