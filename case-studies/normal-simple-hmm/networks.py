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
    







class Sequences(tf.keras.Model):
    def __init__(self, units=128):
        super().__init__()
        self.lstm = LSTM(units, return_sequences=True)

    def __call__(self, x, backward=False):
        if backward:
            x = tf.reverse(x, axis=1)

        output = self.lstm(x)

        if backward:
            output = tf.reverse(output, axis=1)

        return output

class Classifier(tf.keras.Model):
    def __init__(self, n_classes, n_units):
        super().__init__(self)

        inner_layers = Sequential([Dense(n) for n in n_units])
        outer_layer = Dense(n_classes)
        self.net = TimeDistributed(Sequential([inner_layers, outer_layer]))
    
    def __call__(self, x, conditions):
        """
        x - (batch_size, n_observations, n_features)
        conditions - (batch_size, n_conditions)

        output - (batch_size, n_observations, n_classes)
        """ 

        conditions = tf.expand_dims(conditions, 1)
        conditions = tf.tile(conditions, [1, tf.shape(x)[1], 1])

        input = tf.concat([x, conditions], axis=-1)

        output = self.net(input)

        return output

class AmortizedMixture(tf.keras.Model, AmortizedTarget):
    def __init__(self, inference_net, forward_net=None, backward_net=None, loss=CategoricalCrossentropy(from_logits=True)):
        super().__init__()

        self.inference_net=inference_net
        self.forward_net=forward_net
        self.backward_net=backward_net
        self.loss=loss

    def __call__(self, input_dict):
        if self.forward_net:
            forward_summary = self.forward_net(input_dict["observables"])

        summary = forward_summary

        output = self.inference_net(summary, input_dict['parameters'])

        return output

    def compute_loss(self, input_dict, **kwargs):
        logits = self(input_dict)
        loss = self.loss(input_dict["latents"], logits)
        return loss
    
    def log_prob(self, input_dict):
        logits = self(input_dict)
        probs = tf.nn.softmax(logits, axis=-1)
        return tf.math.log(probs)

    def sample(self, observables, parameters):
        pass



class ReverseLayer(tf.keras.layers.Layer):
    def __init__(self, axis, **kwargs):
        super(ReverseLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.reverse(inputs, axis=[self.axis])