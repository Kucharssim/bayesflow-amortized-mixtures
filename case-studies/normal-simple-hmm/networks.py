import tensorflow as tf
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, TimeDistributed
from bayesflow.amortizers import AmortizedTarget
from bayesflow.losses import log_loss
from numpy import expand_dims, zeros, array



class Smoothing(tf.keras.Model):
    def __init__(self, n_classes):
        super().__init__()

        self.n_classes = n_classes
        self.lstm = Bidirectional(LSTM(n_classes*4, return_sequences=True))
        self.dense = TimeDistributed(Dense(n_classes))

    def call(self, observables, conditions):

        conditions = tf.expand_dims(conditions, 1)
        conditions = tf.tile(conditions, [1, tf.shape(observables)[1], 1])

        input = tf.concat([observables, conditions], axis=-1)

        output = self.lstm(input)
        output = self.dense(output)

        return tf.nn.softmax(output, axis=-1)
    

class MixtureAmortizer(tf.keras.Model, AmortizedTarget):
    def __init__(self, inference_net):
        tf.keras.Model.__init__(self)
        self.inference_net = inference_net
    
    def compute_loss(self, input_dict, **kwargs):
        probs = self.inference_net(input_dict['observables'], input_dict['parameters'])
        loss = log_loss(input_dict['latents'], probs)
        return tf.reduce_mean(loss)
    
    def log_prob(self):
        pass

    def sample(self, observables, parameters):
        n_samples = parameters.shape[0]
        n_timestamps = observables.shape[1]
        probs = zeros((n_samples, n_timestamps, self.inference_net.n_classes))

        for s in range(n_samples):
            probs[s] = array(self.inference_net(observables, expand_dims(parameters[s], 0)))

        return probs