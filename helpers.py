import numpy as np
import tensorflow as tf

def classification_entropy(p):
    log_p = np.log(p)
    log_p[log_p == -np.inf] = 0

    ent = np.sum(p * log_p, axis = (-1, -2))

    num_classes = p.shape[-1]
    num_obs = p.shape[-2]
    ent = 1 + (ent/(num_obs*np.log(num_classes)))
    
    return ent

def accuracy(p, latents):
    cls = tf.one_hot(np.argmax(p, axis=-1), p.shape[-1])
    acc = np.mean(latents == cls, axis=1)
    return acc