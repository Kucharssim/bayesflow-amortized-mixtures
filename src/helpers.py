import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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


def plot_bands(x, samples, probs=[0.5, 0.9], color=None, alpha=None, axis=0):
    for p in probs:
        plt.fill_between(
            x,
            np.quantile(samples, q=1-p/2.0, axis=axis),
            np.quantile(samples, q=p/2.0, axis=axis),
            alpha=alpha,
            color=color
        )