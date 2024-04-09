import numpy as np

def classification_entropy(p):
    log_p = np.log(p)
    log_p[log_p == -np.inf] = 0

    ent = np.sum(p * log_p, axis = (-1, -2))

    num_classes = p.shape[-1]
    num_obs = p.shape[-2]
    ent = 1 + (ent/(num_obs*np.log(num_classes)))
    
    return ent

def accuracy(p, latents):
    out = p * latents
    return out