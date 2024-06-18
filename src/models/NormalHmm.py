import numpy as np
from ..Constraints import ordered, simplex
from .NormalMixture import *

class NormalHmm(NormalMixture):
    def __init__(self, n_cls=2, n_obs=None, separation=1.0, ordered=True, alpha=2, init=None, seed=None):
        alpha = [alpha for _ in range(n_cls)]
        alpha = np.array([alpha for _ in range(n_cls)])

        if init is None:
            init = np.array([1.0/n_cls for _ in range(n_cls)])

        self.n_cls = n_cls
        self.n_obs = n_obs
        self.separation = separation
        self.ordered = ordered
        self.alpha = alpha
        self.init = init
        self.rng = np.random.default_rng(seed=seed)

        self._prior_means = None
        self._prior_sds = None

    @property
    def n_par(self):
        return self.n_cls * (self.n_cls-1) + self.n_cls

    def _context(self):
        return {
            "n_obs": self.rng.integers(low=self.n_obs[0], high=self.n_obs[1], endpoint=True)
        }

    def _prior(self):
        p = []
        for k in range(self.n_cls):
            p.append(self.rng.dirichlet(alpha=self.alpha[k]))
        
        hyper_mu = [self.separation * (n - (self.n_cls-1) / 2.0) for n in range(self.n_cls)]
        mu = self.rng.normal(loc=hyper_mu, scale=1, size=self.n_cls)
        if self.ordered:
            while any(np.sort(mu) != mu):
                mu = self.rng.normal(loc=hyper_mu, scale=1, size=self.n_cls)

        return { "p": p, "mu": mu}

    def _simulator(self, parameters, context):
        p, mu = parameters.values()
        n_obs = context['n_obs']

        latents = []

        state = self.rng.choice(a=self.n_cls, size=1, replace=True, p=self.init)
        latents.append(state[0])
        for _ in range(n_obs-1):
            state=self.rng.choice(a=self.n_cls, size=1, replace=True, p=p[state[0]])
            latents.append(state[0])
        latents=np.array(latents)

        mu=mu[latents]
        y = self.rng.normal(loc=mu)

        return {"latents": latents, "observables": y}
    
    def _concatenate_parameters(self, parameters):
        parameters = list2dict(parameters)

        p = simplex.unconstrain(parameters['p'], axis=-1)
        p = p.reshape((p.shape[0], -1))

        parameters['p'] = p

        parameters['mu'] = self._configure_mu(parameters['mu'])

        return concat(parameters)
        
    
    def _configure_observables(self, latents):
        """
        Ensure that the observables are atleast 3D
        """
        return at_least_ndim(latents, ndim=3)
    
    def extract_pars(self, parameters, constrained=True):
        parameters = self._unstandardize_parameters(parameters)

        len_p = self.n_cls * (self.n_cls - 1)
        p  = np.take(parameters, indices=[i for i in range(len_p)], axis=-1)
        p  = np.reshape(p, p.shape[:-1] + (self.n_cls, self.n_cls-1)) 
        mu = np.take(parameters, indices=[i+len_p for i in range(self.n_cls)], axis=-1)

        if constrained:
            p  = simplex.constrain(p,  axis=-1)
            mu = ordered.constrain(mu, axis=-1)
        return {"p": p, "mu": mu}
        
