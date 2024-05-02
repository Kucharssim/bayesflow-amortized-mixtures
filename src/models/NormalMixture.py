import numpy as np
from ..Constraints import ordered, simplex


class NormalMixture:
    def __init__(self, n_cls=2, n_obs=None, n_rep=None, separation=1.0, ordered=True, alpha=None, seed=None):
        if alpha is None:
            alpha = [2 for _ in range(n_cls)]
        
        self.n_cls = n_cls
        self.n_obs = n_obs
        self.n_rep = n_rep
        self.separation = separation
        self.ordered = ordered
        self.alpha = alpha
        self.rng = np.random.default_rng(seed=seed)

    @property
    def n_par(self):
        return self.n_cls*2 - 1

    def __call__(self, batch_size, context=None, parameters=None):
        out = self._sim_batch(batch_size=batch_size, context=context, parameters=parameters)
        out = self._configure(out)
        return out
    
    def _context(self):
        return {
            "n_obs": self.rng.integers(low=self.n_obs[0], high=self.n_obs[1], endpoint=True),
            "n_rep": self.rng.integers(low=self.n_rep[0], high=self.n_rep[1], endpoint=True)
        }
    
    def _prior(self):
        p = self.rng.dirichlet(alpha=self.alpha)
        hyper_mu = [self.separation * (n - (self.n_cls-1) / 2.0) for n in range(self.n_cls)]
        mu = self.rng.normal(loc=hyper_mu, scale=1, size=self.n_cls)
        if self.ordered:
            while any(np.sort(mu) != mu):
                mu = self.rng.normal(loc=hyper_mu, scale=1, size=self.n_cls)
            #mu = np.sort(mu)

        return { "p": p, "mu": mu}
    
    def _simulator(self, parameters, context):
        p, mu = parameters.values()
        n_obs, n_rep = context.values()

        latents = self.rng.choice(a=self.n_cls, size=n_obs, replace=True, p=p)

        mu = mu[latents]
        mu = np.repeat(mu[...,np.newaxis], repeats=n_rep, axis=1)
        y = self.rng.normal(loc=mu)

        return { "latents": latents, "observables": y}
    
    def _sim_once(self, context, parameters=None):
        if parameters is None:
            parameters = self._prior()
            
        sims = self._simulator(parameters=parameters, context=context)

        return {
            "context": context,
            "parameters": parameters,
            "latents": sims['latents'],
            "observables": sims['observables']
        }
    
    def _sim_batch(self, batch_size, context, parameters):
        if context is None:
            context = self._context()
        output = [self._sim_once(context, parameters) for _ in range(batch_size)]
        output = list2dict(output)
        return output
    
    def _configure_parameters(self, parameters):
        """
        Concatenate all parameters
        """
        parameters = list2dict(parameters)
        parameters['p'] = simplex.unconstrain(parameters['p'], axis=1)

        if self.ordered:
            parameters['mu'] = ordered.unconstrain(parameters['mu'], axis=1)
        else:
            parameters['mu'] = parameters['mu']
        return concat(parameters)
    
    def _configure_latents(self, latents):
        """
        Does one-hot encoding and reshapes from (batch_size, n_obs) to (batch_size, n_obs, n_cls)
        """
        batch_size, n_obs = latents.shape
        latents = latents.reshape(-1)
        latents = np.eye(self.n_cls)[latents] # one-hot
        latents = latents.reshape([batch_size, n_obs, self.n_cls])
        return latents
    
    def _configure_observables(self, latents):
        """
        Ensure that the observables are atleast 4D (so that we can apply a 2-level hierarchical summary net)
        """
        return at_least_ndim(latents, ndim=4)
    
    def _configure_context(self, context):
        """
        Concatenate context
        """
        return np.array([list(c.values()) for c in context])
    
    def _configure(self, input):
        return {
            "parameters": self._configure_parameters(input['parameters']).astype(np.float32),
            "latents": self._configure_latents(input["latents"]).astype(np.float32),
            "summary_conditions": self._configure_observables(input["observables"]).astype(np.float32),
            "direct_conditions": self._configure_context(input['context']).astype(np.float32)
        }
    
    def config_bayesflow(self, context, observables):
        """
        Convenience function for reshaping data into bayesflow format (for sampling)
        Assuming a single dataset
        context: A dict with 'n_obs', 'n_rep'
        observables: Array of shape (n_obs, n_rep)
        """
        return {
            "direct_conditions": self._configure_context([context]).astype(np.float32),
            "summary_conditions": self._configure_observables(observables[np.newaxis,...]).astype(np.float32)
        }

    def extract_pars(self, parameters, axis=1, constrained=True):
        p  = np.take(parameters, indices=[i for i in range(self.n_cls-1)], axis=axis)
        mu = np.take(parameters, indices=[i+self.n_cls-1 for i in range(self.n_cls)], axis=axis)

        if constrained:
            p = simplex.constrain(p, axis=axis)
            mu = ordered.constrain(mu, axis=axis)
        
        return {
            "p": p,
            "mu": mu
        }




def at_least_ndim(x, ndim):
    while x.ndim < ndim:
        x = x[...,np.newaxis]
    return x

def list2dict(x):
    return {key: np.array([d[key] for d in x]) for key in x[0]}

def concat(x):
    return np.concatenate(list(x.values()), axis=1)
