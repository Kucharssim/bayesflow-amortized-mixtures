from abc import ABC
import numpy as np
from scipy.special import logit, expit

class Constraints(ABC):
    def unconstrain(self, x, axis=0):
        x = np.array(x)
        return np.apply_along_axis(self.transform, axis, x)
    def constrain(self, y, axis=0):
        x = np.array(y)
        return np.apply_along_axis(self.inv_transform, axis, x)
    @staticmethod
    def transform(x):
        raise NotImplementedError
    @staticmethod
    def inv_transform(y):
        raise NotImplementedError

class Positive(Constraints):
    @staticmethod
    def transform(x):
        return np.log(x)
    @staticmethod
    def inv_transform(x):
        return np.exp(x)

positive = Positive()

class Ordered(Constraints):
    @staticmethod
    def transform(x):
        y = []
        for i, xi in enumerate(x):
            if i == 0:
                y.append(xi)
            else:
                dx = xi - x[i-1]
                y.append(np.log(dx))

        return y
    @staticmethod
    def inv_transform(y):
        x = []
        for i, yi in enumerate(y):
            if i == 0:
                x.append(yi)
            else:
                d = np.exp(yi)
                x.append(x[i-1] + d)
        return x
ordered = Ordered()


class Simplex(Constraints):
    @staticmethod
    def transform(x):
        K = len(x)
        k = np.array([i+1 for i in range(K-1)])
        x = x[0:(K-1)]
        norm = np.cumsum(x)
        norm = 1-np.insert(norm, 0, 0)
        norm = norm[0:(K-1)]

        z = x / norm

        adjust = np.log(1.0 / (K - k))
        y = logit(z) - adjust
        return y
    @staticmethod
    def inv_transform(y):
        K = len(y)+1
        k = np.array([i+1 for i in range(K-1)])
        adjust = np.log(1.0 / (K - k))
        z = expit(y + adjust)

        x = []

        for i, zi in enumerate(z):
            s = 1 - np.sum(x)
            x.append(s * zi)

        x.append(1 - np.sum(x))
        return np.array(x)

simplex = Simplex()