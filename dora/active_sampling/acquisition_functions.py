

import numpy as np
import scipy.stats as stats


class AcqFn:

    def __init__(self, *args):
        self.args = args

    def __call__(self, U, V):
        return self.A(U, V)


class VarSum(AcqFn):

    def A(self, u, v):
        return np.sum(v, axis=1)


class UpperBound(AcqFn):

    def __init__(self, explore_priority=1.0):
        AcqFn.__init__(self)
        self.explore_priority = explore_priority

    def A(self, u, v):
        a = np.max(u + 3 * self.explore_priority * np.sqrt(v), axis=1)
        return a


class ProdMax(AcqFn):

    def __init__(self,  y_mean=0, explore_priority=1.0):
        AcqFn.__init__(self)
        self.y_mean = y_mean
        self.explore_priority = explore_priority

    def A(self, u, v):
        a = np.max(
            (u + (self.y_mean + (self.explore_priority/.1) / 3.0)) * np.sqrt(v),
            axis=1
        )
        return a


class ProbTail(AcqFn):

    def __init__(self, explore_priority=1.0):
        AcqFn.__init__(self)
        self.explore_priority = explore_priority

    def A(self, u, v):
        a = np.max(1 - stats.norm.cdf(
            (self.explore_priority/10000) * np.ones(u.shape), u, np.sqrt(v)
        ), axis=1)
        return a


class Sigmoid(AcqFn):

    def __init__(self, explore_priority=1.0):
        AcqFn.__init__(self)
        self.explore_priority = explore_priority

    def A(self, u, v):
        a = np.abs(
            stats.logistic.cdf(u + np.sqrt(v), loc=0.5,
                               scale=self.explore_priority) -
            stats.logistic.cdf(u - np.sqrt(v), loc=0.5,
                               scale=self.explore_priority)
        ).sum(axis=1)
        return a
