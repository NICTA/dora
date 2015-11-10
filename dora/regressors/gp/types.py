import numpy as np

class OptConfig:
    def __init__(self):
        self.global_opt = False
        self.sigma = None
        self.noise = Range([], [], [])
        self.walltime = 0

class Range: # simply a container
    def __init__(self, lowerBound, upperBound, initialVal):
        self.lowerBound = lowerBound
        self.initialVal = initialVal
        self.upperBound = upperBound

class Folds:
    def __init__(self, n_folds, X, Y, flat_y):
        self.X = X
        self.Y = Y # structured y
        self.flat_y = flat_y # flattened y
        self.n_folds = n_folds

class RegressionParams:
    def __init__(self, X, L, alpha, kernel, y, noise_std, mean=0):
        self.X = X
        self.L = L
        self.alpha = alpha
        self.kernel = kernel
        self.y = y
        self.noise_std = noise_std
        self.mean = mean

class QueryParams:
    def __init__(self, Xs, K_xxs):
        self.Xs = Xs
        self.K_xxs = K_xxs


