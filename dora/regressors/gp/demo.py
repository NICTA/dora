""" simple_regression.py
    This demo shows how to construct a simple regression by composing a kernel
    and optimising its hyperparameters.
"""
import numpy as np
import matplotlib.pyplot as pl
import dora.regressors.gp as gp

def main():
    nTrain = 20
    nQuery = 100
    nDraws = 20
    nDims = 1
    seed = 100

    # Make test dataset:
    np.random.seed(seed)
    X = np.random.uniform(0, 30, size=(nTrain,nDims))
    X = X[np.argsort(X[:,0])]
    noise = np.random.normal(loc=0.0, scale=0.05, size=(nTrain,1))
    def ground_truth(X):
        return np.sin(X-5) + np.sin(X/2-2) + 0.4*np.sin(X/5-2) + 0.4*np.sin(X-3) + 0.2*np.sin(X/0.3-3)
    Y = ground_truth(X)
    Y = Y[:,0]
    Xs = np.linspace(0., 30., nQuery)[:,np.newaxis]

    # Whiten inputs and de-mean outputs:
    Xw = X
    Xsw = Xs
    data_mean = np.mean(Y, axis=0)
    Ys = Y - data_mean

    # Define a GP kernel:
    def mykernel(h, k):
        # a fun pathological example
        a = h(0.1, 5, 0.1) # We can use the same parameter multiple times!
        b = h(0.1, 5, 0.1) # or just define it inline later
        return b*k('matern3on2', a)
        # return a*k('gaussian', b) + b*k('matern3on2', a)

    # We can automatically extract the upper and lower theta vectors
    myKernelFn = gp.compose(mykernel)  # callable covariance underlyingFunction
    myPrintFn = gp.describer(mykernel)

    # Set up optimisation
    opt_config = gp.OptConfig()
    opt_config.sigma = gp.auto_range(mykernel)
    opt_config.noise = gp.Range([0.0001], [0.5], [0.05])
    opt_config.walltime = 3.0

    # Learning signal and noise hyperparameters
    hyper_params = gp.learn(Xw, Ys, myKernelFn, opt_config)
    print('Final kernel:', myPrintFn(hyper_params), '+ noise', hyper_params[1])

    # to extract the hypers:  
    hypers = gp.train.pack(hyper_params[0], hyper_params[1])[0]

    # Reonstitute them, using the kernel definition to define the structure:
    theta0, structure = gp.train.pack(gp.auto_range(mykernel).initialVal, [0])
    reconstitute = gp.train.unpack(hypers, structure)

    regressor = gp.condition(Xw, Ys, myKernelFn, hyper_params)
    query = gp.query(Xsw, regressor)

    # import IPython; IPython.embed(); import sys; sys.exit()
    post_mu = gp.mean(regressor, query)
    post_cov = gp.predict.covariance(regressor, query) # for draws
    post_var = gp.variance(regressor, query)
    draws = gp.predict.draws(nDraws, post_mu, post_cov)

    # Shift outputs back:
    post_mu += data_mean
    draws = [draw+data_mean for draw in draws]

    # Plot
    fig = pl.figure()
    ax = fig.add_subplot(121)
    ax.plot(Xs, post_mu, 'k-')
    post_mu = post_mu[:,np.newaxis]
    real_var = (post_var + noise[0]**2)[:,np.newaxis]
    upper = (post_mu + 2*np.sqrt(real_var))
    lower = (post_mu - 2*np.sqrt(real_var))
    ax.fill_between(Xs.ravel(), upper.ravel(),lower.ravel(), facecolor=(0.9,0.9,0.9), edgecolor=(0.5,0.5,0.5))
    ax.plot(regressor.X[:,0], regressor.y+data_mean,'r.')
    ax = fig.add_subplot(122)
    for i in range(nDraws):
        ax.plot(Xs.ravel(), draws[i])
    pl.show()


if __name__ == "__main__":
    main()
