"""
Bayesian Inference (in style of PyMC)
=====================================

XXX: CURRENTLY THIS DOES NOT RUN (PROOF OF CONCEPT ONLY)

This example illustrates how a stochastic numpy program (one that involves
calls to a RandomState) can be interpreted as a Bayesian model.

The example is adapted from
http://pymc-devs.github.com/pymc/README.html#purpose

"""

import numpy as np
import matplotlib.pyplot as plt
import numba


def model(n, x):
    # Sample unknown parameters
    # N.B. the distributions will matter more than the values
    alpha = np.random.normal(loc=0, scale=0.01)
    beta = np.random.normal(loc=0, scale=0.01)
    theta = 1.0 / (1.0 + np.exp(-(a + b * x)))
    data = np.random.binomial(n=n, p=theta)
    return alpha, beta, data


def main():
    # Create some fake data
    n = np.array([5, 5, 5, 5])
    x = np.array([-0.86, -0.3, -0.05, 0.73])

    # Create a model (by allowing numba to trace execution)
    ctxt = numba.Context()
    alpha, beta, data = ctxt.call(model, (n, x))

    posterior = numba.sample(
            # tell sampler what variables to return
            {'alpha': alpha, 'beta': beta},
            # communicate conditioning data for sampler
            given=[(data, [0.,1.,3.,5.])],
            # MCMC parameters
            iter=10000,
            burn=5000,
            thin=2,
            # tell sampler how data relates to alpha, beta
            ctxt=ctxt,
            # initialize sampler
            rseed=1,
            )

    plt.scatter(posterior['alpha'], posterior['beta'])
    plt.show()


if __name__ == '__main__':
    sys.exit(main())


