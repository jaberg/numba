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
from numba import rv


def model(n, x):
    # Sample unknown parameters
    # N.B. the distributions will matter more than the values
    alpha = np.random.normal(loc=0, scale=0.01)
    beta = np.random.normal(loc=0, scale=0.01)
    theta = 1.0 / (1.0 + np.exp(-(a + b * x)))
    data = np.random.binomial(n=n, p=theta)
    return locals()


def main():
    def specialized():
        d = model(
            # Create some fake data
            np.array([5, 5, 5, 5]),
            np.array([-0.86, -0.3, -0.05, 0.73]))
        # tell rv.eval what posteriors we care about
        return dict(
                alpha=d['alpha'],
                beta=d['beta'],
                data=d['data'])

    samples = rv.mcmc(specialized,
            given={'data': [0.,1.,3.,5.]})

    plt.scatter(samples['alpha'], samples['beta'])
    plt.show()


if __name__ == '__main__':
    np.random.seed(1)
    sys.exit(main())

