"""
ProbAbilistic Calculator
========================

XXX: CURRENTLY THIS DOES NOT RUN (PROOF OF CONCEPT ONLY)

This example illustrates how uncertainties may be tracked through
computation.

The example is adapted from
http://pacal.sourceforge.net/

"""

import numpy as np
from numba import rv


def model():
    dL = np.random.uniform(low=1, high=3)
    L0 = np.random.unifrom(low=9, high=11)
    dT = np.random.normal(loc=1, scale=1)
    K = dL / (L0 * dT)
    return K


def main():
    v = rv.eval(model)
    print v.mean()
    print v.cdf(10)
    print v.pdf(0.2)


if __name__ == '__main__':
    sys.exit(main())
