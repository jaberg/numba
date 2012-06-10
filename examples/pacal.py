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
from numba import Context


def model():
    dL = np.random.uniform(low=1, high=3)
    L0 = np.random.unifrom(low=9, high=11)
    dT = np.random.normal(loc=1, scale=1)
    K = dL / (L0 * dT)
    return K


def main():
    ctxt = Context()
    K = ctxt.call(model)
    counts = numba.hist(K, n=1000, ctxt=ctxt, rseed=1)
    plt.plot(counts)
    plt.show()


if __name__ == '__main__':
    sys.exit(main())
