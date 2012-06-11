"""
Uncertainty Propagation
=======================

XXX: CURRENTLY THIS DOES NOT RUN (PROOF OF CONCEPT ONLY)

This example illustrates how uncertainties may be tracked through
computation.

The example is adapted from
http://packages.python.org/uncertainties/

The original is quite short and elegant:

>>> from uncertainties import ufloat
>>> from uncertainties.umath import *  # sin(), etc.
>>> x = ufloat((1, 0.1))  # x = 1+/-0.1
>>> print 2*x
2.0+/-0.2
>>> sin(2*x)  # In a Python shell, "print" is optional
0.90929742682568171+/-0.083229367309428481
"""

import numpy as np
from numba import rv


def model(x, xvar):
    x = np.random.normal(loc=x, scale=np.sqrt(xvar))
    return 2 * x, np.sin(2 * x)


def main():
    a, b = rv.eval(model, (1, .1))
    print a.mean(), b.variance()
    print b.mean, b.variance()


if __name__ == '__main__':
    sys.exit(main())
