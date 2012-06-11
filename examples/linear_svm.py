"""
Linear SVM
==========

This example illustrates how a function defined purely by numpy operations can
be minimized directly with a gradient-based solver.

This example is a linear classifier (support vector machine) applied to random
data.

"""

import sys
import numpy as np
from numba.ad import fmin


# a Python function with no data-dapendent control flow in the bytecode
#
# (A lot of mathematical programming fits into this limited Python subset.)
def binary_svm_hinge_loss(weights, bias, x, y, l2_regularization):
    margin = y * (np.dot(x, weights) + bias)
    loss = np.maximum(0, 1 - margin) ** 2
    l2_cost = 0.5 * l2_regularization * np.dot(weights, weights)
    # we can still convert this to single-assignment form easily
    loss = np.mean(loss) + l2_cost
    return loss


def main():
    # SVM-specific hack here
    loss_fn = functools.partial(binary_svm_hinge_loss,
            x=np.random.rand(10, 7),
            y=2 * (np.random.rand(10) > 0.5) - 1,
            l2_regularization=0.00001)

    w, b = fmin(loss_fn, [np.zeros(X.shape[1]), np.zeros(())])

    print 'Best-fit SVM:'
    print ' -> cost:', loss_fn(w, b)
    print ' -> weights:', w
    print ' -> bias:', b


if __name__ == '__main__':
    np.random.seed(1)
    sys.exit(main())

