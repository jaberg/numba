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
from scipy.optimize.lbfgsb import fmin_l_bfgs_b
from numba.ad import Context


# XXX: currently has to be a toplevel function to work,
#      numba.ad does not work on nested functions & closures
def binary_svm_hinge_loss(x, y, weights, bias, l2_regularization):
    margin = y * (np.dot(x, weights) + bias)
    losses = np.maximum(0, 1 - margin) ** 2
    l2_cost = 0.5 * l2_regularization * np.dot(weights, weights)
    cost = np.mean(losses) + l2_cost
    return cost


def main():
    np.random.seed(1)

    # create some fake data: features and labels
    X = np.random.rand(10, 7)
    y = 2 * (np.random.rand(10) > 0.5) - 1

    # initialize a linear SVM classification model
    weights = np.zeros(X.shape[1])
    bias = np.zeros(())

    # create a high-level numba interface object thing
    ctxt = Context()

    # evaluate the cost function once to define it (and debug it!)
    # XXX: could the interface be less weird?
    loss = ctxt.call(binary_svm_hinge_loss,
            (X, y, weights, bias, 0.00001))

    # minimize the model loss wrt weights and bias parameters using
    # fmin_l_bfgs_b
    (fit_weights, fit_bias), minloss, info = ctxt.fmin(
            cost=loss,
            wrt=[weights, bias],
            algo=(fmin_l_bfgs_b, dict(iprint=1, factr=1e11, maxfun=1000)))
    print 'Best-fit SVM:'
    print ' -> cost:', minloss
    print ' -> weights:', fit_weights
    print ' -> bias:', fit_bias

if __name__ == '__main__':
    sys.exit(main())

