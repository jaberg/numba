"""
Neural Network
==============

This example illustrates how a function defined purely by numpy operations can
be minimized directly with a stochastic gradient descent solver, which is
efficient in machine learning problems with large redundant data sets.

"""

import sys
import numpy as np
from numba.ad import fmin, sgd

from skdata.mnist.views import Official as OfficialMNIST

def neural_network_classifier(params, x, y, l2_regularization):
    # print x, y
    weights = params[::2]
    biases = params[1::2]
    features = x
    l2_cost = 0
    for w, b in zip(weights, biases):
        features = 1.0 / (1.0 + np.exp(np.dot(w, x) + b))
        l2_cost += 0.5 * l2_regularization * (w ** 2).sum()

    margin = y * features
    loss = np.maximum(0, 1 - margin) ** 2
    loss = np.mean(loss) + l2_cost

    error_rate = np.mean(XXX)
    return loss, error_rate


def train_mnist(algo):

    # load mnist from skdata

    def loss_fn(*params):
        loss, erate = neural_network_classifier(params, x, y, 1e-4)
        return loss

    weights_biases = fmin(loss_fn,
            [
                # weights and bias for each layer
                np.zeros((784, 500)), np.zeros((500,)),
                np.zeros((500, 10)), np.zeros((10,)),
            ],
            algo=algo)

    test_loss, test_erate = neural_network_classifier(
            weights_biases, test_x, test_y, 1e-4)
    print 'Final error rate', erate


def main():
    print 'Batch training (l-bfgs-b)'
    train_mnist(algo=bfgs)

    print 'Online training (sgd)'
    train_mnist(algo=sgd)


if __name__ == '__main__':
    np.random.seed(1)
    sys.exit(main())

