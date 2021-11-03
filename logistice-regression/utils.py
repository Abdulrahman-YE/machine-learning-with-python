import numpy as np
from matplotlib import pyplot as plt
import sys


def plot_decision_boundary(plot_data, theta, X, y):
    """
    Plots the data points X and y into a new figure with the decision boundary defined by theta.
    Plots the data points with * for the positive examples and o for  the negative examples.

    Parameters
    ----------
    plotData : func
        A function reference for plotting the X, y data.

    theta : array_like
        Parameters for logistic regression. A vector of shape (n+1, ).

    X : array_like
        The input dataset. X is assumed to be  a either:
            1) Mx3 matrix, where the first column is an all ones column for the intercept.
            2) MxN, N>3 matrix, where the first column is all ones.

    y : array_like
        Vector of data labels of shape (m, ).
    """
    theta = np.array(theta)

    plot_data(X[:, 1:3], y)

    if X.shape[1] <= 3:
        #2 end-points to define a line
        plot_x = np.array([np.min(X[:, 1]), np.max(X[:, 1]) ])
        print('Suppose theta = [{:.3f}, {:.3f}, {:.3f}]'.format(*theta))
        print('and x end-points(min, max) = [{:.3f}, {:.3f}]'.format(*plot_x))
        print('To calculate decision boundary: (-1. / theta[2]) * (theta[1] * plot_x + theta[0])')
        v = -1. / theta[2]
        print('\t(-1. / theta[2]) = {:.3f}'.format(v))
        u = theta[1] * plot_x
        print('\t(theta[1] * plot_x) = [{:.3f}, {:.3f}]'.format(*u))
        u = u + theta[0]
        print('\t(theta[1] * plot_x) + theta[0] = [{:.3f}, {:.3f}]'.format(*(u)))
        d = v * u
        print('\tDecision boundary = [{:.3f}, {:.3f}] '.format(*d))


        #Calculate the decision boundary
        #Suppose theta[-25,161, 0.21, 0.201]
        # X
        plot_y = (-1. / theta[2]) * (theta[1] * plot_x + theta[0])

        plt.plot(plot_x, plot_y)
        plt.legend(['Admitted', 'Not Admitted', 'Decision Boundary'])
        plt.xlim([20, 120])
        plt.ylim([20, 120])
