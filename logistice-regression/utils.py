import numpy as np
from matplotlib import pyplot as plt
import sys

def map_feature(X1, X2, degree=6):
    """
    Maps the two input features to quadratic features used in the regularization exercise.

    Returns a new feature array with more features, comprising of
    X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..

    Parameters
    ----------
    X1 : array_like
        A vector of shape (m, 1), containing one feature for all examples.

    X2 : array_like
        A vector of shape (m, 1), containing a second feature for all examples.
        Inputs X1, X2 must be the same size.

    degree: int, optional
        The polynomial degree.

    Returns
    -------
    : array_like
        A matrix of of m rows, and columns depend on the degree of polynomial.
    """

    if X1.ndim > 0:
        out = [np.ones(X1.shape[0])]
    else :
        out = [np.ones(1)]
    no_dim = X1.ndim
    it = 0
    for i in range(1, degree + 1):
        for j in range(i + 1):
            v = (X1 ** (i - j)) * (X2 ** j)
            it = it + 1
            if no_dim > 0:
                if (i-j) == 0:
                    print('|| X\u2082^{}'.format(j), end=' ')
                elif j == 0 :
                    print('|| X\u2081^{} ||'.format(i-j), end=' ')
                else:
                    print('|| X\u2081^{}.X\u2082^{}'.format((i-j), j), end=' ')
                
            out.append(v)
        
        if no_dim >0:
            print('Appended\n')
    if no_dim >0:
        print('Our 2 features transformed into {}-dimensional vector'.format(i))
        print('out shape =', len(out))
        print('X1.ndim = ', X1.ndim)
       
    
    if X1.ndim > 0:
        #Create a 118 * 28 matrix
        return np.stack(out, axis=1)
    else :
        #Create a 1 * 28 vector
        return np.array(out, dtype='float')


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
    else:
        #Grid range
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        z = np.zeros((u.size, v.size))
        #Evalute z = theta * X over the grid
        for i, ui in enumerate(u):
            for j, vj in enumerate(v):
                #Create a matrix 50 * 50, each cell contains  X(1,28) * theta(28, 1) = value(1,1)
                z[i, j] = np.dot(map_feature(ui,vj), theta)
        
        #Important to transpose z before calling contour,
        #Without transposing the plot will be flipped upside down
        z = z.T
        #Plot the decision 
        plt.contour(u, v, z, levels=[0], linewidths=2, colors='g')
        #Plot a filled contour
        #From np.min(z) to 0 is from anywhere to the decision boundary border
        #From 0 to np.max(z) is from decision boundary to its center
        #As I understand :)
        plt.contourf(u, v, z, levels=[np.min(z), 0, np.max(z)], cmap='Greens', alpha=0.4)
        plt.legend(['Admitted', 'Not Admitted', 'Decision Boundary'])



