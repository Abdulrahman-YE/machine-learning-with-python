from applicant_admisson import plot_data, read_data, slope, sigmoid
from matplotlib import pyplot as plt
import numpy as np
import utils
import os
from scipy import optimize

#Part of Pr. Andrew Ng course Machine Leaning in coursera
#Week 3, Exercise 2
#In this script we will implement we'll implement regularized logistic regression to predict whether microchips from a fabrication plant passes quality assurance (QA). 
#During QA, each microchip goes through various tests to ensure it is functioning correctly.

def costFunctionReg(theta, X, y, lambda_):
    """
    Compute cost and gradient for logistic regression with regularization.
    
    Parameters
    ----------
    theta : array_like
        Logistic regression parameters. A vector with shape (n, ). n is 
        the number of features including any intercept. If we have mapped
        our initial features into polynomial features, then n is the total 
        number of polynomial features. 
    
    X : array_like
        The data set with shape (m x n). m is the number of examples, and
        n is the number of features (after feature mapping).
    
    y : array_like
        The data labels. A vector with shape (m, ).
    
    lambda_ : float
        The regularization parameter. 
    
    Returns
    -------
    J : float
        The computed value for the regularized cost function. 
    
    grad : array_like
        A vector of shape (n, ) which is the gradient of the cost
        function with respect to theta, at the current values of theta.
    
    Instructions
    ------------
    Compute the cost `J` of a particular choice of theta.
    Compute the partial derivatives and set `grad` to the partial
    derivatives of the cost w.r.t. each parameter in theta.
    """

    # ===================== YOUR CODE HERE ======================
    m = y.shape[0]  # number of training examples
    J = 0
    grad = np.zeros(theta.shape)
    # 1 / (1 + e ^ (X . Theta))
    h = sigmoid(slope(theta, X))
    #1 - y
    u = 1 - y
    # 1 - h(x)
    c = np.ones(m) - h
    #-y * log(h(x))
    right = np.matmul(y, np.log(h))
    right = right * -1
    #(1- y) * log(1 - h(x))
    left = np.matmul(u, np.log(c))
    #-y * log(h(x)) - (1- y) * log(1 - h(x))
    cost = right - left
    #(1/ m) *-y * log(h(x)) - (1- y) * log(1 - h(x))
    J = cost / m
    penalized_theta = np.power(theta[1:], 2)
    penalized_theta= np.sum(penalized_theta)
    scalar = 2*m
    scalar = lambda_ / scalar
    penalized_theta = scalar * penalized_theta
    # cost_term = regularized_term
    J = J + penalized_theta

    #The regularized gradiant cost
    # h(x) - y
    diff = h - y
    # ( h(x) - y ) * x
    grad = np.matmul(X.T, diff)
    #(1/m) * ( h(x) - y ) * x
    grad = grad / m
    #Get all theta except theta0
    penalized_theta = theta[1:]
    #Compute the regularization term
    scalar = lambda_ / m
    penalized_theta = scalar * penalized_theta
    penalized_theta = grad[1:] + penalized_theta
    #add theta0
    grad = np.append(grad[0], penalized_theta)
   
    return J, grad

def main():
    X, y = read_data(os.path.join('data', 'ex2data2.txt'))
    plot_data(X, y)
    plt.xlabel('Test 1 result')
    plt.ylabel('Test 2 result')
    plt.legend(['y=1', 'y=0'], loc='upper right')
    plt.show()

    #One way to fit the data better is to create more features from each data point
    # this higher-dimension feature vector will have a more complex decision boundary
    # and will appear nonlinear when drawn in our 2-dimensional plot.it also more susceptible to overfitting
    # regularization can help combat the overfitting problem.
    # This fucntion also adds a column of ones in the begining , so the intercept term with theta0 is handled
    X = utils.mapFeature(X[:, 0], X[:, 1])

    # Initialize fitting parameters
    initial_theta = np.zeros(X.shape[1])

    # Set regularization parameter lambda to 1
    # DO NOT use `lambda` as a variable name in python
    # because it is a python keyword
    lambda_ = 1

    # Compute and display initial cost and gradient for regularized logistic
    # regression
    cost, grad = costFunctionReg(initial_theta, X, y, lambda_)

    print('Cost at initial theta (zeros): {:.3f}'.format(cost))
    print('Expected cost (approx)       : 0.693\n')

    print('Gradient at initial theta (zeros) - first five values only:')
    print('\t[{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]'.format(*grad[:5]))
    print('Expected gradients (approx) - first five values only:')
    print('\t[0.0085, 0.0188, 0.0001, 0.0503, 0.0115]\n')


    # Compute and display cost and gradient
    # with all-ones theta and lambda = 10
    test_theta = np.ones(X.shape[1])
    cost, grad = costFunctionReg(test_theta, X, y, 10)

    print('------------------------------\n')
    print('Cost at test theta    : {:.2f}'.format(cost))
    print('Expected cost (approx): 3.16\n')

    print('Gradient at test theta - first five values only:')
    print('\t[{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]'.format(*grad[:5]))
    print('Expected gradients (approx) - first five values only:')
    print('\t[0.3460, 0.1614, 0.1948, 0.2269, 0.0922]')

    #OPTIMIZE
    #============================================================
    options = { 'maxiter' : 400} 
    res = optimize.minimize(costFunctionReg, initial_theta , (X, y, lambda_), jac=True, method='TNC', options=options )
    cost = res.fun
    theta = res.x

    print('Cost at theta found by optimize.minimize = {:.4f}'.format(cost))
    print('Optimized theta returned by optimize.minimize =[{:.2f}, {:.2f}, {:.2f}]'.format(*theta))





main()


