import os
from matplotlib import pyplot as plt
import numpy as np
from scipy import optimize
import utils
#Part of Pr. Andrew Ng course Machine Leaning in coursera
#Week 3, Exercise 2
#In this script we will implement logistic regression to whether a student gets admitted into a university

def read_data(path, deli=','):
    """
    Read the data from a file

    Parameters
    path : string
           represent the path of the file
    deli : string
                represent the seperator between values to  read

    """
    data = np.genfromtxt(path, delimiter=deli)
    X = data[:, :-1]
    y = data[:, -1]
    return (X, y)

#
def plot_data(X, y):
    """
    Plots the data points X and y into a new figure. Plots the data 
    points with * for the positive examples and o for the negative examples.
    
    Parameters
    ----------
    X : array_like
        An Mx2 matrix representing the dataset. 
    
    y : array_like
        Label values for the dataset. A vector of size (M, ).
    
    Instructions
    ------------
    Plot the positive and negative examples on a 2D plot, using the
    option 'k*' for the positive examples and 'ko' for the negative examples.    
    """
    #find indices of postive and negative examples
    positive = y == 1
    negative = y == 0
    #Create a new figure
    fig = plt.figure()
    #Plot the data
    plt.plot(X[positive,0], X[positive, 1], 'k*', mfc='b', mec='k', mew=1, lw=2, ms=10)
    plt.plot(X[negative,0], X[negative, 1], 'ko', mfc='y', mec='k', mew=1, ms=8)
    
    



def sigmoid(z):
    """
    Compute sigmoid function given the input z.
    
    Parameters
    ----------
    z : array_like
        The input to the sigmoid function. This can be a 1-D vector 
        or a 2-D matrix. 
    
    Returns
    -------
    g : array_like
        The computed sigmoid function. g has the same shape as z, since
        the sigmoid is computed element-wise on z.
        
    Instructions
    ------------
    Compute the sigmoid of each value of z (z can be a matrix, vector or scalar).
    """

    #Convert the input to numpy array
    z = np.array(z)
    #Returned variable
    g= np.zeros(z.shape)
    e = np.exp(z)
    d = e + 1
    return 1 / d

def design_matrix(X):
    """
    Create the design matrix for data matrix X(m x n)
    where m is the number of examples and n the number of features

    Parameters
    ----------
    X : array_like
        The data matrix with shape m x n

    Returns
    -------
    X_design : array_like
               A matrix of shape m x (n+1).A vector m x 1 of ones is added. 
    """
    m, n = X.shape
    X0 = np.ones((m, 1))
    X_design = np.hstack((X0, X))
    return X_design

def slope(theta, X):
    """
    Compute -1 * (X(m x n+1) . Theta(n+1 x 1))

    Parameters
    ----------
    X : array_like
        Represent the design matrix of the data

    theta : array_like
            A 1xD vector represent thetas values

    Return
    ------
    z : array_like
        a 1xD vector result of multiplying X . theta
    """
    z = np.matmul(X, theta )
    z = -1 * z
    return z

def cost_function(theta, X, y):
    """
    Compute cost and gradient for logistic regression. 
    
    Parameters
    ----------
    theta : array_like
        The parameters for logistic regression. This a vector
        of shape (n+1, ).
    
    X : array_like
        The input dataset of shape (m x n+1) where m is the total number
        of data points and n is the number of features. We assume the 
        intercept has already been added to the input.
    
    y : arra_like
        Labels for the input. This is a vector of shape (m, ).
    
    Returns
    -------
    J : float
        The computed value for the cost function. 
    
    grad : array_like
        A vector of shape (n+1, ) which is the gradient of the cost
        function with respect to theta, at the current values of theta.
        
    Instructions
    ------------
    Compute the cost of a particular choice of theta. You should set J to 
    the cost. Compute the partial derivatives and set grad to the partial
    derivatives of the cost w.r.t. each parameter in theta.
    """
    # Initialize some useful values
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

    #The gradiant cost
    # h(x) - y
    diff = h - y
    # ( h(x) - y ) * x
    grad = np.matmul(X.T, diff)
    #(1/m) * ( h(x) - y ) * x
    grad = grad / m
    return J, grad

def predict(theta, X):
    """
    Predict whether the label is 0 or 1 using learned logistic regression.
    Computes the predictions for X using a threshold at 0.5 
    (i.e., if sigmoid(theta.T*x) >= 0.5, predict 1)
    
    Parameters
    ----------
    theta : array_like
        Parameters for logistic regression. A vecotor of shape (n+1, ).
    
    X : array_like
        The data to use for computing predictions. The rows is the number 
        of points to compute predictions, and columns is the number of
        features.

    Returns
    -------
    p : array_like
        Predictions and 0 or 1 for each row in X. 
    
    Instructions
    ------------
    Complete the following code to make predictions using your learned 
    logistic regression parameters.You should set p to a vector of 0's and 1's    
    """
    m = X.shape[0] # Number of training examples

    p = np.zeros(m)

    p = slope(theta, X)
    p = sigmoid(p)
    p[p >= 0.5] = 1
    p[p < 0.5] = 0

    return p

def main():
    X, y = read_data(os.path.join('data', 'ex2data1.txt'))
    #Plot the data


    # Test the implementation of sigmoid function here
    z = 0
    g = sigmoid(z)

    print('g(', z, ') = ', g)
    X = design_matrix(X)
    # Initialize fitting parameters
    initial_theta = np.zeros(X.shape[1])

    cost, grad = cost_function(initial_theta, X, y)

    print('Cost at initial theta (zeros): {:.3f}'.format(cost))
    print('Expected cost (approx): 0.693\n')

    print('Gradient at initial theta (zeros):')
    print('\t[{:.4f}, {:.4f}, {:.4f}]'.format(*grad))
    print('Expected gradients (approx):\n\t[-0.1000, -12.0092, -11.2628]\n')

    # Compute and display cost and gradient with non-zero theta
    test_theta = np.array([-24, 0.2, 0.2])
    cost, grad = cost_function(test_theta, X, y)

    print('Cost at test theta: {:.3f}'.format(cost))
    print('Expected cost (approx): 0.218\n')

    print('Gradient at test theta:')
    print('\t[{:.3f}, {:.3f}, {:.3f}]'.format(*grad))
    print('Expected gradients (approx):\n\t[0.043, 2.566, 2.647]')

    #OPTIMIZE
    #============================================================
    #Now we'll use optimize.minimize instead of writting gradiant descent function
    #options that we'll pass to the minimize fucntion 
    options = { 'maxiter' : 400} 
    #costFunction is a reference to the function
    # (X, y) is tuple that represent the values we'll pass to costFunction
    # jac=True indicates that our costFunction return the gradiant of  theta along side the cost
    # method='TNC' indicates that we'll use trucated newton algorithem which equavilant to MATLAB function fminunc which was used in the course 
    # the function return object 'OptimizeResult'
    res = optimize.minimize(cost_function, initial_theta , (X, y), jac=True, method='TNC', options=options )
    # the 'fun' property returns the value of costFunction at optimize theta
    cost = res.fun
    # the 'x' property returns the optimize theta
    theta = res.x

    print('Cost at theta found by optimize.minimize = {:.4f}'.format(cost))
    print('Expected cost (approx): 0.203\n');

    # The '*' operator in format function is used to unpack the array_like object
    print('Optimized theta returned by optimize.minimize =\n\t [{:.2f}, {:.2f}, {:.2f}]'.format(*theta))
    print('Expected theta (approx):\n\t[-25.161, 0.206, 0.201]')


    #PLOT the decision boundary
    #============================
    utils.plot_decision_boundary(plot_data, theta, X, y)
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.show()

    #PREDICT
    #==========================================
    #  Predict probability for a student with score 45 on exam 1 
    #  and score 85 on exam 2 
    prob = sigmoid(slope(theta, [1, 45, 85]))
    print('For a student with scores 45 and 85,'
        'we predict an admission probability of {:.3f}'.format(prob))
    print('Expected value: 0.775 +/- 0.002\n')

    # Compute accuracy on our training set
    p = predict(theta, X)
    print('Train Accuracy: {:.2f} %'.format(np.mean(p == y) * 100))
    print('Expected accuracy (approx): 89.00 %')




if __name__ == "__main__":
   main()