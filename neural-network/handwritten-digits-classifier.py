import os
import numpy as np
from scipy.optimize.optimize import show_options
import utils
from matplotlib import pyplot as plt
from scipy import io
from scipy import optimize

# Part of Pr. Andrew Ng course Machine Leaning in coursera
# Week 4, Exercise 3
# In this script we will implement one-vs-all logistic regression
# and neural networks to recognize handwritten digits


def visualize_data(X):
    # randomly select 100 data points to display
    random_indices = np.random.choice(X.shape[0], 100, replace=False)
    selected_data = X[random_indices, :]

    utils.display_data(selected_data)

def lr_cost_function(theta, X, y, lambda_):
    """
    Computes the cost of using theta as the parameter for regularized
    logistic regression and the gradient of the cost w.r.t. to the parameters.
    
    Parameters
    ----------
    theta : array_like
        Logistic regression parameters. A vector with shape (n, ). n is 
        the number of features including any intercept.  
    
    X : array_like
        The data set with shape (m x n). m is the number of examples, and
        n is the number of features (including intercept).
    
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
    Compute the cost of a particular choice of theta. You should set J to the cost.
    Compute the partial derivatives and set grad to the partial
    derivatives of the cost w.r.t. each parameter in theta
    
    Hint 1
    ------
    The computation of the cost function and gradients can be efficiently
    vectorized. For example, consider the computation
    
        sigmoid(X * theta)
    
    Each row of the resulting matrix will contain the value of the prediction
    for that example. You can make use of this to vectorize the cost function
    and gradient computations. 
    
    Hint 2
    ------
    When computing the gradient of the regularized cost function, there are
    many possible vectorized solutions, but one solution looks like:
    
        grad = (unregularized gradient for logistic regression)
        temp = theta 
        temp[0] = 0   # because we don't add anything for j = 0
        grad = grad + YOUR_CODE_HERE (using the temp variable)
    
    Hint 3
    ------
    We have provided the implementatation of the sigmoid function within 
    the file `utils.py`. At the start of the notebook, we imported this file
    as a module. Thus to access the sigmoid function within that file, you can
    do the following: `utils.sigmoid(z)`.
    
    """
    print('lrCostFunction() :')
    print('\tX-dim : ', X.shape)
    print('\ty-dim : ', y.shape)
    print('\ttheta-dim : ', theta.shape)
    print('\tlambda_ : ', lambda_)
    

    #Initialize some useful values
    m = y.size
    print('\tNumber of training examples : ', m)
    # convert labels to ints if their type is bool
    if y.dtype == bool:
        y = y.astype(int)
    
    # You need to return the following variables correctly
    J = 0
    grad = np.zeros(theta.shape)
    print('\tInitial cost J : ', J)
    
    
    # ====================== YOUR CODE HERE ======================
    # h(X)
    h = utils.sigmoid(X @ theta)
    print('\th-dim : ', h.shape)
    # (1 - y)
    y_minus = 1 - y
    # (1 - h(X))
    h_minus = 1 - h
    # Get log(h(x))
    h_log = np.log(h)
    # Get log(1 - h(x))
    h_minus = np.log(h_minus)
    # -y * log(h(X)) - ( 1 - y) * log(1 - h(X)) 
    S = ((-1 * y) * h_log) - (y_minus * h_minus)
    S = np.sum(S)
    J = (1 / m) * S
    print('\tUnregularized cost :', J)
    ## ----Regularized-Cost
    penalized_theta = np.power(theta[1:], 2)
    penalized_theta = np.sum(penalized_theta)
    penalized_theta = (lambda_ / (2 * m)) * penalized_theta
    J = J + penalized_theta 
    print('\tRegularized cost :', J)

    #----Gradiant----
    # h(X) - y
    error = h - y
    # X.T * (h(X) - y)
    # to get partial derivative for all thetas
    grad = X.T @ error
    grad = (1 / m) * grad
    print('\tUnregularized gradiant(1st 5 elements) :')
    print('\t\t', grad[:5])
    ## ----Regularized-Gradiant
    penalized_theta = theta
    # because we don't add anything for j = 0
    penalized_theta[0] = 0 
    grad = grad + ((lambda_ / m) * penalized_theta)
    print('\tRegularized gradiant(1st 5 elements) :')
    print('\t\t', grad[:5])
    print('=============================================')
        
    # =============================================================
    return J, grad


def one_vs_all(X, y, num_labels, lambda_):
    """
    Trains num_labels logistic regression classifiers and returns
    each of these classifiers in a matrix all_theta, where the i-th
    row of all_theta corresponds to the classifier for label i.
    
    Parameters
    ----------
    X : array_like
        The input dataset of shape (m x n). m is the number of 
        data points, and n is the number of features. Note that we 
        do not assume that the intercept term (or bias) is in X, however
        we provide the code below to add the bias term to X. 
    
    y : array_like
        The data labels. A vector of shape (m, ).
    
    num_labels : int
        Number of possible labels.
    
    lambda_ : float
        The logistic regularization parameter.
    
    Returns
    -------
    all_theta : array_like
        The trained parameters for logistic regression for each class.
        This is a matrix of shape (K x n+1) where K is number of classes
        (ie. `numlabels`) and n is number of features without the bias.
    
    Instructions
    ------------
    You should complete the following code to train `num_labels`
    logistic regression classifiers with regularization parameter `lambda_`. 
    
    Hint
    ----
    You can use y == c to obtain a vector of 1's and 0's that tell you
    whether the ground truth is true/false for this class.
    
    Note
    ----
    For this assignment, we recommend using `scipy.optimize.minimize(method='CG')`
    to optimize the cost function. It is okay to use a for-loop 
    (`for c in range(num_labels):`) to loop over the different classes.
    
    Example Code
    ------------
    
        # Set Initial theta
        initial_theta = np.zeros(n + 1)
      
        # Set options for minimize
        options = {'maxiter': 50}
    
        # Run minimize to obtain the optimal theta. This function will 
        # return a class object where theta is in `res.x` and cost in `res.fun`
        res = optimize.minimize(lrCostFunction, 
                                initial_theta, 
                                (X, (y == c), lambda_), 
                                jac=True, 
                                method='TNC',
                                options=options) 
    """
    print('One vs All Classification : ')
    # Some useful variables
    m, n = X.shape
    print('\tNumber of training examples m : ', m)
    print('\tNumber of training examples n : ', n)

    # You need to return the following variables correctly 
    all_theta = np.zeros((num_labels, n + 1))
    print('\tall_theta-dim : ', all_theta.shape)
    print('\tEach row in all_theta represents one classifier parameters (theta)')

    # Add ones to the X data matrix
    X = np.concatenate([np.ones((m, 1)), X], axis=1)

    # ====================== YOUR CODE HERE ======================
    # Set options for minimize
    options = {'maxiter' : 50, 'disp' : True}

    for c in range(num_labels):
        inital_theta = np.zeros(n + 1)
        # Run minimize to obtain the optimal theta. This function will 
        # return a class object where theta is in `res.x` and cost in `res.fun`
        res = optimize.minimize(lr_cost_function, inital_theta, (X, (y == c), lambda_), jac=True, method='TNC', options=options )
        print('Cost of classifying ' + c + ' digit : ' + res.fun)
        inital_theta = res.x
        all_theta[c] = inital_theta

    # ============================================================
    print('==========================')
    return all_theta




def test_cost_function():
    #Test values for the parameters theta
    theta_t = np.array([-2, -1, 1, 2], dtype=float)
    # Test values for the inputs
    X_t = np.concatenate([np.ones((5, 1)), np.arange(1, 16).reshape(5, 3, order='F')/ 10.0], axis=1)
    # Test values for the labels
    y_t = np.array([1, 0, 1, 0, 1])
    # Test values for the regularization parameter
    lambda_t = 3

    J, grad = lr_cost_function(theta_t, X_t, y_t, lambda_t)
    print('Test cost function :')

    print('\tCost         : {:.6f}'.format(J))
    print('\tExpected cost: 2.534819')
    print('\t-----------------------')
    print('\tGradients:')
    print('\t [{:.6f}, {:.6f}, {:.6f}, {:.6f}]'.format(*grad))
    print('\tExpected gradients:')
    print('\t [0.146561, -0.548558, 0.724722, 1.398003]')
    print('=============================================')


def main():
    # 20x20 Imput images of digits
    input_layer_size = 400

    # 10 labels, from 1 to 10 ( "0" is mapped to label "10")
    num_labels = 10

    # Training data stored in arrays X, y
    # loadmat() returns a dictionary with keys containing
    # the variable names within the  .mat file.
    # 5000 training examples , each is 20 x 20 pixels grayscale image of the digit.
    # Each pixil is represented by a floating point number indicating
    # the grayscale intensity at that location.
    # The 20 x 20 pixels in unrolled into 400-dimensional vector.
    # Each of these training examples become a single row in our matrix X.
    # X is (5000 x 400) matrix.
    # The second part of the training set is a 5000-dimensional vector
    # y that conatins labels for the training set
    data = io.loadmat(os.path.join('data', 'ex3data1'))
    print('The keys of the ex3data1.mat ', data.keys())
    X = np.array(data['X'])
    print('X-dimensions : ', X.shape)
    y = np.array(data['y'].ravel())
    # Without using ravel() the shape is : (5000, 1)
    # so ravel() used to change 2-dim or mutli-dim array into a contigous flattend array
    print('y-dimensions : ', y.shape)
    print('y-labels : ')
    print(np.unique(y))
    # The zero digit is labeled as 10 in the original data file
    # because the data was used in MATLAB where there is no index 0
    # Change the zero digit label to 0
    y[y == 10] = 0
    print('y-labels after mapping zero digit to label 0 : ')
    print(np.unique(y))

    #The number of training examples
    m = y.size

    #------------------------------
    #1.2 Visualizing the data
    visualize_data(X)
    #1.2 Victorizing Logistic Regression
    test_cost_function()
    #1.4 One-vs-All Classification


if __name__ == "__main__":
    main()
