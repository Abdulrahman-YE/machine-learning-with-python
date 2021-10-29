from matplotlib import pyplot
import numpy as np
X_mean = np.zeros((0,2))
X_std = np.zeros((0,2))

def read_data(path, delimiter):
    data = np.genfromtxt(path, delimiter = delimiter, dtype=np.int32)
    print('Data  ->', data[1])

    #Feature Matrix ()
    X = data[:, 0:-1]
    print('X  ->', X[1])

    #Normalize the feature, since the size of the house is about 1000 time the number of bedrooms
    X = feature_normalize(X)
    #Output Value
    Y = data[:, -1:]
    print('Y  ->', Y[1])

    #No. of Training Examples
    m = data.shape[0]
    #No. of Features
    n = data.shape[1] -1
    #Vecotor of ones for Theta0
    X0 = np.ones((m,1), dtype=np.int32)
    print('X0  ->', X0[1])

    if X.ndim == 1:
        X = X[:, np.newaxis]

    #Add X0 to matrix
    X = np.hstack((X0, X))
    #Create thetas vector initialized with 0
    T = np.zeros((n + 1,1), dtype='int32')
    print('X  ->', X[1])
    # print('Y shape ->', Y.shape)
    # print('T shape ->', T.shape)

    return (X, Y, T)

def cost(X, Y, T):
    #Get number of traning examples
    m = X.shape[0]
    #J = 1/2 * m * sum((X * theta) - y) ** 2)
    #x.dot(theta)
    H = predict(X, T)
    #(x.dot(theta) - y)
    error = H - Y
    #(x.dot(theta) - y) ** 2
    error_sqr = np.matmul(error.T, error)
    #sum((x.dot(theta) - y) ** 2)
    error_sqr_sum = np.sum(error_sqr)
    #1/2 * m * sum((x.dot(theta) - y) ** 2)
    J =  error_sqr_sum /(2*m)
    return J

def partial_dervative(X, Y, T):
    #Get number of traning examples
    m = X.shape[0]
    #partial_dervative = 1/m * ( (X*theta) - y) * (XT)
    # X*theta
    H = predict(X, T) 
    # X * theta - y
    diff = H - Y
    # ((X*theta) - y) * (X`) )
    dervative = np.matmul(X.T, diff)
    # 1/m * ( (X*theta) - y) * (XT)
    return dervative / m

def gradiant_descent(X, Y, T, iterations=1500, alpha=0.01):
    # Used later to plot the cost for each iteration
    J_history = np.zeros((iterations, 1))
    for i in range(iterations):
        #Compute the cost for new theta'T'
        j = cost(X, Y, T)
        #Traking the cost in each iteration
        J_history[i] = j
        # Batch gradiant descent to compute new theta'T'
        T = T - alpha * partial_dervative(X, Y, T)
        print('T -> ', T.T)    
        print('Cost -> ', j)
        if np.isnan(T).any():
            break   

    return (T, J_history)

#Model or Hypothesis h(x) = X(m, n+1) * Theta(n+1, 1)
def predict(X, T):
    return np.matmul(X, T)

def feature_normalize(X):
    global X_mean, X_std
    X_mean = X.mean(axis=0)
    X_norm = X - X_mean
    X_std = X.std(axis=0)
    X_norm = np.divide(X_norm, X_std)
    return X_norm


def main():
    global X_mean, X_std
    X,Y,T = read_data('ex1data2.txt', ',')

    print('Cost with T initalized to zeros = ', cost(X, Y, T))
    T, J_history =gradiant_descent(X,Y,T, 50, 0.1)
   

    # Plot the convergence graph
    pyplot.plot(np.arange(len(J_history)), J_history, lw=2)
    pyplot.xlabel('Number of iterations')
    pyplot.ylabel('Cost J')
    pyplot.show()

    # Display the gradient descent's result
    print('theta computed from gradient descent: {:s}'.format(str(T)))

    # Estimate the price of a 1650 sq-ft, 3 br house
    # ======================= YOUR CODE HERE ===========================
    # Recall that the first column of X is all-ones. 
    # Thus, it does not need to be normalized.

    house = [[1650, 3]] - X_mean
    house = np.divide(house, X_std)
    house = np.concatenate([np.ones((1,1)) , house],axis=1)
    price =   predict(house, T)

    # ===================================================================
    print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): ${:.0f}'.format(float(price)))



main()