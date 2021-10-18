from matplotlib import pyplot
import numpy as np


def read_data(path, delimiter):
    data = np.genfromtxt(path, delimiter = delimiter)
    #Population Vector
    X = data[:, 0:-1].reshape(-1, 1)
    #Price Vector 
    Y = data[:, -1].reshape(-1, 1)
    #No. of Training Examples
    m = data.shape[0]
    #No. of Features
    n = data.shape[1] -1
    return (X, Y, m, n)

def cost(x, theta, y):
    #cost = 1/2 * m * sum((x.dot(theta) - y) ** 2)
    cost = predict(x, theta)
    cost = cost - y
    cost = np.matmul(cost.T, cost)
    cost = np.sum(cost)
    m = x.shape[0]
    cost = cost / (2*m)
    return cost

def dervative(x, theta, y):
    #gr = 1/m * ( (X*theta) - y) * (XT)
    j = predict(x, theta)
    j = j - y
    j = np.matmul(x.T, j)
    j = j / x.shape[0]
    return j

def gradiant_descent(x,theta, y):
    iter = 2000
    alpha = 0.01
    for i in range(iter):
        theta = theta - alpha * dervative(x, theta, y)
        print(cost(x,theta, y))
    return theta

#Model or Hypothesis h(x) = X(m, n+1) * Theta(n+1, 1)
def predict(x, theta):
    return np.matmul(x, theta)


def main():
    X, Y, m, n = read_data('ex1data1.txt', delimiter=',')
    #Initalize the parameter theta to 0, n + 1 features( + 1 is theta0) 
    theta = np.zeros((n + 1,1))

    #To make the Calculation easy we will add a column of 1's to X 
    #To convert h(x) = Theta0 + Theta1*x
    #To h(x) = X * Theta
    X0 = np.ones((m, 1))
    grad_X = np.concatenate((X0, X), axis=1)


    theta = gradiant_descent(grad_X, theta, Y )
    

    #Plot the data
    #Given data(dots)
    pyplot.scatter(X, Y)
    #Predicted data(line)
    pyplot.plot(X, predict(grad_X, theta))
    pyplot.xlabel('Population in 10,000s')
    pyplot.xlabel('Price in $10,000s')

    pyplot.show()


    #
main()