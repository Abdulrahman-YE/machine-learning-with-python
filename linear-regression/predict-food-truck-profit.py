from matplotlib import pyplot
import numpy as np


def read_data(path, delimiter):
    data = np.genfromtxt(path, delimiter = delimiter)
    #Feature Matrix ()
    X = data[:, 0:-1]
    #Output Value
    Y = data[:, -1:]
    #No. of Training Examples
    m = data.shape[0]
    #No. of Features
    n = data.shape[1] -1
    #Vecotor of ones for Theta0
    X0 = np.ones((m,1))

    if X.ndim == 1:
        X = X[:, np.newaxis]

    #Add X0 to matrix
    X = np.hstack((X0, X))
    #Create thetas vector initialized with 0
    T = np.zeros((n + 1,1))

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
       

    return (T, J_history)

#Model or Hypothesis h(x) = X(m, n+1) * Theta(n+1, 1)
def predict(X, T):
    return np.matmul(X, T)


def plot_surface_and_contour(X, Y, T):
    #create an array starting from start->-10 until end->10 repated number->100 time 
    theta0_values = np.linspace(-10, 10, 100)
    theta1_values = np.linspace(-1, 4, 100)
    #Get the Length of each array
    t0_len = theta0_values.shape[0]
    t1_len = theta1_values.shape[0]
    #create a new matrix with M*M dimensions
    j_values = np.zeros((t0_len, t1_len))

    for i in range(0, t0_len):
        for j in range(0, t1_len):
            j_values[i, j] = cost(X, Y, [[theta0_values[i]], [theta1_values[j]]])

    #Plot the cost function as a surface   
    fig = pyplot.figure(figsize=(12,5))
    ax = fig.add_subplot(121, projection='3d')
    #We transpose j_values to make the surface flipped
    ax.plot_surface(theta0_values, theta1_values, j_values.T, cmap='viridis')
    ax.set_xlabel("$\Theta_0$")
    ax.set_ylabel("$\Theta_1$")
    ax.set_zlabel("$J(\Theta)$")
    ax.set_title('Surface')
    ax.view_init(45, 120)

    #Plot contour
    ax = pyplot.subplot(122)
    pyplot.contour(theta0_values, theta1_values, j_values.T, linewidths=2, cmap='viridis', levels=np.logspace(-2, 3, 20))
    pyplot.xlabel('$\Theta_0$')
    pyplot.ylabel('$\Theta_1$')
    pyplot.plot(T[0], T[1], 'r+', ms=10, lw=2)
    pyplot.title('Contour, showing minimum')
    pyplot.show()
    pyplot.figure().clear()


def plot(X, Y, T, J_history):
    fig , axis = pyplot.subplots(nrows=1, ncols=2, figsize=(12, 5))
    axis[0].scatter(X[:, 1], Y)
    axis[0].set_title('Predict Income of Food Truck Based on Population')
    axis[0].set_xlabel('Population in 10,000')
    axis[0].set_ylabel("Income in $10,000")
    axis[0].plot(X[:, 1], predict(X, T), 'g-')

    axis[1].plot(range(J_history.shape[0]), J_history, 'r-')
    axis[1].set_title('Cost Change in Each Iteration')
    axis[1].set_xlabel('Iters.')
    axis[1].set_ylabel('Cost')

    pyplot.show()

    


def main():
    X, Y, T= read_data('ex1data1.txt', delimiter=',')

    T, J_history = gradiant_descent(X, Y, T, iterations=1000)

    predict1 = np.matmul([1, 3.5] , T)
    print("For population = 35,000, we predict a profit of {}".format(predict1 * 10000))
    predict2 = np.matmul([1, 7] , T)
    print("For population = 70,000, we predict a profit of {}".format(predict2 * 10000))

    plot(X,Y, T, J_history)

    plot_surface_and_contour(X, Y, T)



main()