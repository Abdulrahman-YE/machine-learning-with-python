from matplotlib import pyplot
import numpy as np

X_mean = np.zeros((0,2))
X_std = np.zeros((0,2))

def read_data(path, delimiter):
    data = np.genfromtxt(path, delimiter = delimiter, dtype=np.int32)
    #Feature Matrix ()
    X = data[:, 0:-1]
    #Output Value
    Y = data[:, -1:]
    #No. of Training Examples
    m = data.shape[0]
    #No. of Features
    return (X, Y)

#Add add a vector of ones to the X matrix
def add_ones_column(X):
    m = X.shape[0]
    X0 = np.ones((m, 1))
    return np.hstack((X0, X))

#Initalize theta with zeros
def ini_theta(X):
    T = np.zeros((X.shape[1] + 1, 1))
    return T

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
        if np.isnan(T).any():
            print('Exception occured, T has a NAN value.\nMake sure X faetures are normalized to prevent any type of overflow')
            break   

    return (T, J_history)

#Model or Hypothesis h(x) = X(m, n+1) * Theta(n+1, 1)
def predict(X, T):
    return np.matmul(X, T)

#Perform mean normalization and feature scaling for features in the data set
def feature_normalize(X):
    global X_mean, X_std
    #Caluclate the mean for each feature in the dataset
    X_mean = X.mean(axis=0)
    #Mean normalization
    X_norm = X - X_mean
    #Caluclate the standard devation for each feature in the dataset
    X_std = X.std(axis=0)
    #Feature Scaling
    X_norm = np.divide(X_norm, X_std)
    return X_norm

#Calculate Theta using normal equation
def normal_equation(X, Y):
    #Formula Theta = inv(X`. X) . X`. Y
    inv = np.dot(X.T, X)
    inv = np.linalg.pinv(inv)
    R = np.dot(inv, X.T)
    T = np.dot(R, Y)
    return T

def plot_convergence(J_history):
     # Plot the convergence graph
    pyplot.plot(np.arange(len(J_history)), J_history, lw=2)
    pyplot.xlabel('Number of iterations')
    pyplot.ylabel('Cost J')
    pyplot.show()

#Normalize the price of a house and add a column that contain 1
def process_input(input):
    global X_mean, X_std
    X = input - X_mean
    X = np.divide(X, X_std)
    X = add_ones_column(X)
    #Return the processed input
    #X is ready to perform prediction
    return X




def predict_gradiant():
    print('1#Linear regression using gradiant descent')
    print('==========================================')
    X,Y = read_data('ex1data2.txt', ',')
    T = ini_theta(X)
    X = feature_normalize(X)
    X = add_ones_column(X)
    print('Cost with T initalized to zeros = ', cost(X, Y, T))
    T, J_history =gradiant_descent(X,Y,T, 50, 0.1)
    print('Cost with T after gradiant descent = ', cost(X, Y, T))
   
   #Plot the convergence graph to determine the right value for alpha
    plot_convergence(J_history)

    # Estimate the price of a 1650 sq-ft, 3 br house
    house_data = process_input([[1650, 3]])
    price =   predict(house_data, T)
    print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): ${:.0f}'.format(float(price)))
    print('==========================================')


def predict_normal_eqation():
    print('2#Linear regression using normal equation')
    print('==========================================')
    X,Y = read_data('ex1data2.txt', ',')
    T = ini_theta(X)
    X = add_ones_column(X)
    print('Cost with T initalized to zeros = ', cost(X, Y, T))
    T = normal_equation(X,Y)
    print('Cost with T after normal equation = ', cost(X, Y, T))

    house = add_ones_column(np.array([[1650, 3]]))
    price = predict(house, T)
    print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): ${:.0f}'.format(float(price)))
    print('==========================================')

    

def main():
    predict_gradiant()
    predict_normal_eqation()

main()