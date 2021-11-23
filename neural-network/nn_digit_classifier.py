import os
import random
import numpy as np
import utils
from matplotlib import pyplot as plt
from scipy import io
from scipy import optimize
from digit_classifier import visualize_data
# Part of Pr. Andrew Ng course Machine Leaning in coursera
# Week 4, Exercise 3 , part 2
# In this script we will implement feedforward propagation and prediction of neural networks to recognize handwritten digits

def predict(Theta1, Theta2, X, debug=False):
    """
    Predict the label of an input given a trained neural network.
    
    Parameters
    ----------
    Theta1 : array_like
        Weights for the first layer in the neural network.
        It has shape (2nd hidden layer size x input size)
    
    Theta2: array_like
        Weights for the second layer in the neural network. 
        It has shape (output layer size x 2nd hidden layer size)
    
    X : array_like
        The image inputs having shape (number of examples x image dimensions).
    
    Return 
    ------
    p : array_like
        Predictions vector containing the predicted label for each example.
        It has a length equal to the number of examples.
    
    Instructions
    ------------
    Complete the following code to make predictions using your learned neural
    network. You should set p to a vector containing labels 
    between 0 to (num_labels-1).
     
    Hint
    ----
    This code can be done all vectorized using the numpy argmax function.
    In particular, the argmax function returns the index of the  max element,
    for more information see '?np.argmax' or search online. If your examples
    are in rows, then, you can use np.argmax(A, axis=1) to obtain the index
    of the max for each row.
    
    """
    # Make sure the input has two dimensions
    if X.ndim == 1:
        X = X[None]  # promote to 2-dimensions
    
    # useful variables
    m = X.shape[0]
    num_labels = Theta2.shape[0]

    # You need to return the following variables correctly 
    p = np.zeros(X.shape[0])

    # ====================== YOUR CODE HERE ======================
    # Add intercept column for training set
    a1 = np.concatenate((np.ones((X.shape[0], 1), dtype=float), X), axis=1)
    z2 = a1 @ Theta1.T
    a2 = utils.sigmoid(z2)
    #Add a^0 which is the intercept unit for a2
    a2 = np.concatenate((np.ones((a2.shape[0], 1), dtype=float), a2), axis=1)
    z3 = a2 @ Theta2.T
    a3 = utils.sigmoid(z3)
    # get the highest prediction for each trainging example
    p = np.argmax(a3, axis=1)
    
    if debug:
        print('Predict Function : ')
        print('\tX-dim : ', X.shape)
        print('\ta1-dim : ', a1.shape)
        print('\tTheta1-dim', Theta1.shape)
        print('\tz2-dim : ', z2.shape)
        print('\ta2-dim : ', a2.shape)
        print('\tz3-dim : ', z3.shape)
        print('\ta3-dim : ', a3.shape)
        print('\tp-dim : ', p.shape)
    # =============================================================
    return p

def main():
    """
    Main function of nn_digit_classifier
    """
    debug = False
    # Load the training data
    data = io.loadmat(os.path.join('data', 'ex3data1'))
    X, y = data['X'], data['y'].ravel()
    y[y==10] = 0
    m = y.size
    visualize_data(X)

    #Setup parameter that will be used in this script
    input_layer_size = 400 # Number of input layer units a
    hiden_layer_size = 25 # Number of first hidden layer units a
    num_labels = 10 # Number of output layer units a

    weights = io.loadmat(os.path.join('data', 'ex3weights'))
    # get  the model weight from the dictionary
    # Theta1 has size 25 * 401
    # Theta2 has size 10 * 26
    Theta1, Theta2 = weights['Theta1'], weights['Theta2']
    if debug:
        print('X-dim : ', X.shape)
        print('y-dim', y.shape)
        print('Input layer size : {}\nHidden layer size: {}\nOutput layer size : {}'.format(input_layer_size, hiden_layer_size, num_labels))
        print('weights dictionary keys : ', weights.keys())
        print('Theta1-dim : ', Theta1.shape)
        print('Theta2-dim : ', Theta2.shape)
        # Because of MATLAB legacy the last row represents output unit a of zero is in the last row
        print('First 5 elements of column 1 in Theta2 : ', Theta2[:5, 1])
        print('First 5 elements of last column in Theta2 : ', Theta2[:5, -1])

    Theta2 = np.roll(Theta2, 1, axis=0)
    #2.2 Feedforward Propagation and Prediction
    pred = predict(Theta1, Theta2, X, debug)
    print('Training Set Accuracy : {:.2f}%'.format(np.mean(pred == y) * 100))
    print('Exprected Accuracy : 97.5%')

    # randomly permute examples, to be used for visualizing one 
    # picture at a time
    indices = np.random.permutation(m)
    #Display image one at a time from the training set 
    # while printing the predicted label for the displayed img
    if indices.size > 0:
        i, indices = indices[0], indices[1:]
        utils.display_data(X[i, :], fig_size=(4,4))
        pred = predict(Theta1, Theta2, X[i, :], debug)
        print('Neural Network Prediction : {}'.format(*pred))
    else:
        print('No more images to display!')





if __name__=='__main__':
    main()