import numpy as np
from matplotlib import pyplot as plt

def display_data(X, example_width=None, fig_size=(10, 10)):
    """
    Displays 2D data stored in X in a nice grid
    """
    #Compute rows, cols
    if X.ndim == 2:
        m, n = X.shape
    elif X.ndim == 1:
        n = X.size
        m = 1
        X = X[None] #Promote to a 2 dimensional array
    else:
        raise IndexError('Input X should be 1 or 2 dimensional.')
    
    print('Display Data : ')
    print('Number of training examples m : {}'.format(m))
    print('Number of Features n : {}'.format(n))

    #Convert 400 to 20 width* 20 height pixel img 
    example_width = example_width or int(np.round(np.sqrt(n)))
    print('example_width : {}'.format(example_width))
    example_height = n / example_width
    print('example_height : {}'.format(example_height))

    #Compute number of items to display
    display_rows = int(np.floor(np.sqrt(m)))
    print('display_rows : {}'.format(display_rows))
    display_cols = int(np.ceil(m / display_rows))
    print('display_cols : {}'.format(display_cols))

    fig, ax_array = plt.subplots(nrows=display_rows, ncols=display_cols, figsize=fig_size)
    #Add margin between imgs
    fig.subplots_adjust(wspace=0.025, hspace=0.025)

    ax_array = [ax_array] if m == 1 else ax_array.ravel()

    for i, ax in enumerate(ax_array):
        ax.imshow(X[i].reshape(example_width, example_width, order='F'), cmap='Greys', extent=[0, 1, 0, 1])
        ax.axis('off')

    plt.show()
    print('=============================================')




def sigmoid(z):
    """
    Compute the sigmoid of z
    """
    return 1.0 / (1.0 + np.exp(-z))