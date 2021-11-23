from tkinter import *
from PIL import Image , ImageDraw
import numpy as np
import digit_classifier
import nn_digit_classifier
from scipy import io
import os

#For Neural Network
weights = io.loadmat(os.path.join('data', 'ex3weights'))
    # get  the model weight from the dictionary
    # Theta1 has size 25 * 401
    # Theta2 has size 10 * 26
Theta1, Theta2 = weights['Theta1'], weights['Theta2']
Theta2 = np.roll(Theta2, 1, axis=0)
#For Logistic Classifier
all_theta = np.genfromtxt('thetas.csv', delimiter=',')


height =100
width = 100


#Pil empty Image
digit = Image.new("1", (int(width/5), int(height/5)), 255)
# draw object to draw in memory only, not visable
corrdinates = []
img_name = 'digit.png'

current_x, current_y = 0, 0

def locate_xy(event):
    global current_x, current_y
    current_x = event.x
    current_y = event.y
    print(current_x, current_y)

def create_line(event):
    global current_x, current_y
    print(current_x, current_y)
    canvas.create_line(current_x, current_y, event.x, event.y, width=2)
    corrdinates.append(current_x)
    corrdinates.append(current_y)
    current_x = event.x
    current_y = event.y

def clear_canvas():
    canvas.delete('all')
    corrdinates.clear()


def predict_img():
    global digit, corrdinates, img_name
    #Get  copy of the img
    digit_copy = digit.copy()
    # bind a drawer to the img
    draw = ImageDraw.Draw(digit_copy)
    #draw line
    draw.line(corrdinates, 0, width=1)
    # Get 400-dim vector that represents the grayscale intensity
    pixels = np.array(digit_copy.getdata()).reshape((1,-1))
    # Get all thetas from csv file
    # get predicted value for logistic classifier
    pred = digit_classifier.predict_one_vs_all(all_theta, pixels)
    l_predict_value.config(text=(" LR Pred : " + str(pred[0])))
    pred = digit_classifier.predict_one_vs_all(all_theta, pixels)
    # get predicted value for neural network
    pred = nn_digit_classifier.predict(Theta1, Theta2, pixels )
    nn_predict_value.config(text=(" NN Pred : " + str(pred[0])))
    digit_copy.save(img_name, 0)
    #Clear 
    
    corrdinates.clear()
    canvas.delete('all')

window = Tk()
window.title('Draw Digit')
window.rowconfigure(0, weight=1)
window.columnconfigure(0, weight=1)
window.config(bg='black')
window.geometry(str(width)+'x' + str(height))

canvas = Canvas(window)
canvas.config(width=width/5, height=height/5)
canvas.grid(row=0, column=0,columnspan=2 ,sticky='N')
#bind an event to function
canvas.bind('<Button-1>', locate_xy)
canvas.bind('<B1-Motion>', create_line)

pred_button = Button(window, text='Predict', command=predict_img )
pred_button.grid(row=1, column=0, sticky='NWE')
clr_button = Button(window, text='Clear', command=clear_canvas, )
clr_button.grid(row=1, column=1, sticky='NWE')

l_predict_value = Label(window, text='NN Prediction')
l_predict_value.grid(row=3, column=0, sticky='NSEW')
nn_predict_value = Label(window, text='Logistic Prediction')
nn_predict_value.grid(row=4, column=0 , sticky='NSEW')


window.mainloop()
