from tkinter import *
from PIL import Image , ImageDraw
import numpy as np
import digit_classifier

height =100
width = 20 


#Pil empty Image
digit = Image.new("1", (int(width), int(height/5)), 255)
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
    draw.line(corrdinates, 0, width=2)
    # Get 400-dim vector that represents the grayscale intensity
    pixels = np.array(digit_copy.getdata())
    # Get all thetas from csv file
    all_theta = np.genfromtxt('thetas.csv', delimiter=',')
    # get predicted value
    pred = digit_classifier.predict_one_vs_all(all_theta, pixels.reshape(1, -1))
    predict_value.config(text=str(pred[0]))
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
canvas.config(width=width, height=height/5)
canvas.grid(row=0, column=0,columnspan=2 ,sticky='N')
#bind an event to function
canvas.bind('<Button-1>', locate_xy)
canvas.bind('<B1-Motion>', create_line)

pred_button = Button(window, text='Predict', command=predict_img )
pred_button.grid(row=1, column=0, sticky='NWE')
clr_button = Button(window, text='Clear', command=clear_canvas, )
clr_button.grid(row=1, column=1, sticky='NWE')

predict_value = Label(window, text='Prediction')
predict_value.grid(row=3, column=0)


window.mainloop()
