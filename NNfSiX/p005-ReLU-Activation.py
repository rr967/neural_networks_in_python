'''
Associated YT tutorial: https://www.youtube.com/watch?v=gmjzbpSVY1A&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=6
'''

import numpy as np 
import nnfs

# from nnfs.datasets import spiral_data  # See for code: https://gist.github.com/Sentdex/454cb20ec5acf0e76ee8ab8448e6266c

def spiral_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

nnfs.init()

X, y = spiral_data(100, 3)   

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

# Activation function = step unit

class Activation_Step:
    def forward(self, inputs):
        for i in inputs:
            if i > 0:
                self.output.append(1)
            else:
                self.output.append(0)

# Activation function = sigmoid

class Activation_Sigmoid:
    def forward(self, inputs):
        for i in inputs:
            if i > 0:
                self.output.append(1 / (1 - math.exp(-1*abs(i))))
            else:
                self.output.append(0)

# Activation function = Rectified Linear Unit ( ReLU )

class Activation_ReLU:
    def forward(self, inputs):
        #for i in inputs:
        #    if i > 0:
        #        self.output.append(i)
        #    else:
        #        self.output.append(0)
        
        self.output = np.maximum(0, inputs)


layer1 = Layer_Dense(2,5)
activation1 = Activation_ReLU()

layer1.forward(X)

#print(layer1.output)
activation1.forward(layer1.output)
print(activation1.output)
