import numpy as np

from network_package.layer import Layer
from network_package.activation_functions import Softmax
from network_package.metrics import Mse, Cross_entropy


class NN:
    def __init__(self, input_shape, neurons_num, activations, x_train, y_train, loss):
        self.y_train = y_train
        self.x_train = x_train
        self.input_shape = input_shape
        self.x_train, self.y_train = self.convert_to_numpy_array(self.x_train, self.y_train)
        self.layers_num = len(neurons_num)
        self.neurons_num = neurons_num
        self.activations = activations
        self.loss = loss
        self._build()

    def _build(self):
        self.layers = []

        layer = Layer(shape=(self.input_shape[1], self.neurons_num[0]),
                      activation=self.activations[0])
        self.layers.append(layer)

        for i in range(1, self.layers_num):
            layer = Layer(
                shape=(self.layers[i - 1].shape[1], self.neurons_num[i]),
                activation=self.activations[i])
            self.layers.append(layer)

    def calculate_errors(self):
        preds = self.propagate_forward(self.x_train)
        last_error = self.loss.calculate(self.y_train, preds)
        return last_error

    def update_layers(self, weights):
        for i in range(self.layers_num):
            self.layers[i].update(weights[i])

    def convert_to_numpy_array(self, x_train, y_train):
        return np.reshape(np.array(x_train), (-1, self.input_shape[1])), \
                np.reshape(np.array(y_train), (-1, y_train.shape[1]))
       
    
    def propagate_forward(self, x):
        n = x.shape[0]
        for i in range(0, self.layers_num):
            x = x @ self.layers[i].weights + np.ones(shape=(n, 1)) @ self.layers[i].biases.reshape((1, -1))
            x = self.layers[i].activation.calculate(x)
        return x
