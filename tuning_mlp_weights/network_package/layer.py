import numpy as np

from network_package.activation_functions import Sigmoid


class Layer:
    def __init__(self, shape, activation=Sigmoid()):
        self.shape = shape
        self.activation = activation
        self._initialize_weights()
        self.weighted_input = None
        self.output = None

    def _initialize_weights(self, min_val=-1, max_val=1):
        self.weights = np.random.uniform(min_val, max_val, size=self.shape)
        self.biases = np.random.uniform(min_val, max_val, size=(self.shape[1], 1))

    def calculate(self, x):
        self.weighted_input = (self.weights.T @ x) + self.biases

        return self.weighted_input

    def activate(self, x):
        self.output = self.activation.calculate(x)
        return self.output

    def update(self, weights):
        self.weights = np.array(np.reshape(weights, self.shape))