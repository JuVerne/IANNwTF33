import numpy as np

# Input
x = np.random.uniform(0, 1, (1, 100))

# Target
t = [(i**3)-(i**2) for i in x]


class Layer:

    def __init__(self, n, i):
        self.n_units = n
        self.input_units = i

        self.bias = np.zeros((i, 1), dtype=int)
        self.weights = np.random.rand((i, n))

        self.layer_input = None
        self.preactivation = None
        self.activation = None

    def relu(self, x):
        return x if x > 0 else 0

    def forward_step(self, input):
        layer_input = input
        preactivation = np.dot(layer_input, weights)
        activation = self.relu(preactivation)

    def backward_step:
        #
