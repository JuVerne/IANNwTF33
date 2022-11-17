import numpy as np
import tensorflow as tf
# import tensorflow_datasets as tfds
import random
from matplotlib import pyplot as plt

# create training data
x = np.random.rand(100)
t = np.array([i**3 - i**2 for i in x])
t_data = list(zip(x, t))

# activation functions


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))


def relu(x):
    return max(0, x)


def relu_prime(x):
    return 1 if relu(x) > 0 else 0


# we define two classes: one for the output layer and then one for each hidden layer

class OutputLayer(object):

    def __init__(self, n_units, input_units):

        # weights connecting the last layer to the current one
        self.weights = np.random.rand(n_units, input_units)

        # biases for the current layer
        self.bias = np.array([0]*n_units)

    def forward_step(self, X):

        return np.maximum(0, np.dot(self.weights, X) + self.bias)

    def backward_step(self, target, outputs):

        self.errors = self.cost_derivative(outputs, target)
        return self.errors

    def cost_derivative(self, output_activations, y):

        return (output_activations-y)


class HiddenLayer(object):

    def __init__(self, n_units, input_units):

        # weights connecting the last layer to the current one
        self.weights = np.random.rand(n_units, input_units)

        # biases for the current layer
        self.bias = np.array([0]*n_units)

        self.layer_input = None
        self.layer_preactivation = None
        self.layer_activation = None

    def forward_step(self, X):

        self.activations_last = X
        self.activations = np.maximum(0, np.dot(self.weights, X) + self.bias)

        # bias missing - took out because it caused wrong dimensions

        return np.maximum(0, np.dot(self.weights, X))

    def backward_step(self, next_layer_errors, next_layer_weights, last_layer_activations):

        # first get the error signals of the current layer from the layer before (higher index)

        self.errors = np.dot(next_layer_weights.T, next_layer_errors) * \
            sigmoid_prime(np.dot(self.weights, last_layer_activations.T))

        # transform to a matrix with one zero column (for later calculation)

        errors_matrix = np.column_stack((self.errors, np.zeros(len(self.errors))))

        # second, create a matrix from the activations of the previous layer

        activations_last_matrix = np.row_stack(
            (last_layer_activations, np.zeros(np.shape(last_layer_activations)[0])))

        # third, multiply both matrices to get a matrix containing the partial derivative of the cost function w.r.t. each weight
        # between the two layers.

        weights_jacobi_matrix = np.matmul(errors_matrix, activations_last_matrix)
        return weights_jacobi_matrix


class MLP(object):

    def __init__(self, n_layers=1):

        self.n_layers = n_layers
        self.layers = [HiddenLayer(10, 1) for i in range(n_layers)] + [OutputLayer(1, 10)]
        self.activations = None
        self.errors = None

    def forwardpass(self, x):

        self.activations = []
        for layer in self.layers:

            x = layer.forward_step(x)
            # we store all the activation values for each layer since we need them for backprop.
            self.activations.append(x)

        return x

    def backpropagation(self, learning_r, target, outputs):

        # going backward through the list of layers
        self.errors = []
        for i in range(len(self.layers)-1, -1, -1):

            # we treat the final layer slightly differently than the other layers

            gradient_matrix = self.layers[i].backward_step(self.layers[i+1].errors, self.layers[i+1].weights, self.activations[i-1]) if i < len(
                self.layers)-1 else self.layers[i].backward_step(target, outputs)

            # storing the error signals for each layer for later use
            self.errors.append(self.layers[i].errors)

            # updating weights

            self.layers[i].weights = self.layers[i].weights - learning_r*gradient_matrix

    def train(self, n_epochs, data, learning_r):

        loss_epoches = []
        for x in range(n_epochs):

            loss = 0
            for element in data:

                # the data is stored as a list of input-output-tuples, where the first
                # part of the tuple contains the input and the second the desired output

                self.backpropagation(learning_r, element[1], self.forwardpass(element[0]))
                loss += (1/2)*(element[1] - self.forwardpass(element[0]))**2

            loss_epoches.append(loss[0, 0]/100)

        print("average loss: ", loss_epoches)

        # plotting the loss with respect to epochs

        x_axis = 0.5 + np.arange(n_epochs)
        y_axis = loss_epoches

        fig, ax = plt.subplots()
        ax.plot(x_axis, y_axis)
        ax.set(xlim=(0, n_epochs),
               ylim=(0, 0.25))

        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.show()


test = MLP()

epochs = 100
l_rate = 0.01

test.train(epochs, t_data, l_rate)
