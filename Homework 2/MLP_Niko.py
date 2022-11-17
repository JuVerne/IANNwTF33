import numpy as np
from matplotlib import pyplot as plt

# Input
x = np.random.uniform(0, 1, (100))

# Target
t = np.array([(i**3)-(i**2) for i in x])


class Layer:

    def __init__(self, units, inputs):

        self.bias = np.zeros((1, inputs), dtype=int)
        self.weights = np.random.rand(inputs, units)

        self.layer_input = None
        self.layer_preactivation = None
        self.layer_activation = None

    def relu(self, x):
        return np.maximum(0, x)

    def forward_step(self, input):
        self.layer_input = input
        self.layer_preactivation = np.dot(self.layer_input, self.weights)+self.bias
        self.layer_activation = self.relu(self.layer_preactivation)

        return self.layer_activation


a = Layer(2, 2)

a.forward_step(np.array([2, 2]))


def backward_step(self, t):
    sig_preactivation_der = []
    ReLU_preactivation_der = np.zeros(self.layer_preactivation.shape)
    for ind, val in enumerate(self.layer_preactivation):
        #sigmoid = 1/(1 + np.exp(-i))
        #sig_preactivation_der.append(sigmoid * (1 - sigmoid))
        if val <= 0:
            ReLU_preactivation_der[ind] = 0.0
        else:
            ReLU_preactivation_der[ind] = 1.0

    # print(self.layer_activation)
    # print(self.layer_input)
    # print(self.layer_preactivation)
    # print(self.layer_activation)
    # print(t)
    # ReLU_preactivation_der.astype(numpy.float64)
    grad_w = np.zeros(self.weight.shape)
    # print(grad_w)
    for i in range(len(self.layer_input)):
        for j in range(len(self.layer_activation)):
            grad_w[i, j] = self.layer_input[i] * \
                ReLU_preactivation_der[j] * (self.layer_activation[j] - t[j])

    grad_b = ReLU_preactivation_der * (self.layer_activation - t)
    grad_i = (grad_b) @ np.transpose(self.weight)

    # update weights and bias
    self.weight = self.weight - 0.01 * grad_w
    self.bias = self.bias - 0.01 * grad_b


# x = np.random.rand(100)
# layer = Layer(100, 3)
# layer.forward_step(x)
# layer.backward_step([1, 1, 1])

plt.style.use('_mpl-gallery')

x_axis = 0.5 + np.arange(100)
y_axis = x

# plot
fig, ax = plt.subplots()

ax.plot(x_axis, y_axis)

ax.set(xlim=(0, 100),
       ylim=(0, 1.2))

plt.show()


for i in range(10, 0, -2):
    print(i, end=" ")
print()

r = (1,2,3,4)
range(r)
