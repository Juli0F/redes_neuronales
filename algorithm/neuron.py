import random


class Neuron:
    def __init__(self, num_inputs, activation_func, activation_derivative):
        self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
        self.bias = random.uniform(-1, 1)
        self.activation_func = activation_func
        self.activation_derivative = activation_derivative
        self.output = 0
        self.delta = 0

    def activate(self, inputs):
        z = sum(w * i for w, i in zip(self.weights, inputs)) + self.bias
        self.output = self.activation_func(z)
        return self.output
