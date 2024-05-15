import math

class ActivationFunction:

    def __init__(self):
        pass


    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def tanh(self, x):
        return math.tanh(x)

    def tanh_derivative(self, x):
        return 1 - math.tanh(x) ** 2

    def identity(self, x):
        return x

    def identity_derivative(self, x):
        return 1

    def step(self, x):
        return 1 if x > 0 else 0

    def step_derivative(self, x):
        return 0

