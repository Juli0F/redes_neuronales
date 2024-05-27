import math
class ActivationFunction:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    @staticmethod
    def tanh(x):
        return math.tanh(x)

    @staticmethod
    def tanh_derivative(x):
        return 1 - x**2

    @staticmethod
    def identity(x):
        return x

    @staticmethod
    def identity_derivative(x):
        return 1

    @staticmethod
    def step(x):
        return 1 if x >= 0 else 0

    @staticmethod
    def step_derivative(x):
        return 0

