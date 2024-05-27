from algorithm.neuron import Neuron
class Layer:
    def __init__(self, num_neurons, num_inputs, activation_func, activation_derivative):
        self.neurons = [Neuron(num_inputs, activation_func, activation_derivative) for _ in range(num_neurons)]

    def activate(self, inputs):
        return [neuron.activate(inputs) for neuron in self.neurons]