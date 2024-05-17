from .layer import Layer
from algorithm.activation_function import ActivationFunction



class NeuralNetwork:
    def __init__(self, num_inputs, num_outputs):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.hidden_layers = []
        self.output_layer = None
        self.function = ActivationFunction()
        self.activation_functions = {
            'sigmoid': (self.function.sigmoid, self.function.sigmoid_derivative),
            'tanh': (self.function.tanh, self.function.tanh_derivative),
            'identity': (self.function.identity, lambda x: 1),
            'step': (self.function.step, lambda x: 0)
        }

    def add_hidden_layer(self, num_neurons, activation='sigmoid'):
        print('Adding hidden layer')
        if self.hidden_layers:
            num_inputs = len(self.hidden_layers[-1].neurons)
        else:
            num_inputs = self.num_inputs
        activation_func, activation_derivative = self.activation_functions[activation]
        self.hidden_layers.append(Layer(num_neurons, num_inputs, activation_func, activation_derivative))

    def set_output_layer(self, activation='sigmoid'):
        if self.hidden_layers:
            num_inputs = len(self.hidden_layers[-1].neurons)
        else:
            num_inputs = self.num_inputs
        activation_func, activation_derivative = self.activation_functions[activation]
        self.output_layer = Layer(self.num_outputs, num_inputs, activation_func, activation_derivative)

    def feedforward(self, inputs):
        for layer in self.hidden_layers:
            inputs = layer.activate(inputs)
        return self.output_layer.activate(inputs)

    def backpropagate(self, inputs, expected_outputs, learning_rate):
        outputs = self.feedforward(inputs)

        for i, neuron in enumerate(self.output_layer.neurons):
            error = expected_outputs[i] - neuron.output
            neuron.delta = error * neuron.activation_derivative(neuron.output)
        """
        calcular el error de cada neurona en cada capa
        y enviarlo hacia atras
         en las capas ocultas y enviar el error hacia atra
        """
        for l in reversed(range(len(self.hidden_layers))):
            layer = self.hidden_layers[l]
            for i, neuron in enumerate(layer.neurons):
                error = sum(n.weights[i] * n.delta for n in self.output_layer.neurons)
                neuron.delta = error * neuron.activation_derivative(neuron.output)
        """
        Actualizacion de pesos y bias por cada neurona en cada capa
        """
        for l in range(len(self.hidden_layers)):
            inputs = inputs if l == 0 else [neuron.output for neuron in self.hidden_layers[l - 1].neurons]
            for neuron in self.hidden_layers[l].neurons:
                for j, input in enumerate(inputs):
                    neuron.weights[j] += learning_rate * neuron.delta * input
                neuron.bias += learning_rate * neuron.delta

        inputs = [neuron.output for neuron in self.hidden_layers[-1].neurons] if self.hidden_layers else inputs
        for neuron in self.output_layer.neurons:
            for j, input in enumerate(inputs):
                neuron.weights[j] += learning_rate * neuron.delta * input
            neuron.bias += learning_rate * neuron.delta

    def train(self, training_data, epochs, learning_rate):
        for epoch in range(epochs):
            total_error = 0
            for inputs, expected_outputs in training_data:
                self.backpropagate(inputs, expected_outputs, learning_rate)
                total_error += sum(
                    (expected - output) ** 2 for expected, output in zip(expected_outputs, self.feedforward(inputs)))
            print(f'Epoch {epoch + 1}/{epochs}, Error: {total_error}')

    def predict(self, inputs):
        raw_output = self.feedforward(inputs)
        return [self.function.step(output) for output in raw_output]