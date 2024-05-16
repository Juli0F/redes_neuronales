import random

from algorithm.activation_function import ActivationFunction


class Neuronal(object):
    def __init__(self, input_size, output_size, hidden_layers, activation='sigmoid', output_activation='step'):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.layers = [input_size] + hidden_layers + [output_size]
        self.weights = []
        self.__init_weights__()
        self.__activation_function__(activation, output_activation)

    """
    Inicializando pesos 
    generando valores aleatorios entre -1 y 1
    los pesos se generan para todas las capas
    """
    def __init_weights__(self):
        self.weights = []
        for i in range(len(self.layers) - 1):
            layer_weights = [[random.uniform(-1, 1) for _ in range(self.layers[i + 1])] for _ in
                             range(self.layers[i])]

            self.weights.append(layer_weights)

    """
    asignando funcion de activacion 
    """
    def __activation_function__(self, activation,output_activation):
        function = ActivationFunction()
        if activation == 'sigmoid':
            self.activation = function.sigmoid
            self.activation_derivative = function.sigmoid_derivative
        else :
            self.activation = function.tanh
            self.activation_derivative = function.tanh_derivative

        self.output_activation = function.step if output_activation == 'step' else function.identity
        self.output_activation_derivative = function.step_derivative if output_activation == 'step' else (
            function.identity_derivative)

    """
    Para cada capa, calcula las activaciones de las 
    neuronas basándose en las activaciones de la capa anterior
    """
    def forward(self, inputs):
        activations = [inputs]
        for weights in self.weights:
            new_activations = []
            for neuron_weights in weights:
                activation = sum(w * act for w, act in zip(neuron_weights, activations[-1]))
                new_activations.append(self.activation(activation))
            activations.append(new_activations)
        return activations

    """
    de momento no me esta funcionando, la idea es de que cuando encuentra errror en la capa de salida
    regresar ese error en cada, la idea es ajustar los pesos en cada capa
    
    el error creo que esta en esta parte:
    self.weights[i + 1]], al aplicar zip solo va a iterar hasta donde se lograron hacer parejas,
    es decir la lista mas corta
    """
    def backpropagation(self, activations, target, learning_rate):
        deltas = [None] * len(self.weights)

        deltas[-1] = [(a - t) * self.output_activation_derivative(a)
                      for a, t in zip(activations[-1], target)]

        for i in range(len(deltas) - 2, -1, -1):
            deltas[i] = []
            for j in range(len(self.weights[i])):
                ### Me esta dando error, verificar los pesos
                ### posibles soluciones,
                ### verificar que deltas y  self.weights tengan la misma longitud
                error = sum(d * w for d, w in zip(deltas[i + 1], [w[j] for w in self.weights[i + 1]]))
                deltas[i].append(error * self.activation_derivative(activations[i + 1][j]))


        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    self.weights[i][j][k] -= learning_rate * activations[i][j] * deltas[i][k]


    def calculate_loss(self, X, y):
        total_loss = 0
        for x, target in zip(X, y):
            predicted = self.forward(x)[-1]
            total_loss += sum((p - t) ** 2 for p, t in zip(predicted, target))
        return total_loss / len(X)


    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            for x, target in zip(X, y):
                activations = self.forward(x)
                print("Activacion Correcta[-1]:", activations[-1])
                self.backpropagation(activations, target, learning_rate)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {self.calculate_loss(X, y)}')

