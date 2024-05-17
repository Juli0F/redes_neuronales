import config
from algorithm.neuronal_network import NeuralNetwork
#
# nn = NeuralNetwork(2, 1)
#
# nn.add_hidden_layer(3, activation='tanh') #tanh
# nn.add_hidden_layer(4, activation='tanh')
#
# nn.set_output_layer(activation='identity') #identity
#
# training_data = [
#     ([0, 0], [0]),
#     ([0, 1], [1]),
#     ([1, 0], [1]),
#     ([1, 1], [0])
# ]

def initializer_neuronal_network():
    nn = NeuralNetwork(config.entradas,config.salidas)

    for i in range(config.capas_ocultas):
        neuron_amount = input(f"Cantidad de neuronas en la capa {i}:")
        neuron_amount = int(neuron_amount)
        nn.add_hidden_layer(neuron_amount, config.funcion_de_activacion)
    nn.set_output_layer(config.funcion_de_activacion_salida)
    nn.train(config.training_data, config.epocas)

    print("--"*15)
    for test_input in config.test_data:
        output = nn.predict(test_input)
        print(f'Input: {test_input} -> Output: {output}')

if __name__ == "__main__":
    initializer_neuronal_network()