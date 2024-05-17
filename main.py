from algorithm.neuronal_network import NeuralNetwork

nn = NeuralNetwork(2, 1)

nn.add_hidden_layer(3, activation='tanh') #tanh
nn.add_hidden_layer(4, activation='tanh')

nn.set_output_layer(activation='identity') #identity

training_data = [
    ([0, 0], [0]),
    ([0, 1], [1]),
    ([1, 0], [1]),
    ([1, 1], [0])
]

nn.train(training_data, epochs=10000, learning_rate=0.1)


test_data = [
    [0, 0],
    [1, 1],
    [0, 1],
    [1, 0]
]

for test_input in test_data:
    output = nn.predict(test_input)
    print(f'Input: {test_input} -> Output: {output}')