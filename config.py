entradas = 2
salidas = 1
capas_ocultas = 2
funcion_de_activacion = 'sigmoid'
funcion_de_activacion_salida = 'identity'
epocas = 10000

# prueba para and
training_data = [
    ([0, 0], [0]),
    ([0, 1], [0]),
    ([1, 0], [0]),
    ([1, 1], [1])
]

test_data = [
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
]