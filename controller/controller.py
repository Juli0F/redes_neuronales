from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from algorithm.neuronal_network import NeuralNetwork

app = FastAPI()
nn = NeuralNetwork(3,1)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Layer(BaseModel):
    layer: int
    neurons: int

class NeuronalRequest(BaseModel):
    inputAmount: int
    inputs: List[List[int]]
    outputAmount: int
    outputs: List[List[int]]
    layerAmount: int
    layers: List[Layer]
    activationFunction: str
    outputFunction: str

@app.get("/")
def read_root():
    return {"message": "Hello, World"}

@app.post("/")
def post_root(param: NeuronalRequest):
    nn.num_inputs = param.inputAmount
    nn.num_outputs = param.outputAmount
    for i in range(param.layerAmount):
        nn.add_hidden_layer(param.layers[i].neurons, param.activationFunction)
    nn.set_output_layer(param.outputFunction)
    log = nn.train(list(zip(param.inputs, param.outputs)), 10000, 0.3)

    return log

@app.put("/")
def predict(inputs: List[List[int]]):
    predictions = []
    for test_input in inputs:
        output = nn.predict(test_input)
        print(f'Input: {test_input} -> Output: {output}')
        predictions.append(output)
    return predictions



