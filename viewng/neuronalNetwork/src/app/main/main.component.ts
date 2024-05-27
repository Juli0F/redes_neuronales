import { Component, OnInit } from '@angular/core';
import { RedService } from '../service/red.service';

@Component({
  selector: 'app-main',
  templateUrl: './main.component.html',
  styleUrls: ['./main.component.css']
})
export class MainComponent implements OnInit {

  hiddenLayerConfig!: number;
  testCaseConfig!: number;
  outputConfig!: number;
  inputConfig!: number;
  activationFunction!: string;
  outputFunction!: string;
  predictConfig!: number;
  log!: string;

  inputs: number[] = [];
  outputs: number[] = [];
  test: number[] = [];
  layers: { layer: number, neurons: number }[] = [];
  testInputs: number[][] = [];
  testOutputs: number[][] = [];

  predict: number[] = [];
  predictInputs: number[][] = [];
  predictOutputs: number[][] = [];

  constructor(
    private redService: RedService
  ) {}

  ngOnInit(): void {}

  inputConfigChange() {
    this.inputs = Array(this.inputConfig).fill(0).map((x, i) => i);
    this.initializeTestCases();
    this.initializePredictCases();
  }

  outputConfigChange() {
    this.outputs = Array(this.outputConfig).fill(0).map((x, i) => i);
    this.initializeTestCases();
    this.initializePredictCases();
  }

  testConfigChange() {
    this.test = Array(this.testCaseConfig).fill(0).map((x, i) => i);
    this.initializeTestCases();
  }

  predictConfigChange() {
    this.predict = Array(this.predictConfig).fill(0).map((x, i) => i);
    this.initializePredictCases();
  }

  onHiddenLayerConfigChange() {
    this.layers = Array(this.hiddenLayerConfig).fill(0).map((x, i) => ({ layer: i, neurons: 0 }));
  }

  initializeTestCases() {
    this.testInputs = Array(this.testCaseConfig).fill(0).map(() => Array(this.inputConfig).fill(0));
    this.testOutputs = Array(this.testCaseConfig).fill(0).map(() => Array(this.outputConfig).fill(0));
  }

  initializePredictCases() {
    this.predictInputs = Array(this.predictConfig).fill(0).map(() => Array(this.inputConfig).fill(0));
    this.predictOutputs = Array(this.predictConfig).fill(0).map(() => Array(this.outputConfig).fill(0));
  }

  train() {
    const logMessages: string[] = [];

    logMessages.push('Entradas: ' + this.inputConfig);
    logMessages.push('Inputs: ' + JSON.stringify(this.inputs));

    logMessages.push('Salidas: ' + this.outputConfig);
    logMessages.push('Outputs: ' + JSON.stringify(this.outputs));

    logMessages.push('Capas Ocultas: ' + this.hiddenLayerConfig);
    logMessages.push('Neuronas por capa: ' + JSON.stringify(this.layers));

    logMessages.push('Número de Pruebas: ' + this.testCaseConfig);
    logMessages.push('Función de Activación: ' + this.activationFunction);
    logMessages.push('Función de Salida: ' + this.outputFunction);
    logMessages.push('Test Cases Inputs: ' + JSON.stringify(this.testInputs));
    logMessages.push('Test Cases Outputs: ' + JSON.stringify(this.testOutputs));

    this.log = logMessages.join('\n');

    const red = {
      inputAmount: this.inputConfig,
      inputs: this.testInputs,
      outputAmount: this.outputConfig,
      outputs: this.testOutputs,
      layerAmount: this.hiddenLayerConfig,
      layers: this.layers,
      activationFunction: this.activationFunction,
      outputFunction: this.outputFunction
    };

    this.redService.train(red).subscribe({
      next: response => {
        this.log += '\nResponse: ' + JSON.stringify(response);
      }, error: e => {
        this.log += '\nError: ' + JSON.stringify(e);
      }
    });
  }

  testValues() {
    this.redService.testPredict(this.predictInputs).subscribe({
      next: response => {
        console.log("xx->", response);
        this.predictOutputs = response;
      }, error: e => {
        console.log("error", e);
      }
    });
  }
}
