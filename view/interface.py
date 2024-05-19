import tkinter as tk
from tkinter import simpledialog, messagebox, ttk
from algorithm.neuronal_network import NeuralNetwork


class NeuralNetworkApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Network Interface")

        self.nn = None
        self.hidden_layers = []

        self.create_widgets()

    def create_widgets(self):
        self.create_frame = tk.Frame(self.root)
        self.create_frame.pack(pady=10)

        self.create_nn_button = tk.Button(self.create_frame, text="Create Neural Network",
                                          command=self.create_neural_network)
        self.create_nn_button.grid(row=0, column=0, padx=5)

        self.hidden_layers_button = tk.Button(self.create_frame, text="Add Hidden Layer", command=self.add_hidden_layer)
        self.hidden_layers_button.grid(row=0, column=1, padx=5)

        self.set_output_layer_button = tk.Button(self.create_frame, text="Set Output Layer",
                                                 command=self.set_output_layer)
        self.set_output_layer_button.grid(row=0, column=2, padx=5)


        self.train_frame = tk.Frame(self.root)
        self.train_frame.pack(pady=10)

        self.train_nn_button = tk.Button(self.train_frame, text="Train Neural Network",
                                         command=self.train_neural_network)
        self.train_nn_button.grid(row=0, column=0, padx=5)

        self.test_frame = tk.Frame(self.root)
        self.test_frame.pack(pady=10)

        self.test_nn_button = tk.Button(self.test_frame, text="Test Neural Network", command=self.test_neural_network)
        self.test_nn_button.grid(row=0, column=0, padx=5)

        self.data_frame = tk.Frame(self.root)
        self.data_frame.pack(pady=10)

        self.data_text = tk.Text(self.data_frame, height=10, width=50)
        self.data_text.pack(padx=10, pady=10)

    def create_neural_network(self):
        num_inputs = simpledialog.askinteger("Input", "Enter number of inputs:")
        num_outputs = simpledialog.askinteger("Input", "Enter number of outputs:")

        self.nn = NeuralNetwork(num_inputs, num_outputs)
        self.hidden_layers = []

        self.data_text.insert(tk.END, f"Neural Network Created\nInputs: {num_inputs}, Outputs: {num_outputs}\n")

    def add_hidden_layer(self):
        if not self.nn:
            messagebox.showerror("Error", "Create the neural network first")
            return

        num_neurons = simpledialog.askinteger("Input", "Enter number of neurons in the hidden layer:")

        activation_window = tk.Toplevel(self.root)
        activation_window.title("Select Activation Function")

        tk.Label(activation_window, text="Select activation function:").pack(pady=10)

        activation_var = tk.StringVar()
        activation_combobox = ttk.Combobox(activation_window, textvariable=activation_var)
        activation_combobox['values'] = ('sigmoid', 'tanh')
        activation_combobox.pack(pady=10)

        def on_select():
            activation = activation_var.get()
            if activation:
                self.nn.add_hidden_layer(num_neurons, activation)
                self.hidden_layers.append((num_neurons, activation))
                self.data_text.insert(tk.END, f"Added Hidden Layer\nNeurons: {num_neurons}, Activation: {activation}\n")
                activation_window.destroy()
            else:
                messagebox.showerror("Error", "Select an activation function")

        tk.Button(activation_window, text="Select", command=on_select).pack(pady=10)

    def set_output_layer(self):
        if not self.nn:
            messagebox.showerror("Error", "Create the neural network first")
            return

        activation_window = tk.Toplevel(self.root)
        activation_window.title("Select Activation Function")

        tk.Label(activation_window, text="Select activation function:").pack(pady=10)

        activation_var = tk.StringVar()
        activation_combobox = ttk.Combobox(activation_window, textvariable=activation_var)
        activation_combobox['values'] = ('step', 'identity')
        activation_combobox.pack(pady=10)

        def on_select():
            activation = activation_var.get()
            if activation:
                self.nn.set_output_layer(activation)
                self.data_text.insert(tk.END, f"Output Layer Set\nActivation: {activation}\n")
                activation_window.destroy()
            else:
                messagebox.showerror("Error", "Select an activation function")

        tk.Button(activation_window, text="Select", command=on_select).pack(pady=10)

    def train_neural_network(self):
        if not self.nn:
            messagebox.showerror("Error", "Create the neural network first")
            return

        num_samples = simpledialog.askinteger("Input", "Enter number of training samples:")
        training_data = []
        for i in range(num_samples):
            inputs = simpledialog.askstring("Input", f"Enter inputs for sample {i + 1} (space-separated):")
            inputs = list(map(float, inputs.split()))
            outputs = simpledialog.askstring("Input", f"Enter expected outputs for sample {i + 1} (space-separated):")
            outputs = list(map(float, outputs.split()))
            training_data.append((inputs, outputs))
            self.data_text.insert(tk.END, f"Training Sample {i + 1}\nInputs: {inputs}, Outputs: {outputs}\n")

        epochs = simpledialog.askinteger("Input", "Enter number of epochs:")
        learning_rate = simpledialog.askfloat("Input", "Enter learning rate:")

        self.nn.train(training_data, epochs, learning_rate)
        self.data_text.insert(tk.END, f"Training Completed\nEpochs: {epochs}, Learning Rate: {learning_rate}\n")

    def test_neural_network(self):
        if not self.nn:
            messagebox.showerror("Error", "Create and train the neural network first")
            return

        num_tests = simpledialog.askinteger("Input", "Enter number of test samples:")
        results = []
        for i in range(num_tests):
            inputs = simpledialog.askstring("Input", f"Enter inputs for test sample {i + 1} (space-separated):")
            inputs = list(map(float, inputs.split()))
            output = self.nn.predict(inputs)
            results.append(f"Input: {inputs} -> Output: {output}")
            self.data_text.insert(tk.END, f"Test Sample {i + 1}\nInputs: {inputs}, Output: {output}\n")

        result_text = "\n".join(results)
        messagebox.showinfo("Test Results", result_text)


def main():
    root = tk.Tk()
    app = NeuralNetworkApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
