import os
import numpy as np
from utils import relu, relu_derivative, softmax

class MLP:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.n_of_layers = len(self.layer_sizes)
        self.connections = self.n_of_layers - 1
        self.weights = {}
        self.biases = {}
        self.initialize_parameters()

    def initialize_parameters(self):
        # Initialize parameters using He initialization
        
        for i in range(1, self.n_of_layers):
            self.weights[f'W{i}'] = np.random.randn(self.layer_sizes[i], self.layer_sizes[i - 1]) * np.sqrt(2 / self.layer_sizes[i - 1])
            self.biases[f'B{i}'] = np.zeros((self.layer_sizes[i], 1))

    def forward(self, x):
        # Forward pass

        A = x
        activations = {'A0': x}
        pre_activations = {}
        
        for i in range(1, self.connections):
            Z = self.weights[f'W{i}'] @ A + self.biases[f'B{i}']
            A = relu(Z)
            pre_activations[f'Z{i}'] = Z
            activations[f'A{i}'] = A
        
        Z = self.weights[f'W{self.connections}'] @ A + self.biases[f'B{self.connections}']
        A = softmax(Z)
        pre_activations[f'Z{self.connections}'] = Z
        activations[f'A{self.connections}'] = A

        return activations, pre_activations
        
    def loss(self, y, y_h):
        # Cross-entropy loss
        # y: one-hot encoded by label
        # y_h: model prediction

        return -np.sum(y * np.log(y_h + 1e-9)) / y.shape[1]
        
    def backward(self, activations, pre_activations, y):
        # Backward pass
        
        gradients = {}
        L = self.connections
        m = y.shape[1]
        delta = activations[f'A{L}'] - y

        gradients[f'W{L}'] = 1/m * delta @ activations[f'A{L - 1}'].T
        gradients[f'B{L}'] = 1/m * np.sum(delta, axis=1, keepdims=True)

        for l in reversed(range(1, L)):
            delta = self.weights[f'W{l+1}'].T @ delta
            delta = delta * relu_derivative(pre_activations[f'Z{l}'])

            gradients[f'W{l}'] = 1/m * delta @ activations[f'A{l-1}'].T
            gradients[f'B{l}'] = 1/m * np.sum(delta, axis=1, keepdims=True)
        
        return gradients
    
    def update_parameters(self, gradients, learning_rate):
        for i in range(1, self.n_of_layers):
            self.weights[f'W{i}'] -= learning_rate * gradients[f'W{i}']
            self.biases[f'B{i}'] -= learning_rate * gradients[f'B{i}']

    def save(self, path):
        name, extension = os.path.splitext(path)
        unique_path = path
        count = 2

        while os.path.exists(unique_path):
            unique_path = f'{name}_{count}{extension}'
            count += 1

        parameters = {}
        for key, value in self.weights.items():
            parameters[key] = value
        for key, value in self.biases.items():
            parameters[key] = value

        np.savez(path, **parameters)

    def load(self, path):
        parameters = np.load(path)
        weights = {}
        biases = {}

        for key in parameters.files:
            if key in self.weights:
                weights[key] = parameters[key]
            elif key in self.biases:
                biases[key] = parameters[key]

        self.weights = weights
        self.biases = biases
