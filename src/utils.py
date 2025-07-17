import numpy as np
import pandas as pd

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    x = x - np.max(x, axis=0, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

def encode(labels, n_of_classes):
    m = labels.shape[0]
    encoded = np.zeros((n_of_classes, m))
    encoded[labels, np.arange(m)] = 1
    return encoded

def process_data(csv_path, normalize=True):
    data = pd.read_csv(csv_path)
    data = np.array(data)
    
    x = data[:, 1:].T
    y = data[:, 0]

    if normalize:
        x = x / 255.0

    n_of_classes = len(set(y))
    
    return x, encode(y, n_of_classes), n_of_classes
