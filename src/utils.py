import numpy as np
import pandas as pd
import urllib.request
import os
from mnist import MNIST
import csv
import shutil
import gzip

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

def unzip(path):
    if path.endswith('.gz'):
        unzipped_path = path[:-3]
        if not os.path.exists(unzipped_path):
            with gzip.open(path, 'rb') as inp:
                with open(unzipped_path, 'wb') as out:
                    shutil.copyfileobj(inp, out)
        return unzipped_path
    return path

def cleanup(files, data_folder):
    for file in files:
        zipped_path = os.path.join(data_folder, file)
        unzipped_path = zipped_path[:-3]

        if os.path.exists(zipped_path):
            os.remove(zipped_path)

        if os.path.exists(unzipped_path):
            os.remove(unzipped_path)
        
def download_data(dataset, data_folder='../data'):
    os.makedirs(data_folder, exist_ok=True)

    csv_path = os.path.join(data_folder, f'mnist_{dataset}.csv')
    
    base_url = 'https://ossci-datasets.s3.amazonaws.com/mnist/'

    files = {
        'train':[
            'train-images-idx3-ubyte.gz',
            'train-labels-idx1-ubyte.gz'
        ],
        'test': [
            't10k-images-idx3-ubyte.gz',
            't10k-labels-idx1-ubyte.gz'
        ]
    }

    if dataset not in files:
        raise ValueError("Parameter 'dataset' must be 'train' or 'test'")
    
    if not os.path.exists(csv_path):
        cleanup(files[dataset], data_folder)

        for file in files[dataset]:
            path = os.path.join(data_folder, file)

            if not os.path.exists(path):
                print(f'Downloading {file}...')
                request = urllib.request.Request(base_url + file, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(request) as response, open(path, 'wb') as output:
                    output.write(response.read())
                print(f'Downloaded to {path}')
        
            unzip(path)
    
        mnist_data = MNIST(data_folder)

        images, labels = mnist_data.load_training() if dataset == 'train' else mnist_data.load_testing()
    
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            for label, image in zip(labels, images):
                writer.writerow([label] + image)
    
        cleanup(files[dataset], data_folder)
