# Digit Recognition Neural Network From Scratch

A basic neural network for handwritten digit recognition built with Python with NumPy

## Colab demo notebook

[![Demo notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dev079144/digit-recognition/blob/main/notebooks/demo.ipynb)

## How it works

An MLP model class is defined in the 'model' module with methods for initialization, forward and backward propagation, loss calculation, weight updating and saving and loading weights. The MNIST dataset is downloaded and converted to CSV for training and testing. A training loop is executed with hyperparameters set in a designated cell. Model weights are optionally saved and later loaded at inference.

The model is a standard n-layered MLP with customizable hidden layer configuration. ReLU is used as the activation function in the hidden layers, and Softmax is applied to the output layer. Both functions are defined in the 'utils' module. The loss function used is Categorical Cross-Entropy, implemented as a method in the model class within the 'model' module.

## .gitignore setup

To avoid uploading large datasets to GitHub, add this to `.gitignore`:
```
data/
```

## Run

1. Clone the repo:
```
git clone https://github.com/dev079144/digit-recognition.git
cd digit-recognition
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Run notebook
```
jupyter notebook
```
