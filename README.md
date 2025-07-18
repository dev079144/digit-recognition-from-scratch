# Digit Recognition Neural Network From Scratch

A basic neural network for handwritten digit recognition built with Python with NumPy

## Demo Notebooks

### Training

Colab: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dev079144/digit-recognition/blob/main/notebooks/training.ipynb)

### Inference
Colab: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dev079144/digit-recognition/blob/main/notebooks/inference.ipynb?utm_source=github&utm_medium=badge&utm_campaign=readonly)

## Kaggle API Authorization

Download the MNIST dataset using the Kaggle API:

1. Go to https://www.kaggle.com, generate API token and download `kaggle.json`
2. Create a `.secrets/` folder in the project root, and move `kaggle.json` there
3. Add this to your terminal before using Kaggle CLI:
```
export KAGGLE_CONFIG_DIR=$(pwd)/.secrets  # Mac/Linux
```
or
```
set KAGGLE_CONFIG_DIR=%cd%\.secrets       # Windows
```

4. Then download:
```
kaggle datasets download oddrationale/mnist-in-csv
```

## .gitignore setup

To avoid uploading secrets and large datasets to GitHub, add this to `.gitignore`:
```
.secrets/
data/mnist-in-csv/
```

## Run

1. Clone the repo:
```
git clone https://github.com/dev079144/digit-recognition-from-scratch.git
cd digit-recognition-from-scratch
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Run notebook
```
jupyter notebook
```