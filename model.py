import pandas as pd
import numpy as np

data = pd.read_csv('training_set.csv')
inputs = data['input'].values
outputs = data['output'].values

np.random.seed(1)

weights_i_h1 = np.random.randn(1,10)
weights_h1_h2 = np.random.randn(10, 10)
weights_h2_o = np.random.randn(10, 1)

bias_h1 = np.zeros(10)
bias_h2 = np.zeros(10)
bias_o = np.zeros(1)

# Takes the array x and dots it to the weights matrix and adds it to be.
def weighted_sum(x, w, b):
    return np.dot(x, w) + b

# This function passes through all of the weighted sums and calculates out put for each layer
def leaky_relu(a, alpha = 0.01):
    return np.where(a > 0, a, alpha * a)

# 

