import pandas as pd
import numpy as np

data = pd.read_csv('training_set.csv')
inputs = np.array(data['input'].values).reshape(-1, 1)
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
def leaky_relu(z, alpha = 0.01):
    return np.where(z > 0, z, alpha * z)

def feedforward(inputs):
    z1 = weighted_sum(inputs, weights_i_h1, bias_h1)
    a1 = leaky_relu(z1)

    z2 = weighted_sum(a1, weights_h1_h2, bias_h2)
    a2 = leaky_relu(z2)

    # last layers activation function is going to be a identity function.
    o = weighted_sum(a2, weights_h2_o, bias_o)

    return o

output = feedforward(inputs)
print(output)
