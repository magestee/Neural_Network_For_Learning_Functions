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

def mse(y, y_hat):
    errors = y - y_hat
    squared_error = np.square(errors)
    mse = np.mean(squared_error)
    return mse

output = feedforward(inputs)

# calculate the loss gradient at the output
dL_dy_hat = 2 * (output - outputs) / len(outputs)

# Backpropagate through the output layer:
dL_dz_o = dL_dy_hat # for linear activation
dL_dW_h2_o = np.dot(a2.T, dL_dz_o)
dL_db_o = np.sum(dL_dz_o, axis=0)

# Backpropagate through the second hidden layer:
dL_da2 = np.dot(dL_dz_o, weights_h2_o.T)
dL_dz2 = dL_da2 * (z2 > 0) + alpha * (z2 <= 0)
dL_dw_h1_h2 = np.dot(a1.T, dL_dz2)
dL_db_h2 = np.sum(dL_dz2, axis=0)

# Backpropagate through the first hidden layer
dL_da1 = np.dot(dL_dz2, weights_h1_h2.T)
dl_dz1 = dL_da1 * (z1 > 0) + alpha * (z1 <= 0)
dL_dW_i_h1 = np.dot(inputs.T, dL_dz1)
dL_db_h1 = np.sum(dL_dz1, axis=0)

# update weights and biases
weights_h2_o -= learning_rate * dL_dW_h2_o
bias_o -= learning_rate * dL_db_o
weights_h1_h2 -= learning_rate * dL_dw_h1_h2
bias_h2 -= learning_rate * dL_db_h2
weights_i_h1 -= learning_rate * dL_dW_i_h1
bias_h1 -= learning_rate * dL_db_h1

print(mse(output, outputs))
