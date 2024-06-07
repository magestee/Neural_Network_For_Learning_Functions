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

def leaky_relu(a, alpha = 0.1):
    if a > 0:
        return a
    else:
        return alpha * a
