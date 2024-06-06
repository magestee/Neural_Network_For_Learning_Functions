import pandas as pd

data = pd.read_csv('training_data.csv')
inputs = data['input'].values
outputs = data['output'].values
