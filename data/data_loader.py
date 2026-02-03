import pandas as pd
import numpy as np

def load_data(file_path):
    # Load ECG data from a TSV file
    df = pd.read_csv(file_path, sep='\t', header=None)
    data = df.to_numpy()
    
    y = data[:, 0]
    X = data[:, 1:]
    
    return X, y