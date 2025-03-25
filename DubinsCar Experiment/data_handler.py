import numpy as np
import pandas as pd

def get_data(file_name: str):

    data = pd.read_csv(file_name, index_col=False)

    # Deserialize the strings back into NumPy arrays
    data['x'] = data['x'].apply(lambda x: np.array(x.split(), dtype=float))
    data['y'] = data['y'].apply(lambda x: np.array(x.split(), dtype=float))

    return data