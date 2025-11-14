import torch
import numpy as np
import pandas as pd
from arguments import LAGS, HORIZON
from torch.utils.data import Dataset


def create_sequences(X, timestamps, lags=LAGS, horizon=HORIZON, freq="1min", flatten=False, stride=1):
    if len(X) != len(timestamps):
        print("Inputs lengths are different, should be equal")
        return

    time_index = pd.to_datetime(timestamps)
    input_seq, target_seq = [], []

    for i in range(lags, len(time_index) - horizon, stride):
        expected = pd.date_range(start=time_index[i - lags], periods=lags + horizon, freq=freq)
        actual = time_index[i - lags : i + horizon]

        if actual.equals(expected):
            if flatten==True:
                x_window = X[i - lags:i].drop(columns=["Mesure traitee CaOl Alcatron"]).values.flatten()
            else:
                x_window = X[i - lags:i].drop(columns=["Mesure traitee CaOl Alcatron"]).values
                
            y_future = X.iloc[i : i + horizon]["Mesure traitee CaOl Alcatron"].values
            input_seq.append(x_window)
            target_seq.append(y_future)
            

    return np.array(input_seq), np.array(target_seq)

def extract_target_points_of_interest(target_seq):
    masks = []
    for seq in target_seq:
        rounded_seq = np.round(seq, decimals=4)
        change_mask = np.zeros_like(rounded_seq, dtype=bool)  
        change_mask[1:] = rounded_seq[1:] != rounded_seq[:-1] 
        change_mask[rounded_seq == 0] = False
        masks.append(change_mask)
    return np.array(masks)

def convert_target_seq_to_binary(target_seq, model_type, threshold=3.5):
    if model_type == 'forecasting':
        return target_seq
    elif model_type == 'classifier':
        binary_targets = []
        for seq in target_seq:
            binary_seq = (seq < threshold).astype(float)
            binary_targets.append(binary_seq)
        return np.array(binary_targets)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Use 'forecasting' or 'classifier'.")


class TimeSeriesDataset(Dataset):
    def __init__(self, X_seq, y_seq, mask):
        self.X = torch.tensor(X_seq, dtype=torch.float32)
        self.y = torch.tensor(y_seq, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.bool)
        
    def __len__(self):
        return(len(self.X))
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.mask[idx]