import numpy as np 
import torch
import torch.nn as nn
from arguments import HORIZON 

class LSTMForcast(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layer=1, dropout_rate=0.5, outpout_size=HORIZON, model='forecast'):
        super().__init__()
        self.lstm = nn.LSTM(
                        input_size=input_size, 
                        hidden_size=hidden_size, 
                        num_layers=num_layer,
                        dropout=dropout_rate if num_layer > 1 else 0.0,  # dropout only applies if num_layers > 1 
                        bidirectional=True,
                        batch_first=True)
        self.fc = nn.Linear(hidden_size*2, outpout_size)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate)
        self.model = model
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :]) 
        out = self.fc(out)
        if self.model == 'classifier':
            out = self.sigmoid(out)
        return out
    
if __name__ == "__main__":
    
    model = LSTMForcast(input_size=3, model='classifier')
    x = torch.ones(32 ,100, 3)
    outpout = model(x)
    print(outpout.shape)
    print(x[0,0,:])
    print(outpout[0])
    