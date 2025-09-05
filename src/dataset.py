import torch
from torch.utils.data import Dataset
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class NILMDataset(Dataset):
    def __init__(self, df, input_col="tot_active_pow", target_cols=None, window_size=256, stride=1):
        self.inputs = df[input_col].values.astype(float)
        self.targets = df[target_cols].values.astype(float)
        self.window_size = window_size
        self.stride = stride

    def __len__(self):
        return (len(self.inputs) - self.window_size) // self.stride + 1

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.window_size
        x = self.inputs[start:end]
        y = self.targets[start + self.window_size // 2]
        return torch.tensor(x).unsqueeze(-1).float(), torch.tensor(y).float()
