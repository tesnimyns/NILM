import torch
import torch.nn as nn
from src.dataset import PositionalEncoding  # Assurez-vous que ce fichier existe

class  TransformerMultiOutputNILM(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, num_appliances, dropout):
        super().__init__()
        self.input_embedding = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4, 
            dropout=dropout, batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_mlp = nn.Sequential(
            nn.Linear(d_model, 128), 
            nn.ReLU(), 
            nn.Dropout(dropout), 
            nn.Linear(128, num_appliances)
        )

    def forward(self, main_sequence):
        embedded_seq = self.input_embedding(main_sequence)
        pos_encoded_seq = self.pos_encoder(embedded_seq)
        transformer_out = self.transformer_encoder(pos_encoded_seq)
        center_out = transformer_out[:, main_sequence.shape[1] // 2, :]
        prediction = self.output_mlp(center_out)
        return prediction

