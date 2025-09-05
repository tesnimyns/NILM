import torch
import torch.nn as nn
import math
import torch.nn.functional as F

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
        # x shape: (batch_size, sequence_length, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class GeLU2(nn.Module):
    """
    Squared GELU activation as described in the paper.
    """
    def forward(self, x):
        return F.gelu(x) * F.gelu(x)

class DepthwiseSeparableConv1d(nn.Module):
    """
    1D Depth-wise separable convolution.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, features)
        # Permute to (batch_size, features, sequence_length) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.depthwise(x)
        x = self.pointwise(x)
        # Permute back to (batch_size, sequence_length, features)
        x = x.permute(0, 2, 1)
        return x

class LinearAttention(nn.Module):
    """
    Linear Attention mechanism based on kernel function as described in the paper.
    Instead of QKT, it computes phi(Q) * phi(K)T.
    """
    def __init__(self, d_model, n_heads, dropout=0.1, kernel_size=3):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.conv_q = DepthwiseSeparableConv1d(self.head_dim, self.head_dim, kernel_size, padding=kernel_size//2)
        self.conv_k = DepthwiseSeparableConv1d(self.head_dim, self.head_dim, kernel_size, padding=kernel_size//2)
        self.conv_v = DepthwiseSeparableConv1d(self.head_dim, self.head_dim, kernel_size, padding=kernel_size//2)

        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x, batch_size):
        return x.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

    def _combine_heads(self, x, batch_size):
        return x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

    def _phi_kernel(self, x):
        """
        Feature map phi = ELU(x) + 1, ensuring positive values.
        """
        return F.elu(x) + 1

    def forward(self, query, key, value):
        batch_size = query.shape[0]
        seq_len_q = query.shape[1] # Use for clarity
        seq_len_k = key.shape[1]   # Use for clarity

        Q = self.wq(query)
        K = self.wk(key)
        V = self.wv(value)

        Q_heads = self._split_heads(Q, batch_size) # (B, H, S_q, D_h)
        K_heads = self._split_heads(K, batch_size) # (B, H, S_k, D_h)
        V_heads = self._split_heads(V, batch_size) # (B, H, S_k, D_h)

        # Apply 1D Depth-wise separable convolutions
        # Input to conv_x is (batch_size * n_heads, seq_len, head_dim)
        # Output is (batch_size * n_heads, seq_len, head_dim)
        Q_conv = self.conv_q(Q_heads.reshape(batch_size * self.n_heads, seq_len_q, self.head_dim))
        K_conv = self.conv_k(K_heads.reshape(batch_size * self.n_heads, seq_len_k, self.head_dim))
        V_conv = self.conv_v(V_heads.reshape(batch_size * self.n_heads, seq_len_k, self.head_dim))

        Q_conv = Q_conv.view(batch_size, self.n_heads, seq_len_q, self.head_dim)
        K_conv = K_conv.view(batch_size, self.n_heads, seq_len_k, self.head_dim)
        V_conv = V_conv.view(batch_size, self.n_heads, seq_len_k, self.head_dim)

        # Apply kernel function
        phi_Q = self._phi_kernel(Q_conv) # (B, H, S_q, D_h)
        phi_K = self._phi_kernel(K_conv) # (B, H, S_k, D_h)

        # The core of linear attention: A = (phi(Q) @ (phi(K).T @ V)) / (phi(Q) @ sum(phi(K), dim=seq_len))

        # 1. Compute the "context" vector: phi_K.T @ V
        # Sum over the sequence length of K
        # K_T_V_context shape: (B, H, D_h, D_h)
        K_T_V_context = torch.einsum("bhkd,bhks->bhds", phi_K, V_conv) # d_h * v_h

        # 2. Compute the attention numerator: phi_Q @ K_T_V_context
        # Numerator shape: (B, H, S_q, D_h)
        numerator = torch.einsum("bhqd,bhds->bhqs", phi_Q, K_T_V_context)

        # 3. Compute the normalization term (denominator): phi_Q @ sum(phi_K, dim=seq_len)
        # Sum phi_K over sequence length (k) to get (B, H, D_h)
        sum_phi_K = torch.sum(phi_K, dim=-2) # sum over sequence_length_k, result (B, H, D_h)

        # Denominator: phi_Q @ sum_phi_K
        # The equation for the denominator for linear attention is actually a broadcasted dot product
        # (B, H, S_q, D_h) dot (B, H, D_h) -> (B, H, S_q)
        # This will then be used to normalize each 'head_dim' component.
        denominator = torch.einsum("bhqd,bhd->bhq", phi_Q, sum_phi_K) + 1e-6 # (B, H, S_q)
        
        # 4. Final attention output
        # Divide numerator by denominator. Need to unsqueeze denominator to broadcast correctly.
        # (B, H, S_q, D_h) / (B, H, S_q, 1)
        attn_output = numerator / denominator.unsqueeze(-1)
        
        attn_output = self._combine_heads(attn_output, batch_size)
        attn_output = self.fc_out(attn_output)
        return self.dropout(attn_output)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, ffn_dim=256, kernel_size_conv_ffn=1, kernel_size_attn_conv=3):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = LinearAttention(d_model, n_heads, dropout, kernel_size=kernel_size_attn_conv)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Point-wise Feed-Forward Network with 1D convolutions and GeLU^2
        self.ffn = nn.Sequential(
            DepthwiseSeparableConv1d(d_model, ffn_dim, kernel_size=kernel_size_conv_ffn, padding=kernel_size_conv_ffn//2),
            GeLU2(),
            DepthwiseSeparableConv1d(ffn_dim, d_model, kernel_size=kernel_size_conv_ffn, padding=kernel_size_conv_ffn//2),
            nn.Dropout(dropout) # Apply dropout inside FFN as well
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Layer Normalization before attention
        norm_x = self.norm1(x)
        # Self-attention with residual connection
        attn_output = self.attention(norm_x, norm_x, norm_x)
        x = x + self.dropout(attn_output) # Add dropout to residual path

        # Layer Normalization before FFN
        norm_x = self.norm2(x)
        # FFN with residual connection
        ffn_output = self.ffn(norm_x)
        x = x + self.dropout(ffn_output) # Add dropout to residual path
        return x

class Energformer(nn.Module):
    def __init__(self, input_features=1, d_model=64, n_heads=4, n_layers=4,
                 num_appliances=8, dropout=0.1, max_len=5000,
                 conv_ffn_dim=256, kernel_size_input_conv=3, kernel_size_attn_conv=3, kernel_size_ffn_conv=1):
        super().__init__()
        
        # Input 1D Convolutional Layers (as per Fig. 1)
        # Maps input features (e.g., 1 for tot_active_pow) to d_model
        # Output of first conv layer: Nx64 (from Fig. 1)
        # Output of second conv layer: Nx128 (d_model)
        self.input_conv_layers = nn.Sequential(
            nn.Conv1d(input_features, 64, kernel_size=kernel_size_input_conv, stride=1, padding=kernel_size_input_conv//2),
            nn.ReLU(),
            nn.Conv1d(64, d_model, kernel_size=kernel_size_input_conv, stride=1, padding=kernel_size_input_conv//2),
            nn.ReLU()
        )

        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout, conv_ffn_dim, kernel_size_ffn_conv, kernel_size_attn_conv)
            for _ in range(n_layers)
        ])
        
        # Output Linear Layers (as per Fig. 1, after the 4 stacked blocks)
        # Reduces dimension from d_model to num_appliances
        self.output_linear = nn.Sequential(
            nn.Linear(d_model, 64), # d_model (128) to 64
            nn.ReLU(),
            nn.Linear(64, 32), # 64 to 32
            nn.ReLU(),
            nn.Linear(32, num_appliances) # 32 to num_appliances (8)
        )

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_features) e.g., (32, 256, 1)

        # Permute for Conv1d: (batch_size, input_features, sequence_length)
        x = x.permute(0, 2, 1)
        
        # Input Convolutions
        x = self.input_conv_layers(x) # Output shape: (batch_size, d_model, sequence_length)
        
        # Permute back for Positional Encoding and Transformer: (batch_size, sequence_length, d_model)
        x = x.permute(0, 2, 1)

        # Positional Encoding
        x = self.positional_encoding(x)

        # Transformer Blocks
        for block in self.transformer_blocks:
            x = block(x)

        # The paper's Fig 1 shows the output of the last TransformerBlock (Nx128)
        # goes into a Linear layer (Nx64), then another Linear (Nx32), then final Linear (Nx1).
        # Since we are doing sequence-to-point (or sequence-to-sequence if we predict for each step),
        # and the DataLoader gives target for the middle of the window,
        # we should take the output corresponding to the middle of the sequence.
        
        # Assuming we need to extract a single prediction per window,
        # corresponding to the center of the input sequence.
        # This aligns with the `seq2point` idea mentioned in the paper,
        # where the network predicts "the appliance specific power consumption value in the middle of the window."
        sequence_length = x.size(1)
        middle_index = sequence_length // 2
        x_middle = x[:, middle_index, :] # Shape: (batch_size, d_model)

        # Output Linear Layers
        output = self.output_linear(x_middle) # Shape: (batch_size, num_appliances)
        
        return output