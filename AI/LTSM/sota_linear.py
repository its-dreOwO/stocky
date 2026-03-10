import torch
import torch.nn as nn

class SeriesDecomposition(nn.Module):
    """
    Decouples the input sequence into Trend and Seasonal (Remainder) components.
    """
    def __init__(self, kernel_size):
        super(SeriesDecomposition, self).__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        # x shape: [Batch, Seq_Len, Features] -> [Batch, Features, Seq_Len] for Pool1d
        x_transpose = x.permute(0, 2, 1)
        
        # Padding to maintain sequence length
        front = x_transpose[:, :, 0:1].repeat(1, 1, (self.moving_avg.kernel_size[0] - 1) // 2)
        end = x_transpose[:, :, -1:].repeat(1, 1, self.moving_avg.kernel_size[0] // 2)
        x_padded = torch.cat([front, x_transpose, end], dim=-1)
        
        trend = self.moving_avg(x_padded)
        seasonal = x_transpose - trend
        return seasonal.permute(0, 2, 1), trend.permute(0, 2, 1)

class DLinear(nn.Module):
    def __init__(self, seq_len, pred_len, input_dim):
        super(DLinear, self).__init__()
        self.deconstruction = SeriesDecomposition(kernel_size=25)
        
        # Individual linear layers for each component
        self.linear_seasonal = nn.Linear(seq_len, pred_len)
        self.linear_trend = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: [Batch, Seq_Len, Input_Dim]
        seasonal, trend = self.deconstruction(x)
        
        # Project each feature independently
        seasonal_out = self.linear_seasonal(seasonal.permute(0, 2, 1)).permute(0, 2, 1)
        trend_out = self.linear_trend(trend.permute(0, 2, 1)).permute(0, 2, 1)
        
        # Return only the Close Price prediction (index 3 conventionally)
        # Note: If you want all features predicted, remove the indexing
        return (seasonal_out + trend_out)[:, :, 3] 

# class NLinear(nn.Module):
#     def __init__(self, seq_len, pred_len, input_dim):
#         super(NLinear, self).__init__()
#         self.linear = nn.Linear(seq_len, pred_len)

#     def forward(self, x):
#         # Normalization: Subtract the last observed value
#         seq_last = x[:, -1:, :].detach()
#         x = x - seq_last
        
#         # Linear projection
#         out = self.linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        
#         # Denormalization: Add the last observed value back
#         out = out + seq_last[:, :, :]
#         return out[:, :, 3] # Return predicted Close Price
class NLinear(nn.Module):
    def __init__(self, seq_len, pred_len, input_dim):
        super(NLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Channel Independence: Each of the 13 features gets its own 'brain'
        self.linears = nn.ModuleList([
            nn.Linear(seq_len, pred_len) for _ in range(input_dim)
        ])

    def forward(self, x):
        # x: [Batch, Seq_Len, 13]
        
        # 1. Normalization (NLinear trick)
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        
        # 2. Individual Feature Processing
        # Permute to [Batch, 13, Seq_Len]
        x = x.permute(0, 2, 1)
        
        output = torch.zeros([x.size(0), x.size(1), self.pred_len], dtype=x.dtype).to(x.device)
        for i, layer in enumerate(self.linears):
            output[:, i, :] = layer(x[:, i, :])
        
        # 3. Denormalize
        output = output.permute(0, 2, 1) # [Batch, Pred_Len, 13]
        output = output + seq_last
        
        # Return only the 'Close' price (index 3)
        return output[:, :, 3]