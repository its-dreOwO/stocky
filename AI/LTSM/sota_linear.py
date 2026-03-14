import torch
import torch.nn as nn

class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size):
        super(SeriesDecomposition, self).__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        x_transpose = x.permute(0, 2, 1)
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
        
        self.linear_seasonal = nn.Linear(seq_len * input_dim, pred_len)
        self.linear_trend = nn.Linear(seq_len * input_dim, pred_len)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        batch_size = x.size(0)
        seasonal, trend = self.deconstruction(x)
        
        seasonal = seasonal.reshape(batch_size, -1)
        trend = trend.reshape(batch_size, -1)
        
        return self.dropout(self.linear_seasonal(seasonal)) + self.dropout(self.linear_trend(trend))

class NLinear(nn.Module):
    def __init__(self, seq_len, pred_len, input_dim):
        super(NLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_dim = input_dim
        
        self.temporal_joint = nn.Sequential(
            nn.Linear(seq_len * input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, pred_len * 16) 
        )
        
        self.reversal_mixer = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        
        x_flat = x.reshape(batch_size, -1)
        combined_features = self.temporal_joint(x_flat) 
        
        combined_features = combined_features.reshape(batch_size, self.pred_len, 16)
        
        delta_pred = self.reversal_mixer(combined_features).squeeze(-1) 
        
        target_last = seq_last[:, :, 3] 
        
        return delta_pred + target_last
