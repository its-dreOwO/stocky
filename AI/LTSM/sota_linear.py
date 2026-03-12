import torch
import torch.nn as nn

class SeriesDecomposition(nn.Module):
    """
    Decouples the input sequence into Trend and Seasonal components.
    """
    def __init__(self, kernel_size):
        super(SeriesDecomposition, self).__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        # x: [Batch, Seq_Len, Features]
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
        
        # Joint interaction layers
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
    """
    Optimized NLinear for Multivariate Reversals.
    Uses 'Joint Temporal Projection' to catch when momentum is lying.
    """
    def __init__(self, seq_len, pred_len, input_dim):
        super(NLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_dim = input_dim
        
        # Joint Temporal Layer: Allows model to see all features simultaneously 
        # over time to detect trend exhaustion.
        self.temporal_joint = nn.Sequential(
            nn.Linear(seq_len * input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, pred_len * 16) # Expand to a hidden space for mixing
        )
        
        # Reversal Mixer: Specifically designed to decide if the 'trend'
        # should continue or reverse based on feature interactions.
        self.reversal_mixer = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: [Batch, Seq_Len, Input_Dim]
        batch_size = x.size(0)
        
        # 1. Normalization (Handle non-stationarity)
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        
        # 2. Joint Feature-Temporal Learning
        # Flatten to [Batch, Seq_Len * Input_Dim]
        x_flat = x.reshape(batch_size, -1)
        combined_features = self.temporal_joint(x_flat) # [Batch, Pred_Len * 16]
        
        # 3. Reshape for per-day reversal logic
        combined_features = combined_features.reshape(batch_size, self.pred_len, 16)
        
        # 4. Predict the delta (offset) from the last known price
        # This delta can now be negative even if momentum was positive
        delta_pred = self.reversal_mixer(combined_features).squeeze(-1) # [Batch, Pred_Len]
        
        # 5. Denormalization Anchor (Close price is index 3)
        target_last = seq_last[:, :, 3] # [Batch, 1]
        
        return delta_pred + target_last
