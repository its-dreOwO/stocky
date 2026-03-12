import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class MultivariateStockDataset(Dataset):
    def __init__(self, target_ticker, market_path, sentiment_path, sec_path,
                 seq_len=256, pred_len=1, split='train', train_ratio=0.8, feature_weights=None):
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 1. Load Core Datasets
        market_df = pd.read_csv(market_path)
        sentiment_df = pd.read_csv(sentiment_path)
        market_df['Date'] = pd.to_datetime(market_df['Date'])
        sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
        
        # 2. Load SEC Event Data
        if os.path.exists(sec_path):
            sec_df = pd.read_csv(sec_path)
            sec_df['Date'] = pd.to_datetime(sec_df['Date'])
            ticker_sec = sec_df[sec_df['Ticker'] == target_ticker].copy()
        else:
            ticker_sec = pd.DataFrame(columns=['Date', 'SEC_Event'])

        # 3. Synchronized Merging
        ticker_df = market_df[market_df['Ticker'] == target_ticker].copy()
        merged_df = pd.merge(ticker_df, sentiment_df, on='Date', how='inner')
        merged_df = pd.merge(merged_df, ticker_sec[['Date', 'SEC_Event']], on='Date', how='left')
        merged_df['SEC_Event'] = merged_df['SEC_Event'].fillna(0)
        
        # --- Momentum Feature Engineering ---
        merged_df['ROC_5'] = merged_df['Close'].pct_change(periods=5)
        delta = merged_df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        merged_df['RSI_14'] = 100 - (100 / (1 + rs))
        
        merged_df = merged_df.fillna(0).sort_values('Date').reset_index(drop=True)
        
        # 4. Feature Selection (13 Features)
        self.feature_cols = [
            'Open', 'High', 'Low', 'Close', 'Volume',          
            'Fed_Rate', 'CPI', 'Treasury_10Y',                 
            f'{target_ticker}_News_Volume',                    
            f'{target_ticker}_Sentiment_Tone',                 
            'SEC_Event',
            'ROC_5', 'RSI_14' 
        ]
        
        data_matrix = merged_df[self.feature_cols].values
        self.target_idx = self.feature_cols.index('Close')
        
        # 5. Temporal Splitting
        train_size = int(len(data_matrix) * train_ratio)
        if split == 'train':
            self.data = data_matrix[:train_size]
        else:
            self.data = data_matrix[train_size - self.seq_len:]
            
        # 6. Scaling
        self.scaler = StandardScaler()
        self.scaler.fit(data_matrix[:train_size]) 
        self.data_scaled = self.scaler.transform(self.data)
        
        # --- MULTIVARIATE FEATURE WEIGHTING ---
        # feature_weights should be a dict: {'ROC_5': 1.5, 'Sentiment_Tone': 1.2, ...}
        if feature_weights:
            for col_name, weight in feature_weights.items():
                # Map specific news/sentiment columns if they contain ticker names
                mapped_col = col_name.replace('{TICKER}', target_ticker)
                if mapped_col in self.feature_cols:
                    idx = self.feature_cols.index(mapped_col)
                    self.data_scaled[:, idx] *= weight
                elif col_name in self.feature_cols:
                    idx = self.feature_cols.index(col_name)
                    self.data_scaled[:, idx] *= weight

        self.x_tensors, self.y_tensors = self._generate_sequences()
        
    def _generate_sequences(self):
        x, y = [], []
        for i in range(len(self.data_scaled) - self.seq_len - self.pred_len + 1):
            x.append(self.data_scaled[i : i + self.seq_len])
            y.append(self.data_scaled[i + self.seq_len : i + self.seq_len + self.pred_len, self.target_idx])
        return torch.tensor(np.array(x), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32)

    def __len__(self): return len(self.x_tensors)
    def __getitem__(self, idx): return self.x_tensors[idx], self.y_tensors[idx]
