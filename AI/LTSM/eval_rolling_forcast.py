import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os
import argparse 
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Internal project imports
# Ensure these constants in main.py match our optimized config (SEQ_LEN=192, PRED_LEN=1)
from main import ModelFactory, INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, PRED_LEN, SEQ_LEN, MODEL_TYPE, DEVICE
from dataset_builder import MultivariateStockDataset

warnings.filterwarnings('ignore')

# ==========================================
# EVALUATION CONFIGURATION
# ==========================================
TARGET_EQUITY = 'AAPL'
MARKET_DATA_PATH = "../../data/main_data/tech_macro_aligned.csv"
SENTIMENT_DATA_PATH = "../../data/data_scrapping/temp/gdelt_sentiment_bq_aligned.csv"
SEC_DATA_PATH = "../../data/main_data/sec_events.csv"
MODEL_WEIGHTS = f"multivariate_{MODEL_TYPE.lower()}_{TARGET_EQUITY}.pth"

def execute_rolling_evaluation(forecast_start_date, total_steps):
    print(f"--- Starting {total_steps}-Day ROLLING Evaluation for {MODEL_TYPE} ---")
    print(f"--- Initial Context Date: {forecast_start_date} ---")
    
    # 1. Dataset Initialization 
    # Note: For rolling, the model should ideally be trained with pred_len=1
    dataset = MultivariateStockDataset(
        TARGET_EQUITY, MARKET_DATA_PATH, SENTIMENT_DATA_PATH, SEC_DATA_PATH,
        seq_len=SEQ_LEN, pred_len=30, split='train', train_ratio=1
    )
    
    # 2. Model Reconstruction
    # model pred_len here should be 1 if training was 1-step ahead
    model = ModelFactory(MODEL_TYPE, SEQ_LEN, 30, INPUT_DIM, HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
    model.eval()

    # 3. Data Loading & Alignment
    market_df = pd.read_csv(MARKET_DATA_PATH)
    sentiment_df = pd.read_csv(SENTIMENT_DATA_PATH)
    sec_df = pd.read_csv(SEC_DATA_PATH)
    
    for df in [market_df, sentiment_df, sec_df]:
        df['Date'] = pd.to_datetime(df['Date'])
    
    ticker_df = market_df[market_df['Ticker']==TARGET_EQUITY]
    merged_df = pd.merge(ticker_df, sentiment_df, on='Date', how='inner')
    merged_df = pd.merge(merged_df, sec_df[sec_df['Ticker']==TARGET_EQUITY][['Date', 'SEC_Event']], on='Date', how='left')
    merged_df['SEC_Event'] = merged_df['SEC_Event'].fillna(0)
    merged_df = merged_df.sort_values('Date').reset_index(drop=True)
    
    target_date_obj = pd.to_datetime(forecast_start_date)
    future_slice = merged_df[merged_df['Date'] >= target_date_obj]
    
    if future_slice.empty:
        raise ValueError(f"Date {forecast_start_date} is out of bounds.")
    
    start_idx = future_slice.index[0]
    
    # 4. ROLLING INFERENCE LOOP
    # [Image of Recursive multi-step forecasting process]
    current_window = dataset.data_scaled[start_idx - SEQ_LEN : start_idx].copy()
    rolling_preds_scaled = []
    
    for step in range(total_steps):
        x_tensor = torch.tensor(current_window, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
        # This now returns [1, 30]
            full_pred = model(x_tensor).cpu().numpy().flatten() 
        # Extract only the first day for the rolling feedback
            day_1_pred = full_pred[0] 
        rolling_preds_scaled.append(day_1_pred)
    
    # Update the next window using day_1_pred
        if start_idx + step < len(dataset.data_scaled):
            next_row = dataset.data_scaled[start_idx + step].copy()
            next_row[dataset.target_idx] = day_1_pred 
        else:
            next_row = current_window[-1].copy()
            next_row[dataset.target_idx] = day_1_pred
            
        current_window = np.vstack([current_window[1:], next_row])

    # 5. Inverse Scaling
    num_features = dataset.scaler.n_features_in_
    dummy = np.zeros((total_steps, num_features))
    dummy[:, dataset.target_idx] = rolling_preds_scaled
    predicted_prices = dataset.scaler.inverse_transform(dummy)[:, dataset.target_idx]
    
    # Get ground truth
    true_prices = merged_df['Close'].iloc[start_idx : start_idx + total_steps].values
    actual_dates = merged_df['Date'].iloc[start_idx : start_idx + total_steps].dt.strftime('%Y-%m-%d').tolist()

    # 6. Visualization
    mae = mean_absolute_error(true_prices, predicted_prices)
    print(f"\nRolling Analysis (20 Days):")
    print(f"  > MAE: ${mae:.2f}")
    
    plt.figure(figsize=(12, 6))
    plt.plot(actual_dates, true_prices, 'o-', label='Ground Truth', color='blue')
    plt.plot(actual_dates, predicted_prices, 's--', label='Rolling Forecast', color='red')
    plt.title(f"{TARGET_EQUITY} 30-Day Rolling Forecast ({MODEL_TYPE})")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default='2025-02-03')
    parser.add_argument('--days', type=int, default=20) # Now defaulting to 20
    args = parser.parse_args()
    
    execute_rolling_evaluation(args.date, args.days)