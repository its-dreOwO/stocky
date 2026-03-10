import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os
import argparse 
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Internal project imports
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

def execute_evaluation(forecast_start_date, target_forecast_days):
    print(f"--- Finalizing Evaluation for {MODEL_TYPE} (11 Features) ---")
    print(f"--- Target Prediction Date: {forecast_start_date} ---")
    print(f"--- Forecast Horizon: {target_forecast_days} Days ---")
    
    # 1. Dataset Initialization 
    # Notice we keep pred_len=PRED_LEN here to maintain dataset structure
    dataset = MultivariateStockDataset(
        TARGET_EQUITY, MARKET_DATA_PATH, SENTIMENT_DATA_PATH, SEC_DATA_PATH,
        seq_len=SEQ_LEN, pred_len=PRED_LEN, split='train', train_ratio=1.0
    )
    
    # 2. Model Reconstruction
    # CRITICAL FIX: We MUST instantiate the model using the original PRED_LEN (e.g., 5)
    model = ModelFactory(MODEL_TYPE, SEQ_LEN, PRED_LEN, INPUT_DIM, HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
    model.eval()

    # 3. Temporal Alignment
    market_df = pd.read_csv(MARKET_DATA_PATH)
    sentiment_df = pd.read_csv(SENTIMENT_DATA_PATH)
    sec_df = pd.read_csv(SEC_DATA_PATH)
    
    market_df['Date'] = pd.to_datetime(market_df['Date'])
    sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
    sec_df['Date'] = pd.to_datetime(sec_df['Date'])
    
    # Merge exactly like the training set
    ticker_df = market_df[market_df['Ticker']==TARGET_EQUITY]
    merged_df = pd.merge(ticker_df, sentiment_df, on='Date', how='inner')
    merged_df = pd.merge(merged_df, sec_df[sec_df['Ticker']==TARGET_EQUITY][['Date', 'SEC_Event']], on='Date', how='left')
    merged_df['SEC_Event'] = merged_df['SEC_Event'].fillna(0)
    merged_df = merged_df.sort_values('Date').reset_index(drop=True)
    
    target_date_obj = pd.to_datetime(forecast_start_date)
    future_slice = merged_df[merged_df['Date'] >= target_date_obj]
    
    if future_slice.empty:
        raise ValueError(f"Date {forecast_start_date} is out of bounds or not found in the merged dataset.")
    
    start_idx = future_slice.index[0]
    
    if start_idx + target_forecast_days > len(merged_df):
        raise ValueError(f"Not enough future data in dataset to evaluate {target_forecast_days} days from {forecast_start_date}.")
        
    actual_dates = merged_df['Date'].iloc[start_idx : start_idx + target_forecast_days].dt.strftime('%Y-%m-%d').tolist()
    
    # 4. Model Inference (AUTOREGRESSIVE LOOP)
    predictions = []
    # Start with the historical window right before our forecast date
    current_window = dataset.data_scaled[start_idx - SEQ_LEN : start_idx].copy()
    
    # Loop in chunks of PRED_LEN (e.g., 5 days at a time)
    for step in range(0, target_forecast_days, PRED_LEN):
        x_tensor = torch.tensor(current_window, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            pred_chunk = model(x_tensor).cpu().numpy().flatten()
            
        predictions.extend(pred_chunk)
        
        # If we need to loop again, we must slide the window forward
        if step + PRED_LEN < target_forecast_days:
            # Grab the actual market/sentiment data for the next 5 days
            next_features = dataset.data_scaled[start_idx + step : start_idx + step + PRED_LEN].copy()
            
            # Replace the actual prices with our PREDICTED prices so the model feeds on its own outputs
            next_features[:, dataset.target_idx] = pred_chunk
            
            # Slide window: drop the oldest 5 days, append the new 5 days
            current_window = np.vstack((current_window[PRED_LEN:], next_features))
            
    # Trim predictions to the exact number of days requested (in case it wasn't a perfect multiple of 5)
    scaled_pred = np.array(predictions)[:target_forecast_days]
            
    # 5. Inverse Scaling (StandardScaler)
    num_features = dataset.scaler.n_features_in_
    dummy = np.zeros((target_forecast_days, num_features))
    dummy[:, dataset.target_idx] = scaled_pred
    
    predicted_prices = dataset.scaler.inverse_transform(dummy)[:, dataset.target_idx]
    true_prices = merged_df['Close'].iloc[start_idx : start_idx + target_forecast_days].values

    # 6. Metrics & Visualization
    mae = mean_absolute_error(true_prices, predicted_prices)
    
    print(f"\nFinal Analysis:")
    print(f"  > Start Date: {actual_dates[0]}")
    print(f"  > Forecast Length: {target_forecast_days} Days")
    print(f"  > Mean Absolute Error (MAE): ${mae:.2f}")
    
    plt.figure(figsize=(12, 6))
    plt.plot(actual_dates, true_prices, 'o-', color='blue', label='Actual Price')
    plt.plot(actual_dates, predicted_prices, 's--', color='red', label=f'{MODEL_TYPE} Forecast')
    
    # Highlight SEC Event days if they exist in this window
    window_events = merged_df['SEC_Event'].iloc[start_idx : start_idx + target_forecast_days].values
    for i, is_event in enumerate(window_events):
        if is_event == 1:
            plt.axvspan(i-0.2, i+0.2, color='yellow', alpha=0.3, label='SEC Event' if i == 0 else "")

    plt.title(f"{TARGET_EQUITY} Final Evaluation (11 Features + Event Awareness)")
    plt.ylabel("Price (USD)")
    plt.xticks(rotation=45) 
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate stock prediction model.")
    parser.add_argument('--date', type=str, default='2025-02-03', help="Forecast start date (YYYY-MM-DD)")
    parser.add_argument('--days', type=int, default=PRED_LEN, help="Number of days to forecast")
    args = parser.parse_args()
    
    execute_evaluation(args.date, args.days)