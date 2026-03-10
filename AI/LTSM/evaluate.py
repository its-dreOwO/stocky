import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os
import argparse 
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Internal project imports - NOTE: PRED_LEN is now just used as a default
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

def execute_evaluation(forecast_start_date, pred_len):
    print(f"--- Finalizing Evaluation for {MODEL_TYPE} (11 Features) ---")
    print(f"--- Target Prediction Date: {forecast_start_date} ---")
    print(f"--- Forecast Horizon: {pred_len} Days ---")
    
    # 1. Dataset Initialization 
    dataset = MultivariateStockDataset(
        TARGET_EQUITY, MARKET_DATA_PATH, SENTIMENT_DATA_PATH, SEC_DATA_PATH,
        seq_len=SEQ_LEN, pred_len=pred_len, split='train', train_ratio=1.0
    )
    
    # 2. Model Reconstruction
    model = ModelFactory(MODEL_TYPE, SEQ_LEN, pred_len, INPUT_DIM, HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
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
    
    # Check if we have enough future data to evaluate the requested pred_len
    if start_idx + pred_len > len(merged_df):
        raise ValueError(f"Not enough future data in dataset to evaluate {pred_len} days from {forecast_start_date}.")
        
    actual_dates = merged_df['Date'].iloc[start_idx : start_idx + pred_len].dt.strftime('%Y-%m-%d').tolist()
    
    # 4. Model Inference
    historical_window = dataset.data_scaled[start_idx - SEQ_LEN : start_idx]
    x_tensor = torch.tensor(historical_window, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        scaled_pred = model(x_tensor).cpu().numpy().flatten()
            
    # 5. Inverse Scaling (StandardScaler)
    num_features = dataset.scaler.n_features_in_
    dummy = np.zeros((pred_len, num_features))
    dummy[:, dataset.target_idx] = scaled_pred
    
    predicted_prices = dataset.scaler.inverse_transform(dummy)[:, dataset.target_idx]
    true_prices = merged_df['Close'].iloc[start_idx : start_idx + pred_len].values

    # 6. Metrics & Visualization
    mae = mean_absolute_error(true_prices, predicted_prices)
    
    print(f"\nFinal Analysis:")
    print(f"  > Start Date: {actual_dates[0]}")
    print(f"  > Forecast Length: {pred_len} Days")
    print(f"  > Mean Absolute Error (MAE): ${mae:.2f}")
    
    plt.figure(figsize=(12, 6))
    plt.plot(actual_dates, true_prices, 'o-', color='blue', label='Actual Price')
    plt.plot(actual_dates, predicted_prices, 's--', color='red', label=f'{MODEL_TYPE} Forecast')
    
    # Highlight SEC Event days if they exist in this window
    window_events = merged_df['SEC_Event'].iloc[start_idx : start_idx + pred_len].values
    for i, is_event in enumerate(window_events):
        if is_event == 1:
            plt.axvspan(i-0.2, i+0.2, color='yellow', alpha=0.3, label='SEC Event' if i == 0 else "")

    plt.title(f"{TARGET_EQUITY} Final Evaluation (11 Features + Event Awareness)")
    plt.ylabel("Price (USD)")
    plt.xticks(rotation=45) # Added rotation so dates don't overlap if you pick a large number!
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Evaluate stock prediction model for a specific date and length.")
    parser.add_argument(
        '--date', 
        type=str, 
        default='2025-02-03', 
        help="The start date for the forecast in YYYY-MM-DD format (default: 2025-02-03)"
    )
    parser.add_argument(
        '--days', 
        type=int, 
        default=PRED_LEN, 
        help=f"Number of days to forecast (default: {PRED_LEN} from main.py)"
    )
    
    args = parser.parse_args()
    
    # Pass both parsed arguments to the execution function
    execute_evaluation(args.date, args.days)