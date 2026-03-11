import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os
import argparse 
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Internal project imports - NOW INCLUDING FEATURE_WEIGHTS
from main import ModelFactory, INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, PRED_LEN, SEQ_LEN, MODEL_TYPE, DEVICE, FEATURE_WEIGHTS
from dataset_builder import MultivariateStockDataset

warnings.filterwarnings('ignore')

# ==========================================
# EVALUATION CONFIGURATION
# ==========================================
MARKET_DATA_PATH = "../../data/main_data/tech_macro_aligned.csv"
SENTIMENT_DATA_PATH = "../../data/data_scrapping/temp/gdelt_sentiment_bq_aligned.csv"
SEC_DATA_PATH = "../../data/main_data/sec_events.csv"

def execute_rolling_evaluation(target_equity, forecast_start_date, total_steps, use_actuals=True, model_weights=None):
    """
    Executes a forecast evaluation using synchronized FEATURE_WEIGHTS.
    """
    mode_str = "ONE-STEP (using Actuals)" if use_actuals else "RECURSIVE ROLLING"
    print(f"\n--- Starting {total_steps}-Day {mode_str} Evaluation for {MODEL_TYPE} ---")
    print(f"--- Ticker: {target_equity} | Start Date: {forecast_start_date} ---")
    
    if model_weights is None:
        model_weights = f"multivariate_{MODEL_TYPE.lower()}_{target_equity}.pth"
    
    if not os.path.exists(model_weights):
        raise FileNotFoundError(f"Model weights not found at {model_weights}")

    # 1. Dataset Initialization (CRITICAL: Pass FEATURE_WEIGHTS here)
    dataset = MultivariateStockDataset(
        target_equity, MARKET_DATA_PATH, SENTIMENT_DATA_PATH, SEC_DATA_PATH,
        seq_len=SEQ_LEN, pred_len=PRED_LEN, split='train', train_ratio=1.0,
        feature_weights=FEATURE_WEIGHTS
    )
    
    # 2. Model Reconstruction
    model = ModelFactory(MODEL_TYPE, SEQ_LEN, PRED_LEN, INPUT_DIM, HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
    model.load_state_dict(torch.load(model_weights, map_location=DEVICE))
    model.eval()

    # 3. Data Loading & Temporal Alignment
    market_df = pd.read_csv(MARKET_DATA_PATH)
    market_df['Date'] = pd.to_datetime(market_df['Date'])
    ticker_df = market_df[market_df['Ticker'] == target_equity].sort_values('Date').reset_index(drop=True)
    
    sentiment_df = pd.read_csv(SENTIMENT_DATA_PATH)
    sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
    
    merged_df = pd.merge(ticker_df, sentiment_df, on='Date', how='inner')
    merged_df = merged_df.sort_values('Date').reset_index(drop=True)
    
    target_date_obj = pd.to_datetime(forecast_start_date)
    future_slice = merged_df[merged_df['Date'] >= target_date_obj]
    
    if future_slice.empty:
        raise ValueError(f"Date {forecast_start_date} is out of bounds.")
    
    start_idx = future_slice.index[0]
    
    if start_idx < SEQ_LEN:
        raise ValueError(f"Not enough historical data before {forecast_start_date} for SEQ_LEN={SEQ_LEN}")

    # 4. INFERENCE LOOP
    current_window = dataset.data_scaled[start_idx - SEQ_LEN : start_idx].copy()
    preds_scaled = []
    
    for step in range(total_steps):
        x_tensor = torch.tensor(current_window, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            full_pred = model(x_tensor).cpu().numpy().flatten()
            day_1_pred = full_pred[0]
        
        preds_scaled.append(day_1_pred)
        
        if start_idx + step < len(dataset.data_scaled):
            next_row = dataset.data_scaled[start_idx + step].copy()
            if not use_actuals:
                next_row[dataset.target_idx] = day_1_pred
        else:
            break
            
        current_window = np.vstack([current_window[1:], next_row])

    # 5. Inverse Scaling & Metrics
    total_steps_executed = len(preds_scaled)
    num_features = dataset.scaler.n_features_in_
    dummy = np.zeros((total_steps_executed, num_features))
    dummy[:, dataset.target_idx] = preds_scaled
    predicted_prices = dataset.scaler.inverse_transform(dummy)[:, dataset.target_idx]
    
    anchor_idx = start_idx - 1
    true_prices_for_plot = merged_df['Close'].iloc[anchor_idx : start_idx + total_steps_executed].values
    actual_dates_for_plot = merged_df['Date'].iloc[anchor_idx : start_idx + total_steps_executed]
    
    anchor_price = true_prices_for_plot[0]
    predicted_prices_for_plot = np.insert(predicted_prices, 0, anchor_price)

    mae = mean_absolute_error(true_prices_for_plot[1:], predicted_prices)
    rmse = np.sqrt(mean_squared_error(true_prices_for_plot[1:], predicted_prices))
    
    print(f"\nAnalysis Results ({total_steps_executed} Days):")
    print(f"  > MAE:  ${mae:.2f}")
    print(f"  > RMSE: ${rmse:.2f}")
    
    plt.figure(figsize=(14, 7))
    plt.plot(actual_dates_for_plot, true_prices_for_plot, 'o-', label='Ground Truth', color="#060c8f", linewidth=2)
    plt.plot(actual_dates_for_plot, predicted_prices_for_plot, 's--', label=f'Forecast ({MODEL_TYPE})', color='#e74c3c', linewidth=2)

    plt.axvline(x=merged_df['Date'].iloc[start_idx], color='gray', linestyle=':', label='Forecast Start')
    plt.title(f"Forecast Comparison ({mode_str}): {target_equity} ({total_steps_executed} Days)", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Close Price (USD)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str, default='AAPL')
    parser.add_argument('--date', type=str, default='2025-02-03')
    parser.add_argument('--days', type=int, default=PRED_LEN)
    parser.add_argument('--use-actuals', action='store_true')
    parser.add_argument('--weights', type=str, default=None)
    args = parser.parse_args()
    
    execute_rolling_evaluation(args.ticker, args.date, args.days, args.use_actuals, args.weights)
