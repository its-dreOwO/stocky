
import os
import json
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from main import train_model, TARGET_EQUITY, MARKET_DATA_PATH, SENTIMENT_DATA_PATH, SEC_DATA_PATH, DEVICE, PRED_LEN
from dataset_builder import MultivariateStockDataset

def evaluate_config(model, config):
    """
    Evaluates the model on a one-step-ahead basis using the specified configuration.
    """
    s_len = config['seq_len']
    f_weights = config['feature_weights']
    
    # Initialize dataset for evaluation
    dataset = MultivariateStockDataset(TARGET_EQUITY, MARKET_DATA_PATH, SENTIMENT_DATA_PATH, SEC_DATA_PATH,
                                       seq_len=s_len, pred_len=PRED_LEN, split='test', train_ratio=0.8, feature_weights=f_weights)
    
    model.eval()
    all_preds = []
    all_trues = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            x, y = dataset[i]
            x = x.unsqueeze(0).to(DEVICE)
            pred = model(x).cpu().numpy().flatten()
            
            # Record first day of prediction vs first day of ground truth
            all_preds.append(pred[0])
            all_trues.append(y[0].item())
            
    # Inverse scaling
    preds_np = np.array(all_preds)
    trues_np = np.array(all_trues)
    num_features = dataset.scaler.n_features_in_
    
    dummy_p = np.zeros((len(preds_np), num_features))
    dummy_p[:, dataset.target_idx] = preds_np
    inv_preds = dataset.scaler.inverse_transform(dummy_p)[:, dataset.target_idx]
    
    dummy_t = np.zeros((len(trues_np), num_features))
    dummy_t[:, dataset.target_idx] = trues_np
    inv_trues = dataset.scaler.inverse_transform(dummy_t)[:, dataset.target_idx]
    
    mae = mean_absolute_error(inv_trues, inv_preds)
    rmse = np.sqrt(mean_squared_error(inv_trues, inv_preds))
    
    return mae, rmse

def run_tuning_suite():
    """
    Runs a systematic search over hyperparameters and feature weights.
    """
    # Define hyperparameter grid
    seq_lens = [128, 192]
    lrs = [1e-3, 5e-4]
    
    # Define feature weighting groups to keep tuning manageable
    # You can customize these multipliers
    weight_scenarios = [
        {'ROC_5': 1.0, 'RSI_14': 1.0, 'Sentiment': 1.0, 'SEC': 1.0, 'Macro': 1.0}, # Baseline
        {'ROC_5': 1.5, 'RSI_14': 1.5, 'Sentiment': 1.0, 'SEC': 1.0, 'Macro': 1.0}, # Momentum Focus
        {'ROC_5': 1.2, 'RSI_14': 1.2, 'Sentiment': 1.5, 'SEC': 1.5, 'Macro': 1.0}, # Sentiment/Event Focus
        {'ROC_5': 1.5, 'RSI_14': 1.5, 'Sentiment': 1.2, 'SEC': 1.2, 'Macro': 1.2}, # All boosted
    ]
    
    results = []
    best_mae = float('inf')
    best_config = None
    
    total_combos = len(seq_lens) * len(lrs) * len(weight_scenarios)
    count = 0
    
    print(f"Starting Extended Tuning Suite ({total_combos} combinations)...")
    
    for s_len in seq_lens:
        for lr in lrs:
            for scenario in weight_scenarios:
                count += 1
                
                # Construct the full 13-parameter feature_weights dictionary
                # This maps the scenario groups to the actual 13 columns
                f_weights = {
                    'Open': 1.0, 'High': 1.0, 'Low': 1.0, 'Close': 1.0, 'Volume': 1.0,
                    'Fed_Rate': scenario['Macro'], 
                    'CPI': scenario['Macro'], 
                    'Treasury_10Y': scenario['Macro'],
                    '{TICKER}_News_Volume': scenario['Sentiment'],
                    '{TICKER}_Sentiment_Tone': scenario['Sentiment'],
                    'SEC_Event': scenario['SEC'],
                    'ROC_5': scenario['ROC_5'],
                    'RSI_14': scenario['RSI_14']
                }
                
                config = {
                    'seq_len': s_len,
                    'lr': lr,
                    'feature_weights': f_weights,
                    'epochs': 40 # Fast tuning
                }
                
                print(f"[{count}/{total_combos}] Testing SEQ={s_len}, LR={lr}, Scenario={list(scenario.values())}...")
                
                try:
                    model, _ = train_model(config, verbose=False)
                    mae, rmse = evaluate_config(model, config)
                    
                    result = {
                        'config': config,
                        'mae': mae,
                        'rmse': rmse
                    }
                    results.append(result)
                    print(f"   > Result: MAE=${mae:.2f}, RMSE=${rmse:.2f}")
                    
                    if mae < best_mae:
                        best_mae = mae
                        best_config = config
                        
                except Exception as e:
                    print(f"   > Error: {e}")
                    
    with open('tuning_results_comprehensive.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("\n" + "="*30)
    print("COMPREHENSIVE TUNING COMPLETE")
    print(f"Best MAE: ${best_mae:.2f}")
    print(f"Optimal Config saved to tuning_results_comprehensive.json")
    print("="*30)

if __name__ == "__main__":
    run_tuning_suite()
