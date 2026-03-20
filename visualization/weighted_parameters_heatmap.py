import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import os

# Paths
csv_file_path = os.path.join('data', 'main_data', 'tech_macro_aligned.csv')
sentiment_file_path = os.path.join('data', 'data_scrapping', 'temp', 'gdelt_sentiment_bq_aligned.csv')
output_dir = os.path.join('visualization', 'output')

def generate_weighted_heatmap():
    if not os.path.exists(csv_file_path):
        print(f"Error: {csv_file_path} not found.")
        return

    # Load main data
    df = pd.read_csv(csv_file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Load sentiment data if available
    has_sentiment = os.path.exists(sentiment_file_path)
    if has_sentiment:
        try:
            sent_df = pd.read_csv(sentiment_file_path)
            sent_df['Date'] = pd.to_datetime(sent_df['Date'])
            print("Sentiment data loaded.")
        except Exception as e:
            print(f"Warning: Error loading sentiment data: {e}")
            has_sentiment = False
    else:
        print("Warning: Sentiment data not found. Proceeding without it.")

    tickers = df['Ticker'].unique()
    all_importances = {}
    
    features_base = ['Open', 'High', 'Low', 'Volume', 'Fed_Rate', 'CPI', 'Treasury_10Y']
    
    for ticker in tickers:
        print(f"Processing importance for: {ticker}...")
        ticker_data = df[df['Ticker'] == ticker].sort_values('Date').copy()
        
        current_features = features_base.copy()
        
        # Merge sentiment for this ticker if available
        if has_sentiment:
            vol_col = f'{ticker}_News_Volume'
            tone_col = f'{ticker}_Sentiment_Tone'
            
            if vol_col in sent_df.columns and tone_col in sent_df.columns:
                ticker_sent = sent_df[['Date', vol_col, tone_col]].copy()
                ticker_sent.columns = ['Date', 'News_Volume', 'Sentiment_Tone']
                ticker_data = pd.merge(ticker_data, ticker_sent, on='Date', how='left')
                ticker_data['News_Volume'] = ticker_data['News_Volume'].fillna(0)
                ticker_data['Sentiment_Tone'] = ticker_data['Sentiment_Tone'].fillna(ticker_data['Sentiment_Tone'].mean())
                current_features += ['News_Volume', 'Sentiment_Tone']

        # Target: Magnitude of price change
        ticker_data['Price_Change'] = ticker_data['Close'].pct_change() * 100
        ticker_data = ticker_data.dropna(subset=['Price_Change'] + current_features)
        
        if len(ticker_data) < 20: # Increased threshold for better training
            print(f"Skipping {ticker} due to insufficient data ({len(ticker_data)} points).")
            continue

        X = ticker_data[current_features]
        y = ticker_data['Price_Change'].abs()
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        importances = model.feature_importances_
        for feat, imp in zip(current_features, importances):
            if feat not in all_importances:
                all_importances[feat] = {}
            all_importances[feat][ticker] = imp

    if not all_importances:
        print("No importance data collected. Check data availability.")
        return

    # Create Heatmap DataFrame
    heatmap_df = pd.DataFrame(all_importances).T
    
    # Fill missing values with 0 (e.g. if sentiment wasn't available for some tickers)
    heatmap_df = heatmap_df.fillna(0)

    # Sort features by average importance
    heatmap_df['avg'] = heatmap_df.mean(axis=1)
    heatmap_df = heatmap_df.sort_values(by='avg', ascending=False).drop(columns='avg')

    # Plotting
    plt.figure(figsize=(16, 10))
    sns.heatmap(heatmap_df, annot=True, cmap='YlGnBu', fmt=".3f", linewidths=.5)
    
    plt.title('Parameter "Weights" (Feature Importance) Across Tickers', fontsize=18)
    plt.ylabel('Parameters / Features', fontsize=14)
    plt.xlabel('Tickers', fontsize=14)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    save_path = os.path.join(output_dir, 'weighted_parameters_heatmap.png')
    plt.savefig(save_path, bbox_inches='tight')
    print(f"\nHeatmap saved successfully to: {save_path}")

if __name__ == "__main__":
    generate_weighted_heatmap()
