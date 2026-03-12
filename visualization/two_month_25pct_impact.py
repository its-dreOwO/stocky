import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import os

# Define file paths
csv_file_path = os.path.join('data', 'main_data', 'tech_macro_aligned.csv')
sentiment_file_path = os.path.join('data', 'data_scrapping', 'temp', 'gdelt_sentiment_bq_aligned.csv')

def analyze_2month_25pct_impact():
    if not os.path.exists(csv_file_path):
        print(f"Error: {csv_file_path} not found.")
        return
    
    # Load main data
    df = pd.read_csv(csv_file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Get all available tickers
    tickers = df['Ticker'].unique()
    
    all_results = []
    
    # Check for sentiment data
    has_sentiment_global = os.path.exists(sentiment_file_path)
    if has_sentiment_global:
        sent_df = pd.read_csv(sentiment_file_path)
        sent_df['Date'] = pd.to_datetime(sent_df['Date'])
    
    # For simplicity, we'll analyze all tickers or a specific one? 
    # Let's aggregate for a broader view or pick a representative one.
    # The user didn't specify, so I'll try to find any ticker that hits the 25% threshold.
    
    for selected_ticker in tickers:
        print(f"Processing {selected_ticker}...")
        data = df[df['Ticker'] == selected_ticker].sort_values('Date').copy()
        
        # Merge sentiment
        if has_sentiment_global:
            vol_col = f'{selected_ticker}_News_Volume'
            tone_col = f'{selected_ticker}_Sentiment_Tone'
            
            if vol_col in sent_df.columns and tone_col in sent_df.columns:
                ticker_sent = sent_df[['Date', vol_col, tone_col]].copy()
                ticker_sent.columns = ['Date', 'News_Volume', 'Sentiment_Tone']
                data = pd.merge(data, ticker_sent, on='Date', how='left')
                data['News_Volume'] = data['News_Volume'].fillna(0)
                data['Sentiment_Tone'] = data['Sentiment_Tone'].fillna(data['Sentiment_Tone'].mean())
                has_sentiment = True
            else:
                has_sentiment = False
        else:
            has_sentiment = False
            
        # 1. Calculate 2-month (44 trading days) forward price change
        # We look ahead to see if in 2 months it changed by 25% or more
        look_ahead = 44 # Approx 2 months
        data['Future_Price'] = data['Close'].shift(-look_ahead)
        data['Price_Change_2M'] = (data['Future_Price'] - data['Close']) / data['Close'] * 100
        
        # Threshold: 25% or more
        data['Target_25Pct'] = (data['Price_Change_2M'] >= 25).astype(int)
        
        # 2. Define features
        features = ['Open', 'High', 'Low', 'Volume', 'Fed_Rate', 'CPI', 'Treasury_10Y']
        if has_sentiment:
            features += ['News_Volume', 'Sentiment_Tone']
            
        # Drop rows with NaN (especially at the end because of shift)
        model_data = data.dropna(subset=['Price_Change_2M'] + features)
        
        if len(model_data[model_data['Target_25Pct'] == 1]) < 5:
            print(f"Skipping {selected_ticker}: Not enough instances of 25% increase (only {len(model_data[model_data['Target_25Pct'] == 1])}).")
            continue
            
        print(f"Found {len(model_data[model_data['Target_25Pct'] == 1])} instances for {selected_ticker}.")
        
        # 3. Feature Importance
        X = model_data[features]
        y = model_data['Target_25Pct']
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        importances = model.feature_importances_
        for i, feat in enumerate(features):
            all_results.append({'Ticker': selected_ticker, 'Feature': feat, 'Importance': importances[i]})

    if not all_results:
        print("No tickers had enough instances of 25% growth over 2 months.")
        return

    results_df = pd.DataFrame(all_results)
    # Aggregate importance across all valid tickers
    agg_importance = results_df.groupby('Feature')['Importance'].mean().reset_index().sort_values(by='Importance', ascending=False)
    
    print("\nMost impactful parameters for 25% price growth in 2 months (averaged across tickers):")
    print(agg_importance.to_string(index=False))
    
    # 4. Visualization
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=agg_importance, palette='rocket')
    plt.title('Impactful Parameters for 25%+ Price Increase (2-Month Horizon)')
    plt.xlabel('Average Feature Importance (Random Forest)')
    plt.ylabel('Parameter')
    plt.tight_layout()
    
    output_dir = os.path.join('visualization', 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_file = os.path.join(output_dir, 'impact_25pct_2months.png')
    plt.savefig(output_file)
    print(f"Graph saved as: {output_file}")

if __name__ == "__main__":
    analyze_2month_25pct_impact()
