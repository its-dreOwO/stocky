import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import os

csv_file_path = os.path.join('data', 'main_data', 'tech_macro_aligned.csv')
sentiment_file_path = os.path.join('data', 'data_scrapping', 'temp', 'gdelt_sentiment_bq_aligned.csv')

def analyze_2month_25pct_impact():
    if not os.path.exists(csv_file_path):
        print(f"Error: {csv_file_path} not found.")
        return
    
    df = pd.read_csv(csv_file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    tickers = df['Ticker'].unique()
    
    all_results = []
    
    has_sentiment_global = os.path.exists(sentiment_file_path)
    if has_sentiment_global:
        sent_df = pd.read_csv(sentiment_file_path)
        sent_df['Date'] = pd.to_datetime(sent_df['Date'])
    
    for selected_ticker in tickers:
        print(f"Processing {selected_ticker}...")
        data = df[df['Ticker'] == selected_ticker].sort_values('Date').copy()
        
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
            
        look_ahead = 44 
        data['Future_Price'] = data['Close'].shift(-look_ahead)
        data['Price_Change_2M'] = (data['Future_Price'] - data['Close']) / data['Close'] * 100
        
        data['Target_25Pct'] = (data['Price_Change_2M'] >= 25).astype(int)
        
        features = ['Open', 'High', 'Low', 'Volume', 'Fed_Rate', 'CPI', 'Treasury_10Y']
        if has_sentiment:
            features += ['News_Volume', 'Sentiment_Tone']
            
        model_data = data.dropna(subset=['Price_Change_2M'] + features)
        
        if len(model_data[model_data['Target_25Pct'] == 1]) < 5:
            print(f"Skipping {selected_ticker}: Not enough instances of 25% increase (only {len(model_data[model_data['Target_25Pct'] == 1])}).")
            continue
            
        print(f"Found {len(model_data[model_data['Target_25Pct'] == 1])} instances for {selected_ticker}.")
        
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
    agg_importance = results_df.groupby('Feature')['Importance'].mean().reset_index().sort_values(by='Importance', ascending=False)
    
    print("\nMost impactful parameters for 25% price growth in 2 months (averaged across tickers):")
    print(agg_importance.to_string(index=False))
    
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
