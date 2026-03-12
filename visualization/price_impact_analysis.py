import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import os

# Define file paths
csv_file_path = os.path.join('data', 'main_data', 'tech_macro_aligned.csv')
sentiment_file_path = os.path.join('data', 'data_scrapping', 'temp', 'gdelt_sentiment_bq_aligned.csv')

def analyze_price_impact():
    if not os.path.exists(csv_file_path):
        print(f"Error: {csv_file_path} not found.")
        return
    if not os.path.exists(sentiment_file_path):
        print(f"Warning: {sentiment_file_path} not found. Proceeding without sentiment data.")
        has_sentiment = False
    else:
        has_sentiment = True

    # Load main data
    df = pd.read_csv(csv_file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    tickers = df['Ticker'].unique()
    selected_ticker = 'NVDA' if 'NVDA' in tickers else tickers[0]
    print(f"Analyzing price impact for: {selected_ticker}")
    
    data = df[df['Ticker'] == selected_ticker].sort_values('Date').copy()

    # Load and merge sentiment data
    if has_sentiment:
        sent_df = pd.read_csv(sentiment_file_path)
        sent_df['Date'] = pd.to_datetime(sent_df['Date'])
        
        # Extract columns for the selected ticker
        vol_col = f'{selected_ticker}_News_Volume'
        tone_col = f'{selected_ticker}_Sentiment_Tone'
        
        if vol_col in sent_df.columns and tone_col in sent_df.columns:
            ticker_sent = sent_df[['Date', vol_col, tone_col]].copy()
            # Rename for consistency
            ticker_sent.columns = ['Date', 'News_Volume', 'Sentiment_Tone']
            # Merge with main data
            data = pd.merge(data, ticker_sent, on='Date', how='left')
            print(f"Successfully merged sentiment data for {selected_ticker}.")
        else:
            print(f"Columns for {selected_ticker} not found in sentiment file.")
            has_sentiment = False

    # 1. Calculate Daily Percentage Change (Target variable)
    data['Price_Change'] = data['Close'].pct_change() * 100
    
    # 2. Define features
    features = ['Open', 'High', 'Low', 'Volume', 'Fed_Rate', 'CPI', 'Treasury_10Y']
    if has_sentiment:
        features += ['News_Volume', 'Sentiment_Tone']
    
    # Fill missing sentiment values if any (e.g., non-trading days in sentiment file)
    if has_sentiment:
        data['News_Volume'] = data['News_Volume'].fillna(0)
        data['Sentiment_Tone'] = data['Sentiment_Tone'].fillna(data['Sentiment_Tone'].mean())

    # Handle NaN from pct_change and any other missing values
    data = data.dropna(subset=['Price_Change'] + features)

    # 3. Correlation Matrix
    corr_matrix = data[features + ['Price_Change']].corr()
    
    # 4. Feature Importance using Random Forest
    X = data[features]
    y = data['Price_Change'].abs() # Target: Magnitude of sharp movement
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    # 5. Visualization
    fig, axes = plt.subplots(2, 1, figsize=(12, 14))

    # Plot A: Feature Importance
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, ax=axes[0], palette='viridis')
    axes[0].set_title(f'Parameters Affecting Sharp Price Movements (Magnitude) - {selected_ticker}')
    axes[0].set_xlabel('Importance Score')

    # Plot B: Correlation Heatmap
    # Sort by absolute correlation to Price_Change for better visualization
    corr_to_target = corr_matrix[['Price_Change']].sort_values(by='Price_Change', ascending=False)
    sns.heatmap(corr_to_target, annot=True, cmap='coolwarm', ax=axes[1], center=0)
    axes[1].set_title(f'Correlation of Parameters with Daily Price Change (%) - {selected_ticker}')

    plt.tight_layout()
    
    output_dir = os.path.join('visualization', 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_file = os.path.join(output_dir, f'{selected_ticker}_impact_analysis_with_news.png')
    plt.savefig(output_file)
    print(f"Impact analysis graph (including news) saved as: {output_file}")
    
    # Print findings
    print("\nTop parameters influencing price volatility (including News):")
    print(feature_importance_df.to_string(index=False))

    # Identify Sharp Movements (e.g., > 3% change)
    sharp_movements = data[data['Price_Change'].abs() > 3]
    print(f"\nFound {len(sharp_movements)} instances of sharp movements (>3%).")

if __name__ == "__main__":
    analyze_price_impact()
