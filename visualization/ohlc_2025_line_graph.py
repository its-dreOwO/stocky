import pandas as pd
import matplotlib.pyplot as plt
import os

csv_file_path = os.path.join('data', 'main_data', 'tech_macro_aligned.csv')

if not os.path.exists(csv_file_path):
    print(f"Error: {csv_file_path} not found.")
else:
    df = pd.read_csv(csv_file_path)

    df['Date'] = pd.to_datetime(df['Date'])

    df_jan_2025 = df[(df['Date'].dt.year == 2025) & (df['Date'].dt.month == 1)].copy()

    if df_jan_2025.empty:
        print("No data found for January 2025.")
    else:
        tickers = df_jan_2025['Ticker'].unique()
        
        output_dir = os.path.join('visualization', 'output')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for selected_ticker in tickers:
            print(f"Generating OHLC line graph for: {selected_ticker} (Jan 2025)")

            ticker_data = df_jan_2025[df_jan_2025['Ticker'] == selected_ticker].sort_values('Date')

            plt.figure(figsize=(12, 6))
            plt.plot(ticker_data['Date'], ticker_data['Open'], label='Open', linestyle='-', alpha=0.7)
            plt.plot(ticker_data['Date'], ticker_data['High'], label='High', linestyle='--', alpha=0.7)
            plt.plot(ticker_data['Date'], ticker_data['Low'], label='Low', linestyle='--', alpha=0.7)
            plt.plot(ticker_data['Date'], ticker_data['Close'], label='Close', linestyle='-', alpha=0.7)

            plt.title(f'{selected_ticker} OHLC Prices - January 2025')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.tight_layout()

            output_file = os.path.join(output_dir, f'{selected_ticker}_OHLC_Jan_2025.png')
            plt.savefig(output_file)
            print(f"Graph saved as: {output_file}")
            plt.close() 
