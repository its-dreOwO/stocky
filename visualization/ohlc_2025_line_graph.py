import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the file path
csv_file_path = os.path.join('data', 'main_data', 'tech_macro_aligned.csv')

# Load the data
if not os.path.exists(csv_file_path):
    print(f"Error: {csv_file_path} not found.")
else:
    df = pd.read_csv(csv_file_path)

    # Convert Date to datetime object
    df['Date'] = pd.to_datetime(df['Date'])

    # Filter for the year 2025
    df_2025 = df[df['Date'].dt.year == 2025].copy()

    if df_2025.empty:
        print("No data found for the year 2025.")
    else:
        # Get unique tickers
        tickers = df_2025['Ticker'].unique()
        
        # Ensure output directory exists
        output_dir = os.path.join('visualization', 'output')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for selected_ticker in tickers:
            print(f"Generating OHLC line graph for: {selected_ticker} (Year: 2025)")

            # Filter for the selected ticker
            ticker_data = df_2025[df_2025['Ticker'] == selected_ticker].sort_values('Date')

            # Create the plot
            plt.figure(figsize=(12, 6))
            plt.plot(ticker_data['Date'], ticker_data['Open'], label='Open', linestyle='-', alpha=0.7)
            plt.plot(ticker_data['Date'], ticker_data['High'], label='High', linestyle='--', alpha=0.7)
            plt.plot(ticker_data['Date'], ticker_data['Low'], label='Low', linestyle='--', alpha=0.7)
            plt.plot(ticker_data['Date'], ticker_data['Close'], label='Close', linestyle='-', alpha=0.7)

            # Add labels and title
            plt.title(f'{selected_ticker} OHLC Prices - 2025')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.tight_layout()

            # Save the plot
            output_file = os.path.join(output_dir, f'{selected_ticker}_OHLC_2025.png')
            plt.savefig(output_file)
            print(f"Graph saved as: {output_file}")
            plt.close() # Close figure to free memory
