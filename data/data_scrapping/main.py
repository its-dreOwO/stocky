import os
import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
import warnings

# Suppress warnings related to yfinance's multi-index dataframe updates
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION PARAMETERS
# ==========================================
# Representative tech sector tickers for the baseline study
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META'] 

# Define the study period (aligned with the previous LSTM baseline)
START_DATE = '2020-01-01'
END_DATE = '2025-12-31'

# Output directory configuration based on project structure
OUTPUT_DIR = 'data/main_data'
OUTPUT_FILE = 'tech_macro_aligned.csv'

def fetch_price_data(tickers, start, end):
    """
    Fetches daily OHLCV market data from Yahoo Finance.
    This serves as the primary data source and establishes the master trading calendar.
    """
    print("Fetching market price data from Yahoo Finance...")
    all_data = []
    
    for ticker in tickers:
        df = yf.download(ticker, start=start, end=end, progress=False)
        
        # Flatten multi-index columns introduced in recent yfinance versions
        if isinstance(df.columns, pd.MultiIndex):
             df.columns = df.columns.droplevel(1)
             
        df['Ticker'] = ticker
        df = df.reset_index()
        
        # Standardize the date column nomenclature for future merging operations
        df = df.rename(columns={'index': 'Date', 'Date': 'Date'}) 
        # Remove timezone information to prevent merging conflicts
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None) 
        
        # Retain only essential features to minimize dimensionality
        df = df[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]
        all_data.append(df)
        
    return pd.concat(all_data, ignore_index=True)

def fetch_macro_data(start, end):
    """
    Retrieves macroeconomic indicators from the Federal Reserve Economic Data (FRED) API.
    Handles mixed-frequency data (daily and monthly) using forward-fill methodology.
    """
    print("Fetching macroeconomic indicators from FRED...")
    
    # Define series identifiers
    # DFF: Federal Funds Effective Rate (Daily)
    # CPIAUCSL: Consumer Price Index for All Urban Consumers (Monthly)
    # GS10: 10-Year Treasury Constant Maturity Rate (Daily)
    series_dict = {
        'Fed_Rate': 'DFF',
        'CPI': 'CPIAUCSL', 
        'Treasury_10Y': 'GS10'
    }
    
    macro_df = pd.DataFrame()
    for name, series_id in series_dict.items():
        try:
            data = web.DataReader(series_id, 'fred', start, end)
            data.columns = [name]
            
            if macro_df.empty:
                macro_df = data
            else:
                macro_df = macro_df.join(data, how='outer')
        except Exception as e:
            print(f"Error fetching {name}: {e}")
            
    # CRITICAL STEP: Apply forward-fill (ffill) to prevent data leakage (look-ahead bias).
    # This ensures that low-frequency data (e.g., monthly CPI) is only mapped to daily 
    # records *after* it is officially published.
    macro_df = macro_df.ffill()
    
    macro_df = macro_df.reset_index()
    macro_df = macro_df.rename(columns={'DATE': 'Date'})
    macro_df['Date'] = pd.to_datetime(macro_df['Date'])
    
    return macro_df

def build_dataset():
    """
    Constructs the final multivariate dataset by aligning market and macroeconomic data.
    """
    # Step 1: Construct the master calendar using market trading days
    price_df = fetch_price_data(TICKERS, START_DATE, END_DATE)
    
    # Step 2: Retrieve macroeconomic variables
    macro_df = fetch_macro_data(START_DATE, END_DATE)
    
    print("Aligning data sources to the master calendar...")
    # Step 3: Apply a left join on the Date column. 
    # This ensures only valid market trading days are retained, dropping weekends/holidays.
    final_df = pd.merge(price_df, macro_df, on='Date', how='left')
    
    # Forward-fill any remaining missing values caused by asynchronous holiday schedules 
    # between the equity market and the Federal Reserve
    final_df = final_df.sort_values(by=['Ticker', 'Date'])
    final_df[['Fed_Rate', 'CPI', 'Treasury_10Y']] = final_df.groupby('Ticker')[['Fed_Rate', 'CPI', 'Treasury_10Y']].ffill()
    
    # Drop initial rows where macroeconomic data might be unavailable (prior to the first data point)
    final_df = final_df.dropna() 
    
    return final_df

if __name__ == "__main__":
    # Execute the data pipeline
    dataset = build_dataset()
    print("\nPipeline execution complete. Data sample:")
    print(dataset.head())
    print(f"\nTotal records generated: {len(dataset)}")
    
    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save the consolidated dataset
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    dataset.to_csv(output_path, index=False)
    print(f"Successfully saved multivariate dataset to: {output_path}")