import os
import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
import warnings

warnings.filterwarnings('ignore')

TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META'] 

START_DATE = '2020-01-01'
END_DATE = '2025-12-31'

OUTPUT_DIR = 'data/main_data'
OUTPUT_FILE = 'tech_macro_aligned.csv'

def fetch_price_data(tickers, start, end):
    print("Fetching market price data from Yahoo Finance...")
    all_data = []
    
    for ticker in tickers:
        df = yf.download(ticker, start=start, end=end, progress=False)
        
        if isinstance(df.columns, pd.MultiIndex):
             df.columns = df.columns.droplevel(1)
             
        df['Ticker'] = ticker
        df = df.reset_index()
        
        df = df.rename(columns={'index': 'Date', 'Date': 'Date'}) 
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None) 
        
        df = df[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]
        all_data.append(df)
        
    return pd.concat(all_data, ignore_index=True)

def fetch_macro_data(start, end):
    print("Fetching macroeconomic indicators from FRED...")
    
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
            
    macro_df = macro_df.ffill()
    
    macro_df = macro_df.reset_index()
    macro_df = macro_df.rename(columns={'DATE': 'Date'})
    macro_df['Date'] = pd.to_datetime(macro_df['Date'])
    
    return macro_df

def build_dataset():
    price_df = fetch_price_data(TICKERS, START_DATE, END_DATE)
    
    macro_df = fetch_macro_data(START_DATE, END_DATE)
    
    print("Aligning data sources to the master calendar...")
    final_df = pd.merge(price_df, macro_df, on='Date', how='left')
    
    final_df = final_df.sort_values(by=['Ticker', 'Date'])
    final_df[['Fed_Rate', 'CPI', 'Treasury_10Y']] = final_df.groupby('Ticker')[['Fed_Rate', 'CPI', 'Treasury_10Y']].ffill()
    
    final_df = final_df.dropna() 
    
    return final_df

if __name__ == "__main__":
    dataset = build_dataset()
    print("\nPipeline execution complete. Data sample:")
    print(dataset.head())
    print(f"\nTotal records generated: {len(dataset)}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    dataset.to_csv(output_path, index=False)
    print(f"Successfully saved multivariate dataset to: {output_path}")
