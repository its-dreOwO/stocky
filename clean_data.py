import pandas_datareader.data as web
import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime
from scipy import stats

# 1. Setup paths
downloads_folder = str(Path.home() / "Downloads")

# 2. Configuration
indices = {
    'US_DowJones': '^DJI',            
    'EU_EuroTop50': 'FEZ', 
    'VN_ETF': 'VNM.US'       
}
start_date = datetime(2022, 1, 1)
end_date = datetime(2025, 12, 31)

# 3. Data Acquisition
data_list = []
for name, ticker in indices.items():
    try:
        df = web.DataReader(ticker, 'stooq', start_date, end_date)
        df = df.sort_index()
        if not df.empty:
            series = df['Close'].rename(name)
            data_list.append(series)
    except Exception as e:
        print(f"Error fetching {name}: {e}")

# 4. Scientific Cleaning Process
if data_list:
    # Merge datasets on Date index
    df_raw = pd.concat(data_list, axis=1)
    
    # A. Handle Missing Values: Forward fill then backward fill 
    # (Scientific standard for time-series to maintain continuity)
    df_cleaned = df_raw.ffill().bfill()
    
    # B. Statistical Outlier Detection (Z-Score)
    # Instead of a flat 5% threshold, we use 3 standard deviations
    z_scores = np.abs(stats.zscore(df_cleaned.pct_change().dropna()))
    # Identify rows where any column has a Z-score > 3
    is_outlier = (z_scores > 3).any(axis=1)
    # Align index and filter
    outlier_dates = df_cleaned.pct_change().dropna().index[is_outlier]
    df_final = df_cleaned.drop(index=outlier_dates)

    # 5. Scientific Export Formatting
    # Rounding to 4 decimal places for precision without noise
    df_final = df_final.round(4)
    
    # Prepare File Path
    file_name = f"Financial_Data_Cleaned_{datetime.now().strftime('%Y%m%d')}.csv"
    output_path = os.path.join(downloads_folder, file_name)
    
    # Export with specific scientific parameters:
    # - ISO 8601 Date format
    # - Header naming convention (No spaces)
    # - UTF-8 encoding
    df_final.to_csv(output_path, 
                    index_label='Date', 
                    date_format='%Y-%m-%d', 
                    encoding='utf-8')

    print(f"--- Process Complete ---")
    print(f"Original Rows: {len(df_raw)}")
    print(f"Cleaned Rows:  {len(df_final)}")
    print(f"File Saved:    {output_path}")
else:
    print("No data was retrieved.")