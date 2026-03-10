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

# 4. Processing: Cleaning & Indexing
if data_list:
    df_raw = pd.concat(data_list, axis=1)
    df_cleaned = df_raw.ffill().bfill()
    
    # Outlier Removal
    returns = df_cleaned.pct_change().dropna()
    z_scores = np.abs(stats.zscore(returns))
    is_outlier = (z_scores > 3).any(axis=1)
    df_final_prices = df_cleaned.drop(index=returns.index[is_outlier])

    # 5. Build Combined Scientific Table
    # Create normalized columns
    initial_values = df_final_prices.iloc[0]
    
    combined_data = pd.DataFrame(index=df_final_prices.index)
    
    for col in df_final_prices.columns:
        # Add Original Price Column
        combined_data[f"{col}_Price"] = df_final_prices[col].round(2)
        # Add Normalized % Column
        combined_data[f"{col}_%_of_Base"] = ((df_final_prices[col] / initial_values[col]) * 100).round(2)

    # 6. Export to Excel
    file_name = f"Full_Market_Analysis_{datetime.now().strftime('%Y%m%d')}.xlsx"
    output_path = os.path.join(downloads_folder, file_name)
    
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        combined_data.to_excel(writer, sheet_name='Market_Data', index_label='Date')
        
        workbook  = writer.book
        ws = writer.sheets['Market_Data']
        
        # Formatting
        header_fmt = workbook.add_format({'bold': True, 'bg_color': '#D9D9D9', 'border': 1, 'align': 'center'})
        price_fmt = workbook.add_format({'num_format': '#,##0.00'})
        percent_fmt = workbook.add_format({'num_format': '0.00"%"', 'bg_color': '#EBF1DE'})
        date_fmt = workbook.add_format({'num_format': 'yyyy-mm-dd'})

        # Column Formatting
        ws.set_column('A:A', 15, date_fmt) # Date
        
        # Style Price columns (B, D, F) and Percent columns (C, E, G)
        for i in range(len(indices)):
            price_col_idx = 1 + (i * 2)
            pct_col_idx = 2 + (i * 2)
            ws.set_column(price_col_idx, price_col_idx, 15, price_fmt)
            ws.set_column(pct_col_idx, pct_col_idx, 15, percent_fmt)

        ws.freeze_panes(1, 1)

    print(f"--- Process Complete ---")
    print(f"File includes Raw Prices and % Growth.")
    print(f"Saved to: {output_path}")
else:
    print("No data retrieved.")