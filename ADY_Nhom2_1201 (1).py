import pandas_datareader.data as web
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from datetime import datetime


#! US: NVDA GOOGL AAPL MSFT AMZN
#! VN: ACB VHM FPT PLX 
#! EU: SAP LVMH ASML Nestle  
# Xác định thư mục Downloads
downloads_folder = str(Path.home() / "Downloads")
print(f"Thu muc Downloads: {downloads_folder}")

# 1. Định nghĩa danh sách chỉ số (Theo mã của Stooq)
indices = {
    'My (Dow Jones 30)': '^DJI',            
    'Chau Au (Euro top50 )': 'FEZ', # Chỉ số 50 cty lớn nhất EU
    'Viet Nam': 'VNM.US'       
}

# 2. Tải dữ liệu từ STOOQ
print("Dang tai du lieu tu STOOQ...")
data_frames = []

start_date = datetime(2022, 1, 1)
end_date = datetime(2025, 12, 31)

for name, ticker in indices.items():
    try:
        print(f"  - Dang tai {name} ({ticker})...")
        
        # Sử dụng pandas_datareader với nguồn 'stooq'
        df = web.DataReader(ticker, 'stooq', start_date, end_date)
        
        # Stooq trả về dữ liệu ngược (mới nhất ở trên), cần đảo lại
        df = df.sort_index()
        
        if not df.empty and 'Close' in df.columns:
            # Lấy cột Close và đổi tên
            close_series = df['Close']
            close_series.name = name
            data_frames.append(close_series)
            print(f"    -> Thanh cong! ({len(df)} dong)")
        else:
            print(f"    -> THAT BAI: Khong co du lieu cho {ticker}")
            
    except Exception as e:
        print(f"    -> LOI: {e}")
        print("       (Goi y: Hay cai dat: pip install pandas-datareader)")

# 3. Gộp và Xử lý dữ liệu
if len(data_frames) > 0:
    df_original = pd.concat(data_frames, axis=1)
    
    # Đảm bảo tên cột là String (tránh lỗi Tuple)
    df_original.columns = [str(col) for col in df_original.columns]
    
    print("\n\n" + "="*70)
    print("BANG DU LIEU GOC (ORIGINAL DATA)")
    print("="*70)
    print(f"Tong so dong: {len(df_original)}")
    print(df_original.head())
    print(f"\nSo gia tri thieu (missing) theo cot:")
    print(df_original.isnull().sum())
    
    # 4. FILTERING PROCESS - QUÁ TRÌNH LỌC
    print("\n\n\n" + "="*70)
    print("QUA TRINH LOC DU LIEU")
    print("="*70)
    
    # Bước 1: Xử lý missing values
    df_filtered = df_original.copy()
    rows_with_missing = df_filtered[df_filtered.isnull().any(axis=1)].copy()
    
    # Loại bỏ hàng thiếu
    df_filtered = df_filtered.dropna()
    
    # Bước 2: Loại bỏ outliers (Biến động > 10%)
    daily_returns = df_filtered.pct_change()
    outlier_threshold = 0.05
    outlier_mask = (daily_returns.abs() > outlier_threshold).any(axis=1)
    outlier_rows = df_filtered[outlier_mask].copy()
    
    # Loại bỏ outliers
    df_filtered = df_filtered[~outlier_mask]
    
    # 5. TẠO BẢNG DỮ LIỆU BỊ LOẠI BỎ
    print("\n\n\n" + "="*70)
    print("BANG DU LIEU BI LOAI BO (REMOVED DATA)")
    print("="*70)
    
    removed_summary = []
    
    # Ghi nhận Missing
    if len(rows_with_missing) > 0:
        for idx, row in rows_with_missing.iterrows():
            missing_cols = row[row.isnull()].index.tolist()
            removed_summary.append({
                'Ngay': idx.strftime('%Y-%m-%d'),
                'Ly_do': 'Thieu du lieu',
                'Chi_tiet': f"Thieu {len(missing_cols)} chi so: {', '.join(missing_cols)}",
                'So_cot_thieu': len(missing_cols)
            })
    
    # Ghi nhận Outliers
    if len(outlier_rows) > 0:
        for idx, row in outlier_rows.iterrows():
            if idx in daily_returns.index:
                returns = daily_returns.loc[idx]
                extreme_cols = returns[returns.abs() > outlier_threshold].index.tolist()
                extreme_values = [f"{col}: {returns[col]*100:.2f}%" for col in extreme_cols]
                removed_summary.append({
                    'Ngay': idx.strftime('%Y-%m-%d'),
                    'Ly_do': 'Bien dong bat thuong',
                    'Chi_tiet': f"Thay doi qua lon: {', '.join(extreme_values)}",
                    'So_cot_thieu': 0
                })

    if len(removed_summary) > 0:
        removed_data_detailed = pd.DataFrame(removed_summary)
        removed_data_detailed = removed_data_detailed.drop_duplicates(subset=['Ngay'])
        removed_data_detailed = removed_data_detailed.sort_values('Ngay')
        
        removed_file = os.path.join(downloads_folder, 'removed_data.csv')
        removed_data_detailed.to_csv(removed_file, index=False, encoding='utf-8-sig')
        print(f"\n>>> Da xuat bang du lieu bi loai ra: {removed_file}")
    else:
        print("Khong co du lieu nao bi loai bo!")

    # 6. XUẤT DỮ LIỆU SẠCH
    print("\n\n\n" + "="*70)
    print("KET QUA CUOI CUNG")
    print("="*70)
    filtered_file = os.path.join(downloads_folder, 'filtered_data.csv')
    df_filtered.to_csv(filtered_file)
    print(f"Tong so dong con lai: {len(df_filtered)}")
    print(f">>> Da xuat bang du lieu sach ra: {filtered_file}")
    
    # 7. VISUALIZATION
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Biểu đồ 1: Gốc
        if not df_original.empty:
            valid_idx = df_original.first_valid_index()
            if valid_idx:
                df_norm = (df_original / df_original.loc[valid_idx]) * 100
                df_norm.plot(ax=axes[0, 0], linewidth=2, alpha=0.7)
                axes[0, 0].set_title("DU LIEU GOC (Base=100)")
                axes[0, 0].grid(True, alpha=0.3)
        
        # Biểu đồ 2: Đã lọc
        if not df_filtered.empty:
            df_filt_norm = (df_filtered / df_filtered.iloc[0]) * 100
            df_filt_norm.plot(ax=axes[0, 1], linewidth=2, alpha=0.7)
            axes[0, 1].set_title("DU LIEU DA LOC (Base=100)")
            axes[0, 1].grid(True, alpha=0.3)
            
        # Biểu đồ 3: Heatmap Missing
        missing_map = df_original.isnull().astype(int)
        if missing_map.sum().sum() > 0:
            axes[1, 0].imshow(missing_map.T, aspect='auto', cmap='RdYlGn_r', interpolation='nearest')
            axes[1, 0].set_title("BẢN ĐỒ DỮ LIỆU THIẾU")
        else:
            axes[1, 0].text(0.5, 0.5, 'Khong co missing values', ha='center')
            
        # Biểu đồ 4: Số lượng
        counts = [len(df_original), len(df_filtered), len(df_original)-len(df_filtered)]
        axes[1, 1].bar(['Goc', 'Sach', 'Loai bo'], counts, color=['#3498db', '#2ecc71', '#e74c3c'])
        axes[1, 1].set_title("SỐ LƯỢNG DÒNG DỮ LIỆU")
        for i, v in enumerate(counts):
            axes[1, 1].text(i, v, str(v), ha='center', va='bottom')
            
        chart_file = os.path.join(downloads_folder, 'data_filtering_comparison.png')
        plt.tight_layout()
        plt.savefig(chart_file, dpi=300)
        print(f"\n>>> Da luu bieu do: {chart_file}")
        plt.show()
        
    except Exception as e:
        print(f"Loi ve bieu do: {e}")

    print("\nfinished")
    
else:
    print("\nerror: cannot draw data from stooq")