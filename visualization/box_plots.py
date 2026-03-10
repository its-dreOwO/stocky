import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

# Use non-interactive backend for saving files only
plt.switch_backend('Agg')

# ==========================================
# CONFIGURATION
# ==========================================
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "main_data", "tech_macro_aligned.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# ==========================================
# 1. BOX PLOT
# ==========================================
def plot_box(df: pd.DataFrame, save: bool = False):
    """
    Box Plot to show price distribution by ticker and detect outliers.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data = [df[df['Ticker'] == t]['Close'].values for t in df['Ticker'].unique()]
    ax.boxplot(data, tick_labels=df['Ticker'].unique(), patch_artist=True)
    
    # Color each box
    colors = plt.cm.Set3(np.linspace(0, 1, len(data)))
    for patch, color in zip(ax.artists, colors):
        patch.set_facecolor(color)
    
    ax.set_title('Box Plot - Stock Price Distribution', fontsize=14, fontweight='bold')
    ax.set_ylabel('Price (Close)', fontsize=12)
    ax.set_xlabel('Ticker', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.savefig(os.path.join(OUTPUT_DIR, 'box_plot.png'), dpi=300)
        print(f"Saved box plot to: {os.path.join(OUTPUT_DIR, 'box_plot.png')}")

    plt.close()

# ==========================================
# 2. HISTOGRAM
# ==========================================
def plot_histogram(df: pd.DataFrame, column: str = 'Close', bins: int = 30, save: bool = False):
    """
    Histogram to show distribution of a data column.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(df[column], bins=bins, edgecolor='black', alpha=0.7, color='skyblue')
    
    # Add mean and median lines
    mean = df[column].mean()
    median = df[column].median()
    ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.2f}')
    ax.axvline(median, color='green', linestyle='--', linewidth=2, label=f'Median: {median:.2f}')
    
    ax.set_title(f'Histogram - {column} Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel(column, fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.savefig(os.path.join(OUTPUT_DIR, f'histogram_{column}.png'), dpi=300)
        print(f"Saved histogram to: {os.path.join(OUTPUT_DIR, f'histogram_{column}.png')}")

    plt.close()

# ==========================================
# 3. COMBINED PLOT (BOX + HISTOGRAM)
# ==========================================
def plot_combined(df: pd.DataFrame, column: str = 'Close', bins: int = 30, save: bool = False):
    """
    Combined Box Plot and Histogram on the same figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Box Plot (left)
    data = [df[df['Ticker'] == t][column].values for t in df['Ticker'].unique()]
    ax1.boxplot(data, tick_labels=df['Ticker'].unique(), patch_artist=True)
    colors = plt.cm.Set3(np.linspace(0, 1, len(data)))
    for patch, color in zip(ax1.artists, colors):
        patch.set_facecolor(color)
    ax1.set_title('Box Plot', fontsize=12, fontweight='bold')
    ax1.set_ylabel(column, fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Histogram (right)
    ax2.hist(df[column], bins=bins, edgecolor='black', alpha=0.7, color='skyblue')
    mean = df[column].mean()
    median = df[column].median()
    ax2.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.2f}')
    ax2.axvline(median, color='green', linestyle='--', linewidth=2, label=f'Median: {median:.2f}')
    ax2.set_title('Histogram', fontsize=12, fontweight='bold')
    ax2.set_xlabel(column, fontsize=10)
    ax2.set_ylabel('Frequency', fontsize=10)
    ax2.legend()
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle('Data Distribution - Box Plot & Histogram', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.savefig(os.path.join(OUTPUT_DIR, 'combined_plot.png'), dpi=300)
        print(f"Saved combined plot to: {os.path.join(OUTPUT_DIR, 'combined_plot.png')}")

    plt.close()

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print(f"Loading data from: {DATA_PATH}")
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: File not found - {DATA_PATH}")
        print("Please run data/data_scrapping/main.py first to create the data.")
    else:
        df = pd.read_csv(DATA_PATH)
        print(f"Loaded {len(df)} data records.\n")
        
        print("=" * 50)
        print("1. GENERATING BOX PLOT")
        print("=" * 50)
        plot_box(df, save=True)
        
        print("\n" + "=" * 50)
        print("2. GENERATING HISTOGRAM")
        print("=" * 50)
        plot_histogram(df, column='Close', save=True)
        
        print("\n" + "=" * 50)
        print("3. GENERATING COMBINED PLOT")
        print("=" * 50)
        plot_combined(df, column='Close', save=True)
        
        print("\nDone! All visualizations have been generated and saved to the /visualization/output/.")
