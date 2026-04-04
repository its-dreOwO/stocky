import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def create_pipeline_graph():
    # Setup the figure and axis
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_axis_off()

    # Define some helper functions for boxes and arrows
    def draw_box(x, y, w, h, text, color='lightblue', fontsize=12):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3", 
                                     linewidth=2, edgecolor='black', facecolor=color)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, fontweight='bold', wrap=True)
        return x, y, w, h

    def draw_arrow(start_pos, end_pos):
        ax.annotate('', xy=end_pos, xytext=start_pos,
                    arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=10))

    # --- 1. Data Sources ---
    # Top row
    draw_box(10, 85, 20, 10, "GDELT Sentiment\n(BQ Aligned)", color='lightgreen')
    draw_box(40, 85, 20, 10, "SEC Events\n(sec_scraper.py)", color='lightgreen')
    draw_box(70, 85, 20, 10, "Tech & Macro\nAligned Data", color='lightgreen')

    # --- 2. Preprocessing ---
    # Middle-top row
    draw_box(40, 65, 20, 10, "dataset_builder.py\n(MultivariateStockDataset)", color='skyblue')
    
    draw_arrow((20, 85), (45, 75)) # Arrow from GDELT
    draw_arrow((50, 85), (50, 75)) # Arrow from SEC
    draw_arrow((80, 85), (55, 75)) # Arrow from Market Data

    # --- 3. Modeling ---
    # Middle-bottom row
    draw_box(10, 45, 15, 10, "ARIMA\n(Statistical)", color='orange')
    draw_box(30, 45, 40, 10, "LTSM Pipeline\n(DLinear, NLinear, LSTM, GRU)", color='orange')
    draw_box(75, 45, 15, 10, "Logistic Regression\n(Classification)", color='orange')

    draw_arrow((50, 65), (17, 55)) # Arrow to ARIMA
    draw_arrow((50, 65), (50, 55)) # Arrow to LTSM
    draw_arrow((50, 65), (83, 55)) # Arrow to Logistic Regression

    # --- 4. Evaluation ---
    # Bottom row
    draw_box(35, 25, 30, 10, "eval_rolling_forcast.py\n(Metrics & Forecasting)", color='salmon')
    draw_arrow((50, 45), (50, 35))

    # --- 5. Output/Visualization ---
    # Very bottom row
    draw_box(10, 5, 20, 10, "Heatmaps &\nCorrelation Plots", color='plum')
    draw_box(40, 5, 20, 10, "Power BI\n(Stocks_visual.pbix)", color='plum')
    draw_box(70, 5, 20, 10, "Performance Reports\n(MAE, RMSE)", color='plum')

    draw_arrow((45, 25), (20, 15))
    draw_arrow((50, 25), (50, 15))
    draw_arrow((55, 25), (80, 15))

    # Add Titles
    plt.title("ADy201m Project Pipeline Graph", fontsize=20, fontweight='bold', pad=20)
    
    # Save the plot
    output_dir = os.path.join('visualization', 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    save_path = os.path.join(output_dir, 'project_pipeline_graph.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Pipeline graph saved to: {save_path}")

if __name__ == "__main__":
    create_pipeline_graph()
