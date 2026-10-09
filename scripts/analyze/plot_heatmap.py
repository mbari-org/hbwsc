import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def plot_heatmap(csv_path: str | Path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return

    # Load data
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Filter out anything that isn't perch in case somehow there are multiple embedders in sweep
    if 'embedder_type' in df.columns:
        df = df[df['embedder_type'] == 'perch']
        
    # Take max combined ARI and AMI per window and hop
    pivot = pd.pivot_table(
        df, 
        # values='mean_total_score', 
        values='ari',
        index='hop_pct', 
        columns='window_sec', 
        aggfunc='max'
    )
    
    pivot = pivot.sort_index(ascending=False) 
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap
    cax = ax.imshow(pivot.values, cmap='viridis', aspect='auto')
    
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    
    # Text metrics in plot
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            text_color = "black" if val > pivot.values.max() - 0.15 else "white"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=text_color, fontsize=9)
            
    # Titles and labels
    ax.set_title("Maximum ARI by Window Size and Hop Percentage", pad=15)
    ax.set_xlabel("Window Size (seconds)", labelpad=10)
    ax.set_ylabel("Hop Percentage (%)", labelpad=10)
    
    # Add colorbar
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label("Maximum Adjusted Rand Index (ARI)")
    
    fig.tight_layout()
    
    out_path = csv_path.parent / "alignment_heatmap.png"
    fig.savefig(out_path, dpi=300)
    print(f"Saved heatmap to {out_path}")
    # Don't show interactive if running in headless mode, just save
    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a heatmap of ARI scores from sweep results.")
    parser.add_argument("csv_path", type=str, help="Path to combined_results.csv")
    args = parser.parse_args()
    
    plot_heatmap(args.csv_path)
