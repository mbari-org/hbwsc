import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def plot_umap_mcs(csv_path: str | Path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return

    # Load data
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Filter for perch
    if 'embedder_type' in df.columns:
        df = df[df['embedder_type'] == 'perch']
        
    # Filter for our optimal window and hop size
    df_opt = df[(df['window_sec'] == 2.0) & (df['hop_pct'] == 25) & (df['umap_dims'] > 2)].copy()
    
    if df_opt.empty:
        print("No data found for window_sec=2.0 and hop_pct=25")
        return
        
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot separate line for each mcs
    for mcs in sorted(df_opt['mcs'].unique()):
        # Filter for this specific mcs
        df_mcs = df_opt[df_opt['mcs'] == mcs]
        
        # Sort by umap_dims so the line draws left to right
        df_mcs = df_mcs.sort_values('umap_dims')
        
        # Take max score of each umap/mcs combination
        df_plot = df_mcs.groupby('umap_dims')['mean_total_score'].max().reset_index()
        
        ax.plot(df_plot['umap_dims'], df_plot['mean_total_score'], marker='o', label=f'MCS = {mcs}')

    ax.set_title("Performance across UMAP Dimensions and Min Cluster Size\n(Fixed at 2.0s Window, 25% Hop)", pad=15)
    ax.set_xlabel("UMAP Dimensions", labelpad=10)
    ax.set_ylabel("Averaged ARI and AMI", labelpad=10)
    
    # Make the x-ticks match the exact umap dimensions we tested
    ax.set_xticks(sorted(df_opt['umap_dims'].unique()))
    
    ax.legend(title="Min Cluster Size")
    ax.grid(True, linestyle='--', alpha=0.6)
    
    fig.tight_layout()
    
    out_path = csv_path.parent / "umap_mcs_plot.png"
    fig.savefig(out_path, dpi=300)
    print(f"Saved line plot to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot UMAP dims vs MCS for optimal window/hop.")
    parser.add_argument("csv_path", type=str, help="Path to combined_results.csv")
    args = parser.parse_args()
    
    plot_umap_mcs(args.csv_path)
