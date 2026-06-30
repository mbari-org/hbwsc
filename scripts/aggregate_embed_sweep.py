import sys
import pandas as pd
from pathlib import Path
import re

def main():
    if len(sys.argv) != 3:
        print("Usage: uv run python aggregate_embed_sweep.py <session_dir> <output_csv>")
        sys.exit(1)
        
    session_dir = Path(sys.argv[1])
    out_csv = Path(sys.argv[2])
    
    sweep_out_dir = session_dir / "sweep_embed"
    if not sweep_out_dir.exists():
        print(f"ERROR: {sweep_out_dir} does not exist.")
        sys.exit(1)
        
    all_dfs = []
    
    # regex to parse: win0.25_hop0.25_perch
    pattern = re.compile(r"win([\d\.]+)_hop([\d\.]+)_([a-zA-Z0-9_\-]+)")
    
    for child in sweep_out_dir.iterdir():
        if child.is_dir():
            match = pattern.match(child.name)
            if match:
                window_sec = float(match.group(1))
                hop_sec = float(match.group(2))
                embedder_type = match.group(3)
                
                # find the csv inside
                sweep_dir = child / "sweep"
                if not sweep_dir.exists():
                    continue
                
                # find results_*.csv
                csvs = list(sweep_dir.glob("results_*.csv"))
                if not csvs:
                    continue
                
                # there should be only one, or we take the newest
                csv_path = max(csvs, key=lambda p: p.stat().st_mtime)
                
                try:
                    df = pd.read_csv(csv_path)
                    df.insert(0, "embedder_type", embedder_type)
                    df.insert(1, "window_sec", window_sec)
                    df.insert(2, "hop_sec", hop_sec)
                    all_dfs.append(df)
                except Exception as e:
                    print(f"Error reading {csv_path}: {e}")
                    
    if not all_dfs:
        print("No results found to aggregate.")
        sys.exit(0)
        
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Sort by the mean of detsim,nmi,homog?
    sort_cols = []
    for col in ["detsim", "nmi", "homog"]:
        if col in combined_df.columns:
            sort_cols.append(col)
    
    combined_df["mean_total_score"] = combined_df[sort_cols].mean(axis=1)

    if sort_cols:
        combined_df = combined_df.sort_values(by="mean_total_score", ascending=False)
        
    combined_df.to_csv(out_csv, index=False)
    print(f"Aggregated {len(combined_df)} rows into {out_csv}")

if __name__ == "__main__":
    main()
