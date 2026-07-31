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
    
    for child in sweep_out_dir.iterdir():
        if child.is_dir():
            parts = child.name.split('_')
            
            # Check if this looks like a sweep directory: winX_hopY_embedder[_padding]
            if len(parts) < 3 or not parts[0].startswith('win') or not parts[1].startswith('hop'):
                continue
                
            try:
                window_sec = float(parts[0].replace('win', ''))
                hop_str = parts[1].replace('hop', '')
                
                is_pct = hop_str.endswith('pct')
                if is_pct:
                    hop_raw = float(hop_str.replace('pct', ''))
                else:
                    hop_raw = float(hop_str)
                    
                embedder_type = parts[2]
                perch_padding = parts[3] if len(parts) > 3 else "repeat"
                
                if is_pct:
                    hop_pct = int(hop_raw)
                    hop_sec = round(window_sec * hop_pct / 100, 4)
                else:
                    hop_sec = hop_raw
                    hop_pct = int(round(hop_sec / window_sec * 100))
            except ValueError:
                # If conversion to float fails, it's not a valid sweep directory
                continue
                
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
                    df.insert(3, "hop_pct", hop_pct)
                    df.insert(4, "perch_padding", perch_padding)
                    all_dfs.append(df)
                except Exception as e:
                    print(f"Error reading {csv_path}: {e}")
                    
    if not all_dfs:
        print("No results found to aggregate.")
        sys.exit(0)
        
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Sort by the mean of detsim,nmi,ari,homog?
    sort_cols = []
    for col in ["detsim", "nmi", "ari", "homog"]:
        if col in combined_df.columns:
            sort_cols.append(col)
    
    combined_df["mean_total_score"] = combined_df[sort_cols].mean(axis=1)

    if sort_cols:
        combined_df = combined_df.sort_values(by="mean_total_score", ascending=False)
        
    combined_df.to_csv(out_csv, index=False)
    print(f"Aggregated {len(combined_df)} rows into {out_csv}")

if __name__ == "__main__":
    main()
