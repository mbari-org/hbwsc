"""Evaluate clustering generalization across sessions.

This orchestrator script runs a full pipeline:
1. Filters a sweep CSV from a train session to find the top configurations based on constraints.
2. Generates the cluster assignments (pseudo-labels) for those top configs on the train session.
3. Trains a classifier on those pseudo-labels.
4. Uses the classifier to predict labels for an evaluation session.
5. Compares the predicted labels against the evaluation session's manual ground truth.

Usage:
    uv run python scripts/evaluate_generalization.py <train_session> <eval_session> [options]

Options:
    --top-x          Number of configurations to evaluate (default: 5)
    --min-clusters   Minimum number of clusters required (default: 15)
    --max-clusters   Maximum number of clusters allowed (default: 30)
    --max-noise      Maximum allowed noise percentage (default: 40.0)
    --drop-noise     Drop noise points (-1) when training the classifier
"""

import argparse
import csv
import sys
import subprocess
from pathlib import Path
import numpy as np

# Ensure scripts and src are accessible
sys.path.insert(0, str(Path(__file__).parent))
from session import load_params, cmd_run, mcs_dir

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from hbws_clustering.evaluation import load_raven_labels, map_labels_to_windows, compute_metrics


def run_subprocess(cmd):
    print(f"Running: {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("train_session", type=Path, help="Path to the training session directory")
    parser.add_argument("eval_session", type=Path, help="Path to the evaluation session directory")
    parser.add_argument("--top-x", type=int, default=5, help="Number of configurations to evaluate (default: 5)")
    parser.add_argument("--min-clusters", type=int, default=15, help="Minimum number of clusters (default: 15)")
    parser.add_argument("--max-clusters", type=int, default=30, help="Maximum number of clusters (default: 30)")
    parser.add_argument("--max-noise", type=float, default=40.0, help="Maximum allowed noise percentage (default: 40.0)")
    parser.add_argument("--drop-noise", action="store_true", help="Drop noise points when training classifier")
    args = parser.parse_args()

    train_dir = args.train_session.resolve()
    eval_dir = args.eval_session.resolve()
    
    # Read Sweep CSV
    sweep_dir = train_dir / "sweep"
    csv_files = list(sweep_dir.glob("results*.csv"))
    if not csv_files:
        print(f"Error: No sweep CSV found in {sweep_dir}. Run 'session.py <train_session> sweep' first.")
        sys.exit(1)
    
    # Just take the first matching sweep results
    csv_file = csv_files[0]
    print(f"Using sweep results: {csv_file}")
    
    valid_runs = []
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_clusters = int(row["n_clusters"])
            noise_pct = float(row["noise_pct"])
            # Handle any NaN dbcv values
            dbcv = float(row["dbcv"]) if row["dbcv"] not in ("nan", "NaN") else -2.0
            
            if args.min_clusters <= n_clusters <= args.max_clusters and noise_pct <= args.max_noise:
                row["dbcv_val"] = dbcv # cache for sorting
                valid_runs.append(row)
                
    if not valid_runs:
        print(f"No configurations matched the criteria: {args.min_clusters} <= n_clusters <= {args.max_clusters} AND noise <= {args.max_noise}%")
        sys.exit(1)
        
    valid_runs.sort(key=lambda r: r["dbcv_val"], reverse=True)
    top_runs = valid_runs[:args.top_x]
    
    print(f"\nTop {len(top_runs)} configurations passing heuristics:")
    for i, r in enumerate(top_runs):
        print(f" {i+1}. {r['umap_dims']}D, mcs={r['mcs']}, eps={r['eps']} | n={r['n_clusters']}, noise={r['noise_pct']}%, dbcv={r['dbcv']}")

    # Eval session manual labels
    eval_params = load_params(eval_dir)
    if "manual_labels" not in eval_params:
        print("Error: eval_session must have 'manual_labels' defined in parameters.yml to evaluate generalization.")
        sys.exit(1)
    
    manual_labels_path = Path(eval_params["manual_labels"])
    if not manual_labels_path.is_absolute():
        manual_labels_path = eval_dir / manual_labels_path
        
    print(f"\nLoading eval session manual labels from {manual_labels_path}")
    manual = load_raven_labels(manual_labels_path)
    
    results = []

    for idx, run in enumerate(top_runs):
        print(f"\n{'='*70}")
        print(f"=== Evaluating Config {idx+1}/{len(top_runs)}: {run['umap_dims']}D, mcs={run['mcs']}, eps={run['eps']}")
        print(f"{'='*70}")
        
        train_params = load_params(train_dir)
        train_params["umap_cluster_components"] = int(run["umap_dims"])
        train_params["hdbscan_epsilon"] = float(run["eps"])
        mcs_val = run["mcs"]
        
        # Generate cluster labels on train session
        d, _ = mcs_dir(train_dir, train_params, mcs_val)
        npz = d / "results.npz"
        if not npz.exists():
            print(f"Generating cluster labels for train session...")
            cmd_run(train_dir, train_params, mcs_val)
        else:
            print(f"Found existing clustering results: {npz}")
            
        # Train classifier
        model_pkl = train_dir / "models" / f"{npz.parent.name}_model.pkl"
        if not model_pkl.exists():
            print(f"Training classifier on cluster labels...")
            cmd = ["uv", "run", "python", "scripts/train_classifier.py", str(npz), str(model_pkl)]
            if args.drop_noise:
                cmd.append("--drop-noise")
            run_subprocess(cmd)
        else:
            print(f"Found existing trained model: {model_pkl}")
            
        # Predict on Eval Session
        # Inherit embedding/window settings from the train session
        eval_inherited_params = eval_params.copy()
        for key in ["window_sec", "hop_sec", "perch_padding", "embedder_type", "sample_rate"]:
            if key in train_params:
                eval_inherited_params[key] = train_params[key]
                
        import yaml
        inherited_yml_path = eval_dir / f"parameters_inherited_from_{train_dir.name}.yml"
        with open(inherited_yml_path, "w") as f:
            yaml.dump(eval_inherited_params, f, sort_keys=False)
            
        pred_npz = eval_dir / "predictions" / f"{train_dir.name}_{model_pkl.stem}_predictions.npz"
        if not pred_npz.exists():
            print(f"Running inference on evaluation session (inheriting {train_dir.name} windowing)...")
            cmd = ["uv", "run", "python", "scripts/predict.py", str(inherited_yml_path), str(model_pkl), str(pred_npz)]
            run_subprocess(cmd)
        else:
            print(f"Found existing predictions: {pred_npz}")
            
        # 5. Evaluate against ground truth
        print(f"Evaluating predictions against manual labels...")
        r_pred = np.load(pred_npz, allow_pickle=False)
        labels = r_pred["labels"]
        start_secs = r_pred["start_secs"]
        
        # Infer hop_sec from start_secs array
        if len(start_secs) > 1:
            hop_sec = float(np.median(np.diff(start_secs)))
        else:
            hop_sec = float(eval_params["hop_sec"])
            
        window_sec = float(eval_params["window_sec"])
        end_secs = start_secs + window_sec
        
        manual_window, _ = map_labels_to_windows(manual, start_secs, end_secs)
        metrics = compute_metrics(labels, manual_window)
        
        nmi = metrics.get("NMI", float("nan"))
        ari = metrics.get("ARI", float("nan"))
        homog = metrics.get("Homogeneity", float("nan"))
        detsim = metrics.get("DetSim", float("nan"))
        
        agg = np.nanmean([nmi, ari, homog])
        
        results.append({
            "config": f"{run['umap_dims']}D, mcs={run['mcs']}, eps={run['eps']}",
            "DetSim": detsim,
            "NMI": nmi,
            "ARI": ari,
            "Homogeneity": homog,
            "Aggregated": agg
        })
        
    print(f"\n\n{'='*85}")
    print(f"=== GENERALIZATION REPORT: {train_dir.name} -> {eval_dir.name} ===")
    print(f"{'='*85}")
    print(f"{'Configuration':<25} | {'DetSim':<8} | {'NMI':<8} | {'ARI':<8} | {'Homog':<8} | {'Aggreg':<8}")
    print("-" * 85)
    
    results.sort(key=lambda r: r["Aggregated"] if not np.isnan(r["Aggregated"]) else -2.0, reverse=True)
    
    for res in results:
        print(f"{res['config']:<25} | {res['DetSim']:<8.4f} | {res['NMI']:<8.4f} | {res['ARI']:<8.4f} | {res['Homogeneity']:<8.4f} | {res['Aggregated']:<8.4f}")

    # Save to CSV
    report_csv = eval_dir / f"generalization_report_{train_dir.name}.csv"
    with open(report_csv, "w", newline="") as f:
        fieldnames = ["config", "DetSim", "NMI", "ARI", "Homogeneity", "Aggregated"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\nReport saved to: {report_csv}")


if __name__ == "__main__":
    main()
