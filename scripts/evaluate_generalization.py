"""Evaluate clustering generalization across sessions.

This orchestrator script runs a full pipeline:
1. Reads the combined sweep CSV (from aggregate-sweep-embed) or a single session's sweep CSV.
2. Filters to configurations based on cluster count and noise constraints (optionally limited to top-N by DBCV).
3. Generates the cluster assignments (pseudo-labels) for those top configs on the train session.
4. Trains a classifier on those pseudo-labels.
5. Uses the classifier to predict labels for an evaluation session.
6. Compares the predicted labels against the evaluation session's manual ground truth.

Usage:
    uv run python scripts/evaluate_generalization.py <train_session> <eval_session> [options]

Options:
    --min-clusters   Minimum number of clusters required (default: 15)
    --max-clusters   Maximum number of clusters allowed (default: 30)
    --max-noise      Maximum allowed noise percentage (default: 40.0)
    --min-dbcv       Minimum DBCV score to include (default: 0.2)
    --top-n-dbcv     Optional: limit to top N configs by DBCV (default: run all passing filters)
    --allow-no-overlap  Include configs where window_sec == hop_sec (dropped by default)
    --drop-noise     Drop noise points (-1) when training the classifier
    --eval-audio     Auto-initialize eval session with this audio file
    --eval-labels    Auto-initialize eval session with these Raven labels
    --eval-embeddings-dir  Directory of pre-computed eval embeddings (e.g. PERCH_SWEEP/sweep_embed/)
"""

import argparse
import csv
import os
import sys
import subprocess
from pathlib import Path
import numpy as np
import yaml

# Ensure scripts and src are accessible
sys.path.insert(0, str(Path(__file__).parent))
from session import load_params, cmd_run, mcs_dir, cmd_sweep_embed, cmd_aggregate_sweep_embed

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
    parser.add_argument("--min-clusters", type=int, default=15, help="Minimum number of clusters (default: 15)")
    parser.add_argument("--max-clusters", type=int, default=30, help="Maximum number of clusters (default: 30)")
    parser.add_argument("--max-noise", type=float, default=40.0, help="Maximum allowed noise percentage (default: 40.0)")
    parser.add_argument("--min-dbcv", type=float, default=0.2, help="Minimum DBCV score to include (default: 0.2)")
    parser.add_argument("--top-n-dbcv", type=int, default=None, help="Optional: limit to top N configs by DBCV (default: run all passing filters)")
    parser.add_argument("--allow-no-overlap", action="store_true", help="Include configs where window_sec == hop_sec (dropped by default)")
    parser.add_argument("--drop-noise", action="store_true", help="Drop noise points when training classifier")
    parser.add_argument("--eval-audio", type=Path, default=None, help="Auto-initialize eval session with this audio file")
    parser.add_argument("--eval-labels", type=Path, default=None, help="Auto-initialize eval session with these Raven labels")
    parser.add_argument("--eval-embeddings-dir", type=Path, default=None,
                        help="Directory of pre-computed eval embeddings (e.g. PERCH_SWEEP/sweep_embed/)")
    args = parser.parse_args()

    train_dir = args.train_session.resolve()
    eval_dir = args.eval_session.resolve()
    
    # Auto-initialize eval session if audio and labels are provided
    if args.eval_audio and args.eval_labels:
        eval_dir.mkdir(parents=True, exist_ok=True)
        eval_yml = eval_dir / "parameters.yml"
        if not eval_yml.exists():
            print(f"Auto-initializing eval session at {eval_dir}...")
            minimal_params = {
                "audio_files": [str(args.eval_audio.resolve())],
                "manual_labels": str(args.eval_labels.resolve())
            }
            with open(eval_yml, "w") as f:
                yaml.dump(minimal_params, f, sort_keys=False)

    # ---------------------------------------------------------------------------
    # Run embedding sweep -> aggregate -> read combined CSV
    # ---------------------------------------------------------------------------
    train_params_root = load_params(train_dir)
    if "embedding_sweep" not in train_params_root:
        print("Error: parameters.yml must have an 'embedding_sweep' block.")
        sys.exit(1)

    sweep_embed_dir = train_dir / "sweep_embed"
    combined_csv = sweep_embed_dir / "combined_results.csv"

    print(f"\nRunning sweep-embed pipeline...")
    cmd_sweep_embed(train_dir, train_params_root)
    
    print(f"\nAggregating sweep results...")
    cmd_aggregate_sweep_embed(train_dir)
    
    if not combined_csv.exists():
        print(f"Error: aggregate-sweep-embed failed to produce {combined_csv}")
        sys.exit(1)
    
    print(f"\nUsing sweep results: {combined_csv}")
    csv_file = combined_csv

    # ---------------------------------------------------------------------------
    # Read CSV and filter to top configurations
    # ---------------------------------------------------------------------------
    valid_runs = []
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_clusters = int(row["n_clusters"])
            noise_pct = float(row["noise_pct"])
            # Handle any NaN dbcv values
            dbcv = float(row["dbcv"]) if row["dbcv"] not in ("nan", "NaN") else -2.0
            
            if not args.allow_no_overlap and float(row["window_sec"]) == float(row["hop_sec"]):
                continue
            if args.min_clusters <= n_clusters <= args.max_clusters and noise_pct <= args.max_noise and dbcv >= args.min_dbcv:
                row["dbcv_val"] = dbcv  # cache for sorting
                valid_runs.append(row)
                
    if not valid_runs:
        print(f"No configurations matched the criteria: {args.min_clusters} <= n_clusters <= {args.max_clusters} AND noise <= {args.max_noise}% AND dbcv >= {args.min_dbcv}")
        sys.exit(1)
        
    valid_runs.sort(key=lambda r: r["dbcv_val"], reverse=True)
    
    if args.top_n_dbcv is not None:
        top_runs = valid_runs[:args.top_n_dbcv]
        print(f"\nTop {len(top_runs)} configurations by DBCV (out of {len(valid_runs)} passing filters):")
    else:
        top_runs = valid_runs
        print(f"\nAll {len(top_runs)} configurations passing filters:")
    
    for i, r in enumerate(top_runs):
        prefix = f"win={r['window_sec']}, hop={r['hop_sec']}, {r['embedder_type']}, {r['perch_padding']} | "
        print(f" {i+1}. {prefix}{r['umap_dims']}D, mcs={r['mcs']}, eps={r['eps']} | n={r['n_clusters']}, noise={r['noise_pct']}%, dbcv={r['dbcv']}")

    # ---------------------------------------------------------------------------
    # Load eval session manual labels
    # ---------------------------------------------------------------------------
    eval_params = load_params(eval_dir)
    if "manual_labels" not in eval_params:
        print("Error: eval_session must have 'manual_labels' defined in parameters.yml to evaluate generalization.")
        sys.exit(1)
    
    manual_labels_path = Path(eval_params["manual_labels"])
    if not manual_labels_path.is_absolute():
        manual_labels_path = eval_dir / manual_labels_path
        
    print(f"Loading eval session manual labels from {manual_labels_path}")
    manual = load_raven_labels(manual_labels_path)

    # ---------------------------------------------------------------------------
    # Evaluate each top configuration
    # ---------------------------------------------------------------------------
    master_results = []

    for idx, run in enumerate(top_runs):
        sub_session_name = f"win{run['window_sec']}_hop{run['hop_sec']}_{run['embedder_type']}_{run['perch_padding']}"
        session_dir = sweep_embed_dir / sub_session_name
        window_sec_val = float(run["window_sec"])
        hop_sec_val = float(run["hop_sec"])
        embedder_val = run["embedder_type"]
        padding_val = run["perch_padding"]
        
        print(f"\n{'='*70}")
        print(f"=== [{idx+1}/{len(top_runs)}] {session_dir.name}: {run['umap_dims']}D, mcs={run['mcs']}, eps={run['eps']}")
        print(f"{'='*70}")
        
        train_params = load_params(session_dir)
        train_params["umap_cluster_components"] = int(run["umap_dims"])
        train_params["hdbscan_epsilon"] = float(run["eps"])
        mcs_val = run["mcs"]
        
        # Generate cluster labels on train session
        d, _ = mcs_dir(session_dir, train_params, mcs_val)
        npz = d / "results.npz"
        if not npz.exists():
            print(f"Generating cluster labels for train session...")
            cmd_run(session_dir, train_params, mcs_val)
        else:
            print(f"Found existing clustering results: {npz}")
            
        # Train classifier
        model_pkl = session_dir / "models" / f"{npz.parent.name}_model.pkl"
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
                
        inherited_yml_path = eval_dir / f"parameters_inherited_from_{session_dir.name}.yml"
        with open(inherited_yml_path, "w") as f:
            yaml.dump(eval_inherited_params, f, sort_keys=False)
            
        pred_npz = eval_dir / "predictions" / f"{session_dir.name}_{model_pkl.stem}_predictions.npz"
        if not pred_npz.exists():
            print(f"Running inference on evaluation session (inheriting {session_dir.name} windowing)...")
            cmd = ["uv", "run", "python", "scripts/predict.py", str(inherited_yml_path), str(model_pkl), str(pred_npz)]
            if args.eval_embeddings_dir:
                cmd.extend(["--embeddings-cache-dir", str(args.eval_embeddings_dir)])
            run_subprocess(cmd)
        else:
            print(f"Found existing predictions: {pred_npz}")
            
        # Evaluate against ground truth
        print(f"Evaluating predictions against manual labels...")
        r_pred = np.load(pred_npz, allow_pickle=False)
        labels = r_pred["labels"]
        start_secs = r_pred["start_secs"]
        
        pred_window_sec = float(eval_inherited_params.get("window_sec", 2.0))
            
        end_secs = start_secs + pred_window_sec
        
        manual_window, _ = map_labels_to_windows(manual, start_secs, end_secs)
        metrics = compute_metrics(labels, manual_window)
        
        nmi = metrics.get("NMI", float("nan"))
        ari = metrics.get("ARI", float("nan"))
        homog = metrics.get("Homogeneity", float("nan"))
        detsim = metrics.get("DetSim", float("nan"))
        
        agg = np.nanmean([nmi, ari, homog])
        
        master_results.append({
            "train_session": session_dir.name,
            "window_sec": window_sec_val,
            "hop_sec": hop_sec_val,
            "embedder": embedder_val,
            "padding": padding_val,
            "config": f"{run['umap_dims']}D, mcs={run['mcs']}, eps={run['eps']}",
            "DetSim": detsim,
            "NMI": nmi,
            "ARI": ari,
            "Homogeneity": homog,
            "Aggregated": agg
        })

    # ---------------------------------------------------------------------------
    # Print and save report
    # ---------------------------------------------------------------------------
    print(f"\n\n{'='*115}")
    print(f"=== MASTER GENERALIZATION REPORT: {train_dir.name} -> {eval_dir.name} ===")
    print(f"{'='*115}")
    print(f"{'Session':<30} | {'Win/Hop/Pad':<20} | {'Config':<20} | {'DetSim':<6} | {'NMI':<6} | {'ARI':<6} | {'Homog':<6} | {'Aggreg':<6}")
    print("-" * 115)
    
    master_results.sort(key=lambda r: r["Aggregated"] if not np.isnan(r["Aggregated"]) else -2.0, reverse=True)
    
    for res in master_results:
        win_str = f"{res['window_sec']}/{res['hop_sec']} {res['padding'][:3]}"
        print(f"{res['train_session']:<30} | {win_str:<20} | {res['config']:<20} | {res['DetSim']:<6.4f} | {res['NMI']:<6.4f} | {res['ARI']:<6.4f} | {res['Homogeneity']:<6.4f} | {res['Aggregated']:<6.4f}")

    report_csv = eval_dir / f"master_generalization_report_{train_dir.name}.csv"
    with open(report_csv, "w", newline="") as f:
        fieldnames = ["train_session", "window_sec", "hop_sec", "embedder", "padding", "config", "DetSim", "NMI", "ARI", "Homogeneity", "Aggregated"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(master_results)
        
    print(f"\nMaster report saved to: {report_csv}")


if __name__ == "__main__":
    main()
