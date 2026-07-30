"""Session manager for hbws_clustering experiments.

A session is a directory containing a parameters.yml file that captures
all inputs and parameters for a set of clustering runs on the same audio.

Directory layout
----------------
<session_dir>/
  parameters.yml       all inputs and parameters
  embeddings.npy       AVES embeddings (shared across runs, created by 'run')
  sweep/
    results.csv        hyperparameter sweep results
  mcs<N>/              one subdir per min_cluster_size value
    results.npz
    raven.txt
    raven_aggregate.txt
    timeline.png
    umap.png
    clusters/          audio snippets per cluster

Commands
--------
    init     Create a new session directory with a template parameters.yml.
    run      Run the clustering pipeline. Results go to mcs<N>/.
    sweep    Run the hyperparameter sweep. Results go to sweep/.
    analyze  Generate all analysis outputs for a run (timeline, umap, raven, clusters).
    inspect  Print the cluster summary for a run.

Usage
-----
    uv run python scripts/session.py <session_dir> init
    uv run python scripts/session.py <session_dir> run [mcs]
    uv run python scripts/session.py <session_dir> sweep
    uv run python scripts/session.py <session_dir> analyze [mcs]
    uv run python scripts/session.py <session_dir> inspect [mcs]
"""

import json
import subprocess
import sys
import copy
from datetime import datetime, timezone
from pathlib import Path
from itertools import product

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from hbws_clustering.evaluation import load_raven_labels, map_labels_to_windows, compute_metrics

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML is required. Run: uv add pyyaml")
    sys.exit(1)

# ---------------------------------------------------------------------------

TEMPLATE = """\
# hbws_clustering session parameters
# ---------------------------------------------------------------------------

# Audio input — one or more WAV files
audio_files:
  - /path/to/recording.wav

# Score-guided windowing — choose one or neither
# score_file: /path/to/scores.npy       # explicit score file (single WAV only)
# score_dir:  /path/to/scores/base/dir  # auto-locate by date from WAV filename

score_threshold: 0.7

# Windowing
window_sec: 0.5
hop_sec: 0.25
sample_rate: 16000

# AVES embedding
batch_size: 16               # increase on GPU (e.g. 64, 128)
perch_padding: repeat        # 'repeat' (tile audio) or 'zero' (zero-fill) for Perch

# UMAP
umap_components: 2           # for visualization (2-D scatter plot)
umap_cluster_components: 10  # for HDBSCAN clustering
umap_neighbors: 15

# HDBSCAN
min_cluster_size: 100        # default; can be overridden on 'run' command line
hdbscan_alpha: 1.0           # scaling for distance matrix
hdbscan_epsilon: 0.0         # threshold below which clusters are merged

# Hyperparameter sweep
sweep_dims: "2,5,10,15,20,30"
sweep_mcs: "50,100,200"
sweep_workers: 2

# Timeline plot segmentation (minutes per segment; omit or set to 0 for a single full plot)
# timeline_segment_minutes: 10

# Audio export (samples per cluster)
n_cluster_samples: 10

# Manual labels for comparison (Raven format; optional)
# manual_labels: /path/to/labels.txt
"""

# ---------------------------------------------------------------------------


def load_params(session_dir: Path) -> dict:
    yml = session_dir / "parameters.yml"
    if not yml.exists():
        print(f"ERROR: {yml} not found. Run 'init' first.")
        sys.exit(1)
    with open(yml) as f:
        params = yaml.safe_load(f)
    # Resolve relative paths against the session directory so that parameters.yml
    # is self-contained regardless of the working directory when just is invoked.
    for key in ("score_file", "score_dir", "manual_labels"):
        if key in params:
            params[key] = str((session_dir / params[key]).resolve())
    params["audio_files"] = [str((session_dir / p).resolve()) for p in params["audio_files"]]
    return params


def mcs_dir(session_dir: Path, params: dict, mcs_arg: str) -> tuple[Path, int]:
    mcs = int(mcs_arg) if mcs_arg else int(params.get("min_cluster_size", 100))
    umap_dims = int(params.get("umap_cluster_components", 10))
    
    folder_name = f"umap{umap_dims}_mcs{mcs}"
    
    eps = params.get("hdbscan_epsilon", 0.0)
    if eps != 0.0:
        folder_name += f"_eps{eps}"
        
    alpha = params.get("hdbscan_alpha", 1.0)
    if alpha != 1.0:
        folder_name += f"_alpha{alpha}"
        
    d = session_dir / folder_name
    d.mkdir(parents=True, exist_ok=True)
    return d, mcs


def run_cmd(cmd: list, **kwargs):
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        sys.exit(result.returncode)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_init(session_dir: Path, window_sec: str, hop_sec: str):
    session_dir.mkdir(parents=True, exist_ok=True)
    yml = session_dir / "parameters.yml"
    if yml.exists():
        print(f"Already exists: {yml}")
    else:
        content = TEMPLATE.replace("window_sec: 0.5", f"window_sec: {window_sec}")
        content = content.replace("hop_sec: 0.25", f"hop_sec: {hop_sec}")
        yml.write_text(content)
        print(f"Created: {yml}")
        print("Edit parameters.yml before running other commands.")


def cmd_run(session_dir: Path, params: dict, mcs_arg: str):
    d, mcs = mcs_dir(session_dir, params, mcs_arg)
    embeddings_cache = session_dir / "embeddings.npy"
    npz = d / "results.npz"

    audio_files = [str(p) for p in params["audio_files"]]

    cmd = [
        "uv",
        "run",
        "hbws-cluster",
        "--score-threshold",
        str(params.get("score_threshold", 0.7)),
        "--window-sec",
        str(params["window_sec"]),
        "--hop-sec",
        str(params["hop_sec"]),
        "--sample-rate",
        str(params.get("sample_rate", 16000)),
        "--umap-components",
        str(params.get("umap_components", 2)),
        "--umap-cluster-components",
        str(params.get("umap_cluster_components", 10)),
        "--umap-neighbors",
        str(params.get("umap_neighbors", 15)),
        "--min-cluster-size",
        str(mcs),
        "--alpha",
        str(params.get("hdbscan_alpha", 1.0)),
        "--epsilon",
        str(params.get("hdbscan_epsilon", 0.0)),
        "--batch-size",
        str(params.get("batch_size", 16)),
        "--embeddings-cache",
        str(embeddings_cache),
        "--output",
        str(npz),
    ]

    if "score_file" in params:
        cmd += ["--score-file", str(params["score_file"])]
    elif "score_dir" in params:
        cmd += ["--score-dir", str(params["score_dir"])]

    if "embedder_type" in params:
        cmd += ["--embedder-type", str(params["embedder_type"])]
    if "perch_padding" in params:
        cmd += ["--perch-padding", str(params["perch_padding"])]
    if "model" in params:
        cmd += ["--model", str(params["model"])]

    cmd += audio_files
    run_cmd(cmd)

    if embeddings_cache.exists() and npz.exists():
        import numpy as np

        r = np.load(npz, allow_pickle=False)

        # Embeddings sidecar (session root, shared across runs)
        meta = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "n_windows": int(r["embeddings"].shape[0]),
            "embedding_dim": int(r["embeddings"].shape[1]),
            "source_files": sorted(set(r["source_files"].astype(str).tolist())),
            "window_sec": params["window_sec"],
            "hop_sec": params["hop_sec"],
            "score_threshold": params.get("score_threshold"),
            "score_file": params.get("score_file"),
            "score_dir": params.get("score_dir"),
        }
        meta_path = session_dir / "embeddings_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"\nEmbedding summary saved to {meta_path}")

        # Cluster summary (mcs subdir)
        labels = r["labels"]
        unique, counts = np.unique(labels, return_counts=True)
        summary = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "min_cluster_size": mcs,
            "n_clusters": int((unique >= 0).sum()),
            "n_noise": int(counts[unique == -1][0]) if -1 in unique else 0,
            "clusters": {
                ("noise" if int(lbl) == -1 else f"cluster {int(lbl)}"): int(cnt) for lbl, cnt in zip(unique, counts)
            },
        }

        # If manual labels exist, compute metrics directly for the summary
        if "manual_labels" in params:
            manual_labels = Path(params["manual_labels"])
            if not manual_labels.is_absolute():
                manual_labels = session_dir / manual_labels
            if manual_labels.exists():
                try:
                    manual = load_raven_labels(manual_labels)
                    start_secs = r["start_secs"]
                    
                    window_sec = float(params["window_sec"])
                    end_secs = start_secs + window_sec
                    
                    manual_window, _ = map_labels_to_windows(manual, start_secs, end_secs)
                    metrics = compute_metrics(labels, manual_window)
                    
                    if not np.isnan(metrics.get("DetSim", float("nan"))):
                        summary["evaluation"] = metrics
                        print("Evaluated against manual labels and added metrics to summary.")
                except Exception as e:
                    print(f"Warning: Could not compute metrics for summary: {e}")
        summary_path = d / "cluster_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Cluster summary saved to {summary_path}")


def cmd_sweep(session_dir: Path, params: dict):
    embeddings_cache = session_dir / "embeddings.npy"
    if not embeddings_cache.exists():
        print(f"ERROR: {embeddings_cache} not found. Run 'run' first to generate embeddings.")
        sys.exit(1)

    sweep_out = session_dir / "sweep"
    sweep_out.mkdir(exist_ok=True)
    n_workers = params.get("sweep_workers", 2)
    
    csv_name = f"results_workers{n_workers}"
    eps = params.get("hdbscan_epsilon", 0.0)
    alpha = params.get("hdbscan_alpha", 1.0)
    if eps != 0.0:
        csv_name += f"_eps{eps}"
    if alpha != 1.0:
        csv_name += f"_alpha{alpha}"
    csv_name += ".csv"

    d, _ = mcs_dir(session_dir, params, "")
    npz = d / "results.npz"

    cmd = [
        "uv",
        "run",
        "python",
        "scripts/sweep.py",
        str(embeddings_cache),
        "--dims",
        str(params.get("sweep_dims", "2,5,10,15,20,30")),
        "--mcs",
        str(params.get("sweep_mcs", "50,100,200")),
        "--neighbors",
        str(params.get("umap_neighbors", 15)),
        "--epsilons",
        str(params.get("sweep_epsilons", "0.0")),
        "--epsilon",
        str(eps),
        "--alpha",
        str(alpha),
        "--workers",
        str(n_workers),
        "--out",
        str(sweep_out / csv_name),
    ]

    if "manual_labels" in params:
        manual_labels = Path(params["manual_labels"])
        if not manual_labels.is_absolute():
            manual_labels = session_dir / manual_labels
        if manual_labels.exists() and npz.exists():
            cmd.extend(["--manual-labels", str(manual_labels), "--npz", str(npz), "--window-sec", str(params["window_sec"])])

    run_cmd(cmd)


def cmd_sweep_embed(session_dir: Path, params: dict):
    # toggalable to check if we want to skip existing UMAP/HDBSCANs
    SKIP_EXISTING = True

    sweep_config = params.get("embedding_sweep", {})
    if not sweep_config:
        print("Could nto find 'embedding_sweep' block in parameters.yml")
        return
    
    # sweep session directory
    sweep_out_dir = session_dir / "sweep_embed"
    sweep_out_dir.mkdir(exist_ok=True)

    for window_sec, hop_sec, embedder_type, perch_padding in product(
        sweep_config.get("window_sec"),
        sweep_config.get("hop_sec"),
        sweep_config.get("embedder_type"),
        sweep_config.get("perch_padding", ["repeat"]),
    ):
        # skip if the hop and window combination makes the sweep skip audio
        if hop_sec > window_sec:
            continue
        # make sub directory for each sweep session
        sweep_sesh_name = f"win{window_sec}_hop{hop_sec}_{embedder_type}_{perch_padding}"
        sweep_sesh_dir = sweep_out_dir / sweep_sesh_name
        sweep_sesh_dir.mkdir(exist_ok=True)

        # for each subdirectory, create a new, non-embed_sweep paramater file
        child_params = copy.deepcopy(params)
        if "embedding_sweep" in child_params:
            del child_params["embedding_sweep"]

        child_params["window_sec"] = window_sec
        child_params["hop_sec"] = hop_sec
        child_params["embedder_type"] = embedder_type
        child_params["perch_padding"] = perch_padding
        
        with open(session_dir / "parameters.yml") as f:
            raw_params = yaml.safe_load(f)
            
        for key in ("score_file", "score_dir", "manual_labels"):
            if key in child_params and child_params[key]:
                # Prepend ../../ to account for the nested sweep_embed/<config>/ directory
                child_params[key] = "../../" + raw_params[key]
        if "audio_files" in child_params:
            child_params["audio_files"] = ["../../" + p for p in raw_params["audio_files"]]

        child_yml = sweep_sesh_dir / "parameters.yml"
        with open(child_yml, "w") as f:
            yaml.dump(child_params, f, sort_keys=False)

        if SKIP_EXISTING:
            n_workers = child_params.get("sweep_workers", 2)
            csv_name = f"results_workers{n_workers}"
            eps = child_params.get("hdbscan_epsilon", 0.0)
            alpha = child_params.get("hdbscan_alpha", 1.0)
            if eps != 0.0:
                csv_name += f"_eps{eps}"
            if alpha != 1.0:
                csv_name += f"_alpha{alpha}"
            csv_name += ".csv"

            if (sweep_sesh_dir / "sweep" / csv_name).exists():
                print(f"Skipping {sweep_sesh_name}: results already exist.")
                continue
            
        print(f"Created embedding sweep: {sweep_sesh_name}, running pipeline:")

        cmd_run(sweep_sesh_dir, load_params(sweep_sesh_dir), "")
        
        print(f"Clustering session...")

        cmd_sweep(sweep_sesh_dir, child_params)


def cmd_analyze(session_dir: Path, params: dict, mcs_arg: str):
    d, mcs = mcs_dir(session_dir, params, mcs_arg)
    npz = d / "results.npz"
    if not npz.exists():
        print(f"ERROR: {npz} not found. Run 'run' first.")
        sys.exit(1)

    window_sec = str(params["window_sec"])
    n_samples = str(params.get("n_cluster_samples", 10))

    run_cmd(["uv", "run", "python", "scripts/plot_umap.py", str(npz), str(d / "umap.png")])

    timeline_dir = d / "timeline"
    timeline_dir.mkdir(exist_ok=True)
    timeline_cmd = ["uv", "run", "python", "scripts/plot_timeline.py", str(npz), str(timeline_dir / "timeline.png")]
    seg_minutes = params.get("timeline_segment_minutes", 0)
    if seg_minutes:
        timeline_cmd += ["--segment-minutes", str(seg_minutes)]
    run_cmd(timeline_cmd)
    
    density_cmd = ["uv", "run", "python", "scripts/plot_density.py", str(npz), str(timeline_dir / "density.png")]
    if seg_minutes:
        density_cmd += ["--segment-minutes", str(seg_minutes)]
    run_cmd(density_cmd)

    run_cmd(["uv", "run", "python", "scripts/export_raven_table.py", str(npz), window_sec, str(d / "raven.txt")])

    run_cmd(["uv", "run", "python", "scripts/aggregate_raven.py", str(d / "raven.txt")])

    run_cmd(
        ["uv", "run", "python", "scripts/export_all_clusters.py", str(npz), window_sec, n_samples, str(d / "clusters")]
    )

    if "manual_labels" in params:
        manual_labels = Path(params["manual_labels"])
        if not manual_labels.is_absolute():
            manual_labels = session_dir / manual_labels
        if manual_labels.exists():
            run_cmd([
                "uv", "run", "python", "scripts/compare_labels.py",
                str(npz), str(manual_labels),
                "--out", str(d / "label_comparison.txt"),
            ])
        else:
            print(f"WARNING: manual_labels not found: {manual_labels}")


def cmd_inspect(session_dir: Path, params: dict, mcs_arg: str):
    d, _ = mcs_dir(session_dir, params, mcs_arg)
    npz = d / "results.npz"
    if not npz.exists():
        print(f"ERROR: {npz} not found. Run 'run' first.")
        sys.exit(1)
    run_cmd(["uv", "run", "python", "scripts/inspect_npz.py", str(npz)])


def cmd_aggregate_sweep_embed(session_dir: Path):
    sweep_out_dir = session_dir / "sweep_embed"
    out_csv = sweep_out_dir / "combined_results.csv"
    run_cmd(["uv", "run", "python", "scripts/aggregate_embed_sweep.py", str(session_dir), str(out_csv)])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

COMMANDS = {"init", "run", "sweep", "analyze", "inspect", "sweep-embed", "aggregate-sweep-embed", "evaluate-generalization"}

if len(sys.argv) < 3 or sys.argv[2] not in COMMANDS:
    print(__doc__)
    print(f"Commands: {', '.join(sorted(COMMANDS))}")
    sys.exit(1)

session_dir = Path(sys.argv[1])
command = sys.argv[2]
extra = sys.argv[3] if len(sys.argv) > 3 else ""

if command == "init":
    window_sec = sys.argv[3] if len(sys.argv) > 3 else "0.5"
    hop_sec = sys.argv[4] if len(sys.argv) > 4 else "0.25"
    cmd_init(session_dir, window_sec, hop_sec)
else:
    params = load_params(session_dir)
    if command == "run":
        cmd_run(session_dir, params, extra)
    elif command == "sweep":
        cmd_sweep(session_dir, params)
    elif command == "analyze":
        cmd_analyze(session_dir, params, extra)
    elif command == "inspect":
        cmd_inspect(session_dir, params, extra)
    elif command == "sweep-embed":
        cmd_sweep_embed(session_dir, params)
    elif command == "aggregate-sweep-embed":
        cmd_aggregate_sweep_embed(session_dir)
    elif command == "evaluate-generalization":
        if not extra:
            print("ERROR: evaluate-generalization requires an eval_session directory.")
            sys.exit(1)
        # Pass through to the dedicated script
        cmd = ["uv", "run", "python", "scripts/evaluate_generalization.py", str(session_dir), extra]
        # Check if drop-noise was passed
        if "--drop-noise" in sys.argv:
            cmd.append("--drop-noise")
        run_cmd(cmd)
