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
from datetime import datetime, timezone
from pathlib import Path

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

# UMAP
umap_components: 2           # for visualization (2-D scatter plot)
umap_cluster_components: 10  # for HDBSCAN clustering
umap_neighbors: 15

# HDBSCAN
min_cluster_size: 100        # default; can be overridden on 'run' command line

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
    for key in ("score_file", "score_dir"):
        if key in params:
            params[key] = str((session_dir / params[key]).resolve())
    params["audio_files"] = [str((session_dir / p).resolve()) for p in params["audio_files"]]
    return params


def mcs_dir(session_dir: Path, params: dict, mcs_arg: str) -> Path:
    mcs = int(mcs_arg) if mcs_arg else int(params.get("min_cluster_size", 100))
    d = session_dir / f"mcs{mcs}"
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
        "--embeddings-cache",
        str(embeddings_cache),
        "--output",
        str(npz),
    ]

    if "score_file" in params:
        cmd += ["--score-file", str(params["score_file"])]
    elif "score_dir" in params:
        cmd += ["--score-dir", str(params["score_dir"])]

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

    run_cmd(
        [
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
            "--workers",
            str(n_workers),
            "--out",
            str(sweep_out / f"results_workers{n_workers}.csv"),
        ]
    )


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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

COMMANDS = {"init", "run", "sweep", "analyze", "inspect"}

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
