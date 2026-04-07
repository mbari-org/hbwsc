_list:
    @just --list --unsorted

# Running on the "well-known" 4.5hr recording (from 20161221)
run45:
    uv run hbws-cluster \
      --score-file MARS_20161221_000046_SongSession_10kHz_HPF5Hz_scores.npy \
      --score-threshold 0.7 \
      --window-sec 0.5 \
      --hop-sec 0.25 \
      --min-cluster-size 100 \
      --umap-cluster-components 10 \
      --embeddings-cache output/embeddings_song_w0.5_h0.25.npy \
      --output output/results_song_w0.5_h0.25_mcs100.npz \
      MARS_20161221_000046_SongSession_16kHz_HPF5Hz.wav

run *args:
    uv run hbws-cluster {{ args }}

run_basic *wavs:
    uv run hbws-cluster \
      --window-sec 2.0 \
      --hop-sec 1.0 \
      --min-cluster-size 10 \
      --output output/results.npz \
      {{ wavs }}

# run_scored /mnt/PAM_Analysis/GoogleHumpbackModel/Scores 0.7 500 wav ...
run_scored score_dir score_threshold mcs *wavs:
    uv run hbws-cluster \
      --score-dir {{ score_dir }} \
      --score-threshold {{ score_threshold }} \
      --window-sec 2.0 \
      --hop-sec 1.0 \
      --embeddings-cache output/embeddings.npy \
      --min-cluster-size {{ mcs }} \
      --output output/results_mcs{{ mcs }}.npz \
      {{ wavs }}

export-cluster npz cluster n window_sec out_dir="":
    uv run python scripts/export_cluster.py \
       {{ npz }} {{ cluster }} {{ n }} {{ window_sec }} \
       {{ if out_dir != "" { out_dir } else { "output/cluster_" + cluster } }}

# Export N samples from every cluster into output/clusters_<stem>/cluster_*/
export-all-clusters npz window_sec n="10":
    uv run python scripts/export_all_clusters.py {{ npz }} {{ window_sec }} {{ n }}

# Plot cluster labels over time (saves <stem>_timeline.png)
plot-timeline npz:
    uv run python scripts/plot_timeline.py {{ npz }}

# Export Raven Pro selection table (saves <stem>_raven.txt)
export-raven npz window_sec:
    uv run python scripts/export_raven_table.py {{ npz }} {{ window_sec }}

# Merge consecutive same-cluster selections in a Raven table (saves <stem>_aggregate.txt)
aggregate-raven raven_txt:
    uv run python scripts/aggregate_raven.py {{ raven_txt }}

inspect-npz npz="output/results.npz":
    uv run python scripts/inspect_npz.py {{ npz }}

plot-umap npz="output/results.npz" out="output/results.png":
    uv run python scripts/plot_umap.py {{ npz }} {{ out }}

# Session-based workflow (reads parameters.yml from the session directory)

# Create a new session directory with a template parameters.yml
new-session dir:
    uv run python scripts/session.py {{ dir }} init

# Run the clustering pipeline for a session; mcs overrides parameters.yml
run-session dir mcs="":
    uv run python scripts/session.py {{ dir }} run {{ mcs }}

# Run the hyperparameter sweep for a session (embeddings must exist)
sweep-session dir:
    uv run python scripts/session.py {{ dir }} sweep

# Generate all analysis outputs for a session run (timeline, umap, raven, clusters)
analyze-session dir mcs="":
    uv run python scripts/session.py {{ dir }} analyze {{ mcs }}

# Print cluster summary for a session run
inspect-session dir mcs="":
    uv run python scripts/session.py {{ dir }} inspect {{ mcs }}

dev: test format lint

# Run tests
test *options="":
    uv run pytest {{options}}

# Format source code
format:
    uv run ruff format

# Check source formatting
format-check:
    uv run ruff format --check

# Lint source code
lint:
    uv run ruff check --fix

# Lint check
lint-check:
    uv run ruff check
