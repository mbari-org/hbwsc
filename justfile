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

# Plot proportional cluster density over time (saves <stem>_density.png)
plot-density npz window_sec="5":
    uv run python scripts/plot_density.py {{ npz }} --window-sec {{ window_sec }}

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

# Create a new session directory named <base>_w<window>_h<hop>/ with a template parameters.yml
new-session base window_sec hop_sec:
    uv run python scripts/session.py {{ base }}_w{{ window_sec }}_h{{ hop_sec }} init {{ window_sec }} {{ hop_sec }}

# Run the clustering pipeline for a session; mcs overrides parameters.yml
run-session dir mcs="":
    uv run python scripts/session.py {{ dir }} run {{ mcs }}

# Run the hyperparameter sweep for a session (embeddings must exist)
sweep-session dir:
    uv run python scripts/session.py {{ dir }} sweep

# Run the embedding hyperparameter sweep for a session
sweep-embed-session dir:
    uv run python scripts/session.py {{ dir }} sweep-embed

# Generate all analysis outputs for a session run (timeline, umap, raven, clusters)
analyze-session dir mcs="":
    uv run python scripts/session.py {{ dir }} analyze {{ mcs }}

# Print cluster summary for a session run
inspect-session dir mcs="":
    uv run python scripts/session.py {{ dir }} inspect {{ mcs }}

run-inference src_dir targ_dir train_out="" pred_out="":
    uv run python scripts/train_classifier.py {{ src_dir }}/results.npz {{ train_out }}
    uv run python scripts/predict.py {{ targ_dir }}/parameters.yml {{ src_dir }}/models/results_model.pkl {{ pred_out }}


###########################

browse_timeline npz segment_minutes='2' density_window='5' file_index='0':
    uv run python scripts/browse_timeline.py \
        {{ npz }} \
        --manual-labels ryjo_labels/MARS_20161221_000046_SongSession_16kHz_HPF5HzNorm_labels.txt \
        --segment-minutes {{ segment_minutes }} --density-window-sec {{ density_window}} --file-index {{ file_index }}

browse_timeline_audio npz segment_minutes='2' density_window='5' file_index='0' audio='':
    uv run python scripts/browse_timeline.py \
        {{ npz }} \
        --manual-labels ryjo_labels/MARS_20161221_000046_SongSession_16kHz_HPF5HzNorm_labels.txt \
        --segment-minutes {{ segment_minutes }} --density-window-sec {{ density_window}} --file-index {{ file_index }} --audio {{ audio }}

export_timeline npz segment_minutes='2' density_window='5' file_index='0' file_name='' :
    uv run python scripts/browse_timeline.py \
        {{ npz }} \
        --manual-labels ryjo_labels/MARS_20161221_000046_SongSession_16kHz_HPF5HzNorm_labels.txt \
        --segment-minutes {{ segment_minutes }} --density-window-sec {{ density_window}} --file-index {{ file_index }} --export-pdf {{ file_name }} \

export_timeline_audio npz segment_minutes='2' density_window='5' file_index='0' audio= '' file_name='' :
    uv run python scripts/browse_timeline.py \
        {{ npz }} \
        --manual-labels ryjo_labels/MARS_20161221_000046_SongSession_16kHz_HPF5HzNorm_labels.txt \
        --segment-minutes {{ segment_minutes }} --density-window-sec {{ density_window}} --file-index {{ file_index }} --export-pdf {{ file_name }} \
        --audio {{ audio }}

export_timeline_nolabel npz segment_minutes='2' density_window='5' file_index='0' file_name='' :
    uv run python scripts/browse_timeline.py \
        {{ npz }} \
        --segment-minutes {{ segment_minutes }} --density-window-sec {{ density_window}} --file-index {{ file_index }} --export-pdf {{ file_name }}

# --audio MARS_20161221_000046_SongSession_16kHz_HPF5Hz.wav \


browse_timeline_score0_5 segment_minutes='2':
    just browse_timeline \
      experiments/song_20161221_score0.5_w0.5_h0.25/umap15_mcs50/results.npz \
      {{ segment_minutes }}

browse_timeline_full segment_minutes='2':
    just browse_timeline \
      experiments/song_20161221_full_w0.5_h0.25/umap20_mcs200/results.npz \
      {{ segment_minutes }}

###########################
## GPU

## Ad hoc recipe for some testing on sonus
gpu_embeddings batch_size:
  uv run hbws-cluster \
    --score-threshold 0.5 \
    --window-sec 0.5 \
    --hop-sec 0.25 \
    --sample-rate 16000 \
    --umap-components 2 \
    --umap-cluster-components 15 \
    --umap-neighbors 15 \
    --min-cluster-size 50 \
    --batch-size {{ batch_size }} \
    --embeddings-cache "/tmp/emb_bs{{ batch_size }}.npy" \
    --output /tmp/results_bs{{ batch_size }}.npz \
    --score-file /opt/humpback/hbwsc/MARS_20161221_000046_SongSession_10kHz_HPF5Hz_scores.npy \
    /opt/humpback/hbwsc/MARS_20161221_000046_SongSession_16kHz_HPF5Hz.wav


###########################
## Run Cluster-Eval Pipeline
cluster_eval train_dir eval_dir audio="../../MARS_20161221_000046_32kHz.wav" labels="../../ryjo_labels/MARS_20161221_000046_SongSession_16kHz_HPF5HzNorm_labels.txt":
    uv run python scripts/session.py {{ train_dir }} evaluate-generalization {{ eval_dir }} --drop-noise --eval-audio {{ audio }} --eval-labels {{ labels }}


# Install uv (if not already installed)
install-uv:
    curl -LsSf https://astral.sh/uv/install.sh | sh

sync:
    uv sync

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
