_list:
    @just --list --unsorted

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

export-cluster npz cluster n="10":
    uv run python scripts/export_cluster.py \
       {{ npz }} {{ cluster }} \
       output/cluster_{{ cluster }} {{ n }}

inspect-npz npz="output/results.npz":
    uv run python scripts/inspect_npz.py {{ npz }}

plot-umap npz="output/results.npz" out="output/results.png":
    uv run python scripts/plot_umap.py {{ npz }} {{ out }}

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
