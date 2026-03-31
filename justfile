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

# run_scored /mnt/PAM_Analysis/GoogleHumpbackModel/Scores 0.7 wav ...
run_scored score_dir score_threshold *wavs:
    uv run hbws-cluster \
      --score-dir {{ score_dir }} \
      --score-threshold {{ score_threshold }} \
      --window-sec 2.0 \
      --hop-sec 1.0 \
      --min-cluster-size 500 \
      --embeddings-cache output/embeddings.npy \
      --output output/results_from_scored.npz \
      {{ wavs }}

inspect-npz npz="output/results.npz":
    uv run python scripts/inspect_npz.py {{ npz }}

plot-umap npz="output/results.npz" out="output/results.png":
    uv run python scripts/plot_umap.py {{ npz }} {{ out }}
