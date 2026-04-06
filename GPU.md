## GPU support

The embedder auto-detects CUDA (`"cuda" if torch.cuda.is_available() else "cpu"`), so no code
changes are needed on a GPU machine. The only requirement is installing the CUDA-enabled PyTorch
build instead of the default CPU-only one.

Add the following to `pyproject.toml` (replace `cu124` with the CUDA version reported by
`nvidia-smi` on the target machine — common values: `cu118`, `cu121`, `cu124`):

```toml
[[tool.uv.index]]
name = "pytorch-cu124"
url = "https://download.pytorch.org/whl/cu124"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu124" }
torchaudio = { index = "pytorch-cu124" }
```

Then run `uv sync` to reinstall with the CUDA build. After that, increase the batch size for
better GPU utilisation:

```bash
uv run hbws-cluster --batch-size 64 --embeddings-cache output/embeddings.npy ...
```

> **Note:** Because the index config is hardware-specific, it is not committed to the repo.
> CPU builds work on all machines; apply the override locally on GPU hosts.
