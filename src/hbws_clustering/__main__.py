"""CLI entry point: ``python -m hbws_clustering`` or ``hbws-cluster``."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import typer

from hbws_clustering.embedding import AVES_BASE_BIO
from hbws_clustering.pipeline import ClusteringPipeline
from hbws_clustering.windowing import AudioWindower
from hbws_clustering.embedding import AvesEmbedder
from hbws_clustering.reduction import UmapReducer
from hbws_clustering.clustering import HdbscanClusterer

app = typer.Typer(help="Humpback whale vocalization clustering via AVES + UMAP + HDBSCAN.")


@app.command()
def run(
    audio_files: list[Path] = typer.Argument(..., help="Input WAV/FLAC audio files."),
    window_sec: float = typer.Option(2.0, help="Window duration in seconds."),
    hop_sec: Optional[float] = typer.Option(None, help="Window hop in seconds (default = window_sec)."),
    sample_rate: int = typer.Option(16_000, help="Target sample rate for resampling."),
    model: str = typer.Option(AVES_BASE_BIO, help="URL to a TorchAudio AVES checkpoint (.pt)."),
    pooling: str = typer.Option("mean", help="Embedding pooling: 'mean' or 'max'."),
    umap_components: int = typer.Option(2, help="UMAP output dimensions."),
    umap_neighbors: int = typer.Option(15, help="UMAP n_neighbors."),
    min_cluster_size: int = typer.Option(5, help="HDBSCAN min_cluster_size."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save results as .npz file."),
) -> None:
    """Cluster humpback whale vocalizations in AUDIO_FILES."""
    pipe = ClusteringPipeline(
        windower=AudioWindower(
            window_sec=window_sec,
            hop_sec=hop_sec,
            target_sr=sample_rate,
        ),
        embedder=AvesEmbedder(model_url=model, pooling=pooling),
        reducer=UmapReducer(n_components=umap_components, n_neighbors=umap_neighbors),
        clusterer=HdbscanClusterer(min_cluster_size=min_cluster_size),
    )

    result = pipe.run(audio_files)

    typer.echo("\nCluster summary:")
    for label, count in sorted(result.cluster_summary().items()):
        tag = "noise" if label == -1 else f"cluster {label}"
        typer.echo(f"  {tag}: {count} windows")

    if output is not None:
        np.savez(
            output,
            labels=result.labels,
            probabilities=result.probabilities,
            reduced=result.reduced,
            embeddings=result.embeddings,
        )
        typer.echo(f"\nResults saved to {output}")


if __name__ == "__main__":
    app()
