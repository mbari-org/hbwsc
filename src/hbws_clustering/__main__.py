"""CLI entry point: ``python -m hbws_clustering`` or ``hbws-cluster``."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import typer

from hbws_clustering.embedding import AVES_BASE_BIO, AvesEmbedder
from hbws_clustering.clustering import HdbscanClusterer
from hbws_clustering.pipeline import ClusteringPipeline
from hbws_clustering.reduction import UmapReducer
from hbws_clustering.windowing import AudioWindower, ScoreGuidedWindower

app = typer.Typer(help="Humpback whale vocalization clustering via AVES + UMAP + HDBSCAN.")

# Matches MARS-YYYYMMDDTHHMMSSZ-*kHz.wav, capturing the date token YYYYMMDD.
_WAV_DATE_RE = re.compile(r"MARS-(\d{4})(\d{2})(\d{2})T")


def _score_path(audio_path: Path, score_dir: Path) -> Path:
    """Derive the score .npy path from a WAV filename and score base directory.

    Expected WAV name pattern: ``MARS-YYYYMMDDTHHMMSSZ-<rate>kHz.wav``
    Resulting score path:      ``{score_dir}/YYYY/MM/Scores-YYYYMMDD.npy``
    """
    m = _WAV_DATE_RE.search(audio_path.name)
    if not m:
        raise ValueError(
            f"Cannot parse date from filename {audio_path.name!r}. "
            "Expected pattern: MARS-YYYYMMDDTHHMMSSZ-<rate>kHz.wav"
        )
    yyyy, mm, dd = m.group(1), m.group(2), m.group(3)
    return score_dir / yyyy / mm / f"Scores-{yyyy}{mm}{dd}.npy"


@app.command()
def run(
    audio_files: list[Path] = typer.Argument(..., help="Input WAV files (one or more day files)."),
    score_dir: Optional[Path] = typer.Option(
        None,
        "--score-dir",
        help=(
            "Base directory for HWSD score files "
            "(e.g. /mnt/PAM_Analysis/GoogleHumpbackModel/Scores). "
            "When provided, only high-confidence windows are extracted."
        ),
    ),
    score_threshold: float = typer.Option(
        0.7,
        "--score-threshold",
        help="Minimum HWSD score to include a second in analysis. Default: 0.7.",
    ),
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
    """Cluster humpback whale vocalizations in AUDIO_FILES.

    Without --score-dir: windows every second of every file.\n
    With --score-dir: reads HWSD score files matched by date from the WAV
    filename and only extracts windows where score >= --score-threshold.
    """
    inner_windower = AudioWindower(window_sec=window_sec, hop_sec=hop_sec, target_sr=sample_rate)

    if score_dir is not None:
        windower = ScoreGuidedWindower(windower=inner_windower, threshold=score_threshold)
    else:
        windower = inner_windower

    pipe = ClusteringPipeline(
        windower=windower,
        embedder=AvesEmbedder(model_url=model, pooling=pooling),
        reducer=UmapReducer(n_components=umap_components, n_neighbors=umap_neighbors),
        clusterer=HdbscanClusterer(min_cluster_size=min_cluster_size),
    )

    if score_dir is not None:
        pairs = [(f, _score_path(f, score_dir)) for f in audio_files]
        result = pipe.run_scored(pairs)
    else:
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
