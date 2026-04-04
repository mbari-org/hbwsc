"""CLI entry point: ``python -m hbws_clustering`` or ``hbws-cluster``."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="Humpback whale vocalization clustering via AVES + UMAP + HDBSCAN.")

# Defined here so the default is visible without importing torch.
_AVES_BASE_BIO_URL = "https://storage.googleapis.com/esp-public-files/ported_aves/aves-base-bio.torchaudio.pt"

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
            "Score file is located automatically from the WAV filename date. "
            "When provided, only high-confidence windows are extracted."
        ),
    ),
    score_file: Optional[Path] = typer.Option(
        None,
        "--score-file",
        help=(
            "Explicit path to a score .npy file. "
            "Use instead of --score-dir when the WAV filename does not follow "
            "the MARS-YYYYMMDDTHHMMSSZ convention. Requires exactly one audio file."
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
    model: str = typer.Option(_AVES_BASE_BIO_URL, help="URL to a TorchAudio AVES checkpoint (.pt)."),
    pooling: str = typer.Option("mean", help="Embedding pooling: 'mean' or 'max'."),
    batch_size: int = typer.Option(16, help="AVES inference batch size. Increase (e.g. 64) on GPU."),
    umap_components: int = typer.Option(2, help="UMAP dimensions for visualization (2-D scatter plot)."),
    umap_cluster_components: int = typer.Option(
        10,
        "--umap-cluster-components",
        help="UMAP dimensions for HDBSCAN clustering",
    ),
    umap_neighbors: int = typer.Option(15, help="UMAP n_neighbors."),
    min_cluster_size: int = typer.Option(5, help="HDBSCAN min_cluster_size."),
    embeddings_cache: Optional[Path] = typer.Option(
        None,
        "--embeddings-cache",
        "-e",
        help="Path to a .npy file for caching AVES embeddings. Loaded if it exists, saved otherwise.",
    ),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save results as .npz file."),
) -> None:
    """Cluster humpback whale vocalizations in AUDIO_FILES.

    Without --score-dir/--score-file: windows every second of every file.\n
    With --score-dir: reads HWSD score files matched by date from the WAV filename.\n
    With --score-file: uses the given score file directly (one audio file only).
    """
    # Heavy imports deferred so --help is instant.
    import numpy as np

    from hbws_clustering.clustering import HdbscanClusterer
    from hbws_clustering.embedding import AvesEmbedder
    from hbws_clustering.pipeline import ClusteringPipeline
    from hbws_clustering.reduction import UmapReducer
    from hbws_clustering.windowing import AudioWindower, ScoreGuidedWindower

    if score_dir is not None and score_file is not None:
        raise typer.BadParameter("--score-dir and --score-file are mutually exclusive.")
    if score_file is not None and len(audio_files) != 1:
        raise typer.BadParameter("--score-file requires exactly one audio file.")

    inner_windower = AudioWindower(window_sec=window_sec, hop_sec=hop_sec, target_sr=sample_rate)

    use_scores = score_dir is not None or score_file is not None
    if use_scores:
        windower = ScoreGuidedWindower(windower=inner_windower, threshold=score_threshold)
    else:
        windower = inner_windower

    reducer_viz = UmapReducer(n_components=umap_components, n_neighbors=umap_neighbors)
    if umap_cluster_components != umap_components:
        reducer_cluster = UmapReducer(n_components=umap_cluster_components, n_neighbors=umap_neighbors)
    else:
        reducer_cluster = None

    pipe = ClusteringPipeline(
        windower=windower,
        embedder=AvesEmbedder(model_url=model, pooling=pooling, batch_size=batch_size),
        reducer=reducer_viz,
        reducer_cluster=reducer_cluster,
        clusterer=HdbscanClusterer(min_cluster_size=min_cluster_size),
    )

    if score_file is not None:
        pairs = [(audio_files[0], score_file)]
        result = pipe.run_scored(pairs, embeddings_cache=embeddings_cache)
    elif score_dir is not None:
        pairs = [(f, _score_path(f, score_dir)) for f in audio_files]
        result = pipe.run_scored(pairs, embeddings_cache=embeddings_cache)
    else:
        result = pipe.run(audio_files, embeddings_cache=embeddings_cache)

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
            reduced_cluster=result.reduced_cluster,
            embeddings=result.embeddings,
            start_secs=np.array([w.start_sec for w in result.windows]),
            source_files=np.array([str(w.source_file) for w in result.windows]),
        )
        typer.echo(f"\nResults saved to {output}")


if __name__ == "__main__":
    app()
