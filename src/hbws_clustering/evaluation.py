"""Evaluation utilities for comparing clusters to manual labels."""

import csv
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    homogeneity_completeness_v_measure,
)


def load_raven_labels(path: Path | str) -> list[tuple[float, float, str]]:
    """Load Raven selection table into a list of (begin_sec, end_sec, type)."""
    manual_labels = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            begin_sec = float(row["Begin Time (s)"])
            end_sec = float(row["End Time (s)"])
            label_type = row["Type"].strip()
            manual_labels.append((begin_sec, end_sec, label_type))
    return manual_labels


def map_labels_to_windows(
    manual_labels: list[tuple[float, float, str]],
    start_secs: np.ndarray,
    end_secs: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    """Map manual labels to analysis windows based on maximum overlap.

    Args:
        manual_labels: List of (begin_sec, end_sec, label_type).
        start_secs: Array of window start times in seconds.
        end_secs: Array of window end times in seconds.

    Returns:
        manual_window: np.ndarray (shape N), containing integer indices of the 
                       manual label types, or -1 if no overlap.
        type_to_idx: dict mapping string label types to their integer indices.
    """
    unique_types = sorted({t for _, _, t in manual_labels})
    type_to_idx = {t: i for i, t in enumerate(unique_types)}

    manual_window = np.full(len(start_secs), -1, dtype=int)
    best_overlap = np.zeros(len(start_secs), dtype=float)

    for begin_sec, end_sec, ltype in manual_labels:
        overlap = np.minimum(end_secs, end_sec) - np.maximum(start_secs, begin_sec)
        np.clip(overlap, 0.0, None, out=overlap)

        better = overlap > best_overlap
        best_overlap = np.where(better, overlap, best_overlap)
        manual_window = np.where(better, type_to_idx[ltype], manual_window)

    return manual_window, type_to_idx


def compute_metrics(
    labels: np.ndarray,
    manual_window: np.ndarray,
) -> dict[str, float]:
    """Compute clustering metrics against manual labels.

    Args:
        labels: Integer cluster labels (shape N), where -1 is noise.
        manual_window: Integer manual labels (shape N), where -1 is unlabelled.

    Returns:
        Dictionary containing DetSim, NMI, ARI, Homogeneity, Completeness, V_measure.
        Values are NaN if there is insufficient intersection.
    """
    is_clustered = labels >= 0
    is_labelled = manual_window >= 0

    inter = is_clustered & is_labelled
    union = is_clustered | is_labelled

    detsim = float(inter.sum() / union.sum()) if union.any() else 0.0

    if inter.sum() < 2:
        return {
            "DetSim": detsim,
            "NMI": float("nan"),
            "ARI": float("nan"),
            "Homogeneity": float("nan"),
            "Completeness": float("nan"),
            "V_measure": float("nan"),
        }

    c_in = labels[inter]
    m_in = manual_window[inter]

    nmi = normalized_mutual_info_score(m_in, c_in)
    ari = adjusted_rand_score(m_in, c_in)
    homog, comp, v_meas = homogeneity_completeness_v_measure(m_in, c_in)

    return {
        "DetSim": float(detsim),
        "NMI": float(nmi),
        "ARI": float(ari),
        "Homogeneity": float(homog),
        "Completeness": float(comp),
        "V_measure": float(v_meas),
    }
