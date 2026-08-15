import numpy as np
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt

def get_default_colors(labels: np.ndarray) -> dict[int, tuple]:
    # Default color mapping
    cluster_ids = sorted(k for k in np.unique(labels) if k >= 0)
    colors = {}
    
    if len(cluster_ids) == 0:
        return colors

    palette = plt.get_cmap("tab20").colors
    
    for cid in cluster_ids:
        colors[cid] = palette[cid % len(palette)]
        
    return colors

def get_2d_colors(labels: np.ndarray, reduced: np.ndarray, probabilities: np.ndarray) -> dict[int, tuple]:
    """Map each cluster to a discrete RGB color based on the spatial location of its core point.
    
    Args:
        labels: Array of cluster labels (N,)
        reduced: Array of 2D UMAP coordinates (N, 2)
        probabilities: Array of HDBSCAN probabilities (N,)
        
    Returns:
        Dictionary mapping cluster ID to RGB tuple.
    """
    cluster_ids = sorted(k for k in np.unique(labels) if k >= 0)
    colors = {}
    
    if len(cluster_ids) == 0:
        return colors

    # Find point in cluster with highest probability ("Cluster Centroid in a way")
    core_points = []
    for cid in cluster_ids:
        mask = labels == cid
        # Within the cluster, find the index of max probability
        if probabilities is not None and len(probabilities) == len(labels):
            core_point_idx = np.argmax(probabilities[mask])
            core_point = reduced[mask][core_point_idx]
        else:
            # Error
            print(f"ERROR: Cannot find UMAP Probabilities or they are improperly formatted")
        core_points.append(core_point)
        
    core_points = np.array(core_points) # (N_clusters, 2)
    
    # Find the center of all points, so we can put our color wheel there
    center = np.mean(core_points, axis=0)
    centered = core_points - center
    
    # Polar coordinates
    angles = np.arctan2(centered[:, 1], centered[:, 0]) # -pi to pi
    radii = np.hypot(centered[:, 0], centered[:, 1])

    # Un-evenly spaced angles (might have more similar colors though) normalized to HSV 0-1 range
    hues = (angles + np.pi) / (2 * np.pi)
    
    # Map normalized radius [0, 1] to saturation [0.2, 1.0] to make distances more sensitive
    max_radius = np.max(radii) if np.max(radii) > 0 else 1.0
    norm_radii = radii / max_radius
    sats = 0.2 + 0.8 * norm_radii
    
    # Fixed brightness
    vals = np.full_like(hues, 0.80)
    
    # Convert HSV to RGB
    hsv_array = np.column_stack((hues, sats, vals))
    for i, cid in enumerate(cluster_ids):
        rgb = hsv_to_rgb(hsv_array[i])
        colors[cid] = tuple(rgb)
        
    return colors

def get_3d_colors(labels: np.ndarray, reduced_3d: np.ndarray, probabilities: np.ndarray) -> dict[int, tuple]:
    """Map each cluster to a discrete RGB color based on its spatial location in 3D UMAP.
    
    Args:
        labels: Array of cluster labels (N,)
        reduced_3d: Array of 3D UMAP coordinates (N, 3)
        probabilities: Array of HDBSCAN probabilities (N,)
        
    Returns:
        Dictionary mapping cluster ID to RGB tuple.
    """
    cluster_ids = sorted(k for k in np.unique(labels) if k >= 0)
    colors = {}
    
    if len(cluster_ids) == 0:
        return colors

    # Find point in cluster with highest probability
    core_points = []
    for cid in cluster_ids:
        mask = labels == cid
        if probabilities is not None and len(probabilities) == len(labels):
            core_point_idx = np.argmax(probabilities[mask])
            core_point = reduced_3d[mask][core_point_idx]
        else:
            core_point = np.median(reduced_3d[mask], axis=0)
        core_points.append(core_point)
        
    core_points = np.array(core_points) # (N_clusters, 3)
    
    # Normalize X, Y, Z to [0, 1] for R, G, B
    mins = np.min(core_points, axis=0)
    maxs = np.max(core_points, axis=0)
    
    # Prevent division by zero
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    
    normalized = (core_points - mins) / ranges
    
    for i, cid in enumerate(cluster_ids):
        # The normalized [X, Y, Z] maps directly to [R, G, B]
        rgb = tuple(normalized[i])
        colors[cid] = rgb
        
    return colors

def extract_colors_from_npz(r, color_mode: str = "3D") -> dict[int, tuple]:
    """Extract or compute colors from an npz results dictionary.
    
    If the npz dictionary contains inherited colors (from a prediction pipeline), 
    those are returned. Otherwise, colors are computed based on the requested mode.
    
    Args:
        r: Dictionary or npz file object containing results.
        color_mode: "3D", "2D", or "default".
        
    Returns:
        Dictionary mapping cluster ID to RGB tuple.
    """
    if "cluster_color_keys" in r and "cluster_color_vals" in r:
        keys = r["cluster_color_keys"]
        vals = r["cluster_color_vals"]
        colors = {int(k): tuple(v) for k, v in zip(keys, vals)}
        # Ensure noise color is set if not present
        if -1 not in colors:
            colors[-1] = (0.75, 0.75, 0.75, 0.4)
        return colors
        
    labels = r.get("labels")
    if labels is None:
        return {}
        
    probabilities = r.get("probabilities")
    reduced = r.get("reduced")
    reduced_3d = r.get("reduced_3d")
    
    if color_mode == "2D" and reduced is not None:
        colors = get_2d_colors(labels, reduced, probabilities)
    elif color_mode == "3D" and reduced_3d is not None:
        colors = get_3d_colors(labels, reduced_3d, probabilities)
    elif color_mode == "3D" and reduced_3d is None:
        colors = get_2d_colors(labels, reduced, probabilities) if reduced is not None else get_default_colors(labels)
    else:
        colors = get_default_colors(labels)
        
    colors[-1] = (0.75, 0.75, 0.75, 0.4)
    return colors
