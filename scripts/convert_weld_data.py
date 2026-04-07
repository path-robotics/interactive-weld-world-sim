"""Convert weld-world-model point-cloud data to level-set (SDF) HDF5 episodes.

Reads data_XXXXXX/ folders from the weld-world-model repository, groups samples
into multi-pass trajectories, converts each cross-section into a 128x128 signed
distance field image, and writes HDF5 episodes compatible with the
interactive-weld-world-sim data pipeline.

Usage:
    python scripts/convert_weld_data.py \
        --input_dir ~/Documents/weld-world-model/data/full_training_set \
        --output_dir data/weld_deposition \
        --resolution 128 \
        --val_ratio 0.1 \
        --seed 42
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def _unsigned_distance_to_segments(
    points: np.ndarray,
    seg_a: np.ndarray,
    seg_b: np.ndarray,
    chunk_size: int = 4096,
) -> np.ndarray:
    """Compute unsigned distance from query points to the nearest segment.

    Args:
        points: (M, 2) query points.
        seg_a: (N, 2) segment start points.
        seg_b: (N, 2) segment end points.
        chunk_size: query points processed per batch.

    Returns:
        dist: (M,) unsigned distance to the nearest segment.
    """
    M_total = points.shape[0]

    A = seg_a.astype(np.float32)
    B = seg_b.astype(np.float32)
    AB = B - A
    ab_sq = np.maximum((AB * AB).sum(axis=-1), 1e-30)

    all_dist = np.empty(M_total, dtype=np.float32)

    for start in range(0, M_total, chunk_size):
        end = min(start + chunk_size, M_total)
        P = points[start:end].astype(np.float32)

        AP = P[:, np.newaxis, :] - A[np.newaxis, :, :]
        t = (AP[:, :, 0] * AB[:, 0] + AP[:, :, 1] * AB[:, 1]) / ab_sq
        t = np.clip(t, 0.0, 1.0)

        dx = AP[:, :, 0] - t * AB[:, 0]
        dy = AP[:, :, 1] - t * AB[:, 1]
        dist_sq = dx * dx + dy * dy

        all_dist[start:end] = np.sqrt(dist_sq.min(axis=1))

    return all_dist


def surface_points_to_sdf(
    points: np.ndarray,
    grid_resolution: int = 128,
    x_range: tuple[float, float] = (-0.015, 0.015),
    y_range: tuple[float, float] = (-0.005, 0.015),
    max_dist_m: float = 0.010,
) -> np.ndarray:
    """Convert ordered 2D surface points to a signed distance field.

    Distance is measured to the polyline (open curve).  The inside/outside
    sign is determined by closing the polyline along the grid boundary into
    a polygon and using a robust contains test.

    Sign convention:
      >0.5 → solid (below the surface)
      <0.5 → air   (above the surface)
       0.5 → on the surface

    Args:
        points: (N, 2) ordered surface points in meters.
        grid_resolution: Pixels per side of the output grid.
        x_range: (x_min, x_max) physical bounds in meters.
        y_range: (y_min, y_max) physical bounds in meters.
        max_dist_m: Maximum SDF distance before clipping.

    Returns:
        (grid_resolution, grid_resolution) float32 array in [0, 1].
        Row 0 corresponds to y_max (top of image).
    """
    from matplotlib.path import Path as MplPath

    # --- Unsigned distance to the polyline (open curve) ---
    seg_a = points[:-1]
    seg_b = points[1:]

    x_coords = np.linspace(x_range[0], x_range[1], grid_resolution)
    y_coords = np.linspace(y_range[1], y_range[0], grid_resolution)
    xx, yy = np.meshgrid(x_coords, y_coords)
    grid_pts = np.column_stack([xx.ravel(), yy.ravel()])

    dist = _unsigned_distance_to_segments(grid_pts, seg_a, seg_b)

    # --- Sign via polygon containment ---
    # Close the polyline along the grid boundary (bottom edge) to form
    # a polygon whose interior is "solid".
    first, last = points[0], points[-1]
    boundary_close = np.array([
        [x_range[1], last[1]],     # extend right from last point
        [x_range[1], y_range[0]],  # down to bottom-right corner
        [x_range[0], y_range[0]],  # across bottom edge
        [x_range[0], first[1]],    # up to first point's height
    ])
    polygon_pts = np.concatenate([points, boundary_close], axis=0)
    mpl_path = MplPath(polygon_pts)
    inside = mpl_path.contains_points(grid_pts)

    sign = np.where(inside, 1.0, -1.0)
    sdf = (sign * dist).reshape(grid_resolution, grid_resolution)

    sdf = np.clip(sdf, -max_dist_m, max_dist_m)
    sdf_normalized = (sdf + max_dist_m) / (2.0 * max_dist_m)
    return sdf_normalized.astype(np.float32)


def fuse_surface_and_profile(
    deposition_surface: np.ndarray,
    weld_profile: np.ndarray,
) -> np.ndarray:
    """Fuse a weld bead profile onto a deposition surface.

    The bead's toe points (first and last points of *weld_profile*) are
    located on the deposition surface.  The substrate segments between the
    two toe points are removed and replaced by the bead profile, producing
    a single continuous polyline that represents the outer boundary of the
    fused cross-section.

    Args:
        deposition_surface: (N, 2) ordered substrate polyline.
        weld_profile: (M, 2) ordered bead polyline.  First and last points
            are the toe points that sit on the substrate.

    Returns:
        (K, 2) fused polyline.
    """
    if weld_profile.shape[0] < 2:
        return deposition_surface.copy()

    toe_0 = weld_profile[0]
    toe_1 = weld_profile[-1]

    # Find nearest deposition-surface vertex to each toe
    d0 = np.linalg.norm(deposition_surface - toe_0, axis=1)
    d1 = np.linalg.norm(deposition_surface - toe_1, axis=1)
    idx_0 = int(np.argmin(d0))
    idx_1 = int(np.argmin(d1))

    # Ensure idx_0 < idx_1 along the substrate ordering;
    # if not, reverse the weld profile so it splices in the right direction.
    if idx_0 > idx_1:
        idx_0, idx_1 = idx_1, idx_0
        weld_profile = weld_profile[::-1]

    # Splice: substrate before left toe + bead + substrate after right toe
    fused = np.concatenate([
        deposition_surface[:idx_0],
        weld_profile,
        deposition_surface[idx_1 + 1:],
    ], axis=0)

    return fused


def load_sample(sample_dir: str) -> dict:
    """Load a single weld data sample."""
    deposition_surface = np.load(os.path.join(sample_dir, "deposition_surface.npy"))
    weld_profile = np.load(os.path.join(sample_dir, "weld_profile.npy"))
    weld_params = np.load(os.path.join(sample_dir, "weld_parameter_inputs.npy"))

    with open(os.path.join(sample_dir, "metadata.json")) as f:
        metadata = json.load(f)

    return {
        "deposition_surface": deposition_surface,
        "weld_profile": weld_profile,
        "weld_params": weld_params,
        "metadata": metadata,
        "dir": sample_dir,
    }


def load_all_samples(input_dir: str) -> list[dict]:
    """Load all weld data samples that have a valid weld profile.

    Returns a flat list of samples, each with deposition_surface,
    weld_profile (>= 2 points), weld_params, and metadata.
    """
    sample_dirs = sorted(
        [
            os.path.join(input_dir, d)
            for d in os.listdir(input_dir)
            if d.startswith("data_") and os.path.isdir(os.path.join(input_dir, d))
        ]
    )
    print(f"Found {len(sample_dirs)} data directories")

    samples = []
    skipped = 0
    for sample_dir in tqdm(sample_dirs, desc="Loading samples"):
        try:
            sample = load_sample(sample_dir)
        except Exception as e:
            print(f"Warning: skipping {sample_dir}: {e}")
            skipped += 1
            continue

        if sample["weld_profile"].shape[0] < 2:
            skipped += 1
            continue

        samples.append(sample)

    print(f"Loaded {len(samples)} samples with valid weld profiles "
          f"(skipped {skipped})")
    return samples


def sample_to_sdf_episode(
    sample: dict,
    grid_resolution: int = 128,
    x_range: tuple[float, float] = (-0.015, 0.015),
    y_range: tuple[float, float] = (-0.005, 0.015),
    max_dist_m: float = 0.010,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Convert a single sample into a 2-frame SDF episode.

    Frame 0: SDF of deposition surface (pre-bead)
    Frame 1: SDF of fused surface (post-bead)
    Action:  weld parameters for the bead placement

    Returns:
        (sdf_frames, actions) where:
        - sdf_frames: (2, H, W, 1) float32 in [0, 1]
        - actions: (2, 5) float32
        Returns None if the sample is invalid.
    """
    sdf_kwargs = dict(
        grid_resolution=grid_resolution,
        x_range=x_range,
        y_range=y_range,
        max_dist_m=max_dist_m,
    )

    try:
        sdf_pre = surface_points_to_sdf(sample["deposition_surface"], **sdf_kwargs)
        fused = fuse_surface_and_profile(
            sample["deposition_surface"], sample["weld_profile"]
        )
        sdf_post = surface_points_to_sdf(fused, **sdf_kwargs)
    except Exception as e:
        print(f"Warning: failed to compute SDF for {sample['dir']}: {e}")
        return None

    sdf_frames = np.stack(
        [sdf_pre[:, :, np.newaxis], sdf_post[:, :, np.newaxis]], axis=0
    )  # (2, H, W, 1)

    actions = np.stack(
        [sample["weld_params"].astype(np.float32), np.zeros(5, dtype=np.float32)],
        axis=0,
    )  # (2, 5)

    return sdf_frames, actions


def split_by_experiment(
    samples: list[dict],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split samples into train/val by experiment (dataset_path).

    All samples from the same experiment go to the same split to prevent
    data leakage.
    """
    exp_to_samples: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        exp_name = Path(s["metadata"]["dataset_path"]).name
        exp_to_samples[exp_name].append(s)

    rng = np.random.default_rng(seed)
    exp_names = sorted(exp_to_samples.keys())
    rng.shuffle(exp_names)

    n_val = max(1, int(len(exp_names) * val_ratio))
    val_exps = set(exp_names[:n_val])
    train_exps = set(exp_names[n_val:])

    train = [s for exp in train_exps for s in exp_to_samples[exp]]
    val = [s for exp in val_exps for s in exp_to_samples[exp]]

    print(f"Split: {len(train_exps)} train experiments ({len(train)} samples), "
          f"{len(val_exps)} val experiments ({len(val)} samples)")
    return train, val


def write_episodes(
    samples: list[dict],
    output_dir: str,
    grid_resolution: int = 128,
    x_range: tuple[float, float] = (-0.015, 0.015),
    y_range: tuple[float, float] = (-0.005, 0.015),
    max_dist_m: float = 0.010,
) -> int:
    """Convert samples to single-step SDF episodes and write as HDF5 files."""
    os.makedirs(output_dir, exist_ok=True)
    episode_idx = 0
    skipped = 0

    for sample in tqdm(samples, desc=f"Writing episodes to {output_dir}"):
        result = sample_to_sdf_episode(
            sample,
            grid_resolution=grid_resolution,
            x_range=x_range,
            y_range=y_range,
            max_dist_m=max_dist_m,
        )
        if result is None:
            skipped += 1
            continue

        sdf_frames, actions = result
        episode_path = os.path.join(output_dir, f"episode_{episode_idx}.hdf5")

        with h5py.File(episode_path, "w") as f:
            obs_group = f.create_group("obs")
            images_group = obs_group.create_group("images")
            images_group.create_dataset(
                "cross_section_sdf",
                data=sdf_frames,
                dtype=np.float32,
            )
            f.create_dataset(
                "action",
                data=actions,
                dtype=np.float32,
            )

        episode_idx += 1

    if skipped > 0:
        print(f"Warning: skipped {skipped} samples due to errors")
    print(f"Wrote {episode_idx} episodes to {output_dir}")
    return episode_idx


def visualize_samples(
    samples: list[dict],
    output_path: str,
    n_samples: int = 10,
    grid_resolution: int = 128,
    x_range: tuple[float, float] = (-0.015, 0.015),
    y_range: tuple[float, float] = (-0.005, 0.015),
    max_dist_m: float = 0.010,
) -> None:
    """Generate visualization of single-step SDF conversions.

    Columns: points (substrate + bead) | pre-bead SDF | post-bead SDF
    """
    vis_samples = samples[: min(n_samples, len(samples))]

    n_cols = 3
    fig, axes = plt.subplots(
        len(vis_samples), n_cols,
        figsize=(4 * n_cols, 4 * len(vis_samples)), squeeze=False,
    )

    sdf_kwargs = dict(
        grid_resolution=grid_resolution,
        x_range=x_range,
        y_range=y_range,
        max_dist_m=max_dist_m,
    )
    extent_mm = [x_range[0] * 1000, x_range[1] * 1000,
                 y_range[0] * 1000, y_range[1] * 1000]
    x_contour = np.linspace(x_range[0] * 1000, x_range[1] * 1000, grid_resolution)
    y_contour = np.linspace(y_range[0] * 1000, y_range[1] * 1000, grid_resolution)

    def _plot_sdf(ax: plt.Axes, sdf: np.ndarray, title: str) -> None:
        im = ax.imshow(sdf, cmap="RdBu_r", extent=extent_mm,
                       origin="upper", vmin=0, vmax=1)
        ax.contour(x_contour, y_contour, sdf[::-1],
                   levels=[0.5], colors="black", linewidths=1)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, shrink=0.8)

    for row, sample in enumerate(vis_samples):
        dep = sample["deposition_surface"]
        wp = sample["weld_profile"]

        sdf_pre = surface_points_to_sdf(dep, **sdf_kwargs)
        fused = fuse_surface_and_profile(dep, wp)
        sdf_post = surface_points_to_sdf(fused, **sdf_kwargs)

        # Col 0: point cloud (substrate + bead)
        ax = axes[row, 0]
        ax.plot(dep[:, 0] * 1000, dep[:, 1] * 1000, "b.", markersize=1, label="substrate")
        ax.plot(wp[:, 0] * 1000, wp[:, 1] * 1000, "r.", markersize=1, label="bead")
        ax.set_title(f"Sample {row}: points")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_aspect("equal")
        ax.legend(markerscale=5)

        # Col 1: pre-bead SDF
        _plot_sdf(axes[row, 1], sdf_pre, "Pre-bead SDF")

        # Col 2: post-bead SDF
        _plot_sdf(axes[row, 2], sdf_post, "Post-bead SDF")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved visualization to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert weld data to SDF HDF5 episodes")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to weld data directory (e.g., ~/Documents/weld-world-model/data/full_training_set)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/weld_deposition",
        help="Output directory for HDF5 episodes",
    )
    parser.add_argument("--resolution", type=int, default=128, help="Grid resolution")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    parser.add_argument("--x_min", type=float, default=-0.015, help="X range min (meters)")
    parser.add_argument("--x_max", type=float, default=0.015, help="X range max (meters)")
    parser.add_argument("--y_min", type=float, default=-0.005, help="Y range min (meters)")
    parser.add_argument("--y_max", type=float, default=0.015, help="Y range max (meters)")
    parser.add_argument("--max_dist", type=float, default=0.010, help="Max SDF distance (meters)")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization")
    parser.add_argument("--vis_samples", type=int, default=10, help="Number of samples to visualize")
    args = parser.parse_args()

    x_range = (args.x_min, args.x_max)
    y_range = (args.y_min, args.y_max)

    # Load all samples with valid weld profiles
    samples = load_all_samples(args.input_dir)

    # Visualize before conversion (optional)
    if args.visualize:
        visualize_samples(
            samples,
            output_path=os.path.join(args.output_dir, "sdf_visualization.png"),
            n_samples=args.vis_samples,
            grid_resolution=args.resolution,
            x_range=x_range,
            y_range=y_range,
            max_dist_m=args.max_dist,
        )

    # Split into train/val by experiment
    train_samples, val_samples = split_by_experiment(
        samples, val_ratio=args.val_ratio, seed=args.seed
    )

    # Write single-step episodes
    train_dir = os.path.join(args.output_dir, "train")
    val_dir = os.path.join(args.output_dir, "val")

    n_train = write_episodes(
        train_samples,
        train_dir,
        grid_resolution=args.resolution,
        x_range=x_range,
        y_range=y_range,
        max_dist_m=args.max_dist,
    )
    n_val = write_episodes(
        val_samples,
        val_dir,
        grid_resolution=args.resolution,
        x_range=x_range,
        y_range=y_range,
        max_dist_m=args.max_dist,
    )

    print(f"\nDone! {n_train} train episodes, {n_val} val episodes")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
