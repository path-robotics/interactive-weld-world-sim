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
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm


def surface_points_to_sdf(
    points: np.ndarray,
    grid_resolution: int = 128,
    x_range: tuple[float, float] = (-0.015, 0.015),
    y_range: tuple[float, float] = (-0.005, 0.015),
    max_dist_m: float = 0.010,
) -> np.ndarray:
    """Convert 2D surface points to a signed distance field on a regular grid.

    Args:
        points: (N, 2) array of (x, y) surface points in meters.
        grid_resolution: Number of pixels per side of the output grid.
        x_range: (x_min, x_max) physical bounds in meters.
        y_range: (y_min, y_max) physical bounds in meters.
        max_dist_m: Maximum distance in meters for clipping the SDF.

    Returns:
        (grid_resolution, grid_resolution) float32 array in [0, 1],
        where 0.5 is the zero level set, >0.5 is inside (solid), <0.5 is outside (air).
    """
    # Sort points by x
    sorted_idx = np.argsort(points[:, 0])
    pts = points[sorted_idx]

    # Build interpolation of surface height y(x)
    interp_fn = interp1d(
        pts[:, 0],
        pts[:, 1],
        kind="linear",
        bounds_error=False,
        fill_value=(pts[0, 1], pts[-1, 1]),
    )

    # Create grid
    x_coords = np.linspace(x_range[0], x_range[1], grid_resolution)
    y_coords = np.linspace(y_range[0], y_range[1], grid_resolution)

    # Surface height at each column
    surface_heights = interp_fn(x_coords)

    # Build occupancy grid: (H, W) where H is y-axis (row 0 = y_max), W is x-axis
    # Using convention: row index 0 = top of image = y_max
    yy, xx = np.meshgrid(y_coords[::-1], x_coords, indexing="ij")
    surface_heights_grid = interp_fn(xx)
    occupancy = (yy <= surface_heights_grid).astype(np.float64)

    # Compute unsigned distances for both regions
    dist_inside = distance_transform_edt(occupancy)
    dist_outside = distance_transform_edt(1.0 - occupancy)

    # Signed distance: positive inside (solid), negative outside (air)
    pixel_spacing_x = (x_range[1] - x_range[0]) / grid_resolution
    pixel_spacing_y = (y_range[1] - y_range[0]) / grid_resolution
    avg_spacing = (pixel_spacing_x + pixel_spacing_y) / 2.0

    sdf = (dist_inside - dist_outside) * avg_spacing

    # Clip and normalize to [0, 1]
    sdf = np.clip(sdf, -max_dist_m, max_dist_m)
    sdf_normalized = (sdf + max_dist_m) / (2.0 * max_dist_m)

    return sdf_normalized.astype(np.float32)


def merge_surface_and_profile(
    deposition_surface: np.ndarray,
    weld_profile: np.ndarray,
) -> np.ndarray:
    """Merge deposition surface and weld profile into a combined boundary.

    The combined boundary takes the maximum y at each x position, effectively
    representing the cumulative surface after the weld pass.

    Args:
        deposition_surface: (N, 2) deposition surface points.
        weld_profile: (M, 2) weld bead profile points.

    Returns:
        (K, 2) combined surface points sorted by x.
    """
    if weld_profile.shape[0] == 0:
        return deposition_surface.copy()

    # Combine all points
    all_points = np.vstack([deposition_surface, weld_profile])

    # Sort by x
    sorted_idx = np.argsort(all_points[:, 0])
    all_points = all_points[sorted_idx]

    # For overlapping x regions, take the max y (highest point)
    # Use a fine grid to resample both surfaces and take the max
    x_min = min(deposition_surface[:, 0].min(), weld_profile[:, 0].min())
    x_max = max(deposition_surface[:, 0].max(), weld_profile[:, 0].max())

    n_sample = max(200, deposition_surface.shape[0] + weld_profile.shape[0])
    x_grid = np.linspace(x_min, x_max, n_sample)

    # Interpolate deposition surface
    dep_sorted = deposition_surface[np.argsort(deposition_surface[:, 0])]
    dep_interp = interp1d(
        dep_sorted[:, 0],
        dep_sorted[:, 1],
        kind="linear",
        bounds_error=False,
        fill_value=(dep_sorted[0, 1], dep_sorted[-1, 1]),
    )
    dep_heights = dep_interp(x_grid)

    # Interpolate weld profile
    wp_sorted = weld_profile[np.argsort(weld_profile[:, 0])]
    wp_interp = interp1d(
        wp_sorted[:, 0],
        wp_sorted[:, 1],
        kind="linear",
        bounds_error=False,
        fill_value=-np.inf,  # No weld outside its x range
    )
    wp_heights = wp_interp(x_grid)

    # Mark weld profile as -inf outside its actual x range
    wp_x_min, wp_x_max = weld_profile[:, 0].min(), weld_profile[:, 0].max()
    mask_outside = (x_grid < wp_x_min) | (x_grid > wp_x_max)
    wp_heights[mask_outside] = -np.inf

    # Take maximum height at each x
    combined_heights = np.maximum(dep_heights, wp_heights)

    return np.column_stack([x_grid, combined_heights])


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


def build_trajectories(
    input_dir: str,
) -> list[list[dict]]:
    """Group samples into multi-pass trajectories.

    Returns a list of trajectories, each being a list of samples
    sorted by pass_index.
    """
    # Scan all data directories
    sample_dirs = sorted(
        [
            os.path.join(input_dir, d)
            for d in os.listdir(input_dir)
            if d.startswith("data_") and os.path.isdir(os.path.join(input_dir, d))
        ]
    )

    print(f"Found {len(sample_dirs)} data samples")

    # Load metadata and group
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)

    for sample_dir in tqdm(sample_dirs, desc="Loading metadata"):
        try:
            sample = load_sample(sample_dir)
        except Exception as e:
            print(f"Warning: skipping {sample_dir}: {e}")
            continue

        md = sample["metadata"]
        dataset_name = Path(md["dataset_path"]).name
        waypoint_idx = md["waypoint_index"]
        key = (dataset_name, waypoint_idx)
        groups[key].append(sample)

    # Sort each group by pass_index and build trajectory list
    trajectories = []
    for key, samples in groups.items():
        samples.sort(key=lambda s: s["metadata"]["pass_index"])
        trajectories.append(samples)

    # Sort trajectories by length (descending) for better progress tracking
    trajectories.sort(key=len, reverse=True)

    traj_lengths = [len(t) for t in trajectories]
    print(f"Built {len(trajectories)} trajectories")
    print(f"  Length distribution: min={min(traj_lengths)}, max={max(traj_lengths)}, "
          f"mean={np.mean(traj_lengths):.1f}, median={np.median(traj_lengths):.0f}")
    print(f"  Multi-pass (len>=2): {sum(1 for l in traj_lengths if l >= 2)}")
    print(f"  Single-pass (len==1): {sum(1 for l in traj_lengths if l == 1)}")

    return trajectories


def trajectory_to_sdf_episode(
    trajectory: list[dict],
    grid_resolution: int = 128,
    x_range: tuple[float, float] = (-0.015, 0.015),
    y_range: tuple[float, float] = (-0.005, 0.015),
    max_dist_m: float = 0.010,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Convert a trajectory of samples into SDF frames and actions.

    Returns:
        (sdf_frames, actions) where:
        - sdf_frames: (T, H, W, 1) float32 in [0, 1]
        - actions: (T, 5) float32
        Returns None if the trajectory is invalid.
    """
    sdf_kwargs = dict(
        grid_resolution=grid_resolution,
        x_range=x_range,
        y_range=y_range,
        max_dist_m=max_dist_m,
    )

    K = len(trajectory)
    sdf_frames = []
    actions = []

    # Frames 0..K-1: SDF of each pass's deposition surface (pre-weld state)
    for i in range(K):
        try:
            sdf = surface_points_to_sdf(trajectory[i]["deposition_surface"], **sdf_kwargs)
            sdf_frames.append(sdf[:, :, np.newaxis])
            actions.append(trajectory[i]["weld_params"].astype(np.float32))
        except Exception as e:
            print(f"Warning: failed to compute SDF for {trajectory[i]['dir']}: {e}")
            return None

    # Frame K: final state after the last pass (merge deposition_surface + weld_profile)
    last = trajectory[-1]
    try:
        if last["weld_profile"].shape[0] > 0:
            combined = merge_surface_and_profile(
                last["deposition_surface"], last["weld_profile"]
            )
            sdf = surface_points_to_sdf(combined, **sdf_kwargs)
        else:
            # No weld profile for the last pass — just use deposition surface
            sdf = surface_points_to_sdf(last["deposition_surface"], **sdf_kwargs)
        sdf_frames.append(sdf[:, :, np.newaxis])
        # Pad action for the final frame (no action taken)
        actions.append(np.zeros(5, dtype=np.float32))
    except Exception as e:
        print(f"Warning: failed to compute final SDF for {last['dir']}: {e}")
        return None

    sdf_frames = np.stack(sdf_frames, axis=0)  # (T, H, W, 1)
    actions = np.stack(actions, axis=0)  # (T, 5)

    return sdf_frames, actions


def split_by_experiment(
    trajectories: list[list[dict]],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[list[dict]], list[list[dict]]]:
    """Split trajectories into train/val by experiment (dataset_path).

    All trajectories from the same experiment go to the same split to prevent
    data leakage.
    """
    # Group trajectories by experiment
    exp_to_trajs: dict[str, list[list[dict]]] = defaultdict(list)
    for traj in trajectories:
        exp_name = Path(traj[0]["metadata"]["dataset_path"]).name
        exp_to_trajs[exp_name].append(traj)

    # Shuffle experiments
    rng = np.random.default_rng(seed)
    exp_names = sorted(exp_to_trajs.keys())
    rng.shuffle(exp_names)

    # Split
    n_val = max(1, int(len(exp_names) * val_ratio))
    val_exps = set(exp_names[:n_val])
    train_exps = set(exp_names[n_val:])

    train_trajs = [t for exp in train_exps for t in exp_to_trajs[exp]]
    val_trajs = [t for exp in val_exps for t in exp_to_trajs[exp]]

    print(f"Split: {len(train_exps)} train experiments ({len(train_trajs)} trajectories), "
          f"{len(val_exps)} val experiments ({len(val_trajs)} trajectories)")

    return train_trajs, val_trajs


def write_episodes(
    trajectories: list[list[dict]],
    output_dir: str,
    grid_resolution: int = 128,
    x_range: tuple[float, float] = (-0.015, 0.015),
    y_range: tuple[float, float] = (-0.005, 0.015),
    max_dist_m: float = 0.010,
) -> int:
    """Convert trajectories to SDF episodes and write as HDF5 files.

    Returns the number of episodes written.
    """
    os.makedirs(output_dir, exist_ok=True)
    episode_idx = 0
    skipped = 0

    for traj in tqdm(trajectories, desc=f"Writing episodes to {output_dir}"):
        result = trajectory_to_sdf_episode(
            traj,
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
        print(f"Warning: skipped {skipped} trajectories due to errors")
    print(f"Wrote {episode_idx} episodes to {output_dir}")
    return episode_idx


def visualize_samples(
    trajectories: list[list[dict]],
    output_path: str,
    n_samples: int = 10,
    grid_resolution: int = 128,
    x_range: tuple[float, float] = (-0.015, 0.015),
    y_range: tuple[float, float] = (-0.005, 0.015),
    max_dist_m: float = 0.010,
) -> None:
    """Generate visualization of SDF images alongside original point data."""
    # Pick multi-pass trajectories for more interesting visualizations
    multi_pass = [t for t in trajectories if len(t) >= 2]
    samples = multi_pass[: min(n_samples, len(multi_pass))]

    fig, axes = plt.subplots(
        len(samples), 4, figsize=(16, 4 * len(samples)), squeeze=False
    )

    for row, traj in enumerate(samples):
        # Show first state SDF
        sdf_first = surface_points_to_sdf(
            traj[0]["deposition_surface"],
            grid_resolution=grid_resolution,
            x_range=x_range,
            y_range=y_range,
            max_dist_m=max_dist_m,
        )

        # Show last state SDF (after final weld)
        if traj[-1]["weld_profile"].shape[0] > 0:
            combined = merge_surface_and_profile(
                traj[-1]["deposition_surface"], traj[-1]["weld_profile"]
            )
        else:
            combined = traj[-1]["deposition_surface"]
        sdf_last = surface_points_to_sdf(
            combined,
            grid_resolution=grid_resolution,
            x_range=x_range,
            y_range=y_range,
            max_dist_m=max_dist_m,
        )

        # Column 0: first state point cloud
        ax = axes[row, 0]
        pts = traj[0]["deposition_surface"]
        ax.plot(pts[:, 0] * 1000, pts[:, 1] * 1000, "b.", markersize=1)
        ax.set_title(f"Traj {row}: Initial surface (points)")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_aspect("equal")

        # Column 1: first state SDF
        # Array has row 0 = y_max, so use origin="upper" to display correctly
        ax = axes[row, 1]
        im = ax.imshow(
            sdf_first,
            cmap="RdBu_r",
            extent=[x_range[0] * 1000, x_range[1] * 1000, y_range[0] * 1000, y_range[1] * 1000],
            origin="upper",
            vmin=0, vmax=1,
        )
        ax.contour(
            np.linspace(x_range[0] * 1000, x_range[1] * 1000, grid_resolution),
            np.linspace(y_range[0] * 1000, y_range[1] * 1000, grid_resolution),
            sdf_first[::-1],
            levels=[0.5],
            colors="black",
            linewidths=1,
        )
        ax.set_title("Initial SDF")
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Column 2: last state point cloud
        ax = axes[row, 2]
        dep = traj[-1]["deposition_surface"]
        wp = traj[-1]["weld_profile"]
        ax.plot(dep[:, 0] * 1000, dep[:, 1] * 1000, "b.", markersize=1, label="substrate")
        if wp.shape[0] > 0:
            ax.plot(wp[:, 0] * 1000, wp[:, 1] * 1000, "r.", markersize=1, label="bead")
        ax.set_title(f"Final state (points, {len(traj)} passes)")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_aspect("equal")
        ax.legend(markerscale=5)

        # Column 3: last state SDF
        ax = axes[row, 3]
        im = ax.imshow(
            sdf_last,
            cmap="RdBu_r",
            extent=[x_range[0] * 1000, x_range[1] * 1000, y_range[0] * 1000, y_range[1] * 1000],
            origin="upper",
            vmin=0, vmax=1,
        )
        ax.contour(
            np.linspace(x_range[0] * 1000, x_range[1] * 1000, grid_resolution),
            np.linspace(y_range[0] * 1000, y_range[1] * 1000, grid_resolution),
            sdf_last[::-1],
            levels=[0.5],
            colors="black",
            linewidths=1,
        )
        ax.set_title("Final SDF")
        plt.colorbar(im, ax=ax, shrink=0.8)

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

    # Build trajectories
    trajectories = build_trajectories(args.input_dir)

    # Visualize before conversion (optional)
    if args.visualize:
        visualize_samples(
            trajectories,
            output_path=os.path.join(args.output_dir, "sdf_visualization.png"),
            n_samples=args.vis_samples,
            grid_resolution=args.resolution,
            x_range=x_range,
            y_range=y_range,
            max_dist_m=args.max_dist,
        )

    # Split into train/val
    train_trajs, val_trajs = split_by_experiment(
        trajectories, val_ratio=args.val_ratio, seed=args.seed
    )

    # Write episodes
    train_dir = os.path.join(args.output_dir, "train")
    val_dir = os.path.join(args.output_dir, "val")

    n_train = write_episodes(
        train_trajs,
        train_dir,
        grid_resolution=args.resolution,
        x_range=x_range,
        y_range=y_range,
        max_dist_m=args.max_dist,
    )
    n_val = write_episodes(
        val_trajs,
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
