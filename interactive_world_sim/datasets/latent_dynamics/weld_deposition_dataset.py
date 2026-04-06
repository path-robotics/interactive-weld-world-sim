"""Dataset for weld deposition SDF data.

Loads HDF5 episodes produced by scripts/convert_weld_data.py, where each episode
is a multi-pass weld trajectory with signed distance field (SDF) images and weld
parameter actions.
"""

import copy
import glob
import multiprocessing
import os
import shutil
from pathlib import Path
from typing import Dict, Optional

import h5py
import numpy as np
import torch
import zarr
import zarr.storage
from filelock import FileLock
from numcodecs import Blosc
from omegaconf import DictConfig
from tqdm import tqdm

from interactive_world_sim.utils.normalizer import (
    LinearNormalizer,
    array_to_stats,
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
)
from interactive_world_sim.utils.pytorch_util import dict_apply
from interactive_world_sim.utils.replay_buffer import ReplayBuffer
from interactive_world_sim.utils.sampler import SequenceSampler

from .base_dataset import BaseImageDataset


def _convert_weld_to_dp_replay(
    store: zarr.storage.Store,
    shape_meta: dict,
    dataset_dir: str,
) -> ReplayBuffer:
    """Convert weld HDF5 episodes to a Zarr replay buffer."""
    # Parse shape_meta for observation types
    sdf_keys = []
    obs_shape_meta = shape_meta["obs"]
    for key, attr in obs_shape_meta.items():
        obs_type = attr.get("type", "low_dim")
        if obs_type == "sdf":
            sdf_keys.append(key)

    root = zarr.group(store)
    data_group = root.require_group("data", overwrite=True)
    meta_group = root.require_group("meta", overwrite=True)

    # Find and sort episodes
    episodes_paths = glob.glob(os.path.join(dataset_dir, "episode_*.hdf5"))
    episodes_stem_name = [Path(path).stem for path in episodes_paths]
    episodes_idx = [int(stem_name.split("_")[-1]) for stem_name in episodes_stem_name]
    episodes_idx = sorted(episodes_idx)

    episode_ends = []
    prev_end = 0
    action_data_list = []
    sdf_data_dict: dict[str, list] = {}

    for epi_idx in tqdm(episodes_idx, desc="Loading episodes"):
        dataset_path = os.path.join(dataset_dir, f"episode_{epi_idx}.hdf5")
        with h5py.File(dataset_path) as file:
            episode_length = file["action"].shape[0]
            episode_end = prev_end + episode_length
            prev_end = episode_end
            episode_ends.append(episode_end)

            # Load actions directly (no conversion needed)
            action_data_list.append(file["action"][()].astype(np.float32))

            # Load SDF data
            for key in sdf_keys:
                if key not in sdf_data_dict:
                    sdf_data_dict[key] = []
                sdf_data = file["obs"]["images"][key][()]  # (T, H, W, 1)
                sdf_data_dict[key].append(sdf_data.astype(np.float32))

    # Dump metadata
    n_steps = episode_ends[-1]
    meta_group.array(
        "episode_ends", episode_ends, dtype=np.int64, compressor=None, overwrite=True
    )

    # Dump actions (lowdim, no compression)
    action_data = np.concatenate(action_data_list, axis=0)
    data_group.array(
        name="action",
        data=action_data,
        shape=action_data.shape,
        chunks=action_data.shape,
        compressor=None,
        dtype=action_data.dtype,
    )

    # Dump SDF data (float32, lossless compression)
    compressor = Blosc(cname="lz4", clevel=5, shuffle=Blosc.BITSHUFFLE)
    for key, data_list in sdf_data_dict.items():
        sdf_arr = np.concatenate(data_list, axis=0)  # (N_total, H, W, 1)
        shape = tuple(shape_meta["obs"][key]["shape"])
        c, h, w = shape
        data_group.require_dataset(
            name=key,
            shape=(n_steps, h, w, c),
            chunks=(1, h, w, c),
            compressor=compressor,
            dtype=np.float32,
        )
        # Write data in bulk
        data_group[key][:] = sdf_arr

    replay_buffer = ReplayBuffer(root)
    return replay_buffer


def load_replay_buffer(
    dataset_dir: str, use_cache: bool, shape_meta: dict
) -> ReplayBuffer:
    """Load or create a cached replay buffer from weld HDF5 episodes."""
    replay_buffer = None
    if use_cache:
        cache_zarr_path = os.path.join(dataset_dir, "cache.zarr.zip")
        cache_lock_path = cache_zarr_path + ".lock"
        print("Acquiring lock on cache.")
        with FileLock(cache_lock_path):
            if not os.path.exists(cache_zarr_path):
                try:
                    print("Cache does not exist. Creating!")
                    replay_buffer = _convert_weld_to_dp_replay(
                        store=zarr.MemoryStore(),
                        shape_meta=shape_meta,
                        dataset_dir=dataset_dir,
                    )
                    print("Saving cache to disk.")
                    with zarr.ZipStore(cache_zarr_path) as zip_store:
                        replay_buffer.save_to_store(store=zip_store)
                except Exception as e:
                    if os.path.exists(cache_zarr_path):
                        os.remove(cache_zarr_path)
                    raise e
            else:
                print("Loading cached ReplayBuffer from Disk.")
                with zarr.ZipStore(cache_zarr_path, mode="r") as zip_store:
                    replay_buffer = ReplayBuffer.copy_from_store(
                        src_store=zip_store, store=zarr.MemoryStore()
                    )
                print("Loaded!")
    else:
        replay_buffer = _convert_weld_to_dp_replay(
            store=zarr.MemoryStore(),
            shape_meta=shape_meta,
            dataset_dir=dataset_dir,
        )
    return replay_buffer


class WeldDepositionDataset(BaseImageDataset):
    """Dataset for weld deposition SDF data."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        shape_meta = cfg.shape_meta
        dataset_dir = cfg.dataset_dir
        horizon = cfg.horizon * cfg.skip_frame
        pad_before = cfg.pad_before
        pad_after = cfg.pad_after
        use_cache = cfg.use_cache
        self.val_horizon = (
            cfg.val_horizon * cfg.skip_frame if "val_horizon" in cfg else horizon
        )
        self.skip_idx = cfg.skip_idx if "skip_idx" in cfg else 1
        self.aug_mode = "none"

        train_dir = os.path.join(dataset_dir, "train")
        self.replay_buffer = load_replay_buffer(train_dir, use_cache, shape_meta)

        # Classify observation keys by type
        sdf_keys = []
        lowdim_keys = []
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            obs_type = attr.get("type", "low_dim")
            if obs_type == "sdf":
                sdf_keys.append(key)
            elif obs_type == "low_dim":
                lowdim_keys.append(key)

        train_mask = np.ones((self.replay_buffer.n_episodes,), dtype=bool)
        all_keys = list(self.replay_buffer.keys())

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            goal_sample=cfg.goal_sample,
            keys=all_keys,
            skip_frame=cfg.skip_frame,
            keys_to_keep_intermediate=["action"],
        )

        self.shape_meta = shape_meta
        self.sdf_keys = sdf_keys
        self.lowdim_keys = lowdim_keys
        self.train_mask = train_mask
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.dataset_dir = dataset_dir
        self.skip_frame = cfg.skip_frame
        self.goal_sample = cfg.goal_sample
        self.use_cache = use_cache
        self.resolution = cfg.resolution

    def get_normalizer(self, mode: str = "none", **kwargs: dict) -> LinearNormalizer:
        """Return a normalizer for the dataset."""
        normalizer = LinearNormalizer()

        # Action normalization: map to [-1, 1] based on data statistics
        stat = array_to_stats(self.replay_buffer["action"])
        normalizer["action"] = get_range_normalizer_from_stat(stat)

        # SDF image normalization: data is already in [0, 1], map to [-1, 1]
        for key in self.sdf_keys:
            normalizer[key] = get_image_range_normalizer()

        return normalizer

    def __len__(self) -> int:
        if self.is_val:
            return self.replay_buffer.n_episodes // self.skip_idx
        else:
            return len(self.sampler)

    def get_validation_dataset(self) -> "BaseImageDataset":
        """Return a validation dataset."""
        val_set = copy.copy(self)
        val_set.is_val = True
        val_dir = os.path.join(self.dataset_dir, "val")
        val_set.replay_buffer = load_replay_buffer(
            val_dir, self.use_cache, self.shape_meta
        )
        val_mask = np.ones((val_set.replay_buffer.n_episodes,), dtype=bool)
        val_set.sampler = SequenceSampler(
            replay_buffer=val_set.replay_buffer,
            sequence_length=self.val_horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=val_mask,
            skip_idx=self.skip_idx,
            goal_sample=self.goal_sample,
            skip_frame=self.skip_frame,
            keys_to_keep_intermediate=["action"],
        )
        val_set.train_mask = val_mask
        return val_set

    def _sample_to_data(self, sample: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        obs_dict = {}
        final_dict = {}

        for key in self.sdf_keys:
            # SDF data: (T, H, W, 1) float32 in [0, 1]
            # Move to channels-first: (T, 1, H, W)
            obs_dict[key] = np.moveaxis(sample[key], -1, 1).astype(np.float32)
            final_dict[key] = np.moveaxis(
                sample[f"{key}_final"], -1, 0
            ).astype(np.float32)
            del sample[f"{key}_final"]
            del sample[key]

        for key in self.lowdim_keys:
            obs_dict[key] = sample[key].astype(np.float32)
            final_dict[key] = sample[f"{key}_final"].astype(np.float32)
            del sample[f"{key}_final"]
            del sample[key]

        actions = sample["action"].astype(np.float32)

        data = {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "goal": dict_apply(final_dict, torch.from_numpy),
            "action": torch.from_numpy(actions),
            "is_early_stop": torch.from_numpy(np.array([sample["is_early_stop"]])),
            "rel_stop_idx": torch.from_numpy(np.array([sample["rel_stop_idx"]])),
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.is_val:
            epi_idx = idx * self.skip_idx
            epi_start = (
                self.replay_buffer.episode_ends[epi_idx - 1] if epi_idx > 0 else 0
            )
            epi_end = self.replay_buffer.episode_ends[epi_idx]
            val_horizon = self.val_horizon
            seq_end = min(epi_end, epi_start + val_horizon)
            sample = {}
            for key in self.sampler.keys:
                sample[key] = self.replay_buffer[key][epi_start:seq_end]
                if sample[key].shape[0] < val_horizon:
                    pad_len = val_horizon - sample[key].shape[0]
                    pad_shape = (pad_len, *np.ones_like(sample[key].shape[1:]).tolist())
                    sample_pad = np.tile(sample[key][-1:], pad_shape)
                    sample[key] = np.concatenate([sample[key], sample_pad], axis=0)
                if key in self.sampler.keys_to_keep_intermediate:
                    inter_frames = sample[key].shape[0] // self.skip_frame
                    sample_shape = list(sample[key].shape[1:])
                    sample_shape[0] = sample_shape[0] * self.skip_frame
                    sample[key] = sample[key].reshape(
                        inter_frames, self.skip_frame, *sample[key].shape[1:]
                    )
                    sample[key] = sample[key].reshape(-1, *sample_shape)
                else:
                    sample[key] = sample[key][:: self.skip_frame]
                sample[f"{key}_final"] = sample[key][-1]
                sample["is_early_stop"] = False
                sample["rel_stop_idx"] = val_horizon - 1
        else:
            sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        return data
