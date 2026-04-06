# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Interactive World Simulator for robot policy training and evaluation. Trains latent dynamics world models (VAE encoder + diffusion-based dynamics + diffusion decoder) for robot manipulation tasks. Supports real ALOHA robot hardware and MuJoCo simulation.

Forked from [Boyuan Chen's research template](https://github.com/buoyancy99/research-template). MIT license requires keeping attribution.

## Commands

### Setup
```bash
mamba env create -f conda_env.yaml && conda activate iws
uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126/
pip install -e .
git submodule update --init --recursive && uv pip install -e external/gym-aloha/
```

### Training (three-stage pipeline)
```bash
# All training goes through main.py with Hydra config overrides
# Stage 1: Autoencoder (image <-> latent)
python main.py +name=NAME algorithm.training_stage=1 dataset.dataset_dir=DATA_PATH

# Stage 2: Dynamics (predict future latents from actions)
python main.py +name=NAME algorithm.training_stage=2 algorithm.load_ae=STAGE1_CKPT

# Stage 3: Decoder finetuning (robustness to latent noise)
python main.py +name=NAME algorithm.training_stage=3 algorithm.load_ae=STAGE2_CKPT
```

A run name is required (`+name=...`). W&B entity must be set in `configurations/config.yaml` or via `wandb.entity=...`.

### Inference
```bash
# Keyboard teleoperation
python scripts/inference/teleoperate_keyboard.py "+ckpt_paths=['path/to/best.ckpt']" dataset=real_aloha_dataset

# Real robot teleoperation
python scripts/inference/teleoperate_aloha.py +scene=SCENE "+ckpt_paths=['path/to/best.ckpt']"

# Web demo
bash deploy/start_demo.sh
```

### Data
```bash
bash scripts/download_mini_data.sh       # Mini dataset (few episodes, for testing)
bash scripts/download_full_data.sh       # Full training data
bash scripts/download_checkpoints.sh     # Pretrained checkpoints -> outputs/
```

### Code Quality
```bash
pre-commit run --all-files    # Runs ruff, black, mypy, clang-format, etc.
ruff check --fix .            # Lint only
black .                       # Format only
```

## Architecture

### Entry Point & Config
- `main.py` — Hydra app entry point. Registers `eval` and `torch` OmegaConf resolvers, dispatches to local or SLURM execution.
- `configurations/` — Hydra YAML configs with three main groups: `algorithm/`, `dataset/`, `experiment/`. Root config is `config.yaml`.

### Core Package: `interactive_world_sim/`

**Experiments** (`experiments/`): Training loop orchestration via PyTorch Lightning.
- `exp_base.py` — `BaseExperiment` abstract class
- `exp_latent_dyn.py` — `LatentDynExperiment`, the main training experiment
- `__init__.py` — `exp_registry` dict maps YAML names to experiment classes

**Algorithms** (`algorithms/`): Model definitions and training logic.
- `latent_dynamics/latent_world_model.py` — `LatentWorldModel`, the main LightningModule. Implements three training stages with separate forward passes and loss functions.
- `latent_dynamics/models/` — `cm_controlnet.py` (dynamics predictor), `cm_decoder.py` (diffusion image decoder), `diffae_unet.py` (UNet backbone), `attention.py`, `embeddings.py`
- `common/` — shared components: `base_pytorch_algo.py` (LightningModule base), `diffusion_helper.py`, metrics (`fvd.py`, `fid.py`, `lpips.py`)

**Datasets** (`datasets/`): HDF5 trajectory loading, cached via Zarr.
- `latent_dynamics/real_aloha_dataset.py` — real robot data
- `latent_dynamics/sim_aloha_dataset.py` — MuJoCo simulation data

**Environments** (`environments/`): MuJoCo simulation wrappers for ALOHA tasks (PushT, grasping, sweeping, rope).

**Real World** (`real_world/`): ALOHA robot hardware interface — camera sync (`multi_realsense.py`), master/puppet control, calibration scripts.

**Utils** (`utils/`): Normalizers, samplers, replay buffers, diffusion schedulers (`cm_utils.py`), W&B logging, checkpoint management, SLURM submission, TensorRT wrapper.

### Registry Pattern
Experiments are registered by name in `experiments/__init__.py`. The YAML filename under `configurations/experiment/` must match the registry key. Same pattern for algorithms and datasets resolved through Hydra.

### Data Flow
HDF5 trajectories -> Zarr cache -> Dataset (fixed-length sequences) -> DataLoader -> LatentWorldModel (encoder -> dynamics -> decoder) -> W&B logging + checkpointing

## Linting & Type Checking

Configured in `pyproject.toml` (ruff) and `.pre-commit-config.yaml`. Key settings:
- **Ruff**: E, W, F, I, B, D (partial), SLF001, RET, RUF, PL (partial), PIE, FLY
- **Black**: default settings
- **MyPy**: `--disallow-untyped-defs --disallow-incomplete-defs --disallow-untyped-calls`
- `external/` directory is excluded from all checks
