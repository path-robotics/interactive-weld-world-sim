# Weld World Model

Trains a latent dynamics world model on weld deposition cross-sections. Given
a pre-bead surface (as a signed distance field image) and weld parameters, the
model predicts the post-bead surface after a single weld pass.

## Setup

```bash
# Create venv (uv downloads Python 3.11 automatically)
uv venv --python 3.11
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements-weld.txt \
    --extra-index-url https://download.pytorch.org/whl/cu126/

# Install this package
uv pip install -e .
```

Verify CUDA works:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Data Conversion

Convert the weld-world-model point-cloud data into SDF HDF5 episodes:

```bash
python scripts/convert_weld_data.py \
    --input_dir ~/Documents/weld-world-model/data/full_training_set \
    --output_dir data/weld_deposition \
    --visualize
```

This will:
- Load all samples with valid weld profiles (~40k+)
- Convert each into a 2-frame episode: pre-bead SDF → post-bead SDF
- Split 90/10 by experiment into `data/weld_deposition/{train,val}/`
- Save a `sdf_visualization.png` for inspection (pass `--vis_samples 20` for more)

### Conversion options

| Flag | Default | Description |
|------|---------|-------------|
| `--resolution` | 128 | SDF grid resolution (pixels per side) |
| `--x_min` / `--x_max` | -0.015 / 0.015 | Physical x bounds (meters) |
| `--y_min` / `--y_max` | -0.005 / 0.015 | Physical y bounds (meters) |
| `--max_dist` | 0.010 | SDF clipping distance (meters) |
| `--val_ratio` | 0.1 | Validation split fraction |
| `--seed` | 42 | Random seed for experiment split |

## Training

All training uses the three-stage pipeline via `main.py` with Hydra config
overrides. Set your W&B entity first:

```bash
# In configurations/config.yaml, set:
#   wandb:
#     entity: YOUR_WANDB_ENTITY
#
# Or pass on the command line: wandb.entity=YOUR_ENTITY
```

### Stage 1: Autoencoder

Trains the encoder (SDF → latent) and diffusion decoder (latent → SDF):

```bash
python main.py +name=weld_ae \
    dataset=weld_deposition_dataset \
    algorithm.training_stage=1 \
    algorithm.action_dim=5
```

### Stage 2: Dynamics

Trains the latent dynamics model (predict next latent from current + weld params).
Requires a Stage 1 checkpoint:

```bash
python main.py +name=weld_dyn \
    dataset=weld_deposition_dataset \
    algorithm.training_stage=2 \
    algorithm.action_dim=5 \
    algorithm.load_ae=outputs/weld_ae/checkpoints/best.ckpt
```

### Stage 3: Decoder finetuning

Finetunes the decoder for robustness to latent noise.
Requires a Stage 2 checkpoint:

```bash
python main.py +name=weld_dec \
    dataset=weld_deposition_dataset \
    algorithm.training_stage=3 \
    algorithm.action_dim=5 \
    algorithm.load_ae=outputs/weld_dyn/checkpoints/best.ckpt
```

## Data Format

Each episode is an HDF5 file with:

```
episode_N.hdf5
├── obs/images/cross_section_sdf    (2, 128, 128, 1) float32 in [0, 1]
└── action                          (2, 5) float32
```

- **Frame 0**: Pre-bead SDF (deposition surface only)
- **Frame 1**: Post-bead SDF (substrate with weld bead fused in)
- **Action**: `[travel_speed, wire_feed_speed, CTWD, work_angle, bead_area]`

SDF values: 0.5 = surface boundary, >0.5 = solid (below surface), <0.5 = air (above surface).
