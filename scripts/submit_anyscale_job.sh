#!/bin/bash
# Submit an Anyscale training job, loading W&B API key from ~/.wandb/keys.
# Usage: bash scripts/submit_anyscale_job.sh [extra anyscale flags...]

set -euo pipefail

# Load WANDB_API_KEY from local keys file
KEYS_FILE="${HOME}/.wandb/keys"
if [ ! -f "$KEYS_FILE" ]; then
    echo "Error: W&B keys file not found at $KEYS_FILE"
    echo "Create it with: echo 'WANDB_API_KEY=your_key_here' > ~/.wandb/keys"
    exit 1
fi
source "$KEYS_FILE"

if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "Error: WANDB_API_KEY not set in $KEYS_FILE"
    exit 1
fi

anyscale job submit \
    -f anyscale_job.yaml \
    --env "WANDB_API_KEY=${WANDB_API_KEY}" \
    "$@"
