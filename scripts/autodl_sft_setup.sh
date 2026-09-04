#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
MODEL_DIR=${1:-/root/autodl-tmp/models/Qwen3-1.7B}
PYTHON_BIN=${PYTHON_BIN:-python3}

cd "$REPO_DIR"

required_files=(
  "data/scheduler/v1/sft/train.jsonl"
  "data/scheduler/v1/sft/validation.jsonl"
  "$MODEL_DIR/config.json"
  "$MODEL_DIR/tokenizer.json"
  "$MODEL_DIR/model.safetensors.index.json"
  "$MODEL_DIR/model-00001-of-00002.safetensors"
  "$MODEL_DIR/model-00002-of-00002.safetensors"
)

for path in "${required_files[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
done

if [[ ! -x .venv/bin/python ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -e '.[scheduler-train]'
.venv/bin/python -m pip check

.venv/bin/python - <<'PY'
import json
from pathlib import Path

for split, expected in (("train", 1367), ("validation", 110)):
    path = Path(f"data/scheduler/v1/sft/{split}.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != expected:
        raise SystemExit(f"{split} sample count mismatch: {len(rows)} != {expected}")
    if any(row["target_action"] not in row["valid_actions"] for row in rows):
        raise SystemExit(f"{split} contains an invalid target action")

import accelerate
import peft
import torch
import transformers

print(
    {
        "train_samples": 1367,
        "validation_samples": 110,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "accelerate": accelerate.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
)
PY

echo "Environment is ready. Run the GPU smoke test before training."
