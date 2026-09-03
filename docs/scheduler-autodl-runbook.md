# Scheduler AutoDL RTX 4090 Runbook

This runbook starts from a fresh checkout on an AutoDL RTX 4090 instance. It does not contain credentials or a selected scheduler base model.

## 1. Prepare the environment

```bash
nvidia-smi
uv venv .venv-train --python 3.12
source .venv-train/bin/activate
uv pip install -e '.[dev,scheduler-train]'
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected CUDA check:

```text
True NVIDIA GeForce RTX 4090
```

Run the full offline verification suite before configuring paid APIs:

```bash
python -m pytest -q
ruff check tradingagents training tests cli
```

## 2. Configure secrets safely

Create a new OpenRouter key because any key pasted into chat must be treated as exposed. Set it only in the shell or an uncommitted `.env` file:

```bash
export OPENROUTER_API_KEY='REPLACE_WITH_NEW_KEY'
```

The Strong Teacher client uses:

```text
https://openrouter.ai/api/v1
google/gemini-3.8-flash
OPENROUTER_API_KEY
```

The same OpenRouter account may also serve the frozen Expert Agents. Configure their provider and models separately so Teacher and Expert versions remain auditable:

```bash
export TRADINGAGENTS_LLM_PROVIDER='openrouter'
export TRADINGAGENTS_LLM_BACKEND_URL='https://openrouter.ai/api/v1'
export TRADINGAGENTS_QUICK_THINK_LLM='REPLACE_WITH_EXPERT_MODEL'
export TRADINGAGENTS_DEEP_THINK_LLM='REPLACE_WITH_EXPERT_MODEL'
```

Never commit `.env`, print the key, or copy it into trajectory metadata.

## 3. Prepare and validate the task snapshot

The repository already contains the frozen 18-task pilot. Validate it before any model call:

```bash
python -m training.scheduler.validate_task_manifest \
  --tasks data/scheduler/tasks/static_langgraph_pilot_v1.jsonl \
  --manifest data/scheduler/tasks/static_langgraph_pilot_v1.manifest.json
```

To audit or rebuild the selection method from yfinance without using OpenRouter:

```bash
python -m training.scheduler.build_task_seeds \
  --start 2026-04-01 \
  --end 2026-08-28 \
  --output-tasks data/scheduler/tasks/static_langgraph_pilot_v1.rebuild.jsonl \
  --output-manifest data/scheduler/tasks/static_langgraph_pilot_v1.rebuild.manifest.json
```

Do not silently overwrite the frozen pilot: providers may revise historical records. Keep train, validation, and test ticker/time blocks separate when scaling beyond it.

The expanded 300-task pool is already stored in
`data/scheduler/tasks/static_langgraph_300_v1.jsonl`. Validate it with:

```bash
python -m training.scheduler.validate_task_manifest \
  --tasks data/scheduler/tasks/static_langgraph_300_v1.jsonl \
  --manifest data/scheduler/tasks/static_langgraph_300_v1.manifest.json
```

For a balanced 60-trajectory cost pilot, add these arguments to the Static generation command:

```text
--tasks data/scheduler/tasks/static_langgraph_300_v1.jsonl
--task-split train
--tasks-per-family 10
```

## 4. Generate Static Teacher trajectories

This directly streams the original compiled `setup_static_graph()` and records actual Agent, Tool, control-node, and state events. It does not run a learned graph or invent a trajectory prompt:

```bash
python -m training.scheduler.generate_data \
  --tasks data/scheduler/tasks/static_langgraph_300_v1.jsonl \
  --task-split train \
  --tasks-per-family 10 \
  --output-dir data/scheduler/static_langgraph_300_v1/train \
  --policy static \
  --trajectories-per-task 1 \
  --resume
```

This first run produces 60 balanced Train trajectories. Inspect `accepted.jsonl`, `rejected.jsonl`, and `generation_manifest.json`, then run:

```bash
python -m training.scheduler.trajectory_audit \
  --input data/scheduler/static_langgraph_300_v1/train/accepted.jsonl \
  --input data/scheduler/static_langgraph_300_v1/train/rejected.jsonl \
  --output data/scheduler/static_langgraph_300_v1/train/audit_report.json
```

After this gate passes, rerun generation without `--tasks-per-family`; `--resume` skips the first 60 and completes the 210-task Train split. Generate the 42-task Validation split in a separate directory before SFT. Do not generate Test for training.

## 5. Generate Strong Teacher candidates

This step makes paid OpenRouter calls. Run it only after setting a fresh key and approving the pilot budget:

```bash
python -m training.scheduler.generate_data \
  --tasks data/scheduler/tasks/static_langgraph_pilot_v1.jsonl \
  --output-dir data/scheduler/strong_teacher \
  --policy teacher \
  --trajectories-per-task 2
```

Gemini returns one next-Agent action per step. TradingAgents executes that Expert Agent before Gemini sees the next state. Only verifier-accepted trajectories are SFT positives.

## 6. Audit and build split-aware SFT data

This command audits accepted records again, blocks malformed positives, and writes split files plus checksums and a manifest:

```bash
python -m training.scheduler.prepare_sft_dataset \
  --input data/scheduler/static_langgraph_300_v1/train/accepted.jsonl \
  --input data/scheduler/static_langgraph_300_v1/validation/accepted.jsonl \
  --output-dir data/scheduler/sft/static_langgraph_300_v1 \
  --dataset-id tradingagents-scheduler-sft-v1
```

Rejected trajectories remain audit/error-analysis data and never become SFT positive labels. Agent reports and Tool observations remain context; only the Scheduler action and EOS are supervised.

## 7. Configure and run SFT

Copy the example and fill in an actual model ID or local path plus pinned revisions:

```bash
cp training/scheduler/configs/sft.example.json training/scheduler/configs/sft.json
```

No base model is selected by this repository. After editing the config:

```bash
accelerate launch -m training.scheduler.train_sft \
  --config training/scheduler/configs/sft.json
```

The base model remains frozen. LoRA plus the action-token embedding/output modules are trained and saved under the configured output directory. When `validation_path` is set, the trainer reports a real validation loss.

## 8. Collect grouped RL rollouts

Copy and fill in the rollout config:

```bash
cp training/scheduler/configs/rollout.example.json training/scheduler/configs/rollout.json

python -m training.scheduler.collect_rollouts \
  --config training/scheduler/configs/rollout.json
```

Each Train task produces four trajectories. The active and frozen reference policies score actions under the same valid-action mask. Output includes complete trajectories, action-level GRPO rows, Reward components, and `rollout_manifest.json`; an existing iteration directory is rejected instead of silently receiving duplicates.

## 9. Run one GRPO-style update stage

```bash
cp training/scheduler/configs/grpo.example.json training/scheduler/configs/grpo.json

accelerate launch -m training.scheduler.train_grpo \
  --config training/scheduler/configs/grpo.json
```

Only Scheduler action tokens contribute to policy loss. Expert and Tool observations remain context.

## 10. Run Static/SFT/RL evaluation

```bash
python -m training.scheduler.evaluate_policy \
  --input static=data/scheduler/static_langgraph_pilot_v1/accepted.jsonl \
  --input learned-sft=artifacts/scheduler/rollouts/sft/trajectories.jsonl \
  --input learned-rl=artifacts/scheduler/rollouts/rl/trajectories.jsonl \
  --output artifacts/scheduler/evaluations/ab-report.json
```

Report completion, invalid actions, fallback, rating/action agreement, Agent and Tool calls, tokens, latency, and path length. Do not report unmeasured improvements.

## 11. Run the trained scheduler in the application

```bash
export TRADINGAGENTS_SCHEDULER_MODE='learned'
export TRADINGAGENTS_SCHEDULER_BASE_MODEL='REPLACE_WITH_MODEL_ID_OR_PATH'
export TRADINGAGENTS_SCHEDULER_ADAPTER_PATH='artifacts/scheduler/checkpoints/grpo-iteration-001'

tradingagents analyze --scheduler-mode learned
```

If the model or adapter cannot load, the application records the reason and uses the original Static Graph when fallback is enabled.
