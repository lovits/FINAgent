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

## 3. Prepare tasks

Copy the example and replace it with frozen historical tasks:

```bash
mkdir -p data/scheduler/tasks
cp data/scheduler/tasks/pilot.example.jsonl data/scheduler/tasks/pilot.jsonl
```

Each row requires `task_id`, `ticker`, `trade_date`, and `asset_type`. Keep train, validation, and test tickers/event windows separate before scaling beyond the pilot.

## 4. Generate Static Teacher trajectories

This runs the scheduler-centered graph with `StaticSchedulerPolicy`, records each decision state, and compares the result with the original Static Graph:

```bash
python -m training.scheduler.generate_data \
  --tasks data/scheduler/tasks/pilot.jsonl \
  --output-dir data/scheduler/static_teacher \
  --policy static \
  --trajectories-per-task 1
```

Inspect `accepted.jsonl` and `rejected.jsonl` before any paid Teacher batch.

## 5. Generate Strong Teacher candidates

This step makes paid OpenRouter calls. Run it only after setting a fresh key and approving the pilot budget:

```bash
python -m training.scheduler.generate_data \
  --tasks data/scheduler/tasks/pilot.jsonl \
  --output-dir data/scheduler/strong_teacher \
  --policy teacher \
  --trajectories-per-task 2
```

Gemini returns one next-Agent action per step. TradingAgents executes that Expert Agent before Gemini sees the next state. Only verifier-accepted trajectories are SFT positives.

## 6. Build SFT data

Build separate files first, then combine them after reviewing source proportions:

```bash
python -m training.scheduler.build_sft_dataset \
  --input data/scheduler/static_teacher/accepted.jsonl \
  --output data/scheduler/sft/static.jsonl

python -m training.scheduler.build_sft_dataset \
  --input data/scheduler/strong_teacher/accepted.jsonl \
  --output data/scheduler/sft/teacher.jsonl
```

Create `train.jsonl` and `validation.jsonl` using task-level splits. Do not split steps from the same trajectory across both files.

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

The base model remains frozen. LoRA plus the action-token embedding/output modules are trained and saved under the configured output directory.

## 8. Collect grouped RL rollouts

Copy and fill in the rollout config:

```bash
cp training/scheduler/configs/rollout.example.json training/scheduler/configs/rollout.json

python -m training.scheduler.collect_rollouts \
  --config training/scheduler/configs/rollout.json
```

Each task produces four trajectories. The active and frozen reference policies score actions under the same valid-action mask. The output includes complete trajectories and action-level GRPO rows.

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
  --input static=data/scheduler/static_teacher/accepted.jsonl \
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
