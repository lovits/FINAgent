# TradingAgents Scheduler执行手册

> 适用环境：本地数据生成 + AutoDL RTX 4090训练
> 模型：`Qwen/Qwen3-1.7B`

## 1. 安装与回归

本地开发环境：

```bash
uv venv --python /usr/local/bin/python3.12 .venv
uv pip install --python .venv/bin/python -e '.[dev,scheduler-train]'
.venv/bin/pytest -q
.venv/bin/ruff check tradingagents training tests/scheduler cli/main.py
```

AutoDL：

```bash
cd /root/autodl-tmp/TradingAgents-RL
python -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -e '.[scheduler-train]'
```

## 2. 配置API

使用环境变量：

```bash
export OPENROUTER_API_KEY='你的OpenRouter Key'
export TRADINGAGENTS_LLM_PROVIDER='openrouter'
export TRADINGAGENTS_QUICK_THINK_LLM='google/gemini-3.8-flash'
export TRADINGAGENTS_DEEP_THINK_LLM='google/gemini-3.8-flash'
export TRADINGAGENTS_TEACHER_MODEL='google/gemini-3.8-flash'
```

Key不写入配置、Prompt、日志、轨迹或Git。

先验证Teacher模型能够进行一次结构化Scheduler决策：

```bash
.venv/bin/python -m training.scheduler.provider_preflight \
  --model google/gemini-3.8-flash
```

交互式LangGraph不能使用带`:batch`后缀的模型ID；Batch模型只能通过OpenRouter Batch API运行。

## 3. 准备股票列表

创建一个纯文本文件，每行一个Ticker，例如：

```text
AAPL
MSFT
NVDA
AMZN
GOOGL
META
```

为了稳定选出每个场景50条任务，正式任务池建议提供至少50个、覆盖多个行业的Ticker。

## 4. 生成历史特征

```bash
.venv/bin/python -m training.scheduler.market_features \
  --tickers data/scheduler/tickers.txt \
  --start 2024-01-01 \
  --end 2026-08-31 \
  --sectors data/scheduler/sectors.json \
  --output data/scheduler/v1/features.jsonl
```

输出字段包括：

- 5日收益；
- 20日年化波动；
- 20日成交量Z-score；
- 财报窗口；
- 数据快照ID；
- 信息截止时间。

## 5. 构建300任务池

```bash
.venv/bin/python -m training.scheduler.task_seeds \
  --features data/scheduler/v1/features.jsonl \
  --output-dir data/scheduler/v1/tasks
```

输出：

```text
data/scheduler/v1/tasks/pool_300.jsonl
data/scheduler/v1/tasks/train_60.jsonl
data/scheduler/v1/tasks/validation_12.jsonl
data/scheduler/v1/tasks/manifest.json
```

## 6. 先跑一条Static Pilot

```bash
.venv/bin/python -m training.scheduler.generate \
  --tasks data/scheduler/v1/tasks/pool_300.jsonl \
  --split train \
  --limit 1 \
  --mode static \
  --run-id static-train-pilot \
  --output-dir data/scheduler/v1/static/train
```

检查：

```text
raw.jsonl
accepted.jsonl 或 rejected.jsonl
generation_manifest.json
```

Pilot通过后删除`--limit 1`运行60条Train。

## 7. 跑一条Teacher Pilot

```bash
.venv/bin/python -m training.scheduler.generate \
  --tasks data/scheduler/v1/tasks/pool_300.jsonl \
  --split train \
  --limit 1 \
  --mode teacher \
  --run-id teacher-train-pilot \
  --output-dir data/scheduler/v1/teacher/train
```

Teacher每次只输出一个Agent Action。首次非法输出会在相同AgentState上纠正一次。

Pilot通过后删除`--limit 1`运行60条Train。

## 8. 生成Validation轨迹

Static：

```bash
.venv/bin/python -m training.scheduler.generate \
  --tasks data/scheduler/v1/tasks/pool_300.jsonl \
  --split validation \
  --mode static \
  --run-id static-validation-v1 \
  --output-dir data/scheduler/v1/static/validation
```

Teacher：

```bash
.venv/bin/python -m training.scheduler.generate \
  --tasks data/scheduler/v1/tasks/pool_300.jsonl \
  --split validation \
  --mode teacher \
  --run-id teacher-validation-v1 \
  --output-dir data/scheduler/v1/teacher/validation
```

## 9. 配对Static与Teacher

Train：

```bash
.venv/bin/python -m training.scheduler.pair_dataset \
  --static data/scheduler/v1/static/train/accepted.jsonl \
  --teacher data/scheduler/v1/teacher/train/accepted.jsonl \
  --output data/scheduler/v1/paired/train.jsonl
```

Validation：

```bash
.venv/bin/python -m training.scheduler.pair_dataset \
  --static data/scheduler/v1/static/validation/accepted.jsonl \
  --teacher data/scheduler/v1/teacher/validation/accepted.jsonl \
  --output data/scheduler/v1/paired/validation.jsonl
```

## 10. 在AutoDL下载Qwen3-1.7B

```bash
cd /root/autodl-tmp/TradingAgents-RL
.venv/bin/python -m training.scheduler.download_model \
  --destination /root/autodl-tmp/models/Qwen3-1.7B \
  --model-id Qwen/Qwen3-1.7B \
  --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
```

下载完成后检查：

```text
/root/autodl-tmp/models/Qwen3-1.7B/model_manifest.json
```

## 11. 构造SFT数据

Train：

```bash
.venv/bin/python -m training.scheduler.prepare_sft \
  --static data/scheduler/v1/static/train/accepted.jsonl \
  --teacher data/scheduler/v1/teacher/train/accepted.jsonl \
  --comparisons data/scheduler/v1/paired/train.jsonl \
  --output data/scheduler/v1/sft/train.jsonl \
  --model-id /root/autodl-tmp/models/Qwen3-1.7B
```

Validation：

```bash
.venv/bin/python -m training.scheduler.prepare_sft \
  --static data/scheduler/v1/static/validation/accepted.jsonl \
  --teacher data/scheduler/v1/teacher/validation/accepted.jsonl \
  --comparisons data/scheduler/v1/paired/validation.jsonl \
  --output data/scheduler/v1/sft/validation.jsonl \
  --model-id /root/autodl-tmp/models/Qwen3-1.7B
```

## 12. 运行2K→32K显存Smoke

```bash
.venv/bin/python -m training.scheduler.smoke_qwen \
  --model-path /root/autodl-tmp/models/Qwen3-1.7B \
  --lengths 2048,8192,16384,32768 \
  --output artifacts/scheduler/qwen3-1p7b/smoke/memory_report.json
```

四个长度全部`passed`后再运行正式SFT。

## 13. 准备AutoDL训练配置

复制仓库配置，并把`base_model`改为AutoDL本地路径、`base_revision`改为`null`：

```text
configs/scheduler/sft-qwen3-1p7b.json
configs/scheduler/rollout-qwen3-1p7b.json
configs/scheduler/grpo-qwen3-1p7b.json
```

本地模型路径：

```text
/root/autodl-tmp/models/Qwen3-1.7B
```

## 14. 运行SFT

```bash
.venv/bin/python -m training.scheduler.train_sft \
  --config configs/scheduler/sft-qwen3-1p7b.json
```

最佳Adapter：

```text
artifacts/scheduler/qwen3-1p7b/sft/checkpoint-best
```

## 15. 收集第一轮GRPO轨迹

```bash
.venv/bin/python -m training.scheduler.collect_rollouts \
  --config configs/scheduler/rollout-qwen3-1p7b.json
```

首次Rollout中，Active Adapter与Reference Adapter都指向最佳SFT Adapter。每个Train任务生成4条轨迹。

## 16. 运行一次GRPO更新

```bash
.venv/bin/python -m training.scheduler.train_grpo \
  --config configs/scheduler/grpo-qwen3-1p7b.json
```

下一轮Rollout：

- Active指向上一轮GRPO Adapter；
- Reference继续指向最佳SFT Adapter；
- 输出目录改为`iteration-002`。

## 17. 生成A/B报告

```bash
.venv/bin/python -m training.scheduler.evaluate \
  --static data/scheduler/v1/static/validation/accepted.jsonl \
  --teacher data/scheduler/v1/teacher/validation/accepted.jsonl \
  --sft artifacts/scheduler/qwen3-1p7b/rollouts/sft-validation/trajectories.jsonl \
  --grpo artifacts/scheduler/qwen3-1p7b/rollouts/grpo-validation/trajectories.jsonl \
  --output artifacts/scheduler/qwen3-1p7b/evaluation/ab_report.json
```

## 18. 在应用中切换模式

Static：

```bash
.venv/bin/tradingagents analyze --orchestration-mode static
```

Teacher：

```bash
.venv/bin/tradingagents analyze --orchestration-mode teacher
```

Learned：

```bash
export TRADINGAGENTS_SCHEDULER_BASE_MODEL='/root/autodl-tmp/models/Qwen3-1.7B'
.venv/bin/tradingagents analyze \
  --orchestration-mode learned \
  --scheduler-adapter-path artifacts/scheduler/qwen3-1p7b/grpo/iteration-001
```

## 19. 完成检查

```bash
.venv/bin/pytest -q
.venv/bin/ruff check tradingagents training tests/scheduler cli/main.py
.venv/bin/pip check
```

交付产物：

- Static/Teacher数据与审核结果；
- SFT Train/Validation数据；
- Qwen下载Manifest和显存报告；
- SFT与GRPO Adapter；
- GRPO轨迹与Reward；
- Static/Teacher/SFT/GRPO A/B报告。
