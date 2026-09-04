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
export TRADINGAGENTS_QUICK_THINK_LLM='z-ai/glm-5.3-flash'
export TRADINGAGENTS_DEEP_THINK_LLM='z-ai/glm-5.3-flash'
export TRADINGAGENTS_TEACHER_MODEL='z-ai/glm-5.3-flash'
export TRADINGAGENTS_TEMPERATURE='0.0'
```

Key不写入配置、Prompt、日志、轨迹或Git。

先验证Teacher模型能够进行一次结构化Scheduler决策：

```bash
.venv/bin/python -m training.scheduler.provider_preflight \
  --model z-ai/glm-5.3-flash
```

当前方案的数据生成、Expert执行与交互调试统一使用同步模型`z-ai/glm-5.3-flash`。

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
  --static data/scheduler/v1/static/train/raw.jsonl \
  --teacher data/scheduler/v1/teacher/train/raw.jsonl \
  --output data/scheduler/v1/paired/train.jsonl
```

Validation：

```bash
.venv/bin/python -m training.scheduler.pair_dataset \
  --static data/scheduler/v1/static/validation/raw.jsonl \
  --teacher data/scheduler/v1/teacher/validation/raw.jsonl \
  --output data/scheduler/v1/paired/validation.jsonl
```

### 9.1 并行生成100条完整训练轨迹

100条指审核通过的完整轨迹，不是100个SchedulerStep。结构为20条Static和80条Teacher：20个配对任务同时运行Static/Teacher，另外60个任务只运行Teacher。因此需要80个不重复的任务。

从`reserve`复制任务时只把这一份采集计划标记为Train，原始`pool_300.jsonl`和12条Validation不变：

```bash
.venv/bin/python -m training.scheduler.collection_plan \
  --pool data/scheduler/v1/tasks/pool_300.jsonl \
  --output data/scheduler/v1/collection/sft100/tasks.jsonl \
  --paired-count 20 \
  --teacher-only-count 60 \
  --source-split reserve
```

用独立子进程并行生成，每个任务保存自己的轨迹、审核结果、报告和日志：

```bash
.venv/bin/python -m training.scheduler.batch_generate \
  --repo . \
  --tasks data/scheduler/v1/collection/sft100/tasks.jsonl \
  --output-dir data/scheduler/v1/collection/sft100/runs \
  --run-prefix sft100-v1 \
  --max-workers 100
```

`--max-workers 100`使100个轨迹任务全部同时启动，不在批处理器内部排队。`batch_manifest.json`持续记录已完成、待运行、accepted、rejected和failed数量。如果有拒绝轨迹，用`collection_plan --offset ...`从未使用的Reserve任务中生成小批量替补；不把拒绝轨迹改名为通过数据。

### 9.2 汇总为精确20:80轨迹源

```bash
.venv/bin/python -m training.scheduler.assemble_sft_sources \
  --paired-root data/scheduler/v1/collection/sft100/runs/paired \
  --teacher-only-root data/scheduler/v1/collection/sft100/runs/teacher-only \
  --output-dir data/scheduler/v1/sft100/sources \
  --paired-task-target 20 \
  --teacher-trajectory-target 80
```

汇总器只读取`accepted.jsonl`，抽取20条Static、20条配对Teacher和60条Teacher-only，同时生成`comparisons.jsonl`。数量不足时直接报错，不会用失败数据补数。

### 9.3 增量合并后续轨迹

后续批次不直接拼接SFT JSONL，而是先与已审核的规范轨迹源合并：

```bash
.venv/bin/python -m training.scheduler.extend_sft_sources \
  --base-dir data/scheduler/v1/sft100/sources \
  --paired-root data/scheduler/v1/experiments/<batch>/paired \
  --teacher-only-root data/scheduler/v1/experiments/<batch>/teacher-only \
  --output-dir data/scheduler/v1/sft-extended/sources
```

该入口只合并`accepted`轨迹，拒绝与基础数据任务重叠，为完整Static/Teacher对生成新的配对记录，并把没有Static对应项但本身审核通过的Teacher归入`teacher_audited`。完成后重新运行`prepare_sft`执行全局去重和32K过滤。

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

100条轨迹采集方案使用：

```bash
.venv/bin/python -m training.scheduler.prepare_sft \
  --static data/scheduler/v1/sft100/sources/static.accepted.jsonl \
  --teacher data/scheduler/v1/sft100/sources/teacher-paired.accepted.jsonl \
  --teacher-audited data/scheduler/v1/sft100/sources/teacher-audited.accepted.jsonl \
  --comparisons data/scheduler/v1/sft100/sources/comparisons.jsonl \
  --output data/scheduler/v1/sft/train.jsonl \
  --model-id /root/autodl-tmp/models/Qwen3-1.7B \
  --max-tokens 32768
```

每个合法SchedulerStep会变成一条SFT样本。完整报告和已执行Agent的结果位于`input_text`中；唯一监督标签是`target_action`，即下一个应调用的Agent。转换器会去重、拒绝超过32K Token的输入，并保留`task_id/trajectory_id/step_id/source`用于追溯。

Train：

```bash
.venv/bin/python -m training.scheduler.prepare_sft \
  --static data/scheduler/v1/static/train/accepted.jsonl \
  --teacher data/scheduler/v1/teacher/train/accepted.jsonl \
  --teacher-audited data/scheduler/v1/teacher/train-only/accepted.jsonl \
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

直接使用已准备的AutoDL配置：

```text
configs/scheduler/sft-qwen3-1p7b-autodl.json
```

该配置固定使用`/root/autodl-tmp/models/Qwen3-1.7B`，且`base_revision/tokenizer_revision`为`null`，避免对本地模型路径再解析Hugging Face修订号。

```text
configs/scheduler/sft-qwen3-1p7b.json
configs/scheduler/rollout-qwen3-1p7b.json
configs/scheduler/grpo-qwen3-1p7b.json
configs/scheduler/eval-sft-qwen3-1p7b.json
configs/scheduler/eval-grpo-qwen3-1p7b.json
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

先分别用SFT和GRPO Adapter在同一组12条Validation任务上各生成一条确定性轨迹：

```bash
.venv/bin/python -m training.scheduler.collect_evaluation \
  --config configs/scheduler/eval-sft-qwen3-1p7b.json

.venv/bin/python -m training.scheduler.collect_evaluation \
  --config configs/scheduler/eval-grpo-qwen3-1p7b.json
```

再比较四种模式。必须传入`raw.jsonl`，不能只使用accepted子集；这样失败任务也会进入完成率，评测器会强制四组`task_id + data_snapshot_id`完全一致。

```bash
.venv/bin/python -m training.scheduler.evaluate \
  --static data/scheduler/v1/static/validation/raw.jsonl \
  --teacher data/scheduler/v1/teacher/validation/raw.jsonl \
  --sft artifacts/scheduler/qwen3-1p7b/evaluation/sft-validation/raw.jsonl \
  --grpo artifacts/scheduler/qwen3-1p7b/evaluation/grpo-validation/raw.jsonl \
  --output artifacts/scheduler/qwen3-1p7b/evaluation/ab_report.json
```

## 18. 在应用中切换模式

直接运行时，CLI首先显示编排模式菜单，可选择`Static LangGraph`或
`Teacher Dynamic Scheduler`：

```bash
.venv/bin/tradingagents
```

显式参数会跳过菜单，适合脚本或自动化运行。

Static：

```bash
.venv/bin/tradingagents --orchestration-mode static
```

Teacher：

```bash
.venv/bin/tradingagents --orchestration-mode teacher
```

Learned：

```bash
export TRADINGAGENTS_SCHEDULER_BASE_MODEL='/root/autodl-tmp/models/Qwen3-1.7B'
.venv/bin/tradingagents \
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
