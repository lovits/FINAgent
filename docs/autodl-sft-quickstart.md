# AutoDL Scheduler SFT快速启动

## 1. 上传后的目录

```text
/root/autodl-tmp/
├── TradingAgents-RL/
│   ├── data/scheduler/v1/sft/train.jsonl
│   ├── data/scheduler/v1/sft/validation.jsonl
│   └── configs/scheduler/sft-qwen3-1p7b-autodl.json
└── models/
    └── Qwen3-1.7B/
        ├── config.json
        ├── tokenizer.json
        ├── model.safetensors.index.json
        ├── model-00001-of-00002.safetensors
        └── model-00002-of-00002.safetensors
```

SFT训练不需要OpenRouter API Key。

## 2. 无卡模式准备环境

```bash
cd /root/autodl-tmp/TradingAgents-RL
bash scripts/autodl_sft_setup.sh
```

脚本会校验模型分片、1683条Train和110条Validation，创建可复用基础镜像PyTorch的虚拟环境，并安装`.[scheduler-train]`依赖。

## 3. GPU模式运行显存Smoke

```bash
cd /root/autodl-tmp/TradingAgents-RL
.venv/bin/python -m training.scheduler.smoke_qwen \
  --model-path /root/autodl-tmp/models/Qwen3-1.7B \
  --lengths 2048,8192,16384,32768 \
  --output artifacts/scheduler/qwen3-1p7b/smoke/memory_report.json
```

只有四个长度都显示`passed`才进入正式训练。

## 4. 启动SFT

```bash
cd /root/autodl-tmp/TradingAgents-RL
screen -S scheduler-sft
.venv/bin/python -m training.scheduler.train_sft \
  --config configs/scheduler/sft-qwen3-1p7b-autodl.json
```

输出位于：

```text
artifacts/scheduler/qwen3-1p7b/sft/checkpoint-best/
artifacts/scheduler/qwen3-1p7b/sft/training_manifest.json
```

## 5. 训练数据口径

- Train：1683条Scheduler动作样本，162个任务；
- Validation：110条Scheduler动作样本，8个独立任务；
- Train/Validation无`task_id`重叠；
- 每轮按原始来源比例训练全部样本，每条恰好一次；
- 最大上下文32768 Token；
- 仅下一Agent动作产生Loss，报告和工具结果只作为输入上下文。

当前训练配置启用无放回全量洗牌：每轮559条Static、1124条Teacher，共1683次；两轮3366次样本呈现，约212次参数更新。无需补采样；后续通过真实新增数据扩充训练集。

每轮查看`sampling_plan.json`与`coverage-epoch-N.json`：遗漏数必须为0，实际抽取数须与计划一致。

## 6. 每轮五场景自动监督

本地已提供`training.scheduler.train_with_evaluation`入口，与独立SFT使用同一训练函数；使用`configs/scheduler/sft-cycle-qwen3-1p7b-autodl.json`并传入5个与训练/验证隔离的任务JSONL即可。每轮保存checkpoint，再执行5个Learned和5个Static任务，然后继续下一轮，优化器状态在同一进程中保留。完整进程中断后的断点恢复不是此回调的能力。

```bash
.venv/bin/python -m training.scheduler.train_with_evaluation \
  --config configs/scheduler/sft-cycle-qwen3-1p7b-autodl.json \
  --tasks artifacts/scheduler/sft-cycle-20260905/eval-tasks-5.jsonl
```

真实场景和质量评审需要`OPENROUTER_API_KEY`；纯离线SFT不需要。评审与奖励记录在每轮`scenarios/epoch-NN/quality_reviews.json`和`comparison.json`中。程序校验必需报告和合法动作，大模型按证据/逻辑/风险评分并引用原文；评审失败不会记为0分冒充坏策略。GRPO主配置复用同一自动评审，保留原始轨迹与跳过组原因。
