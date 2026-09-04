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
- 训练采样固定为20% Static和80% Teacher；
- 最大上下文32768 Token；
- 仅下一Agent动作产生Loss，报告和工具结果只作为输入上下文。
