# 训练数据、评测产物与本地模型快照

## GitHub代码

代码、配置、测试和设计文档位于仓库`main`分支：

https://github.com/lovits/TradingAgents-RL

## GitHub数据快照

训练数据、轨迹、日志、指标和评测报告位于Release：

https://github.com/lovits/TradingAgents-RL/releases/tag/training-data-2026-09-06

Release资产：

- `training-data.tar.zst`：完整`data/`目录；
- `training-artifacts-no-models.tar.zst`：移除模型文件后的`artifacts/`目录；
- `SHA256SUMS`：两个压缩包的SHA-256。

该Release明确不包含模型权重、优化器状态、`.env`、API Key、虚拟环境和机器缓存。

下载并恢复：

```bash
gh release download training-data-2026-09-06 \
  --repo lovits/TradingAgents-RL \
  --pattern 'training-*' --pattern 'SHA256SUMS'
shasum -a 256 -c SHA256SUMS
zstd -dc training-data.tar.zst | tar -xf -
zstd -dc training-artifacts-no-models.tar.zst | tar -xf -
```

已从GitHub重新下载并验证：两个压缩包校验通过、可完整解压，包含5181个数据条目和1644个非模型产物条目；没有`.safetensors`、`.pt`、`.bin`、`.env`或私钥文件。

## 本地模型

模型只保存在本地项目，不上传GitHub：

- 基模：`dist/autodl-scheduler/models/Qwen3-1.7B/`；
- 六份SFT模型：`artifacts/scheduler/qwen3-1p7b/sft-six-segments-20260905/`；
- 五份RL模型：`artifacts/scheduler/qwen3-1p7b/grpo-three-rounds-20260905/`；
- 校验清单：`artifacts/scheduler/model-sync-20260906/SHA256SUMS`。

本地已对基模两个权重分片、十一份SFT/RL模型的权重和训练状态进行校验，共24个大文件；其SHA-256与服务器完全一致。

模型检查点是PEFT/LoRA适配器，加载时仍需搭配本地Qwen3-1.7B基模。模型目录包含Tokenizer、Adapter配置、训练状态和训练清单。

## 同步边界

- GitHub保存可公开、可复现的代码和数据快照；
- 本地与AutoDL保存模型和训练状态；
- GitHub Release是快照，不会跟随本地文件自动实时变化；新增数据需要发布新Release或更新现有资产；
- `.env`和API Key始终由各机器单独配置。
