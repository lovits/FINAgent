# TradingAgents Scheduler训练设计

> 阶段：第二阶段——Scheduler训练与评测
> 版本：v1.0
> 日期：2026-09-03
> 基模：`Qwen/Qwen3-1.7B`
> 硬件：单张RTX 4090 24GB

关联文档：

- [数据生成设计](scheduler-data-generation-design.md)
- [完整技术设计](scheduler-technical-design.md)
- [代码优化实施计划](scheduler-implementation-plan.md)

## 1. 训练目标

训练一个本地中央Scheduler，根据当前TradingAgents状态选择下一位Expert Agent。

模型输入：

```text
Agent Catalog
+ Completion Contract
+ 当前完整AgentState
+ Scheduler动作历史
+ 已使用与剩余预算
+ valid_actions
```

模型输出：

```text
一个Agent Action Token
```

训练路线：

```text
审核后的Static/Teacher监督数据
→ LoRA SFT
→ SFT真实Harness验证
→ 在线Learned Rollout
→ 终局Reward
→ 组内相对优势
→ GRPO-style LoRA更新
→ Static/Learned A/B评测
```

## 2. 为什么先SFT再GRPO

SFT先让小模型掌握：

- 13个动作Token；
- Agent依赖和完成信号；
- 基本协作顺序；
- 合法STOP时机；
- Teacher给出的动态路径模式。

GRPO再优化：

- 相同任务下哪条完整调用轨迹更好；
- 如何在保持最终输出质量的同时减少冗余调用；
- 如何避免无进展循环、非法动作和过早STOP。

这对应DeepSeek-R1“先建立可读、稳定的cold-start行为，再进入强化学习”的阶段思想，但本项目只训练Agent动作，不训练长思维链。[DeepSeek-R1](https://arxiv.org/abs/2501.12948)

## 3. 基模配置

### 3.1 模型信息

| 项目 | 固定值 |
|---|---|
| Hugging Face ID | `Qwen/Qwen3-1.7B` |
| Revision | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| 架构 | `Qwen3ForCausalLM` |
| 参数量 | 1.7B |
| Transformer层数 | 28 |
| 原生上下文 | 32,768 Token |
| 官方仓库文件总量 | 4,079,450,110 bytes |
| 精度 | BF16 |

模型来源：[Qwen3-1.7B官方模型卡](https://huggingface.co/Qwen/Qwen3-1.7B)

### 3.2 下载位置

模型下载到AutoDL数据盘：

```text
/root/autodl-tmp/models/Qwen3-1.7B/
```

训练配置同时保存模型ID和revision。代码只允许从该目录或Hugging Face缓存读取权重，不把模型文件提交到Git。

### 3.3 推理形式

Qwen输入统一通过`tokenizer.apply_chat_template(..., enable_thinking=False, add_generation_prompt=True)`构造。Scheduler不使用自由文本`generate()`生成理由；运行时读取Chat Template末尾位置的logits，只抽取当前`valid_actions`对应的Token，再进行greedy选择或温度采样。

Qwen3的Thinking内容不进入训练标签，SFT、GRPO和在线推理复用同一Prompt Builder与Chat Template参数。

## 4. 动作Token

13个Scheduler动作注册为Tokenizer特殊Token：

```text
<ACT_MARKET>
<ACT_SENTIMENT>
<ACT_NEWS>
<ACT_FUNDAMENTALS>
<ACT_BULL>
<ACT_BEAR>
<ACT_RESEARCH_MANAGER>
<ACT_TRADER>
<ACT_AGGRESSIVE>
<ACT_CONSERVATIVE>
<ACT_NEUTRAL>
<ACT_PORTFOLIO_MANAGER>
<ACT_STOP>
```

训练前验证每个动作编码后恰好是一个Token。

Qwen3词表较大，因此只训练13个新增Token对应的Embedding行。使用PEFT的`trainable_token_indices`，其余Embedding和LM Head保持冻结。[PEFT LoRA文档](https://huggingface.co/docs/peft/main/en/package_reference/lora#efficiently-train-tokens-alongside-lora)

## 5. Scheduler输入模板

```text
<SCHEDULER_ROLE>
Choose exactly one next Expert Agent action.

<AGENT_CATALOG>
每个Agent的action、purpose、reads、writes、prerequisites、completion_signal

<COMPLETION_CONTRACT>
完成条件、STOP条件、最大步数

<CURRENT_STATE>
ticker、日期、四类报告、Research Debate、Investment Plan、Trader Plan、Risk Debate、Final Decision、past_context

<EXECUTION>
history、step、remaining_steps、no_progress_count

<VALID_ACTIONS>
当前硬Mask允许的动作

<SCHEDULER_ACTION>
```

监督标签示例：

```text
<ACT_RESEARCH_MANAGER>
```

Static、Teacher、SFT、GRPO和在线推理必须复用同一个序列化器版本。

## 6. SFT数据

输入文件来自第一阶段：

```text
data/scheduler/v1/sft/train.jsonl
data/scheduler/v1/sft/validation.jsonl
data/scheduler/v1/sft/manifest.json
```

`train.jsonl`来自冻结的60个Train任务，`validation.jsonl`来自独立的12个Validation任务，均使用`scheduler-sft-v1`字段契约。训练阶段不重新拆分任务。

每条样本包含：

| 字段 | 训练用途 |
|---|---|
| `input_text` | 模型上下文 |
| `target_action` | 唯一监督标签 |
| `valid_actions` | 验证标签合法性并计算动作指标 |
| `source` | Static/Teacher分层采样 |
| `task_id` | 防泄漏与任务级统计 |
| `trajectory_id`、`step_id` | 回溯执行证据 |
| `input_token_count` | 长度过滤和分桶 |

训练加载器按真实长度动态Padding。`input_token_count > 32768`的样本不进入训练；不允许从左侧静默截断并丢失早期Analyst证据。

## 7. SFT样本混合

先用Static稳住依赖，再以Teacher动态路径为主：

```text
前10% optimizer updates：Static 50% / Teacher 50%
剩余90% optimizer updates：Static 20% / Teacher 80%
```

采样比例由分层Sampler实现，不复制JSONL样本。每次先选择Static或Teacher来源，再选择task/trajectory，最后选择其中的SchedulerStep，防止步骤更多的长轨迹天然获得更高训练权重。

Teacher数据不足时使用实际合格数量，不重复少量样本凑比例。

## 8. SFT动作分类目标

Scheduler不是自由文本生成器。SFT直接训练Prompt末尾位置的“下一动作分布”：

```text
模型输出最后一个位置的全词表logits
→ 只取13个Action Token的logits
→ 把不在valid_actions中的动作设为不可选
→ 在剩余合法动作上计算Cross-Entropy
→ target是selected_action
```

Prompt、AgentState、报告和Tool observation只提供上下文，不作为语言模型标签。每个SFT样本只有一个Scheduler决策位置产生Loss。

这个目标与在线推理和GRPO使用同一个`valid_actions`分布，避免“训练时在全词表生成、推理时只在动作集合分类”的目标错位。Agent和Tool observation是下一次决策的上下文，不是模型输出；这一边界与Search-R1把环境返回作为上下文、只优化策略动作的原则一致。[Search-R1](https://arxiv.org/abs/2503.09516)

## 9. LoRA设计

### 9.1 可训练参数

```text
冻结：Qwen3-1.7B Base Model
训练：Attention LoRA
训练：13个新增Action Token Embedding行
```

### 9.2 LoRA目标层

```text
q_proj
k_proj
v_proj
o_proj
```

第一版不训练MLP层，不做全参数微调，也不默认量化。

### 9.3 SFT初始参数

```yaml
model:
  model_id: Qwen/Qwen3-1.7B
  revision: 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
  dtype: bfloat16
  max_length: 32768
  attention_backend: sdpa
  use_cache: false

lora:
  rank: 16
  alpha: 32
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj]
  trainable_token_indices: <13个动作Token ID>

sft:
  micro_batch_size: 1
  gradient_accumulation_steps: 16
  epochs: 2
  learning_rate: 5.0e-5
  warmup_ratio: 0.05
  weight_decay: 0.01
  max_grad_norm: 1.0
  scheduler: cosine
  gradient_checkpointing: true
  packing: false
  seed: 42
```

## 10. 32K显存Smoke

在正式SFT前依次运行：

```text
2K forward/backward
→ 8K forward/backward
→ 16K forward/backward
→ 32K forward/backward
```

每一级保存：

- 输入Token数；
- 峰值显存；
- forward和backward耗时；
- Loss是否有限；
- 是否出现NaN；
- Attention后端；
- Adapter能否保存和重新加载。

32K不通过时先启用FlashAttention 2或降低实际样本长度分布；仍然OOM时才切换4-bit QLoRA。数据处理不会静默从左侧截断状态。

## 11. SFT执行流程

```text
1. 加载固定revision的Qwen3-1.7B
2. 注册13个Action Token并调整Embedding尺寸
3. 标记13个新增Token行可训练
4. 注入q/k/v/o LoRA并验证目标层命中
5. 读取Train与Validation SFT数据
6. 拒绝超过32K的样本
7. 按长度分桶并执行5:5→2:8来源采样
8. 在同一valid_actions Mask上计算下一动作Cross-Entropy
9. 每个Epoch运行Validation
10. 保存最佳Adapter、Tokenizer、配置和指标
```

## 12. SFT验证门

### 12.1 离线指标

- Validation Loss；
- Action Top-1 Accuracy；
- Mask内Action Accuracy；
- 各Agent动作Recall；
- STOP Precision；
- STOP Recall；
- Static与Teacher来源分项指标。

### 12.2 真实Harness指标

- 完整任务完成率；
- 非法动作率；
- 过早STOP率；
- 最大步数失败率；
- fallback率；
- Trader结果可解析率；
- Portfolio结果可解析率；
- 平均Agent调用数、Tool调用数、Token和延迟。

通过验证后冻结最佳SFT Adapter作为`πref`。

## 13. GRPO在线轨迹

### 13.1 策略角色

| 策略 | 作用 |
|---|---|
| `πref` | 冻结的最佳SFT策略，用于Reference KL |
| `πold` | 生成当前一批Rollout的策略快照 |
| `πθ` | 正在更新的Scheduler策略 |

### 13.2 成组采样

同一Train任务生成4条轨迹：

```text
task_id
├── rollout-1
├── rollout-2
├── rollout-3
└── rollout-4
```

每一步：

```text
序列化当前状态
→ 计算valid_actions
→ πold在合法Action Token中采样
→ 保存old_logprob
→ πref对同一动作和同一Mask计算ref_logprob
→ 执行Expert Agent及其内部Tool
→ 保存新状态与成本
→ 返回Scheduler
```

`πold`、`πref`和训练时重新计算的`πθ` logprob必须使用同一组Action Token、同一`valid_actions` Mask和同一温度变换。第一版Rollout温度为0.8，因此保存的是温度缩放后真实采样分布中的`old_logprob`，不能保存另一套未缩放概率。

直到STOP、无合法动作、超过最大步数、Expert失败或fallback。

Agent执行与训练分离：TradingAgents负责真实环境执行和轨迹，AutoDL训练器消费转换后的Scheduler步骤。这一边界与Agent Lightning的执行—训练解耦思想一致。[Agent Lightning](https://arxiv.org/abs/2508.03680)

## 14. Reward设计

### 14.1 目标

```text
保持任务完成与最终决策质量
+ 保持输出可解析和流程合法
- 冗余Agent、Tool、Token和失败成本
```

### 14.2 初始Reward分量

| 分量 | 权重或惩罚 |
|---|---:|
| Portfolio五级评级与Static相似度 | `+0.70 × similarity` |
| Trader Buy/Hold/Sell与Static一致 | `+0.30` |
| 唯一STOP、动作合法且无错误 | `+0.10` |
| Research Plan、Trader、Portfolio完整 | `+0.20` |
| 每个Agent调用 | `-0.02` |
| 每个Tool调用 | `-0.005` |
| 每1000 Token | `-0.005` |
| 每个无进展动作 | `-0.10` |
| 出现非法动作 | `-1.00` |
| 未完成 | `-1.00` |
| fallback | `-0.25`，且回退结果不获得完成与质量奖励 |

总Reward裁剪到`[-1.5, 1.2]`。

Static用于构造稳定质量参照。未来股票收益不进入第一版训练Reward。

### 14.3 `RewardBreakdown`

```json
{
  "total": 0.61,
  "portfolio_quality": 0.70,
  "trader_quality": 0.30,
  "format_compliance": 0.10,
  "completion": 0.20,
  "agent_cost": 0.22,
  "tool_cost": 0.04,
  "token_cost": 0.08,
  "no_progress": 0.00,
  "invalid": 0.00,
  "incomplete": 0.00,
  "fallback": 0.00
}
```

## 15. 轨迹级信用分配

同一任务4条轨迹先分别得到终局Reward，再在组内标准化：

```text
高于本组平均 → 正优势
低于本组平均 → 负优势
四条Reward相同 → 本组优势为0，不产生有效更新
```

一条轨迹中的所有Scheduler动作共享该轨迹优势。

这表示：完成质量高且成本低的整条Agent协作路径被整体鼓励；失败、冗余或高成本路径被整体抑制。

## 16. GRPO训练行

```json
{
  "schema_version": "scheduler-grpo-v1",
  "rollout_group_id": "...",
  "trajectory_id": "...",
  "task_id": "...",
  "step_id": 2,
  "serialized_state": "...",
  "valid_actions": ["<ACT_BULL>", "<ACT_BEAR>"],
  "selected_action": "<ACT_BEAR>",
  "old_logprob": -0.62,
  "ref_logprob": -0.77,
  "reward_total": 0.54,
  "reward_components": {},
  "advantage": 0.83
}
```

## 17. GRPO Loss语义

每个动作更新包含三部分：

1. 相对优势决定提高还是降低该动作概率；
2. Clip限制新策略单次变化过大；
3. Reference KL限制策略快速偏离SFT行为。

Loss只作用于Scheduler Action Token。AgentState、报告、Tool observation和环境返回全部只作为上下文。

如果某个任务的4条轨迹Reward完全相同，该组优势为0并跳过更新；不人为加入噪声制造梯度。

## 18. GRPO初始参数

```yaml
grpo:
  group_size: 4
  rollout_temperature: 0.8
  max_steps_per_trajectory: 16
  learning_rate: 1.0e-5
  clip_epsilon: 0.2
  kl_beta: 0.01
  micro_batch_size: 1
  gradient_accumulation_steps: 8
  epochs_per_rollout_batch: 1
  max_length: 32768
  iterations: 1-3
  seed: 42
```

每次更新后使用新策略重新生成下一轮轨迹，旧Rollout只归档，不长期重复作为on-policy数据。

## 19. 单卡4090执行策略

SFT：

```text
一份BF16 Base
+ 一份LoRA Adapter
+ Batch 1
+ Gradient Checkpointing
+ 内存高效Attention
```

GRPO不同时加载三份完整模型：

- Rollout时直接保存`πold` logprob；
- `πref`使用冻结SFT Adapter顺序计算；
- GPU训练阶段只激活`πθ` Adapter；
- 同一任务4条轨迹顺序生成；
- Expert LLM调用与Scheduler训练分时执行。

## 20. Checkpoint与产物

```text
artifacts/scheduler/qwen3-1p7b/
├── model_download/
│   └── model_manifest.json
├── smoke/
│   └── memory_report.json
├── sft/
│   ├── checkpoint-best/
│   ├── tokenizer/
│   └── metrics.json
├── rollouts/
│   ├── iteration-001.jsonl
│   └── iteration-002.jsonl
├── grpo/
│   ├── iteration-001/
│   └── best/
└── evaluation/
    └── ab_report.json
```

每个Checkpoint保存：

- Base模型ID和revision；
- Tokenizer revision与Action Token IDs；
- PyTorch、Transformers、PEFT和Accelerate版本；
- 数据Manifest；
- 完整训练参数；
- Random Seed；
- 训练与Validation指标；
- Reward配置版本。

## 21. A/B评测

在相同Validation任务、数据快照、Expert模型、Debate轮数、Risk轮数和最大步数下运行：

```text
Static LangGraph
SFT Scheduler
GRPO Scheduler
```

Teacher结果作为数据来源质量参考单独报告，不和本地模型混成同一Checkpoint指标。

评测指标：

| 维度 | 指标 |
|---|---|
| 完成 | 完整任务完成率、STOP正确率 |
| 合法 | 非法动作率、最大步数失败率、fallback率 |
| 输出 | Trader可解析率、Portfolio可解析率 |
| 稳定 | Trader一致率、Portfolio评级距离 |
| 成本 | Agent调用、Tool调用、输入/输出Token、延迟 |
| 动态性 | 与Static完全相同路径比例、不同合法路径数量 |

第一版目标：Learned模式完成率相对Static下降不超过5个百分点，同时平均Agent调用或Token成本至少降低10%。这是验收目标，最终结果以实际A/B报告为准。

## 22. 训练完成标准

### SFT完成

- Qwen3-1.7B完成32K backward；
- 13个Action Token均为单Token；
- LoRA和新增Token行产生梯度；
- 最佳Adapter可重新加载；
- Validation真实任务可以形成完整合法轨迹。

### GRPO完成

- 每个任务可生成4条带old/ref logprob的轨迹；
- Reward分量和组内优势可复算；
- Loss只覆盖Action Token；
- 至少完成一次在线Rollout→更新→Validation闭环；
- GRPO模型相对SFT没有显著降低完成率和合法率；
- Static、SFT和GRPO的A/B指标可复现。
