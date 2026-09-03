# Scheduler轨迹、SFT与GRPO数据流水线

> 版本：v1
> 日期：2026-09-03
> 状态：代码与离线测试已完成；真实LLM轨迹尚未生成

## 1. 先分清四种数据

| 数据 | 是什么 | 当前状态 |
|---|---|---|
| Task Seed | 股票、历史日期、split和市场场景 | 300条已生成 |
| Static Trajectory | 原Static LangGraph在一个任务上的真实Agent/Tool执行记录 | 采集代码完成，真实数据未生成 |
| SFT Example | 某个状态下应选择的下一个Agent动作 | 审核与转换代码完成，等待轨迹 |
| RL Rollout | 当前小模型动态调度产生的多条候选路径 | 代码完成，等待SFT模型 |

因此，300条任务种子不是300条训练数据。必须先让原Static LangGraph真正运行，才能得到轨迹；轨迹通过审核后，才会被拆成SFT样本。

## 2. 完整流程

```text
300条任务种子
  ↓
原Static LangGraph逐任务执行
  ↓
accepted.jsonl + rejected.jsonl + generation_manifest.json
  ↓
轨迹审计：动作、节点、时间截止、STOP、输出、重复ID
  ↓
按Train/Validation/Test生成action-only SFT数据与manifest
  ↓
小模型LoRA SFT，并真实计算validation loss
  ↓
同一任务采样4条Learned Scheduler轨迹
  ↓
终局质量、格式、完成性、成本与失败Reward
  ↓
组内相对优势 + Action Token掩码 + Reference KL
  ↓
GRPO-style LoRA更新
```

## 3. 如何从LangGraph拿到轨迹

`training/scheduler/static_trace.py`直接运行原项目的`setup_static_graph()`编译结果，并订阅LangGraph的`updates + values`事件。

一条Static轨迹保存：

- `task`：ticker、trade_date、split、行业与采样场景；
- `node_steps`：真实Agent节点、ToolNode、消息清理节点及状态前后摘要；
- `scheduler_examples`：从固定路径投影得到的“状态—合法动作—实际Agent动作”；
- 四类Analyst报告、Research辩论、Trader方案、Risk辩论和Portfolio结果；
- Tool名称、执行状态、是否拿到数据和结果摘要；
- Agent/Tool调用数、Token和延迟；
- 模型、provider、代码提交和信息截止日期；
- `generation_key`、`trajectory_id`与`sample_index`。

每个任务和样本编号都有稳定`generation_key`。批次中断后增加`--resume`，已经写入accepted或rejected的任务会跳过，不会重复付费执行。若不使用`--resume`却指向已有输出，程序会拒绝追加。

Agent内部Tool调用不属于Scheduler动作。对Scheduler而言：

```text
Scheduler动作：下一步选择哪个Agent
Agent/Tool结果：环境返回的observation
```

## 4. 轨迹审核

`training/scheduler/trajectory_audit.py`提供独立、确定性的审核。

硬错误包括：

- 缺少task、split或Schema版本；
- 信息截止日期与交易日期不一致；
- LangGraph节点步号或Scheduler步号断裂；
- 目标动作不在合法动作集合；
- 动作与真实Agent节点不匹配；
- 序列化状态中的ticker/date/action mask与轨迹不一致；
- 最后没有唯一的`<ACT_STOP>`；
- Trader动作或Portfolio评级不可解析；
- accepted轨迹自身的运行质量检查失败；
- 轨迹ID重复。

警告包括数据稀缺和人工复核尚未完成。警告保留在audit report中；硬错误禁止进入SFT。

```bash
python -m training.scheduler.trajectory_audit \
  --input data/scheduler/static_langgraph_300_v1/train/accepted.jsonl \
  --input data/scheduler/static_langgraph_300_v1/train/rejected.jsonl \
  --output data/scheduler/static_langgraph_300_v1/train/audit_report.json
```

## 5. 如何转换成SFT数据

推荐使用`prepare_sft_dataset.py`，它会再次执行审核，而不是绕过审核直接转换。

```bash
python -m training.scheduler.prepare_sft_dataset \
  --input data/scheduler/static_langgraph_300_v1/train/accepted.jsonl \
  --input data/scheduler/static_langgraph_300_v1/validation/accepted.jsonl \
  --output-dir data/scheduler/sft/static_langgraph_300_v1 \
  --dataset-id tradingagents-scheduler-sft-v1
```

输出：

```text
train.jsonl
validation.jsonl（输入包含Validation轨迹时产生）
test.jsonl（输入包含Test轨迹时产生）
audit_report.json
manifest.json
```

一条SFT样本只有：

```json
{
  "input_text": "压缩后的AgentState与合法动作",
  "target_action": "<ACT_NEWS>",
  "metadata": {
    "trajectory_id": "...",
    "task_id": "...",
    "task_split": "train",
    "source": "original_static_langgraph"
  },
  "schema_version": "v1"
}
```

模型读取全部状态，但Loss只落在Scheduler Action Token和EOS上。Analyst报告、Tool结果、用户输入和环境observation的标签均为`-100`，不会被当作模型应生成的内容。

这对应两篇工作的可迁移思想：

- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)：先用少量高质量cold-start数据建立稳定输出行为，再进入RL；
- [Search-R1](https://arxiv.org/abs/2503.09516)：环境检索结果参与后续决策上下文，但检索内容本身不参与策略Token的训练损失。

本项目没有照搬长CoT。冷启动目标不是让小模型写金融分析，而只是让它学会读状态并输出一个合法Agent动作。

## 6. Static轨迹是不是唯一正确答案

不是。Static轨迹的作用是：

1. 教会Scheduler动作语法；
2. 教会最低依赖关系和正常终止；
3. 提供一条可以完成任务的安全路径；
4. 提供Static/Learned的比较基线。

它不证明固定路径最优。因此后续可以加入Strong Teacher提出、真实Harness执行并审核通过的动态路径。Teacher只提议下一个Agent，不能直接编造整条轨迹。

## 7. Reward如何打分

Reward只给Learned Scheduler生成的整条轨迹，不给单个Tool内容打分。

正向部分：

- Portfolio最终评级与Static参考的一致程度，权重0.70；
- Trader Buy/Hold/Sell与Static参考一致，权重0.30；
- 动作格式、唯一STOP和无执行错误，奖励0.10；
- Investment Plan、Trader动作和Portfolio评级完整可解析，奖励0.20。

成本与失败部分：

- 每多调用一个Agent扣0.02；
- 每次Tool调用扣0.005；
- 每1000 Token扣0.005；
- 重复无进展、非法动作、未完成和回退分别扣分。

特别规则：如果Learned Scheduler失败并回退到Static图，Static图产生的最终答案不能算成Learned Scheduler的质量。回退轨迹的质量分和完成奖励归零，只保留失败惩罚。这避免模型通过“主动失败”骗取高Reward。

这里的最终质量是可重复的Static一致性代理，不是未来市场收益真值。项目第一版不直接用单次股票涨跌训练Scheduler，以免噪声盖过协作质量。

## 8. GRPO-style信用分配

对同一个股票日期，用当前Scheduler采样4条完整路径：

```text
同一任务
├── 路径A → Reward A
├── 路径B → Reward B
├── 路径C → Reward C
└── 路径D → Reward D
```

四个Reward在组内标准化。高于组平均的路径得到正优势，低于平均的路径得到负优势。同一条轨迹内所有Scheduler动作共享该轨迹优势，这是粗粒度信用分配。

训练时只更新Scheduler动作Token：

- 当前策略提供`old_logprob`；
- 冻结SFT策略提供`ref_logprob`；
- PPO-style clipping限制单次更新幅度；
- Reference KL防止策略偏离SFT模型过快；
- Agent报告和Tool observation只作为上下文。

这属于适合简历项目的GRPO-style实现：保留同任务多轨迹、组相对优势与策略约束，不复刻DeepSeek-R1的大规模分布式训练系统。

## 9. 按计划执行

### 第一步：先生成60条Static轨迹

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

### 第二步：审核轨迹

运行第4节命令，检查`blocking_records`、数据稀缺、失败类型和accepted比例。

### 第三步：扩完Train并生成Validation

继续使用同一Train目录与`--resume`，去掉`--tasks-per-family`后补齐剩余150条Train轨迹。然后把`--task-split`改为`validation`，输出到`data/scheduler/static_langgraph_300_v1/validation`，生成42条Validation轨迹。Test此时不运行。

### 第四步：打包SFT数据

运行第5节命令，同时传入Train与Validation的accepted文件。只允许audit通过的轨迹进入SFT。

### 第五步：在AutoDL训练SFT

```bash
accelerate launch -m training.scheduler.train_sft \
  --config training/scheduler/configs/sft.json
```

训练输出同时包含`train_loss`和真实`validation_loss`。

### 第六步：收集Learned RL轨迹

```bash
python -m training.scheduler.collect_rollouts \
  --config training/scheduler/configs/rollout.json
```

Rollout配置默认只读取`train` split，防止Validation/Test进入RL训练。输出包含完整轨迹、GRPO rows与Reward manifest。

### 第七步：GRPO-style更新

```bash
accelerate launch -m training.scheduler.train_grpo \
  --config training/scheduler/configs/grpo.json
```

## 10. 当前已经完成与尚未完成

已经完成：

- 300条任务种子；
- 原Static LangGraph轨迹采集；
- 稳定轨迹ID、重复保护和中断续跑；
- Static/Learned轨迹审核；
- 按split生成SFT文件、audit report、checksum和manifest；
- action-only SFT Loss mask；
- SFT validation loss；
- Learned分组Rollout、Reward、组相对优势和GRPO-style Loss；
- Static回退Reward隔离。

尚未完成：

- 使用新API Key真实生成Static轨迹；
- 审核真实accepted/rejected比例；
- 确定并下载Scheduler基模；
- 在AutoDL运行真实SFT和GRPO；
- 获得Static/Learned A/B结果。
