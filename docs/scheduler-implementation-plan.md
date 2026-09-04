# TradingAgents中央Scheduler代码优化实施计划

> 版本：v1.0
> 日期：2026-09-03
> 代码基线：TradingAgents 0.3.1，提交`a33fd4c`

设计依据：

- [完整技术设计](scheduler-technical-design.md)
- [数据生成设计](scheduler-data-generation-design.md)
- [训练设计](scheduler-training-design.md)

## 1. 优化目标

在原始TradingAgents上增加可训练的中央Agent Scheduler，同时保留原始Static LangGraph。

最终交付：

```text
static  → 原始固定编排
teacher → GLM 5.3 Flash动态编排
learned → Qwen3-1.7B本地动态编排
```

Teacher与Learned只选择下一位Expert Agent。Expert Agent内部继续使用原LLM、Prompt和Tool。

第一版训练范围锁定为`multi-analyst-shallow-v1`：任务输入指定1～4个必须完成的Analyst，Scheduler只学习其执行顺序和后续Agent动态路由；Shallow作为短路径、少重复讨论的Prompt目标，所有Analyst集合大小共用同一套Reward。Single与Multi不拆成不同模式，不增加Medium、Deep或模式化Reward。

## 2. 原代码的优化点

| 原始实现 | 优化内容 | 技术价值 |
|---|---|---|
| `GraphSetup.setup_graph()`只有固定拓扑 | 保留Static图，新增Scheduler动态图 | 固定基线与动态策略并存 |
| 下一节点由LangGraph边决定 | 增加`SchedulerPolicy + ActionMask` | 把编排变成可学习决策 |
| Agent能力散落在节点代码中 | 建立版本化Agent Registry | 模型明确知道每个Agent的输入、输出和依赖 |
| 没有中央决策轨迹 | 记录SchedulerStep与NodeExecution | 为SFT、GRPO和调试提供统一数据 |
| 正常运行会读写Memory Log | 数据生成使用时间过滤的只读快照 | 避免Static结果污染Teacher和历史泄漏 |
| Static/Teacher无统一数据协议 | 增加审核、配对与SFT转换 | 形成可追踪监督数据 |
| 没有本地Scheduler模型 | 接入Qwen3-1.7B LoRA | 在单卡4090上训练 |
| 没有轨迹Reward和信用分配 | 增加终局Reward、组内优势和GRPO-style Loss | 从真实Agent协作结果优化调用策略 |
| 没有编排A/B | 增加Static/SFT/GRPO同任务评测 | 量化质量、成本和动态性 |

## 3. 代码改造总路径

```text
运行协议与动作空间
→ Scheduler动态图
→ 轨迹记录
→ Static/Teacher数据生成
→ 审核、配对与SFT数据
→ Qwen3-1.7B SFT
→ 在线Rollout与GRPO
→ 三模式接入与A/B
```

系统逻辑仍然只有两个阶段：数据生成、模型训练。上面的步骤是代码落地顺序。

## 4. 第一步：建立Scheduler基础协议

### 新增文件

```text
tradingagents/scheduler/actions.py
tradingagents/scheduler/registry.py
tradingagents/scheduler/contracts.py
tradingagents/scheduler/action_mask.py
tradingagents/scheduler/prompt.py
```

### 实现内容

1. 定义13个`SchedulerAction`；
2. 建立Action Token到LangGraph节点的一一映射；
3. 为12个Expert Agent建立`AgentSpec`；
4. 定义`SchedulerContext`、`PolicyDecision`和`SchedulerPolicy`；
5. 实现确定性`compute_valid_actions()`；
6. 实现Canonical Scheduler Prompt Builder；
7. 固定Schema和Prompt版本常量。

### 测试

```text
tests/scheduler/test_actions.py
tests/scheduler/test_registry.py
tests/scheduler/test_action_mask.py
tests/scheduler/test_prompt.py
tests/scheduler/test_contracts.py
```

### 验收

- 每个动作恰好映射一个Expert节点；
- Registry覆盖所有动态Expert节点；
- ToolNode不进入动作空间；
- 所有依赖、最大步数和STOP规则有测试；
- Prompt包含Agent Catalog、完整状态、历史、预算和valid_actions。

## 5. 第二步：嵌入原始LangGraph

### 修改`tradingagents/graph/setup.py`

保留当前`setup_graph()`作为Static图。新增：

```python
def setup_scheduler_graph(
    self,
    selected_analysts,
    scheduler_policy,
    scheduler_config,
    on_decision=None,
):
    ...
```

动态图实现：

- START进入Scheduler；
- Scheduler根据动作映射进入Expert；
- Analyst需要Tool时仍进入原ToolNode；
- Analyst完成报告后回Scheduler；
- Researcher、Manager、Trader和Risk Agent完成后回Scheduler；
- STOP进入END。

### 新增`tradingagents/scheduler/state.py`

定义继承原`AgentState`的`SchedulerAgentState`，增加运行字段：

```text
scheduler_action
scheduler_step
scheduler_history
scheduler_valid_actions
scheduler_policy_id
scheduler_no_progress_count
scheduler_last_state_signature
scheduler_agent_calls
```

原`tradingagents/agents/utils/agent_states.py`保持不变；Static图继续使用原`AgentState`，只有Scheduler动态图使用扩展状态。

### 修改`tradingagents/graph/trading_graph.py`

增加：

- `orchestration_mode`解析；
- Policy Factory；
- Static或Scheduler图装配；
- Scheduler Recorder；
- 仅动态模式把模式写入checkpoint签名，Static签名保持向后兼容；
- Learned加载失败与运行失败的Static回退；
- 三模式统一输出。

### 测试

```text
tests/scheduler/test_graph_topology.py
tests/scheduler/test_scheduler_node.py
tests/scheduler/test_runtime_modes.py
tests/scheduler/test_tool_return.py
tests/scheduler/test_fallback.py
```

### 验收

- Static图节点和边与原始实现一致；
- Mock Scheduler可以选择不同合法顺序完成任务；
- 每个Expert完成后正确返回Scheduler；
- Analyst的Tool循环不被Scheduler截断；
- 无合法动作、无进展和超步数能够终止；
- fallback原因写入运行结果。

## 6. 第三步：实现统一轨迹

### 新增文件

```text
tradingagents/scheduler/recorder.py
tradingagents/scheduler/store.py
```

### 实现内容

- 保存`NodeExecution`；
- 保存`SchedulerStep`；
- 聚合`SchedulerTrajectory`；
- 统计Agent、Tool、Token和延迟；
- 保存执行状态与审核状态；
- JSONL追加写入；
- 使用`run_id + task_id + mode`保证幂等和续跑；
- 保存代码、Prompt、模型、任务和数据快照版本。

### 接入位置

- Scheduler决策前记录`state_before`与Mask；
- Policy返回后记录动作和logprob；
- Expert执行后记录`state_after`；
- LangGraph stream记录原始节点增量；
- LLM/Tool callbacks记录成本。

### 测试

```text
tests/scheduler/test_recorder.py
tests/scheduler/test_store.py
tests/scheduler/test_trajectory_schema.py
tests/scheduler/test_cost_tracking.py
```

### 验收

- 每个Scheduler动作能关联到真实Expert执行；
- STOP没有伪造Expert节点；
- step编号连续；
- 失败轨迹同样落盘；
- 同一generation key不会重复写入。

## 7. 第四步：实现第一阶段数据生成

### 新增文件

```text
training/scheduler/market_features.py
training/scheduler/task_seeds.py
training/scheduler/environment.py
training/scheduler/generate.py
training/scheduler/provenance.py
training/scheduler/audit.py
training/scheduler/pair.py
training/scheduler/pair_dataset.py
training/scheduler/build_sft.py
training/scheduler/prepare_sft.py
```

### 7.1 任务生成

- 生成300个股票×日期×场景任务；
- 冻结60 Train和12 Validation；
- 检查ticker/date跨split泄漏；
- 保存数据快照和信息截止时间。

### 7.2 Static轨迹

- 调用原始Static图；
- 通过事件流记录节点；
- 把实际Expert顺序投影为Scheduler动作；
- 增加最终STOP；
- 输出raw、accepted和failed记录。

### 7.3 Teacher轨迹

新增：

```text
tradingagents/scheduler/teacher_policy.py
```

实现：

- OpenRouter `z-ai/glm-5.3-flash`；
- Structured Output：`{"action":"<ACT_...>"}`；
- 每个任务一条候选；
- 同状态最多纠正一次；
- 每个动作后真实执行Expert；
- 不读取当前任务Static路径或答案。

### 7.4 Memory与Checkpoint

- 两条路线使用同一份、只包含`trade_date`之前记录的Memory快照；
- 数据运行期间不写全局Memory Log；
- 数据生成关闭原应用checkpoint写入，使用确定性trajectory ID和JSONL幂等续跑；
- 应用中的Static checkpoint签名保持原样，动态模式额外区分orchestration mode；
- Static与Teacher从独立初始状态启动。

### 7.5 审核与配对

- 分别执行结构审核与完成审核；
- 按`task_id + data_snapshot_id`配对；
- 比较Trader动作、Portfolio评级和成本；
- 把`static accepted`、配对通过的`teacher_verified`以及结构审核通过的`teacher_audited`转换为SFT数据；配对不一致作为A/B指标而非Teacher轨迹硬删除条件；GRPO仍只使用有Static参考的任务。

### 测试

```text
tests/scheduler/test_task_seeds.py
tests/scheduler/test_environment.py
tests/scheduler/test_generate.py
tests/scheduler/test_teacher_policy.py
tests/scheduler/test_teacher_context.py
tests/scheduler/test_memory_snapshot.py
tests/scheduler/test_provenance.py
tests/scheduler/test_audit.py
tests/scheduler/test_pairing.py
tests/scheduler/test_build_sft.py
```

### 验收

- Mock模式可以完成数据全链路；
- 真实API不是单元测试依赖；
- 一条任务最多一条Static和一条Teacher候选；
- Train/Validation无泄漏；
- 每条SFT样本能回溯至轨迹和节点证据；
- API Key不出现在任何文件中。

## 8. 第五步：适配Qwen3-1.7B并实现SFT

### 修改`pyproject.toml`

增加可选依赖：

```toml
scheduler-train = [
  "torch",
  "transformers",
  "peft",
  "accelerate",
  "datasets",
]
```

AutoDL Smoke通过后用lock文件冻结精确版本。

### 新增文件

```text
training/scheduler/model.py
training/scheduler/sft_dataset.py
training/scheduler/sft_collator.py
training/scheduler/train_sft.py
training/scheduler/evaluate_sft.py
configs/scheduler/sft-qwen3-1p7b.json
```

### 模型加载

- 下载`Qwen/Qwen3-1.7B` revision `70d244cc...`到AutoDL数据盘；
- 使用`AutoModelForCausalLM`并验证实际类型为`Qwen3ForCausalLM`；
- 注册13个动作Token；
- 使用`trainable_token_indices`训练新增Token行；
- 对`q_proj/k_proj/v_proj/o_proj`注入LoRA；
- 启用BF16、Gradient Checkpointing和内存高效Attention。

### SFT目标

```text
最后决策位置logits
→ gather 13个动作Token
→ valid_actions Mask
→ masked Cross-Entropy(target_action)
```

### 数据采样

```text
每个Epoch固定：Static 20% / Teacher 80%
```

Sampler按来源→任务→轨迹→步骤选择，避免长轨迹支配训练。

### 测试

```text
tests/scheduler/test_action_tokens.py
tests/scheduler/test_sft_dataset.py
tests/scheduler/test_sft_collator.py
tests/scheduler/test_masked_action_loss.py
tests/scheduler/test_model_adapter.py
tests/scheduler/test_sft_smoke.py
```

### 验收

- Tiny模型完成一次参数更新；
- Qwen逐级完成2K、8K、16K、32K backward；
- 超长样本被拒绝而非截断；
- LoRA目标层与13个Token行产生梯度；
- Adapter保存、加载和推理结果一致；
- Validation真实Harness能够完整结束。

## 9. 第六步：实现在线Rollout与GRPO

### 新增文件

```text
training/scheduler/rollout.py
training/scheduler/reward.py
training/scheduler/advantage.py
training/scheduler/grpo_dataset.py
training/scheduler/grpo_loss.py
training/scheduler/train_grpo.py
configs/scheduler/reward-v1.json
configs/scheduler/grpo-qwen3-1p7b.json
```

### Rollout

- 每个Train任务生成4条完整Learned轨迹；
- old/ref/new使用相同Action Token、Mask和温度定义；
- 保存成功、失败、fallback和超步数轨迹；
- 每次策略更新后重新采样。

### Reward

- Portfolio五级评级相似度；
- Trader动作一致；
- 完成与格式奖励；
- Agent、Tool、Token和无进展成本；
- 非法、未完成和fallback惩罚；
- 总Reward裁剪到`[-1.5, 1.2]`。

### 信用分配

- 同任务4条轨迹组内标准化Reward；
- 整条轨迹的Scheduler动作共享优势；
- 全组Reward相同时跳过更新；
- 第一版不训练Critic。

### GRPO Loss

- clipped importance ratio；
- Reference KL；
- 只对Scheduler动作位置更新；
- Prompt和环境Observation不进入Loss。

### 测试

```text
tests/scheduler/test_rollout.py
tests/scheduler/test_reward.py
tests/scheduler/test_advantage.py
tests/scheduler/test_grpo_dataset.py
tests/scheduler/test_grpo_loss.py
tests/scheduler/test_grpo_smoke.py
```

### 验收

- 一组4条轨迹可完整生成；
- Reward可以从轨迹重新计算；
- old/ref/new logprob定义一致；
- Tiny模型完成一次GRPO更新；
- Qwen Adapter完成至少一轮Rollout→更新→Validation。

## 10. 第七步：三模式入口与A/B

### 修改`tradingagents/default_config.py`

增加：

```text
orchestration_mode
scheduler_max_steps
scheduler_action_temperature
scheduler_fallback_enabled
scheduler_max_context_tokens
teacher_provider
teacher_model
teacher_prompt_version
scheduler_base_model
scheduler_base_revision
scheduler_adapter_path
scheduler_dtype
scheduler_device
```

### 修改`cli/main.py`

增加：

```text
--orchestration-mode static|teacher|learned
--scheduler-adapter-path PATH
```

### 新增评测

```text
training/scheduler/evaluate.py
training/scheduler/collect_evaluation.py
```

在相同任务和环境中比较：

```text
Static
Teacher
SFT Scheduler
GRPO Scheduler
```

报告：完成率、合法率、STOP、Trader/Portfolio解析、评级距离、Agent/Tool/Token成本、延迟和路径动态性。

### 验收

- 三种模式可以显式切换；
- 默认Static与原项目行为一致；
- Learned加载失败可以回退Static；
- A/B固定任务、数据、Expert配置和预算；
- 每种模式每个Validation任务恰好一条轨迹，失败轨迹不得从评测分母剔除；
- 结果文件包含完整配置和模型版本。

## 11. 提交顺序

| 提交 | 内容 | 验证 |
|---|---|---|
| 1 | 动作、Registry、Contracts、Mask、Prompt | Scheduler单元测试 |
| 2 | Static保留与Scheduler动态图 | 原始回归+图测试 |
| 3 | Recorder、Store、成本 | 轨迹与幂等测试 |
| 4 | Task、Static、Teacher生成 | Mock数据E2E |
| 5 | Audit、Pair、SFT转换 | 数据质量测试 |
| 6 | Qwen模型、SFT数据与训练 | Tiny+4090 Smoke |
| 7 | Rollout、Reward、GRPO | Tiny+真实轨迹闭环 |
| 8 | 三模式CLI与A/B | 全量测试与报告 |

每个提交保持可运行，不把全部代码堆到最后一次提交。

## 12. 完整验收清单

### 数据生成

- [x] 300任务池、60 Train、12 Validation
- [x] Static/Teacher独立运行
- [x] 每任务每路线一条轨迹
- [x] Memory和Checkpoint隔离
- [x] Teacher一次纠正
- [x] 轨迹审核与事后配对
- [x] SFT样本可回溯且无泄漏

### 训练

- [ ] Qwen3-1.7B固定revision下载
- [x] 13个单Token动作
- [ ] 32K BF16 LoRA Smoke
- [x] Masked Action SFT Tiny反向更新
- [ ] SFT真实Harness验证
- [x] 每任务4条GRPO Rollout Mock闭环
- [x] Reward、优势和Loss可复算
- [ ] 至少一次在线GRPO闭环

### 集成

- [x] Static原始图无回归
- [x] Teacher和Learned复用同一Scheduler图
- [x] Tool保持Expert内部调用
- [x] 三模式CLI和配置
- [ ] Static/SFT/GRPO公平A/B
- [ ] Adapter、Manifest、指标和轨迹产物齐全

## 13. 当前执行顺序

```text
1. 实现Scheduler基础协议
2. 加入第二张动态图并完成Static回归
3. 完成轨迹Recorder
4. 完成数据生成、审核和SFT转换
5. 在AutoDL下载并适配Qwen3-1.7B
6. 完成SFT
7. 完成Rollout、Reward和GRPO
8. 完成三模式A/B
```
