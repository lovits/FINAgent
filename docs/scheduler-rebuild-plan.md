# TradingAgents中央Scheduler重建实施计划

> 文档版本：v1.0
> 日期：2026-09-03
> 当前分支：`codex/rebuild-scheduler-from-original`
> 原始代码基线：`a33fd4c`
> 文档性质：从原始TradingAgents重新建设的执行计划，不代表Scheduler代码已经实现

## 1. 当前起点

当前工作树中的受版本管理文件已经与原始提交`a33fd4c`完全一致。

已撤销并从当前代码中移除：

- `0f8f917`中的第一版Scheduler框架；
- `d9d6e41`中的第二版轨迹与数据流水线；
- 原有Scheduler设计、训练、运行和AutoDL文档；
- 原有`tradingagents/scheduler/`与`training/scheduler/`代码；
- 原有Scheduler测试、配置和任务数据；
- 原有`TradingAgents-Harness-Architecture*`可视化产物。

基线验证结果：

```text
576 passed
2 skipped
69 subtests passed
```

GitHub远程`origin/main`仍指向`d9d6e41`。本轮没有改写或强推远程历史；所有重建工作先在当前`codex/`分支完成。

## 2. 唯一目标

在不改写Expert Agent专业逻辑的前提下，为TradingAgents增加一个可训练的中央Agent编排Scheduler。

Scheduler只完成一件事：

```text
读取当前AgentState、Agent说明书、执行历史、剩余预算和valid_actions
→ 选择下一位Expert Agent
→ 或在完成条件满足时选择STOP
```

Scheduler不负责：

- 生成Market、News、Sentiment或Fundamentals报告；
- 生成Bull/Bear或Risk观点；
- 直接生成交易结论；
- 选择或调用Expert内部Tool；
- 修改Expert Agent自身Prompt与模型参数。

## 3. 系统只分两个阶段

后续设计文档、技术文档、代码目录和流程图必须统一使用下面两个阶段，不再创建互相重叠的阶段体系。

### 阶段一：训练前数据生成

```text
任务池
├── Static：运行原始LangGraph固定路径
└── Teacher：API模型动态选择下一位Expert Agent
        ↓
Expert Agents真实执行并自行调用Tool
        ↓
保存完整AgentState、报告、动作和成本轨迹
        ↓
结构审核 + 任务内Static/Teacher事后配对
        ↓
生成Action-only SFT数据
```

阶段一的终点是经过审核、可追踪、无数据泄漏的SFT训练集和Validation集。阶段一不训练Qwen模型。

### 阶段二：Scheduler训练与评测

```text
Qwen3-1.7B
→ LoRA SFT冷启动
→ SFT离线验证与真实Harness验证
→ 每个任务在线采样4条Learned轨迹
→ 计算终局Reward和组内相对优势
→ GRPO-style LoRA更新
→ Static / Teacher / Learned统一评测
```

阶段二的终点是可加载的本地Scheduler Adapter、训练记录和A/B评测报告。

计划中的WP0、WP1等只是代码开发工作包，不是新的系统阶段。

## 4. 最终保留的三种编排模式

目标运行接口统一为：

```text
orchestration_mode = static | teacher | learned
```

| 模式 | 决策者 | 主要用途 | 是否训练 |
|---|---|---|---|
| `static` | 原始LangGraph固定边和条件边 | 稳定生产路径、数据基准、A/B基线、回退 | 否 |
| `teacher` | Strong Teacher API | 生成动态编排候选、人工检查或调试 | 否 |
| `learned` | Qwen3-1.7B + Scheduler Adapter | 本地动态Agent编排 | 是 |

三种模式复用相同的Expert Agent、Tool、AgentState与完成条件。不同模式只替换“下一位Agent由谁选择”。

## 5. 必须永久保留的原始系统

原始Static LangGraph不能被改造成Scheduler，也不能被删除。

它继续负责：

- 原始分析师固定执行顺序；
- Bull/Bear固定辩论；
- Research Manager汇总；
- Trader生成交易方案；
- Aggressive/Conservative/Neutral风险讨论；
- Portfolio Manager形成最终决策；
- ToolNode在各Analyst内部往返；
- 结果报告、checkpoint和现有记忆机制。

动态模式使用另一张图：

```text
Scheduler
→ 选中的Expert Agent
→ 该Agent内部ToolNode与返回逻辑
→ Scheduler
→ 重复直到STOP
```

这样Static基线与动态Scheduler互不污染，也能进行公平A/B实验。

## 6. 新文档体系

旧设计不恢复。后续只建立以下文档，并形成单向引用关系：

```text
scheduler-two-stage-design.md            唯一总设计入口
├── scheduler-data-generation-design.md  阶段一细化
├── scheduler-training-design.md         阶段二细化
├── scheduler-technical-design.md        技术栈、模块、接口与代码映射
├── scheduler-rebuild-plan.md            本实施计划
└── scheduler-autodl-runbook.md          代码完成后的真实执行命令
```

总设计文档负责冻结机制与字段；阶段文档负责解释细节；技术文档负责把设计映射到代码；计划文档负责安排实现顺序；Runbook只保存已经验证可执行的命令。

## 7. 文档对齐规则

以下内容只能有一个定义，其他文档通过链接引用，不允许复制后产生多个版本：

| 内容 | 唯一来源 |
|---|---|
| 两阶段边界、三模式语义、Scheduler职责 | 总设计文档 |
| Static/Teacher数据生成、审核和配对 | 数据生成文档 |
| Qwen3-1.7B、SFT、Reward、GRPO、A/B | 训练文档 |
| JSONL字段、Schema版本和生产者/消费者 | 总设计文档的字段契约章节 |
| Python模块、配置键和CLI | 技术文档 |
| 工作顺序、验收门槛和测试 | 本计划 |
| 实际服务器命令 | Runbook |

任何代码字段或默认值变化，都必须先更新唯一来源，再同步代码与测试。

## 8. 第一阶段设计要求：数据生成

### 8.1 任务池

任务只描述需要运行的市场场景，不提供正确Agent路径或最终交易答案。

第一版目标：

- 建立300条任务种子；
- 从中冻结60条Train任务，每个场景族10条；
- 另冻结12条Validation任务，每个场景族2条；
- 同一个ticker与日期不能跨Train和Validation；
- 只使用决策日及以前的信息构造任务。

场景族：

```text
earnings_window
positive_momentum
negative_momentum
high_volatility
volume_shock
quiet_control
```

### 8.2 Static路线

Static路线真实运行原始LangGraph。Recorder监听节点事件，但不得改变节点、Prompt、Tool或条件边。

Static轨迹需要同时保存：

- 原始节点执行记录；
- 每次Expert执行前后的完整业务状态；
- 从固定节点顺序投影得到的Scheduler状态—动作Transition；
- 完整报告、Trader结果与Portfolio结果；
- Agent调用量、Tool调用量、Token和延迟；
- 失败原因和数据稀疏标记。

ToolNode不是中央Scheduler动作，只属于被调用Expert Agent的内部执行。

### 8.3 Teacher路线

Teacher与Static使用相同任务、数据快照、Expert模型、Prompt、Tool和执行预算，但从独立初始状态开始。

Teacher每次只接收：

- Scheduler职责；
- Agent Catalog；
- 当前完整业务AgentState；
- 已执行动作与剩余预算；
- 硬依赖和完成条件；
- 当前代码计算的`valid_actions`；
- 单动作JSON输出Schema。

Teacher每次只输出：

```json
{"action": "<ACT_NEWS>"}
```

首次输出非法时，Agent不执行、状态不变化，并把错误类型和同一组`valid_actions`反馈给Teacher；只允许纠正一次。第二次仍非法则拒绝整条Teacher轨迹。

### 8.4 审核与配对

Static和Teacher先独立生成、独立审核，再按`task_id`事后配对。

Teacher正样本至少满足：

- 每个动作都位于当时的`valid_actions`；
- 状态确实由上一位Expert真实执行得到；
- 最终只有一个STOP；
- Research Plan、Trader与Portfolio结果完整可解析；
- Trader的Buy/Hold/Sell与Static一致；
- Portfolio五级评级与Static最多相差一级；
- 总成本不超过Static的1.5倍；
- 不存在无进展循环或超过最大步数。

Static是稳定参照，不被描述为绝对金融真值。

### 8.5 SFT数据

一条完整轨迹拆成多个决策样本：

```text
当前完整Scheduler状态 + valid_actions → selected_action
```

只保留合法动作。Teacher首次非法输出、Agent报告和Tool输出都不能成为Scheduler标签。

采样策略：

```text
前10%更新：Static 50% / Teacher 50%
剩余90%更新：Static 20% / Teacher 80%
```

## 9. 第二阶段设计要求：模型训练

### 9.1 基模

固定模型：

```text
Hugging Face ID: Qwen/Qwen3-1.7B
Revision: 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
架构: Qwen3ForCausalLM
上下文: 32,768 Token
官方仓库文件总量: 4,079,450,110 bytes
```

模型权重只下载到AutoDL数据盘或Hugging Face缓存，不进入Git仓库。

### 9.2 SFT

第一版使用BF16 LoRA，不做全参数微调。

建议起始配置：

```text
LoRA rank = 16
LoRA alpha = 32
LoRA dropout = 0.05
target modules = q_proj, k_proj, v_proj, o_proj
micro batch = 1
gradient accumulation = 16
epochs = 2
learning rate = 5e-5
max length = 32768
gradient checkpointing = true
packing = false
```

13个Agent Action注册为单Token，只训练这些新增Token对应的Embedding行，不训练整张Embedding与LM Head。

Loss只作用于`selected_action + EOS`，Prompt、AgentState、报告、执行历史和Tool observation全部使用`-100`屏蔽。

### 9.3 SFT进入RL的门槛

必须同时通过：

- Validation Action Top-1与Mask内准确率；
- STOP Precision和Recall；
- 所有13个动作的覆盖情况；
- 真实Harness任务完成率；
- 非法动作率与fallback率；
- Adapter保存、重新加载和推理一致性；
- 2K、8K、16K、32K逐级forward/backward显存Smoke。

### 9.4 GRPO-style在线训练

SFT完成后：

```text
πref = 冻结的最佳SFT策略
πold = 生成当前Rollout批次的策略快照
πθ   = 正在更新的Scheduler策略
```

每个Train任务在线生成4条完整轨迹。每一步只在`valid_actions`范围内采样，记录`old_logprob`和`ref_logprob`，然后真实执行对应Expert Agent。

失败轨迹也必须保留并获得负Reward，不能只训练成功轨迹。

### 9.5 Reward与信用分配

Reward目标：在保持任务完成和最终决策稳定的前提下，减少冗余Agent、Tool、Token和延迟成本。

第一版分量：

```text
+ Portfolio五级评级与Static的相似度
+ Trader Buy/Hold/Sell一致
+ 唯一STOP、格式合法且无错误
+ Research Plan、Trader和Portfolio完整
- Agent调用成本
- Tool调用成本
- Token成本
- 无进展动作
- 非法动作
- 未完成
- fallback
```

采用轨迹级粗粒度信用分配：同一任务4条轨迹组内标准化Reward；一条轨迹中的所有Scheduler动作共享该轨迹的相对优势。第一版不训练Critic或逐Agent贡献模型。

Loss只计算Scheduler Action Token，使用裁剪策略目标与Reference KL。每次参数更新后必须重新采样新轨迹。

## 10. 字段契约必须重新设计

总设计文档必须从零定义以下五类数据，不复用旧代码中的字段定义。

### 10.1 `TaskSeed`

| 字段 | 类型 | 作用 |
|---|---|---|
| `task_id` | string | 全局唯一任务ID |
| `ticker` | string | 标的代码 |
| `trade_date` | `YYYY-MM-DD` | 决策日期 |
| `asset_type` | enum | 第一版固定`stock` |
| `split` | enum | `train`、`validation`或`test` |
| `seed_family` | enum | 六类代表性市场场景 |
| `data_snapshot_id` | string | 固定数据快照 |
| `selection_features` | object | 仅用于任务分层的历史特征 |

### 10.2 `SchedulerStep`

| 字段 | 类型 | 作用 |
|---|---|---|
| `step_id` | integer | 轨迹内连续编号 |
| `state_before` | object/string | 决策前完整编排状态 |
| `valid_actions` | string[] | 代码计算的合法动作集合 |
| `selected_action` | string | Static、Teacher或Learned选中的动作 |
| `agent_node` | string/null | 实际执行的Expert节点；STOP为null |
| `state_after` | object/string | Expert执行后的完整编排状态 |
| `decision_attempts` | integer | Teacher输出尝试次数，最大2 |
| `old_logprob` | number/null | RL Rollout时的旧策略动作对数概率 |
| `ref_logprob` | number/null | RL Rollout时的参考策略动作对数概率 |
| `cost` | object | Agent、Tool、Token和延迟增量 |
| `error` | string/null | 当前步骤错误 |

### 10.3 `SchedulerTrajectory`

| 字段 | 类型 | 作用 |
|---|---|---|
| `trajectory_id` | string | 全局唯一轨迹ID |
| `run_id` | string | 本次批处理ID |
| `task_id` | string | 关联任务 |
| `mode` | enum | `static`、`teacher`或`learned` |
| `policy_id` | string | 固定路径、Teacher模型或本地Adapter版本 |
| `status` | enum | `completed`、`accepted`、`rejected`、`failed`或`fallback` |
| `steps` | `SchedulerStep[]` | 全部Scheduler决策与执行证据 |
| `final_outputs` | object | Research、Trader、Portfolio输出 |
| `cost_total` | object | 全轨迹成本 |
| `reward` | object/null | GRPO阶段Reward分量 |
| `provenance` | object | 代码、Prompt、模型、数据与配置版本 |
| `failure_reason` | string/null | 失败或拒绝原因 |

### 10.4 `SFTExample`

| 字段 | 类型 | 作用 |
|---|---|---|
| `sample_id` | string | 唯一训练样本ID |
| `input_text` | string | Agent Catalog、完整状态、历史、预算和合法动作 |
| `target_action` | string | 唯一动作标签 |
| `source` | enum | `static`或`teacher_verified` |
| `task_id` | string | 防止数据泄漏和分组统计 |
| `trajectory_id` | string | 回溯原始执行 |
| `step_id` | integer | 回溯轨迹步骤 |
| `schema_version` | string | 数据兼容版本 |

### 10.5 `GRPORow`

| 字段 | 类型 | 作用 |
|---|---|---|
| `rollout_group_id` | string | 同一任务的四条候选组 |
| `trajectory_id` | string | 所属完整轨迹 |
| `step_id` | integer | 所属动作步骤 |
| `serialized_state` | string | 动作发生前输入 |
| `valid_actions` | string[] | 相同动作Mask |
| `selected_action` | string | 策略实际采样动作 |
| `old_logprob` | number | πold动作概率 |
| `ref_logprob` | number | πref动作概率 |
| `reward_total` | number | 终局Reward |
| `reward_components` | object | 可审核的Reward分量 |
| `advantage` | number | 组内相对优势 |

## 11. 基于原始代码的修改位置

### 11.1 必须保留并最小修改的现有文件

| 原始文件 | 计划修改 |
|---|---|
| `tradingagents/graph/setup.py` | 保留原Static构图；新增独立Scheduler图构建入口 |
| `tradingagents/graph/trading_graph.py` | 增加三模式选择、Policy装载、轨迹Recorder和fallback |
| `tradingagents/agents/utils/agent_states.py` | 只增加Scheduler运行字段，不改业务报告字段 |
| `tradingagents/default_config.py` | 增加Scheduler模式、预算、模型与轨迹配置 |
| `cli/main.py` | 增加`static/teacher/learned`模式参数 |
| `pyproject.toml` | 增加独立`scheduler-train`可选依赖，不污染核心安装 |

### 11.2 计划新增的运行模块

```text
tradingagents/scheduler/
├── actions.py              13个动作Token与节点映射
├── registry.py             Agent说明书与依赖
├── policy.py               Static/Teacher/Learned统一接口
├── action_mask.py          硬合法性约束
├── state_serializer.py     32K完整业务状态输入
├── scheduler_node.py       动态图中央节点
├── teacher_policy.py       Teacher API决策与一次纠正
├── hf_policy.py            本地Qwen动作概率与采样
├── trajectory.py           Canonical轨迹Schema
├── recorder.py             State/Action/Observation/Cost记录
└── store.py                JSONL与Manifest持久化
```

### 11.3 计划新增的数据与训练模块

```text
training/scheduler/
├── task_seeds.py           任务分层与split冻结
├── generate_static.py      原LangGraph轨迹投影
├── generate_teacher.py     Teacher动态完整轨迹
├── audit.py                独立结构审核
├── pair.py                 Static/Teacher事后配对
├── build_sft.py            Action-only数据转换
├── model.py                Qwen3-1.7B与LoRA加载
├── collator.py             Action-only Loss Mask
├── train_sft.py            SFT训练
├── rollout.py              四轨迹在线采样
├── reward.py               终局质量与成本Reward
├── advantage.py            组内相对优势
├── grpo_loss.py            Clip与Reference KL
├── train_grpo.py           GRPO-style更新
└── evaluate.py             Static/Teacher/Learned评测
```

模块名将在技术文档冻结后再创建；实现过程中不得先写代码、后倒推文档。

## 12. 实施工作包

### WP0：恢复原始基线——已完成

- 当前代码内容与`a33fd4c`一致；
- 新建独立`codex/`分支；
- 原始测试全部通过；
- 旧Scheduler代码与设计已移除。

### WP1：冻结两阶段总设计与字段Schema

产物：

- `scheduler-two-stage-design.md`；
- JSON Schema或等价Python数据类设计；
- 完整端到端流程图；
- 文档一致性清单。

退出条件：三种模式、13个动作、状态字段、轨迹字段、审核条件、SFT字段与GRPO字段无歧义。

### WP2：编写两份阶段细化文档

产物：

- `scheduler-data-generation-design.md`；
- `scheduler-training-design.md`。

退出条件：两份文档与总设计字段逐项一致，不重复定义冲突默认值。

### WP3：编写技术设计

产物：`scheduler-technical-design.md`。

必须包含：

- 原始运行调用链；
- 目标控制面/执行面；
- Python模块与依赖；
- 三种模式装配方式；
- 数据生产者与消费者；
- 配置、CLI、目录和异常语义；
- 每项改造对应的源文件和测试。

退出条件：技术文档中不存在当前依赖版本不支持的虚构API。

### WP4：实现Scheduler最小运行骨架

先实现动作、Registry、Policy、Action Mask、SchedulerNode与第二张动态图。

退出条件：

- Static输出与原基线回归一致；
- 可用确定性Mock Policy运行动态图；
- ToolNode始终属于Expert内部；
- 非法动作无法进入Expert节点；
- 最大步数和无进展循环能够终止。

### WP5：实现阶段一数据生成

依次实现：

1. 任务种子与split；
2. Static真实轨迹采集；
3. Teacher单动作策略；
4. 一次纠正；
5. 独立审核；
6. 事后配对；
7. SFT转换与Manifest。

退出条件：使用Mock Expert和Mock Teacher可以完整生成、拒绝、续跑和审计；真实API Key仍不是单元测试前置条件。

### WP6：生成首批真实数据

先运行少量Pilot，确认成本和字段，再运行冻结的60个Train任务与12个Validation任务。

退出条件：

- Static与Teacher各任务最多一条候选；
- accepted/rejected均保留；
- API Key未进入任何产物；
- SFT数据可以100%回溯至原始轨迹；
- Train与Validation无ticker/date泄漏。

### WP7：适配Qwen3-1.7B并完成SFT

在AutoDL下载官方模型，冻结revision，完成2K→8K→16K→32K Smoke，再运行LoRA SFT。

退出条件：最佳Adapter可重新加载，并在Validation真实Harness中稳定生成合法动作和STOP。

### WP8：实现并运行GRPO-style闭环

实现四轨迹采样、Reward、组内优势、动作Token Loss和Adapter更新。

退出条件：至少完成一次：

```text
新策略 → 新轨迹 → Reward → Advantage → GRPO更新 → Validation复测
```

### WP9：三模式接入与A/B交付

在同一任务、数据快照、Expert配置和预算下比较：

```text
Static
Teacher
SFT Scheduler
GRPO Scheduler
```

退出条件：生成完成率、决策一致性、Agent/Tool/Token成本、延迟、非法率和fallback率报告。

## 13. 测试策略

### 单元测试

- Action Token与节点一一映射；
- Action Mask依赖和STOP条件；
- Teacher JSON解析和一次纠正；
- Schema验证、ID唯一性和step连续性；
- 数据split无泄漏；
- Reward每个分量；
- 相同Reward组的优势为0；
- SFT与GRPO Loss Mask只覆盖动作Token。

### 图集成测试

- Static图没有Scheduler节点；
- Teacher/Learned图每次Expert执行后返回Scheduler；
- Analyst内部Tool调用可以往返并最终回Scheduler；
- Scheduler失败按配置停止或回退Static；
- 原始报告和记忆行为不被破坏。

### 训练Smoke

- Tiny模型执行一次SFT反向传播；
- Tiny模型执行一次GRPO反向传播；
- Qwen3-1.7B 2K、8K、16K、32K显存阶梯；
- Adapter保存、加载和动作Token一致性；
- old/ref/new logprob使用相同Action Mask。

### 全量回归

每个工作包结束后运行受影响测试；WP4、WP5、WP7、WP8和WP9结束后运行完整测试套件。原始`576 passed, 2 skipped`是最低回归基线。

## 14. 实现纪律

- 不直接修改原Static图的业务顺序；
- 不把Teacher输出当作未经验证的正确答案；
- 不在数据阶段训练模型；
- 不在训练阶段重新定义数据Schema；
- 不把Tool纳入中央Scheduler动作空间；
- 不静默截断超过32K的完整状态；
- 不使用Test集调Reward或超参数；
- 不把API Key、模型权重、轨迹大文件或checkpoint提交到Git；
- 不声称尚未真实运行的训练效果；
- 不因为某个Mock测试通过就声称Qwen3训练已经可用。

## 15. 当前状态与下一步

| 项目 | 状态 |
|---|---|
| 恢复原始`a33fd4c`代码内容 | 已完成 |
| 原始测试回归 | 已完成：576 passed，2 skipped |
| 删除旧设计与Scheduler代码 | 已完成 |
| 重建原始代码CodeGraph索引 | 已完成：139 files，1,894 nodes，4,056 edges |
| 新重建计划 | 本文已完成 |
| 两阶段总设计 | 下一步 |
| 数据生成细化设计 | 未开始 |
| 训练细化设计 | 未开始 |
| 技术设计 | 未开始 |
| 新总流程图 | 未开始 |
| Scheduler代码重建 | 未开始 |
| Hugging Face模型下载 | 未开始，只在AutoDL执行 |
| SFT / GRPO训练 | 未开始 |

下一步严格按顺序执行：

```text
总设计与字段契约
→ 数据生成文档
→ 训练文档
→ 技术文档
→ 总流程图
→ 你确认设计
→ 按WP4开始修改代码
```
