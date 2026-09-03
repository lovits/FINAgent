# TradingAgents Scheduler数据生成设计

> 阶段：第一阶段——训练前数据生成
> 版本：v1.0
> 日期：2026-09-03

关联文档：

- [完整技术设计](scheduler-technical-design.md)
- [Scheduler训练设计](scheduler-training-design.md)
- [代码优化实施计划](scheduler-implementation-plan.md)

## 1. 阶段目标

为中央Scheduler构造“当前状态应该调用哪一个Expert Agent”的监督数据。

一条训练样本表达的是：

```text
当前完整业务状态 + 已执行历史 + 剩余预算 + 合法动作集合
→ 下一位Expert Agent，或者STOP
```

数据阶段只生产任务、执行轨迹、审核结果和SFT样本，不更新任何模型参数。

## 2. Scheduler学习边界

Scheduler学习：

- 下一位调用哪个Expert Agent；
- 哪些分析或辩论还缺失；
- 什么时候进入Research Manager、Trader和Portfolio Manager；
- 什么时候任务已经完成并输出`<ACT_STOP>`。

Scheduler不学习：

- 股票分析内容；
- Agent内部Prompt；
- Tool选择和Tool参数；
- Buy、Hold、Sell结论本身；
- Analyst、Researcher、Trader或Risk Agent的模型参数。

## 3. 两条数据路线

同一批任务分别走两条独立路线。

```text
                         ┌─ Static路线：原始LangGraph固定编排
TaskSeed + 数据快照 ─────┤
                         └─ Teacher路线：API模型动态选择Agent
                                      │
                                      ▼
                         共用原始Expert Agent与Tool
                                      │
                                      ▼
                              完整执行轨迹与报告
```

两条路线共用：

- ticker、trade_date和asset_type；
- 数据快照和信息截止时间；
- Analyst选择；
- Expert LLM配置；
- Expert Prompt；
- Tool和数据源；
- Debate、Risk和最大步数预算；
- AgentState字段；
- Agent Catalog与完成条件。

两条路线从各自的新初始状态开始。Teacher不能读取Static动作序列或Static最终答案。

### 3.1 Memory与Checkpoint隔离

原始`TradingAgentsGraph._run_graph()`会在任务开始时读取`TradingMemoryLog.get_past_context()`，在任务完成后调用`store_decision()`写入新决策。如果先跑Static再跑Teacher而不隔离，Teacher会看到Static刚写入的记忆，双路线就不再公平。

数据生成器必须在两条路线启动前冻结同一份`past_context`，只允许包含日期早于当前`trade_date`的历史记录，并在数据任务中关闭全局Memory Log写入。没有满足时间条件的历史时使用空字符串。两条路线完成后只把使用的Memory快照ID写入`provenance`。

原始Checkpoint的线程ID由ticker、日期和图结构签名生成。数据生成时采用包含`mode + run_id + task_id`的独立签名，防止Static与Teacher复用同一LangGraph checkpoint。成功轨迹清理自己的checkpoint，失败轨迹保留恢复信息但不能影响另一条路线。

### 3.2 Expert随机性与数据快照

同一任务的两条路线使用同一Expert模型配置与数据快照。Provider支持seed时记录并复用Expert seed；不支持确定性seed时记录模型、温度、请求时间和响应ID，用于解释非路由因素产生的差异。

## 4. 任务数据集

### 4.1 总规模

第一版建立300条`TaskSeed`作为任务池，再冻结：

| Split | 数量 | 用途 |
|---|---:|---|
| Train | 60 | Static/Teacher数据生成、SFT与GRPO Rollout |
| Validation | 12 | SFT选模、GRPO调参和A/B验证 |
| Reserve | 228 | 失败补位与后续扩展 |

Test集在代码训练闭环稳定后再冻结，不能从Train或Validation中抽取。

### 4.2 场景分层

Train每类10条，Validation每类2条：

| `seed_family` | 代表场景 | 选择依据 |
|---|---|---|
| `earnings_window` | 财报窗口 | 决策日附近存在财报事件 |
| `positive_momentum` | 短期上涨 | 决策日前5日收益为正 |
| `negative_momentum` | 短期下跌 | 决策日前5日收益为负 |
| `high_volatility` | 高波动 | 决策日前20日年化波动较高 |
| `volume_shock` | 成交量异常 | 决策日前成交量Z-score较高 |
| `quiet_control` | 平稳对照 | 动量、波动和成交量均不极端 |

任务筛选只使用`trade_date`当日及以前的数据。

### 4.3 `TaskSeed`字段

| 字段 | 类型 | 必填 | 含义 |
|---|---|---:|---|
| `task_id` | string | 是 | 全局唯一任务ID |
| `ticker` | string | 是 | 股票代码 |
| `trade_date` | string | 是 | `YYYY-MM-DD`决策日期 |
| `asset_type` | enum | 是 | 第一版固定`stock` |
| `split` | enum | 是 | `train`、`validation`或`reserve` |
| `seed_family` | enum | 是 | 六类场景之一 |
| `sector` | string | 是 | 行业类别 |
| `data_snapshot_id` | string | 是 | 固定市场数据快照 |
| `information_cutoff` | string | 是 | 可使用信息的截止时间 |
| `selection_features` | object | 是 | 任务分层使用的历史特征 |
| `dataset_version` | string | 是 | 任务集版本 |

`selection_features`至少包含：

```json
{
  "trailing_return_5d": 0.031,
  "annualized_volatility_20d": 0.42,
  "volume_zscore_20d": 2.17,
  "earnings_window": false
}
```

## 5. Agent动作空间

中央Scheduler只有13个动作：

| Action Token | Expert Agent | 完成信号 |
|---|---|---|
| `<ACT_MARKET>` | Market Analyst | `market_report`非空 |
| `<ACT_SENTIMENT>` | Sentiment Analyst | `sentiment_report`非空 |
| `<ACT_NEWS>` | News Analyst | `news_report`非空 |
| `<ACT_FUNDAMENTALS>` | Fundamentals Analyst | `fundamentals_report`非空 |
| `<ACT_BULL>` | Bull Researcher | Bull观点写入Research Debate |
| `<ACT_BEAR>` | Bear Researcher | Bear观点写入Research Debate |
| `<ACT_RESEARCH_MANAGER>` | Research Manager | `investment_plan`非空 |
| `<ACT_TRADER>` | Trader | `trader_investment_plan`非空 |
| `<ACT_AGGRESSIVE>` | Aggressive Analyst | Aggressive观点写入Risk Debate |
| `<ACT_CONSERVATIVE>` | Conservative Analyst | Conservative观点写入Risk Debate |
| `<ACT_NEUTRAL>` | Neutral Analyst | Neutral观点写入Risk Debate |
| `<ACT_PORTFOLIO_MANAGER>` | Portfolio Manager | `final_trade_decision`非空 |
| `<ACT_STOP>` | 终止动作 | 最终决策已经生成 |

ToolNode、消息清理节点和数据接口不属于Scheduler动作空间。

## 6. Agent Catalog

每个Expert Agent向Scheduler暴露统一说明书：

```json
{
  "action": "<ACT_NEWS>",
  "name": "News Analyst",
  "purpose": "分析公司新闻、宏观事件和内部人交易信息",
  "reads": ["company_of_interest", "trade_date", "instrument_context"],
  "writes": ["news_report"],
  "prerequisites": [],
  "completion_signal": "news_report非空",
  "tool_policy_owner": "expert_agent"
}
```

字段含义：

| 字段 | 作用 |
|---|---|
| `action` | Scheduler输出动作 |
| `name` | LangGraph Expert节点名 |
| `purpose` | Agent能力说明 |
| `reads` | Agent需要读取的业务状态 |
| `writes` | Agent执行后写入的字段 |
| `prerequisites` | 调用前硬依赖 |
| `completion_signal` | 判断Agent是否已经完成 |
| `tool_policy_owner` | 固定为`expert_agent` |

Scheduler不需要读取Expert Agent内部Prompt或Tool参数。

## 7. Static路线

### 7.1 执行机制

Static路线直接运行原始`GraphSetup.setup_graph()`：

```text
START
→ 已选择Analysts固定顺序
→ Bull/Bear Debate
→ Research Manager
→ Trader
→ Aggressive/Conservative/Neutral Risk Debate
→ Portfolio Manager
→ END
```

Analyst仍按原逻辑执行：

```text
Analyst
→ 如果产生Tool Call则进入对应ToolNode
→ Tool observation返回同一Analyst
→ 报告完成后进入消息清理节点
→ 下一个Expert
```

Recorder通过LangGraph事件流保存执行证据，不改变原图路由。

### 7.2 Static到Scheduler监督样本的投影

原图没有Scheduler节点，因此把实际Expert节点顺序投影成统一动作：

```text
执行Market Analyst前的状态       → <ACT_MARKET>
执行News Analyst前的状态         → <ACT_NEWS>
执行Research Manager前的状态     → <ACT_RESEARCH_MANAGER>
执行Portfolio Manager前的状态    → <ACT_PORTFOLIO_MANAGER>
最终决策完成后的状态             → <ACT_STOP>
```

只有真实执行过的Expert节点才能产生Static动作标签。

## 8. Teacher路线

### 8.1 执行机制

Teacher使用动态Scheduler图：

```text
START
→ Teacher Scheduler
→ 一个合法Expert Agent
→ Agent内部Tool循环
→ AgentState更新
→ Teacher Scheduler
→ ...
→ <ACT_STOP>
```

Teacher模型只选择下一动作，不生成完整Agent调用计划。

第一版Teacher固定为OpenRouter上的`google/gemini-3.8-flash`，使用其Structured Outputs能力约束单动作JSON；API Key只从`OPENROUTER_API_KEY`环境变量读取。

### 8.2 Teacher输入

每个决策点输入六部分：

```text
SCHEDULER_ROLE
AGENT_CATALOG
COMPLETION_CONTRACT
CURRENT_AGENT_STATE
EXECUTION_HISTORY_AND_BUDGET
VALID_ACTIONS
```

`CURRENT_AGENT_STATE`包含完整业务内容：

- 四类Analyst报告；
- Bull/Bear完整讨论历史；
- Research Manager计划；
- Trader计划；
- 三类Risk完整讨论历史；
- Portfolio最终决策；
- ticker、trade_date、asset_type和instrument_context；
- 原项目注入的`past_context`。

不把原始ToolMessage、原始行情表或新闻列表重复喂给Teacher；这些数据由Expert整理为报告。

### 8.3 Teacher输出

唯一合法格式：

```json
{"action": "<ACT_NEWS>"}
```

`action`必须属于当前`valid_actions`。

### 8.4 一次纠正

```text
首次合法
→ 执行Expert

首次非法
→ 不执行Expert
→ AgentState不变化
→ 返回错误类型、原输出和同一valid_actions
→ 允许再输出一次

第二次合法
→ 执行Expert

第二次非法
→ 结束并拒绝整条Teacher轨迹
```

第一次非法输出只保存在审核元数据中，不进入SFT标签。

## 9. Action Mask

Action Mask只负责硬合法性，不负责规定完整顺序。

基本规则：

- 已完成的Analyst报告对应动作不再开放；
- 至少存在一份Analyst报告后才能进入Bull/Bear；
- Research Debate形成内容后才能调用Research Manager；
- `investment_plan`存在后才能调用Trader；
- `trader_investment_plan`存在后才能调用Risk Agents；
- Risk Debate形成内容后才能调用Portfolio Manager；
- `final_trade_decision`存在后只允许STOP；
- 超过最大步数直接结束失败；
- 无进展后禁止立即重复同一动作。

Mask保证可执行性；Teacher和未来Learned模型学习在多个合法动作之间进行选择。

## 10. 完整轨迹结构

### 10.1 `SchedulerStep`

```json
{
  "step_id": 3,
  "state_before": {},
  "serialized_state": "...",
  "valid_actions": ["<ACT_BULL>", "<ACT_BEAR>"],
  "selected_action": "<ACT_BULL>",
  "agent_node": "Bull Researcher",
  "state_after": {},
  "decision_attempts": 1,
  "correction_succeeded": false,
  "observation_ref": "node-step-17",
  "cost": {
    "agent_calls": 1,
    "tool_calls": 0,
    "input_tokens": 2100,
    "output_tokens": 420,
    "latency_ms": 1800
  },
  "error": null
}
```

### 10.2 `NodeExecution`

`NodeExecution`保留LangGraph真实节点级证据，SchedulerStep只保留中央编排视角。两者通过`observation_ref`关联。

```json
{
  "node_step_id": 17,
  "node_name": "News Analyst",
  "node_type": "expert",
  "state_before": {},
  "state_update": {},
  "state_after": {},
  "tool_events": [],
  "input_tokens": 2100,
  "output_tokens": 420,
  "latency_ms": 1800,
  "error": null
}
```

`node_type`取值：

```text
expert | tool | message_cleanup | scheduler
```

### 10.3 `SchedulerTrajectory`

```json
{
  "schema_version": "scheduler-trajectory-v1",
  "trajectory_id": "...",
  "run_id": "...",
  "task_id": "...",
  "mode": "static",
  "policy_id": "static-langgraph-v1",
  "execution_status": "completed",
  "audit_status": "accepted",
  "steps": [],
  "node_executions": [],
  "final_outputs": {
    "investment_plan": "...",
    "trader_investment_plan": "...",
    "final_trade_decision": "...",
    "trader_action": "Hold",
    "portfolio_rating": "Neutral"
  },
  "cost_total": {},
  "provenance": {},
  "audit": {},
  "failure_reason": null
}
```

执行状态与审核状态分开：

| 字段 | 取值 |
|---|---|
| `execution_status` | `completed`、`failed`、`fallback`、`budget_exhausted`、`context_overflow` |
| `audit_status` | `pending`、`accepted`、`rejected`、`warning` |

### 10.4 `provenance`

至少保存：

| 字段 | 含义 |
|---|---|
| `git_commit` | 生成轨迹的代码提交 |
| `task_dataset_version` | 任务集版本 |
| `data_snapshot_id` | 数据快照 |
| `expert_provider` | Expert LLM提供方 |
| `quick_model` | Analyst/Trader模型 |
| `deep_model` | Manager模型 |
| `teacher_model` | Teacher路线使用的模型；Static为空 |
| `teacher_prompt_version` | Teacher Prompt版本 |
| `agent_catalog_version` | Agent说明书版本 |
| `action_schema_version` | 动作空间版本 |
| `state_schema_version` | 状态序列化版本 |
| `generation_config_hash` | 本次生成配置标识 |

API Key不属于`provenance`。

### 10.5 `GenerationManifest`

```json
{
  "run_id": "...",
  "mode": "teacher",
  "task_dataset_version": "tasks-v1",
  "task_count": 60,
  "trajectories_per_task": 1,
  "selected_analysts": ["market", "social", "news", "fundamentals"],
  "graph_config": {},
  "expert_models": {},
  "teacher_model": "...",
  "schema_versions": {},
  "started_at": "...",
  "completed_at": "...",
  "counts": {
    "completed": 0,
    "failed": 0,
    "accepted": 0,
    "rejected": 0
  }
}
```

## 11. 轨迹审核

### 11.1 结构审核

- ID和Schema版本存在；
- `step_id`从0连续递增；
- 每个动作属于13个动作之一；
- `selected_action`位于当时`valid_actions`；
- 动作与实际执行节点一致；
- `state_after`来自真实节点执行结果；
- STOP只出现一次并且位于最后；
- ticker、trade_date和data snapshot与任务一致。

### 11.2 业务完成审核

- 所选Analyst对应报告存在；
- `investment_plan`非空；
- `trader_investment_plan`可以解析Buy/Hold/Sell；
- `final_trade_decision`可以解析五级Portfolio评级；
- 没有无进展循环；
- 没有超过执行预算。

### 11.3 审核状态

| 状态 | 含义 | 是否进入SFT |
|---|---|---:|
| `accepted` | 结构和执行有效 | 是 |
| `rejected` | 非法、未完成或配对不合格 | 否 |
| `failed` | Agent、Tool、Provider或运行异常 | 否 |
| `warning` | 可用但存在稀疏数据等提示 | 人工确认后决定 |

### 11.4 `AuditRecord`

```json
{
  "trajectory_id": "...",
  "audit_status": "rejected",
  "checks": {
    "schema_valid": true,
    "steps_contiguous": true,
    "actions_legal": false,
    "state_progress_valid": true,
    "completion_valid": false,
    "task_snapshot_match": true
  },
  "errors": ["selected_action_not_in_valid_actions"],
  "warnings": [],
  "auditor_version": "scheduler-audit-v1"
}
```

## 12. Static与Teacher配对

配对键是`task_id + data_snapshot_id`。

配对只发生在两条路线分别完成和审核之后，用于筛选Teacher正样本与形成A/B基线。

Teacher进入`teacher_verified`需要：

- Static和Teacher自身均为`accepted`；
- Trader动作一致；
- Portfolio五级评级距离不超过一级；
- Teacher成本不超过Static的1.5倍。

Teacher与Static路径完全相同仍是有效轨迹，但重复的状态—动作样本只保留一份。

配对记录：

```json
{
  "task_id": "...",
  "data_snapshot_id": "...",
  "static_trajectory_id": "...",
  "teacher_trajectory_id": "...",
  "trader_action_match": true,
  "portfolio_rating_distance": 1,
  "teacher_to_static_cost_ratio": 0.82,
  "pair_status": "teacher_verified",
  "rejection_reasons": []
}
```

## 13. SFT-ready数据

### 13.1 样本结构

```json
{
  "schema_version": "scheduler-sft-v1",
  "sample_id": "...",
  "task_id": "...",
  "trajectory_id": "...",
  "step_id": 3,
  "source": "teacher_verified",
  "input_text": "完整Scheduler输入",
  "valid_actions": ["<ACT_BULL>", "<ACT_BEAR>"],
  "target_action": "<ACT_BULL>",
  "input_token_count": 12480
}
```

### 13.2 转换规则

- Static只转换`accepted`轨迹；
- Teacher只转换`teacher_verified`轨迹；
- 每个合法SchedulerStep转换为一条样本；
- Teacher纠正成功只保留第二次合法动作；
- 超过32,768 Token的样本标记`context_overflow`并拒绝；
- Train与Validation按`task_id`隔离后再拆样本；
- 相同`input_text + target_action`去重；
- 每条样本保留到原始轨迹的引用。

### 13.3 SFT Manifest

```json
{
  "schema_version": "scheduler-sft-manifest-v1",
  "task_split": "train",
  "sample_count": 0,
  "task_count": 0,
  "source_counts": {
    "static": 0,
    "teacher_verified": 0
  },
  "action_counts": {},
  "length_percentiles": {
    "p50": 0,
    "p90": 0,
    "p95": 0,
    "p99": 0,
    "max": 0
  },
  "rejected_context_overflow": 0,
  "deduplicated_samples": 0,
  "source_run_ids": []
}
```

## 14. 数据目录

```text
data/scheduler/v1/
├── tasks/
│   ├── pool_300.jsonl
│   ├── train_60.jsonl
│   ├── validation_12.jsonl
│   └── manifest.json
├── static/
│   ├── raw.jsonl
│   ├── accepted.jsonl
│   ├── rejected.jsonl
│   └── audit_report.json
├── teacher/
│   ├── raw.jsonl
│   ├── accepted.jsonl
│   ├── rejected.jsonl
│   └── audit_report.json
├── paired/
│   └── comparison.jsonl
└── sft/
    ├── train.jsonl
    ├── validation.jsonl
    └── manifest.json
```

## 15. 阶段完成标准

第一阶段完成时必须得到：

1. 固定版本的300任务池、60 Train和12 Validation；
2. 每个Train任务最多一条Static和一条Teacher轨迹；
3. 两条路线独立运行并保存完整报告；
4. Teacher只输出动作并支持一次纠正；
5. accepted、rejected和failed轨迹均可追踪；
6. Static/Teacher事后配对结果；
7. 无数据泄漏、无重复标签、无静默截断的SFT数据；
8. 数据Manifest包含代码、模型、Prompt、Schema和任务版本。
