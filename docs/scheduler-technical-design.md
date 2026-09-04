# TradingAgents可训练中央Scheduler完整技术设计

> 版本：v1.0
> 日期：2026-09-03
> 原始代码基线：TradingAgents 0.3.1，提交`a33fd4c`

配套文档：

- [第一阶段：数据生成设计](scheduler-data-generation-design.md)
- [第二阶段：训练设计](scheduler-training-design.md)
- [代码优化实施计划](scheduler-implementation-plan.md)

## 1. 项目目标

原始TradingAgents使用LangGraph固定编排多个金融Expert Agent。本优化增加一个可训练的中央Scheduler，让模型根据当前完整状态动态选择下一位Expert Agent，同时永久保留原始Static LangGraph作为基线和回退路径。

核心改造：

```text
原系统：LangGraph固定边决定下一节点

优化后：
static  → 原始LangGraph固定边
teacher → Strong Teacher API选择下一Agent
learned → 本地Qwen3-1.7B选择下一Agent
```

Scheduler只负责编排Agent；Tool仍由每个Expert Agent内部的大模型调用。

## 2. 原始系统架构

### 2.1 启动流程

原始入口位于`tradingagents/graph/trading_graph.py`：

```text
TradingAgentsGraph.__init__
→ 创建Quick Thinking LLM
→ 创建Deep Thinking LLM
→ 创建TradingMemoryLog
→ 创建4组ToolNode
→ 创建ConditionalLogic
→ GraphSetup.setup_graph(selected_analysts)
→ StateGraph(AgentState).compile()
```

运行入口：

```text
propagate(ticker, trade_date)
→ 解析instrument_context
→ 读取past_context
→ Propagator.create_initial_state
→ graph.invoke / graph.stream
→ final_trade_decision
→ SignalProcessor提取五级评级
→ 保存状态与Memory Log
```

### 2.2 原始固定编排

```text
START
→ Market Analyst
→ Sentiment Analyst
→ News Analyst
→ Fundamentals Analyst
→ Bull Researcher ⇄ Bear Researcher
→ Research Manager
→ Trader
→ Aggressive → Conservative → Neutral Risk Analysts
→ Portfolio Manager
→ END
```

Analyst顺序会根据`selected_analysts`缩减，但选中的Analyst仍按固定顺序执行。

### 2.3 原始Agent与LLM分层

| Agent | 原始模型层 | 主要输入 | 主要输出 |
|---|---|---|---|
| Market Analyst | Quick | 行情、指标、instrument context | `market_report` |
| Sentiment Analyst | Quick | 新闻/社交信息 | `sentiment_report` |
| News Analyst | Quick | 公司、宏观、内部人事件 | `news_report` |
| Fundamentals Analyst | Quick | 财务报表与基本面 | `fundamentals_report` |
| Bull Researcher | Quick | 四类报告、Bear观点、历史 | Bull Debate |
| Bear Researcher | Quick | 四类报告、Bull观点、历史 | Bear Debate |
| Research Manager | Deep | Research Debate | `investment_plan` |
| Trader | Quick | 报告与Research Plan | `trader_investment_plan` |
| Aggressive Analyst | Quick | Trader Plan与Risk历史 | Aggressive观点 |
| Conservative Analyst | Quick | Trader Plan与Risk历史 | Conservative观点 |
| Neutral Analyst | Quick | Trader Plan与Risk历史 | Neutral观点 |
| Portfolio Manager | Deep | Research Plan、Trader Plan、Risk Debate、past context | `final_trade_decision` |

### 2.4 原始Tool执行

| ToolNode | Tool |
|---|---|
| Market | `get_stock_data`、`get_indicators`、`get_verified_market_snapshot` |
| Sentiment | `get_news` |
| News | `get_news`、`get_global_news`、`get_insider_transactions`、`get_macro_indicators`、`get_prediction_markets` |
| Fundamentals | `get_fundamentals`、`get_balance_sheet`、`get_cashflow`、`get_income_statement` |

原始Analyst循环：

```text
Analyst LLM
→ 需要数据：Tool Call → ToolNode → Tool observation → 同一Analyst
→ 报告完成：消息清理节点 → 下一个Expert
```

中央Scheduler永远看不到“调用哪个Tool”这一动作，只看到Expert执行后的业务状态。

## 3. 原始上下文与记忆

### 3.1 `AgentState`

原始共享状态包含：

```text
messages
company_of_interest
asset_type
instrument_context
trade_date
sender
market_report
sentiment_report
news_report
fundamentals_report
investment_debate_state
investment_plan
trader_investment_plan
risk_debate_state
final_trade_decision
past_context
```

### 3.2 Research Debate状态

```text
bull_history
bear_history
history
current_response
judge_decision
count
```

原始路由在`count >= 2 × max_debate_rounds`时进入Research Manager，否则根据最后说话者在Bull/Bear之间切换。

### 3.3 Risk Debate状态

```text
aggressive_history
conservative_history
neutral_history
history
latest_speaker
current_aggressive_response
current_conservative_response
current_neutral_response
judge_decision
count
```

原始路由在`count >= 3 × max_risk_discuss_rounds`时进入Portfolio Manager，否则按Aggressive→Conservative→Neutral循环。

### 3.4 Memory

`TradingMemoryLog`保存历史最终决策，后续补充实际收益与Reflection。每次运行前，`get_past_context()`提取同ticker历史和跨ticker经验，注入`AgentState.past_context`。

Scheduler轨迹存储与TradingMemoryLog分开：

- TradingMemoryLog是业务经验上下文；
- SchedulerTrajectory是训练数据与信用分配记录；
- 数据生成时使用只包含`trade_date`之前记录的冻结Memory快照；
- Static和Teacher共用该快照且不在配对运行中写全局Memory。

## 4. 目标架构

第一版动态策略只训练`multi-analyst-shallow-v1`。四个Analyst是候选集合而不是固定必经节点；不实现Single、Medium、Deep或多Profile条件化策略。

### 4.1 控制面与执行面

```text
控制面
Mode Resolver
→ SchedulerPolicy
→ Action Mask
→ SchedulerNode
→ Trajectory Recorder

执行面
原始Expert Agent
→ 原始ToolNode
→ 原始AgentState
→ 原始报告与最终决策
```

控制面只改变“下一位Expert是谁”。执行面保持原来的分析、Tool和报告逻辑。

### 4.2 三种模式

| 模式 | 图 | Policy | 使用场景 |
|---|---|---|---|
| `static` | 原始固定图 | 无Scheduler模型 | 正常基线、Static数据、回退 |
| `teacher` | Scheduler动态图 | `TeacherSchedulerPolicy`，`z-ai/glm-5.3-flash` | Teacher数据和在线调试 |
| `learned` | Scheduler动态图 | `HFSchedulerPolicy` | SFT/GRPO后的本地编排 |

### 4.3 两张图

Static模式继续调用原始`setup_graph()`。

Teacher和Learned模式使用新`setup_scheduler_graph()`：

```text
START → Scheduler

Scheduler --<ACT_MARKET>------------> Market Analyst
Scheduler --<ACT_SENTIMENT>---------> Sentiment Analyst
Scheduler --<ACT_NEWS>--------------> News Analyst
Scheduler --<ACT_FUNDAMENTALS>------> Fundamentals Analyst
Scheduler --<ACT_BULL>--------------> Bull Researcher
Scheduler --<ACT_BEAR>--------------> Bear Researcher
Scheduler --<ACT_RESEARCH_MANAGER>--> Research Manager
Scheduler --<ACT_TRADER>------------> Trader
Scheduler --<ACT_AGGRESSIVE>--------> Aggressive Analyst
Scheduler --<ACT_CONSERVATIVE>------> Conservative Analyst
Scheduler --<ACT_NEUTRAL>-----------> Neutral Analyst
Scheduler --<ACT_PORTFOLIO_MANAGER>-> Portfolio Manager
Scheduler --<ACT_STOP>--------------> END

每个Expert完成后 → Scheduler
Analyst ToolNode完成后 → 原Analyst → 完成报告后 → Scheduler
```

## 5. 两阶段流程

### 5.1 第一阶段：数据生成

```text
300任务池
→ 冻结60 Train + 12 Validation
→ Static与Teacher独立运行
→ 保存NodeExecution与SchedulerTrajectory
→ 独立审核
→ task_id + data_snapshot_id事后配对
→ 拆分Action-only SFT样本
```

详见[数据生成设计](scheduler-data-generation-design.md)。

### 5.2 第二阶段：训练

```text
Qwen3-1.7B
→ valid_actions监督分类SFT
→ Harness验证
→ 每任务4条在线Rollout
→ 终局Reward
→ 轨迹级组内优势
→ GRPO-style更新
→ Static/SFT/GRPO A/B
```

详见[训练设计](scheduler-training-design.md)。

## 6. Scheduler动作协议

### 6.1 动作Token

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

### 6.2 `AgentSpec`

```python
@dataclass(frozen=True)
class AgentSpec:
    key: str
    action: SchedulerAction
    node_name: str
    purpose: str
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    prerequisites: tuple[str, ...]
    completion_signal: str
    tool_policy_owner: Literal["expert_agent"]
```

Registry是动作Token、节点名、依赖和完成信号的唯一映射源。

### 6.3 `SchedulerContext`

```python
@dataclass(frozen=True)
class SchedulerContext:
    task_id: str
    state: Mapping[str, Any]
    serialized_state: str
    valid_actions: tuple[SchedulerAction, ...]
    selected_analysts: tuple[str, ...]
    history: tuple[SchedulerAction, ...]
    step: int
    max_steps: int
    no_progress_count: int
```

### 6.4 `PolicyDecision`

```python
@dataclass(frozen=True)
class PolicyDecision:
    action: SchedulerAction
    logprob: float | None
    decision_attempts: int
    correction_succeeded: bool
    policy_id: str
    metadata: Mapping[str, Any]
```

### 6.5 `SchedulerPolicy`

```python
class SchedulerPolicy(Protocol):
    policy_id: str

    def select_action(self, context: SchedulerContext) -> PolicyDecision:
        ...
```

Teacher和Learned都实现`SchedulerPolicy`。需要为GRPO提供概率的本地策略额外实现：

```python
class ActionLogprobPolicy(SchedulerPolicy, Protocol):
    def action_logprobs(
        self,
        context: SchedulerContext,
    ) -> Mapping[SchedulerAction, float]:
        ...
```

Static运行原始图，不伪装成动态Policy。

## 7. Action Mask机制

Action Mask由确定性代码计算，模型不能绕过。

```text
AgentState + selected_analysts + step + budget + history
→ compute_valid_actions()
→ tuple[SchedulerAction, ...]
```

规则分两类：

### 7.1 硬依赖

- 未选择的Analyst动作不存在；
- 已完成报告的Analyst默认不重复；
- Research Manager需要Research Debate内容；
- Trader需要`investment_plan`；
- Risk Agents需要`trader_investment_plan`；
- Portfolio Manager需要Risk Debate内容；
- STOP需要`final_trade_decision`。

### 7.2 运行约束

- Shallow不设置Bull/Bear或Risk固定轮数；
- 统一`max_steps=16`只作为异常防循环安全阀；
- 同一动作执行后状态签名不变，增加`no_progress_count`；
- 出现无进展时临时屏蔽立即重复动作；
- 没有合法动作时产生明确失败，不随机选节点。

Mask不规定Market必须先于News，也不规定Bull一定先于Bear；合法集合内的顺序由Teacher或Learned策略决定。

## 8. 状态序列化

### 8.1 输入内容

Scheduler读取：

- `multi-analyst-shallow-v1`动态目标和可用Analyst候选集合；
- `company_of_interest`、`asset_type`、`trade_date`；
- `instrument_context`；
- 四类完整Analyst报告；
- `investment_debate_state`完整历史；
- `investment_plan`；
- `trader_investment_plan`；
- `risk_debate_state`完整历史；
- `final_trade_decision`；
- 时间过滤后的`past_context`；
- Scheduler历史、步数、剩余预算和no-progress计数；
- `valid_actions`。

不重复加入原始ToolMessage和原始数据表。

### 8.2 Canonical文本模板

```text
<SCHEDULER_ROLE>
You select exactly one next Expert Agent action.

<ORCHESTRATION_PROFILE version="multi-analyst-shallow-v1">
{available_analysts_and_minimum_sufficient_path_objective}

<AGENT_CATALOG version="agent-catalog-v1">
{agent_catalog_json}

<COMPLETION_CONTRACT version="completion-v1">
{completion_rules_json}

<TASK>
task_id={task_id}
ticker={ticker}
trade_date={trade_date}
asset_type={asset_type}

<CURRENT_STATE version="scheduler-state-v1">
{business_state_json}

<EXECUTION>
history={history_json}
step={step}
max_steps={max_steps}
remaining_steps={remaining_steps}
no_progress_count={no_progress_count}

<VALID_ACTIONS>
{valid_actions_json}

<SCHEDULER_ACTION>
```

Teacher API接收等价的System/User messages；Qwen使用`apply_chat_template(..., enable_thinking=False, add_generation_prompt=True)`生成最终Token序列。

超过32,768 Token的状态不截断，记录为`context_overflow`。

## 9. Teacher提示协议

### 9.1 System Prompt

```text
You are the central Agent Scheduler for TradingAgents.

Your only job is to choose exactly one next Expert Agent action.
The selected Expert Agent will execute before you receive the next state.

Rules:
1. Choose only from VALID_ACTIONS.
2. Do not call tools. Tools belong to Expert Agents.
3. Do not write financial analysis or a trading decision.
4. Do not output a full future route.
5. Choose STOP only when the completion contract is satisfied.
6. Return one JSON object and no surrounding text.

AGENT_CATALOG={agent_catalog_json}
COMPLETION_CONTRACT={completion_rules_json}
POSITIVE_EXAMPLES={few_shot_examples_json}
CORRECTED_FAILURE_EXAMPLES={failure_examples_json}
PROMPT_VERSION=teacher-scheduler-v2
```

Few-shot只展示通用状态—动作关系，不包含当前任务的Static轨迹或答案。

### 9.2 User Prompt

```text
TASK={task_json}
CURRENT_STATE={business_state_json}
EXECUTION={execution_json}
VALID_ACTIONS={valid_actions_json}
OUTPUT_SCHEMA={"action":"<ACT_...>"}
```

### 9.3 输出Schema

```json
{
  "type": "object",
  "properties": {
    "action": {
      "type": "string",
      "enum": ["<当前valid_actions中的值>"]
    }
  },
  "required": ["action"],
  "additionalProperties": false
}
```

### 9.4 纠正Prompt

```text
Your previous response was invalid.
ERROR_TYPE={error_type}
PREVIOUS_OUTPUT={previous_output}

No Expert Agent was executed and CURRENT_STATE is unchanged.
Choose exactly one action from VALID_ACTIONS={valid_actions_json}.
Return only {"action":"<ACT_...>"}.
This is the final correction attempt.
```

### 9.5 Completion Contract

```json
{
  "max_steps": 16,
  "stop_requires": ["final_trade_decision"],
  "research_manager_requires": ["investment_debate_state.history"],
  "trader_requires": ["investment_plan"],
  "risk_agents_require": ["trader_investment_plan"],
  "portfolio_manager_requires": ["risk_debate_state.history"],
  "one_action_per_decision": true,
  "tools_owned_by": "expert_agent"
}
```

### 9.6 Few-shot示例

正例只解释通用状态关系：

```json
{
  "state_fact": "market_report和news_report为空",
  "valid_actions": ["<ACT_MARKET>", "<ACT_NEWS>"],
  "output": {"action": "<ACT_MARKET>"}
}
```

```json
{
  "state_fact": "final_trade_decision已经生成",
  "valid_actions": ["<ACT_STOP>"],
  "output": {"action": "<ACT_STOP>"}
}
```

纠正失败示例用于说明格式和职责边界：

```json
{
  "invalid_output": {"action": "<ACT_TRADER>"},
  "error": "investment_plan_missing",
  "lesson": "只从VALID_ACTIONS选择"
}
```

```json
{
  "invalid_output": "Call get_stock_data first",
  "error": "scheduler_must_not_call_tools",
  "lesson": "Scheduler只输出一个Agent Action"
}
```

Few-shot文件最多保留少量覆盖性示例，防止Teacher照抄一条固定完整路径。

## 10. Learned Scheduler推理

```text
SchedulerContext
→ Canonical Prompt Builder
→ Qwen3-1.7B + Adapter forward
→ `logits_to_keep=1`只保留最后决策位置
→ 获取最后位置logits
→ gather 13个Action Token logits
→ invalid action设为负无穷
→ softmax / log_softmax
→ greedy或temperature采样
→ PolicyDecision
```

运行温度：

| 场景 | 温度 |
|---|---:|
| A/B确定性评测 | `0.0` |
| GRPO Rollout | `0.8` |

## 11. Canonical数据字段

字段定义以本章为主，阶段文档说明其生成和消费方式。

### 11.1 `TaskSeed`

| 字段 | 类型 | 生产者 | 消费者 |
|---|---|---|---|
| `task_id` | string | Task Builder | 所有阶段 |
| `ticker` | string | Task Builder | TradingAgentsGraph |
| `trade_date` | date string | Task Builder | TradingAgentsGraph |
| `asset_type` | enum | Task Builder | Propagator |
| `split` | enum | Splitter | DataLoader/Evaluator |
| `seed_family` | enum | Task Builder | 分层统计 |
| `sector` | string | Task Builder | 分布统计 |
| `data_snapshot_id` | string | Snapshot Manager | 双路线与A/B |
| `information_cutoff` | datetime | Snapshot Manager | 泄漏检查 |
| `selection_features` | object | Task Builder | 数据审计 |
| `dataset_version` | string | Manifest | 可复现 |

### 11.2 `NodeExecution`

| 字段 | 类型 | 含义 |
|---|---|---|
| `node_step_id` | integer | LangGraph节点执行序号 |
| `node_name` | string | 节点名 |
| `node_type` | enum | scheduler/expert/tool/message_cleanup |
| `state_before` | object | 节点前状态 |
| `state_update` | object | 节点返回增量 |
| `state_after` | object | 合并后状态 |
| `tool_events` | array | Tool名、成功状态和引用 |
| `input_tokens` | integer | 节点输入Token |
| `output_tokens` | integer | 节点输出Token |
| `latency_ms` | number | 节点耗时 |
| `error` | string/null | 执行错误 |

### 11.3 `SchedulerStep`

| 字段 | 类型 | 含义 |
|---|---|---|
| `step_id` | integer | Scheduler决策序号 |
| `state_before` | object | 决策前完整业务状态 |
| `serialized_state` | string | 模型实际输入 |
| `valid_actions` | string[] | 硬Mask结果 |
| `selected_action` | string | 策略动作 |
| `agent_node` | string/null | 实际Expert；STOP为空 |
| `state_after` | object | Expert执行后状态 |
| `decision_attempts` | integer | Teacher尝试次数 |
| `correction_succeeded` | boolean | 是否纠正成功 |
| `observation_ref` | string/null | 对应NodeExecution |
| `old_logprob` | number/null | GRPO旧策略概率 |
| `ref_logprob` | number/null | GRPO参考概率 |
| `cost` | object | 本步成本 |
| `error` | string/null | 决策或执行错误 |

### 11.4 `SchedulerTrajectory`

| 字段 | 类型 | 含义 |
|---|---|---|
| `schema_version` | string | `scheduler-trajectory-v1` |
| `trajectory_id` | string | 全局唯一轨迹ID |
| `run_id` | string | 批次ID |
| `task_id` | string | 任务ID |
| `mode` | enum | static/teacher/learned |
| `policy_id` | string | Policy版本 |
| `execution_status` | enum | running/completed/failed/fallback/budget_exhausted/context_overflow |
| `audit_status` | enum | pending/accepted/rejected/warning |
| `steps` | SchedulerStep[] | 中央编排轨迹 |
| `node_executions` | NodeExecution[] | 原始节点证据 |
| `final_outputs` | object | Plan、Trader、Portfolio与解析值 |
| `cost_total` | object | 总调用、Token和延迟 |
| `reward` | object/null | Reward分量 |
| `provenance` | object | 全版本来源 |
| `audit` | object | 结构检查、错误与审核器版本 |
| `failure_reason` | string/null | 失败原因 |

### 11.5 `SFTExample`

| 字段 | 类型 | 含义 |
|---|---|---|
| `schema_version` | string | `scheduler-sft-v1` |
| `sample_id` | string | 唯一样本ID |
| `task_id` | string | split与统计 |
| `trajectory_id` | string | 回溯轨迹 |
| `step_id` | integer | 回溯步骤 |
| `source` | enum | static/teacher_verified |
| `input_text` | string | Canonical模型输入 |
| `valid_actions` | string[] | 分类Mask |
| `target_action` | string | 监督动作 |
| `input_token_count` | integer | 长度审计 |

### 11.6 `GRPORow`

| 字段 | 类型 | 含义 |
|---|---|---|
| `schema_version` | string | `scheduler-grpo-v1` |
| `rollout_group_id` | string | 同任务候选组 |
| `trajectory_id` | string | 完整轨迹 |
| `task_id` | string | 任务ID |
| `step_id` | integer | 动作步骤 |
| `serialized_state` | string | 策略输入 |
| `valid_actions` | string[] | 动作Mask |
| `selected_action` | string | 已采样动作 |
| `old_logprob` | number | 行为策略logprob |
| `ref_logprob` | number | 参考策略logprob |
| `reward_total` | number | 轨迹Reward |
| `reward_components` | object | Reward分量 |
| `advantage` | number | 组内相对优势 |

### 11.7 `RewardBreakdown`

```text
total
portfolio_quality
trader_quality
format_compliance
completion
agent_cost
tool_cost
token_cost
latency_cost
no_progress
invalid
incomplete
fallback
```

## 12. 技术栈

### 12.1 原项目

| 层 | 技术 | 当前环境 |
|---|---|---|
| 语言 | Python | 3.12.9，项目要求>=3.10 |
| 多Agent图 | LangGraph | 1.2.11 |
| LLM抽象 | LangChain Core | 1.6.1 |
| OpenAI/OpenRouter | langchain-openai | 1.6.0 |
| Gemini | langchain-google-genai | 4.3.7 |
| Teacher API | OpenRouter `z-ai/glm-5.3-flash` | 1,310,720 Token上下文，支持Structured Outputs；[模型页面](https://openrouter.ai/z-ai/glm-5.3-flash) |
| 数据处理 | pandas | 3.0.5 |
| 市场数据 | yfinance、stockstats | yfinance 1.7.0 |
| Checkpoint | langgraph-checkpoint-sqlite | SQLite |
| 配置与CLI | python-dotenv、Typer、Questionary | 原项目现有 |
| 测试 | pytest | 9.1.1 |
| 静态检查 | Ruff | 0.16.5 |

### 12.2 Scheduler训练

| 层 | 技术 | 当前开发环境 |
|---|---|---|
| 深度学习 | PyTorch | 2.14.0 |
| 模型 | Hugging Face Transformers | 5.16.1 |
| 参数高效训练 | PEFT LoRA | 0.20.0 |
| 单卡训练封装 | Accelerate | 1.14.0 |
| 数据集 | datasets + JSONL | 5.0.1 |
| 基模 | Qwen3-1.7B | revision `70d244...` |
| 训练硬件 | AutoDL RTX 4090 | 24GB |

AutoDL首次Smoke通过后生成精确lock文件，训练Manifest记录相同版本。

## 13. 代码结构

### 13.1 原代码修改点

| 文件 | 修改内容 |
|---|---|
| `tradingagents/graph/setup.py` | 原`setup_graph()`保持原样，仅追加独立Scheduler动态图构图 |
| `tradingagents/graph/trading_graph.py` | 解析模式、装配Policy、Recorder和fallback |
| `tradingagents/default_config.py` | 增加Scheduler配置 |
| `cli/main.py` | 增加三模式参数 |
| `pyproject.toml` | 增加`scheduler-train`可选依赖 |

`tradingagents/agents/utils/agent_states.py`不修改。Static继续使用原`AgentState`和原checkpoint签名；动态模式使用独立的`SchedulerAgentState`。

### 13.2 新运行模块

```text
tradingagents/scheduler/
├── actions.py
├── registry.py
├── contracts.py
├── action_mask.py
├── prompt.py
├── state.py
├── scheduler_node.py
├── teacher_policy.py
├── hf_policy.py
├── recorder.py
└── store.py
```

职责：

| 模块 | 职责 |
|---|---|
| `actions.py` | 动作Enum、Token与节点映射 |
| `registry.py` | AgentSpec与Catalog |
| `contracts.py` | Context、Decision、Step与Trajectory数据类 |
| `action_mask.py` | 确定性合法动作 |
| `prompt.py` | Teacher与Qwen共用语义模板 |
| `state.py` | 只供动态图使用的Scheduler状态扩展 |
| `scheduler_node.py` | LangGraph中央节点和路由 |
| `teacher_policy.py` | API调用、JSON校验和一次纠正 |
| `hf_policy.py` | 本地模型动作logits、采样与logprob |
| `recorder.py` | 决策、节点、Tool和成本记录 |
| `store.py` | JSONL、Manifest、续跑与幂等写入 |

### 13.3 新数据与训练模块

```text
training/scheduler/
├── market_features.py
├── task_seeds.py
├── environment.py
├── generate.py
├── provenance.py
├── audit.py
├── pair.py
├── pair_dataset.py
├── build_sft.py
├── prepare_sft.py
├── model.py
├── sft_dataset.py
├── sft_loss.py
├── train_sft.py
├── rollout.py
├── collect_rollouts.py
├── collect_evaluation.py
├── reward.py
├── advantage.py
├── grpo_dataset.py
├── grpo_loss.py
├── train_grpo.py
├── evaluate.py
├── download_model.py
└── smoke_qwen.py
```

`provenance.py`统一记录代码提交、任务/数据快照、Expert与Teacher模型、Prompt/Action/State/Catalog版本和不含密钥的生成配置指纹。

## 14. 配置设计

### 14.1 应用配置

```yaml
orchestration_mode: static
scheduler_max_steps: 16
scheduler_action_temperature: 0.0
scheduler_fallback_enabled: true
scheduler_max_context_tokens: 32768

teacher_provider: openrouter
teacher_model: z-ai/glm-5.3-flash
teacher_prompt_version: teacher-scheduler-v2

scheduler_base_model: Qwen/Qwen3-1.7B
scheduler_base_revision: 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
scheduler_adapter_path: null
scheduler_dtype: bfloat16
scheduler_device: cuda
```

默认`static`保证不配置Teacher或本地模型时，原项目行为不变。

### 14.2 训练配置

SFT与GRPO参数分别放在：

```text
configs/scheduler/sft-qwen3-1p7b.json
configs/scheduler/rollout-qwen3-1p7b.json
configs/scheduler/grpo-qwen3-1p7b.json
configs/scheduler/reward-v1.json
configs/scheduler/eval-sft-qwen3-1p7b.json
configs/scheduler/eval-grpo-qwen3-1p7b.json
```

### 14.3 密钥

Teacher API Key只从`OPENROUTER_API_KEY`读取。配置、Prompt、日志、轨迹和Manifest只保存环境变量名，不保存密钥值。

## 15. 运行时流程

### 15.1 Static

```text
CLI / Python API
→ orchestration_mode=static
→ GraphSetup.setup_graph
→ 原始固定图
→ 原始报告和最终决策
```

### 15.2 Teacher

```text
CLI / Data Generator
→ orchestration_mode=teacher
→ TeacherSchedulerPolicy
→ setup_scheduler_graph
→ SchedulerContext
→ Teacher Action JSON
→ Action Mask验证
→ Expert执行
→ Recorder
→ Scheduler循环
```

### 15.3 Learned

```text
CLI / Rollout / Evaluation
→ orchestration_mode=learned
→ 加载Qwen Base + Adapter
→ HFSchedulerPolicy
→ setup_scheduler_graph
→ Masked Action logits
→ Expert执行
→ Recorder
→ Scheduler循环
```

## 16. 失败与回退

| 失败 | Teacher数据生成 | Learned应用运行 |
|---|---|---|
| JSON格式错误 | 同状态纠正一次 | 不适用，模型直接输出Token |
| Mask外动作 | 同状态纠正一次 | 代码Mask阻止 |
| 无合法动作 | 拒绝轨迹 | 按配置失败或回退Static |
| 超过最大步数 | 拒绝轨迹 | 按配置失败或回退Static |
| Expert/Tool错误 | 保存failed轨迹 | 按原异常策略处理或回退 |
| 上下文超过32K | 拒绝轨迹 | 结束并记录context_overflow |
| Adapter加载失败 | 不适用 | 启用Static回退并记录原因 |

回退运行产生的Static结果不能计入Learned的质量Reward，否则模型会通过失败获取成功奖励。

## 17. 数据与模型产物

```text
data/scheduler/v1/                    任务、轨迹、审核、配对、SFT数据
artifacts/scheduler/qwen3-1p7b/       模型、Smoke、SFT、Rollout、GRPO、评测
configs/scheduler/                     Agent Catalog、Prompt、训练与Reward配置
```

大体积数据、模型和Checkpoint通过`.gitignore`排除；Git只保存Schema、配置模板、小型Fixture和代码。

## 18. 测试与验收

### 18.1 原系统回归

- `static`输出路径保持原始LangGraph语义；
- 原始576项测试继续通过；
- ToolNode、Checkpoint、报告和Memory Log行为保持可用。

### 18.2 Scheduler单元测试

- 13个动作与节点一一映射；
- Agent Catalog与动作Registry一致；
- Action Mask覆盖所有依赖和STOP；
- Prompt字段完整且不泄漏当前Static答案；
- Teacher最多纠正一次；
- 状态无进展检测；
- Schema、Manifest与JSONL验证；
- SFT Masked Classification；
- Reward分量、优势和GRPO Loss。

### 18.3 集成测试

- Mock Policy驱动Scheduler图完成任务；
- Analyst Tool循环结束后返回Scheduler；
- Static和Teacher共用同一数据与Memory快照；
- Learned Adapter可加载并运行；
- fallback不污染Learned Reward；
- 三种模式输出相同的最终报告接口。

## 19. 设计结果

本设计把TradingAgents的原始Expert执行能力保留为Harness，把可学习部分限制在中央Agent调度策略：

```text
原始LangGraph = 稳定基线与回退
Teacher API = 动态编排数据来源
Qwen3-1.7B SFT = 调度冷启动
GRPO-style = 基于完整轨迹质量与成本的策略优化
Action Mask = 执行安全边界
Trajectory Schema = 数据、训练与评测的统一接口
```
