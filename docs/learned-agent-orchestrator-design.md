# TradingAgents 可学习 Agent 编排调度器设计方案

> 文档版本：v3.0
> 文档状态：方案已冻结；离线代码框架已实现，真实数据生成与基模训练未执行
> 目标版本：TradingAgents 0.3.1 兼容改造
> 更新日期：2026-09-03

## 1. 摘要

本方案将 TradingAgents 当前由 LangGraph 手工定义的固定多 Agent 路径，改造为由轻量语言模型控制的动态 Agent 编排系统。新的中央调度器（Scheduler）读取全局 `AgentState`、已有报告、执行历史与推理成本，从当前合法的 Expert Agent 中选择下一执行节点。被选中的 Expert Agent 继续使用原有大模型完成推理，并在自身内部通过既有 `ToolNode` 调用行情、新闻、基本面和宏观工具。

项目只训练 Scheduler，不训练十二个 Expert Agent，也不改变 Expert Agent 的工具调用策略。Scheduler 采用已有小型指令模型作为基模；冷启动数据由 Static Graph 基础轨迹和 Strong Route Teacher 提议、真实 TradingAgents Harness 验证通过的多样化轨迹组成。小模型先通过监督微调（SFT）掌握状态格式与动作语法，再通过多轨迹、结果驱动的 GRPO-style 后训练学习动态 Agent 选择策略。训练奖励同时衡量最终结构化决策质量与 Agent 调用、Tool 调用、Token 和延迟成本。信用分配采用粗粒度轨迹级组相对优势，不引入额外 Critic 或细粒度过程奖励模型。

改造后保留 `static` 和 `learned` 两种运行模式。`static` 完整复现原始编排，`learned` 加载 Scheduler checkpoint 动态生成执行路径。两种模式使用相同的 Expert Agent、Prompt、工具、数据快照与输出 Schema，从而支持可解释的 A/B 对比。

### 1.1 术语约定

| 术语 | 本文定义 |
|---|---|
| Scheduler | 唯一可训练的小语言模型中央调度策略 |
| Expert Agent | 当前十二个冻结的分析、研究、交易、风险与最终决策 Agent |
| ToolNode | Expert Agent 内部执行工具函数的 LangGraph 节点，不属于 Scheduler 动作空间 |
| Static Graph | 当前固定或手写规则驱动的 Agent 编排图 |
| Learned Graph | Scheduler 在固定节点集合上动态形成的实际执行路径 |
| Trajectory | Scheduler 状态、动作、Agent observation、成本和终局结果的完整序列 |
| Static Teacher | Static Graph，为冷启动、结构化质量评分和 A/B 提供稳定参考 |
| Strong Route Teacher | 较强大模型，在部分状态上提出多样化下一 Agent 候选 |
| Environment Verifier | 真实 TradingAgents Harness，执行并过滤 Strong Route Teacher 候选轨迹 |

### 1.2 冻结的 MVP 决策

后续实现以本表为准。若其中任何一项发生变化，必须提升对应 Schema、Prompt 或策略版本，不能在数据采集过程中静默修改。

| 决策点 | 冻结选择 | 说明 |
|---|---|---|
| 产品运行模式 | `static` 与 `learned` 两种 | `static` 永久保留；`learned` 失败可回退 |
| 运行底座 | 两种模式都由 LangGraph 执行 | Learned 不是绕开 LangGraph，而是改变 Agent 间路由 |
| Scheduler 职责 | 只选择下一 Expert Agent 或 STOP | 不选择工具、不生成交易结论 |
| Expert Agent | 原 Agent、Prompt、ToolNode 全部冻结 | 保证 A/B 唯一主要变量是 Agent 编排 |
| Scheduler 基模 | 当前不指定，由 AutoDL 训练配置注入 | 代码不得下载模型；训练前再冻结 model/tokenizer revision |
| 可训练参数 | Scheduler LoRA | 不从头预训练，不做 Expert 联合训练 |
| Strong Teacher | 冻结的强模型 API，只做推理 | 不微调 Teacher，不把 API 输出直接当真值 |
| Teacher 决策粒度 | 每次只输出一个 next-agent action | Agent 实际执行后，Teacher 才观察新状态并继续决策 |
| 冷启动数据 | Static 轨迹 + verified Teacher 轨迹 + 边界状态 | 正样本必须通过真实 Harness 执行验证 |
| 失败样本 | Teacher 纠错示例、测试样例、RL 负轨迹 | 不作为普通 SFT 的错误动作正标签 |
| SFT 目标 | 单个 Agent Action Token | Teacher reason 只用于审计，不进入生成目标 |
| RL 算法 | 轨迹级 GRPO-style 组相对优化 | 同任务默认采样 4 条轨迹 |
| Reward | 完成与质量优先，成本次级 | 不单独训练 Critic、PRM 或神经 Reward Model |
| 工具边界 | Expert Agent 内部继续调用既有 ToolNode | Tool trace 参与成本统计，但不参与 Scheduler 动作空间 |
| 对比对象 | Static、Learned-SFT、Learned-RL | 最终结果来自冻结测试集，不能使用训练集数字 |

Strong Teacher 的具体供应商和模型名称属于运行配置，不属于架构本身；它们必须在首次正式数据生成前写入 manifest 并冻结。API Key 只提供调用权限，不代表 Teacher 被本项目训练。

## 2. 当前系统基线

### 2.1 Expert Agent 清单

当前系统包含十二个参与决策的 LLM Agent：

| 类别 | Agent | 主要职责 | 主要状态输出 |
|---|---|---|---|
| 分析 | Market Analyst | 行情、技术指标与价格快照分析 | `market_report` |
| 分析 | Sentiment Analyst | 新闻、StockTwits 与 Reddit 情绪分析 | `sentiment_report` |
| 分析 | News Analyst | 公司新闻、全球新闻、宏观和预测市场分析 | `news_report` |
| 分析 | Fundamentals Analyst | 基本面、资产负债表、现金流和利润表分析 | `fundamentals_report` |
| 研究 | Bull Researcher | 基于已有报告形成多头论证 | `investment_debate_state` |
| 研究 | Bear Researcher | 形成空头论证并反驳多头观点 | `investment_debate_state` |
| 研究 | Research Manager | 汇总多空历史并生成投资计划 | `investment_plan` |
| 交易 | Trader | 将投资计划转换为 Buy/Hold/Sell 提案 | `trader_investment_plan` |
| 风险 | Aggressive Analyst | 强调高收益机会与积极风险偏好 | `risk_debate_state` |
| 风险 | Conservative Analyst | 强调本金保护和下行风险 | `risk_debate_state` |
| 风险 | Neutral Analyst | 平衡激进与保守观点 | `risk_debate_state` |
| 决策 | Portfolio Manager | 汇总研究、交易、风险与历史记忆 | `final_trade_decision` |

### 2.2 当前固定编排

当前 `GraphSetup.setup_graph()` 在运行前编译固定拓扑。默认路径为：

```text
START
  → Market Analyst
  → Sentiment Analyst
  → News Analyst
  → Fundamentals Analyst
  → Bull Researcher
  → Bear Researcher
  → Research Manager
  → Trader
  → Aggressive Analyst
  → Conservative Analyst
  → Neutral Analyst
  → Portfolio Manager
  → END
```

每个 Analyst 内部存在工具循环：

```text
Analyst
  ├─ 产生 tool_calls → 对应 ToolNode → 返回原 Analyst
  └─ 不再产生 tool_calls → Msg Clear → 固定下一 Agent
```

Sentiment Analyst 是当前实现中的例外：它在节点内部预取新闻、StockTwits 和 Reddit 数据，再调用 LLM 生成报告，不依赖多轮 ToolNode。对于 Scheduler，这仍属于 Expert Agent 内部执行，不进入 Scheduler 动作空间。

多空研究和风险讨论虽然通过条件边运行，但具体顺序仍由 `ConditionalLogic` 手工规定：

```text
Bull ↔ Bear → Research Manager

Aggressive → Conservative → Neutral → Portfolio Manager
```

因此，当前系统的 Agent 间编排是固定或规则驱动的；动态性主要存在于 Expert Agent 内部的工具选择和配置化辩论轮数。

### 2.3 当前系统的入口与运行生命周期

当前系统有两类入口：

1. Python API 通过 `TradingAgentsGraph.propagate(ticker, date)` 运行完整生命周期；
2. CLI 收集 ticker、日期、Analyst 选择、模型和研究深度后，直接 stream 已编译 LangGraph，并在终端持续展示节点状态和 Tool 调用。

`TradingAgentsGraph` 是现有系统的 Composition Root，构造时完成：

```text
读取配置
  ↓
初始化 Quick/Deep LLM Client
  ↓
注册 Market/Social/News/Fundamentals ToolNode
  ↓
创建 ConditionalLogic
  ↓
创建 GraphSetup
  ↓
编译 StateGraph(AgentState)
```

Python API 一次运行的大致生命周期为：

```text
propagate(ticker, date)
  ↓
处理该 ticker 过去待结算的决策记录
  ↓
可选：为该 ticker 打开 SQLite LangGraph checkpointer
  ↓
解析 ticker 对应的真实公司/资产身份
  ↓
读取同 ticker 历史决策与跨 ticker 经验
  ↓
创建初始 AgentState
  ↓
执行 graph.invoke() 或 graph.stream()
  ↓
获得 final_trade_decision
  ↓
写入 JSON 状态、Markdown 决策日志和报告树
  ↓
成功时清理本次 checkpoint
```

Scheduler 必须被集成在 compiled graph 内，而不是只放进 `propagate()`，否则 CLI 直接 stream graph 时可能绕过 Scheduler。轨迹采集也应作为图级或共享运行组件接入，保证 Python API 与 CLI 使用相同的 Scheduler 路径。

### 2.4 当前 LLM 分层

现有系统将模型分成两类：

| 模型层 | 使用者 | 当前目的 |
|---|---|---|
| Quick LLM | 四类 Analyst、Bull/Bear、Trader、三个 Risk Agent、Reflector | 高频分析、对话和工具调用 |
| Deep LLM | Research Manager、Portfolio Manager | 多报告汇总和最终裁决 |

`create_llm_client()` 根据 provider 创建 OpenAI、Google、Anthropic、Azure、Bedrock 或 OpenAI-compatible 客户端。Scheduler 改造不替换这套 Provider Layer。新的 Scheduler 基模是独立于 Quick/Deep LLM 的第三类模型，只负责输出 Agent Action Token。

```text
Quick/Deep LLM：完成专业任务
Scheduler LM：选择下一位 Expert Agent
```

### 2.5 当前共享状态与上下文流

`AgentState` 是现有多 Agent 系统的共享黑板（blackboard）。不同 Agent 不直接调用彼此，而是读取和更新同一个状态：

```text
任务上下文
├── company_of_interest
├── asset_type
├── instrument_context
└── trade_date

分析报告
├── market_report
├── sentiment_report
├── news_report
└── fundamentals_report

研究与交易
├── investment_debate_state
├── investment_plan
└── trader_investment_plan

风险与最终决策
├── risk_debate_state
├── final_trade_decision
└── past_context
```

现有 Expert Agent 的典型节点逻辑为：

```text
读取 AgentState 中与角色相关的字段
  ↓
构造角色 Prompt
  ↓
调用 Quick 或 Deep LLM
  ↓
返回部分状态更新
  ↓
LangGraph 合并到全局 AgentState
```

Learned Scheduler 将直接消费这块共享状态，但不会修改 Expert Agent 的专业 Prompt。它只增加调度历史、动作、成本和 trajectory 引用。

### 2.6 当前 Analyst 与 ToolNode 执行链

Market、News 和 Fundamentals Analyst 使用 LangChain tool binding。LLM 先决定是否调用工具，`ConditionalLogic` 检查最后一条 AIMessage 是否包含 `tool_calls`：

```text
Analyst LLM
  ├─ tool_calls 非空
  │     ↓
  │   对应 ToolNode
  │     ↓
  │   执行 Python 工具函数
  │     ↓
  │   ToolMessage 写回 messages
  │     ↓
  │   返回同一个 Analyst
  │
  └─ tool_calls 为空
        ↓
      写入最终 report
        ↓
      Msg Clear
        ↓
      当前固定图中的下一 Agent
```

四类工具集合：

| ToolNode | 工具示例 | 数据来源 |
|---|---|---|
| `tools_market` | 股票行情、技术指标、验证快照 | yfinance、Alpha Vantage |
| `tools_social` | 新闻接口（保留的 legacy 工具节点） | 配置的 news vendor |
| `tools_news` | 公司新闻、全球新闻、内部人、宏观、预测市场 | yfinance、Alpha Vantage、FRED、Polymarket |
| `tools_fundamentals` | 基本面、资产负债表、现金流、利润表 | yfinance、Alpha Vantage |

Sentiment Analyst 当前不依赖多轮 ToolNode，而是在节点内部预取新闻、StockTwits 和 Reddit 数据，再调用 LLM 生成结构化情绪报告。对于 Scheduler，两种执行方式都属于 Expert Agent 内部实现。

数据工具最终通过 `route_to_vendor()` 选择配置的 vendor chain。核心数据失败会显式报错，可选的宏观/预测市场数据可以降级为“不可用”说明。这些数据和异常边界继续由原系统负责，不进入 Scheduler 训练逻辑。

### 2.7 当前研究、交易与风险链

分析报告完成后，当前系统按固定组织结构处理：

```text
四类 Analyst report
  ↓
Bull Researcher ↔ Bear Researcher
  ↓
Research Manager
  ↓
Trader
  ↓
Aggressive → Conservative → Neutral
  ↓
Portfolio Manager
```

关键状态传递：

- Bull/Bear 读取四类 Analyst report 和对方上一轮观点；
- Research Manager 只汇总 `investment_debate_state.history`，生成结构化 `ResearchPlan`；
- Trader 读取 `investment_plan`，生成结构化 `TraderProposal`；
- 三个 Risk Agent 读取 Trader plan 和已有报告，形成不同风险立场；
- Portfolio Manager 读取 Research plan、Trader plan、Risk history 和 `past_context`，生成结构化 `PortfolioDecision`。

默认研究轮数为一轮 Bull+Bear，风险轮数为 Aggressive+Conservative+Neutral 各一次。运行顺序由 `ConditionalLogic` 的计数器和 speaker label 决定，而不是从任务状态中学习。

### 2.8 当前结构化输出、记忆与报告

现有系统已经提供 Scheduler 训练所需的三个稳定评价锚点：

| 输出 | 枚举 | 用途 |
|---|---|---|
| `ResearchPlan.recommendation` | Buy/Overweight/Hold/Underweight/Sell | 研究结果结构化 |
| `TraderProposal.action` | Buy/Hold/Sell | 交易动作结构化 |
| `PortfolioDecision.rating` | Buy/Overweight/Hold/Underweight/Sell | 最终决策结构化 |

这些 Schema 使质量评分不必依赖额外 LLM Judge 或脆弱的自然语言相似度。

当前跨运行记忆使用 append-only Markdown 决策日志：

```text
本轮结束
  ↓
保存 final_trade_decision，状态为 pending
  ↓
下次运行相同 ticker
  ↓
计算过去决策后的 raw return 和 benchmark alpha
  ↓
Quick LLM 生成简短 reflection
  ↓
作为 past_context 注入 Portfolio Manager
```

LangGraph checkpoint 使用每 ticker SQLite 数据库保存节点状态，用于可选的中断恢复。报告模块将 Analyst、Research、Trading、Risk 和 Portfolio 内容写成分层 Markdown 目录和 consolidated report。

Learned Scheduler 不替换现有 memory/checkpoint/reporting，而是让 trajectory store 与它们并存：业务状态继续由 AgentState/checkpoint 管理，训练状态由 Scheduler trajectory 和 policy checkpoint 管理。

### 2.9 当前系统的可改造问题

| 当前性质 | 对调度优化的限制 |
|---|---|
| Analyst 顺序在编译图时确定 | 运行中不能根据已有报告重排或跳过 Analyst |
| Bull/Bear 按 speaker label 固定轮转 | 不能根据观点冲突程度决定继续谁或何时交给 Manager |
| Risk Agent 固定三人轮转 | 不能根据 Trader plan 选择更相关的风险视角 |
| Agent 完成后直接进入固定下一节点 | 没有统一的中央控制点 |
| 没有 Scheduler action/logprob | 无法形成可训练的策略轨迹 |
| 没有 Agent/Tool/Token 统一成本 Reward | 调用成本只被观察，不能优化 |
| 没有 policy checkpoint | 目录名虽含 RL，当前源码并不存在调度器训练闭环 |

### 2.10 当前源码证据

| 证据 | 当前职责 | 设计中的改造点 |
|---|---|---|
| [`tradingagents/graph/setup.py`](../tradingagents/graph/setup.py) | 注册 Agent、ToolNode 和固定边 | 拆分 `setup_static_graph()` 与 `setup_learned_graph()` |
| [`tradingagents/graph/conditional_logic.py`](../tradingagents/graph/conditional_logic.py) | Analyst 工具路由、多空和风险轮转 | Learned 模式只保留 Analyst 内部 ToolNode 路由 |
| [`tradingagents/graph/trading_graph.py`](../tradingagents/graph/trading_graph.py) | 装配 LLM、ToolNode、GraphSetup 和运行生命周期 | 加载 Scheduler policy 与运行模式配置 |
| [`tradingagents/agents/utils/agent_states.py`](../tradingagents/agents/utils/agent_states.py) | 定义全局 AgentState 与辩论状态 | 增加 Scheduler action、轨迹和成本字段 |
| [`tradingagents/agents/schemas.py`](../tradingagents/agents/schemas.py) | 定义 ResearchPlan、TraderProposal、PortfolioDecision | 为 Static Teacher 评分提供稳定结构化字段 |
| [`tradingagents/default_config.py`](../tradingagents/default_config.py) | 定义当前运行配置 | 增加 Scheduler mode、模型、checkpoint 和步数配置 |

## 3. 优化目标与范围

### 3.1 目标

1. 将 Agent 间调用顺序从固定边转换为小模型策略决策。
2. 允许 Scheduler 根据当前报告和历史路径选择、跳过或重复 Expert Agent。
3. 保持 Expert Agent、工具权限、工具参数生成和 ToolNode 执行逻辑不变。
4. 从 Scheduler 自己生成的多轮轨迹中计算 Reward，并后训练 Scheduler。
5. 保存完整调度轨迹，使每个动作、成本和终局结果可回放。
6. 保留 Static Graph，形成 Static/Learned 同条件 A/B 对比。

### 3.2 非目标

本项目不包含：

- Expert Agent 参数训练；
- ToolNode 选择或工具参数策略训练；
- 专家 Prompt 自动优化；
- 真实券商订单执行；
- 以未来股票收益为主要训练标签；
- 细粒度 Process Reward Model；
- 多策略联合训练或完整 MARL；
- 论文级大规模 baseline 和消融矩阵。

### 3.3 核心架构决策

| 决策 | 选择 | 原因 |
|---|---|---|
| Scheduler 类型 | 小型 Causal Language Model + LoRA | 能直接读取文本化全局状态，并输出离散 Agent action token |
| 训练对象 | 仅 Scheduler | 隔离调度变量，控制训练成本和非平稳性 |
| 工具策略 | 冻结在 Expert Agent 内部 | 保持职责分离，使 A/B 只比较 Agent 编排 |
| Reward | Static Teacher 决策一致性 + 推理成本 | 无需等待市场结果，评分稳定且可重复 |
| 信用分配 | 轨迹级 Group-Relative Advantage | 不增加 Critic，符合项目复杂度 |
| 运行模式 | `static` / `learned` 双模式 | 保证向后兼容、回退和 A/B 对比 |

### 3.4 从原系统到优化系统的改造映射

| 原系统模块 | 原行为 | 改造后行为 | 是否训练 |
|---|---|---|---|
| `TradingAgentsGraph` | 只装配 Quick/Deep LLM 和固定图 | 额外装配 Scheduler policy、mode 和 trajectory recorder | 否 |
| `GraphSetup` | 编译固定 Agent 边 | 保留 Static Graph，并新增以 Scheduler 为中心的 Learned Graph | 否 |
| `ConditionalLogic` | 同时控制 Tool、Bull/Bear、Risk 路由 | Learned 模式只保留 Expert Agent 内部 Tool 路由 | 否 |
| `AgentState` | 保存报告、辩论、交易和风险状态 | 增加 Scheduler action、历史、成本和 trajectory 引用 | 否 |
| Expert Agent | 根据固定顺序被调用 | 根据 Scheduler action 被动态调用 | 否，冻结 |
| ToolNode | 接收 Expert Agent tool_calls 并执行 | 完全保持原行为 | 否，冻结 |
| Structured Schemas | 规范 Manager/Trader/PM 输出 | 同时作为 Static Teacher 质量评分字段 | 否 |
| Static Graph | 唯一生产路径 | 成为 baseline、fallback 和 Static Teacher 结果参考 | 否 |
| Scheduler LM | 不存在 | 读取全局状态并输出下一个 Agent Action Token | 是，SFT+GRPO |
| Trajectory Store | 不存在 | 保存状态、action、logprob、observation、成本和 Reward | 否 |
| Reward/Trainer | 不存在 | 对调度轨迹评分并更新 Scheduler LoRA | 是，更新 Scheduler |

改造不是重写 TradingAgents，而是在现有 Harness 外层增加可学习控制面：

```text
原有数据源、Expert Agent、ToolNode、报告、记忆、checkpoint
                          │
                          │ 全部复用
                          ▼
               Learned Scheduler 控制面
                          │
                trajectory/reward/trainer
```

## 4. 目标架构

### 4.1 总体执行链

```text
Task(ticker, date, asset_type)
                │
                ▼
        Small-LM Scheduler
                │
                │ 生成一个 Agent action token
                ▼
          Selected Expert Agent
                │
                ├─ Agent 内部 tool_calls
                │        ↓
                │      ToolNode
                │        ↓
                │   Tool observation
                │        ↓
                └──── 返回 Expert Agent
                         │
                         ▼
                 更新 AgentState
                         │
                         ▼
                   Scheduler
                         │
                 继续选择或 STOP
```

### 4.2 两层策略边界

高层 Scheduler 学习：

\[
\pi_\theta(a_t^{agent}\mid s_t^{global})
\]

Expert Agent 保留已有工具策略：

\[
\pi_{expert}(a_t^{tool}, args_t\mid s_t^{local})
\]

工具调用会作为 observation 和成本进入 Scheduler 轨迹，但工具名称、参数和调用顺序不属于 Scheduler 动作空间，也不参与 Scheduler 的策略梯度。

### 4.3 动态实际执行图

系统不设置显式 `analysis/research/risk` 阶段变量。每次任务由 Scheduler 形成不同路径，例如：

```text
Market → Fundamentals → Bull → News → Bear
→ Research Manager → Trader → Conservative
→ Portfolio Manager → STOP
```

或：

```text
Sentiment → News → Market → Bull → Bear → Bull
→ Research Manager → Trader → Aggressive → Neutral
→ Portfolio Manager → STOP
```

固定部分仅为节点集合、数据依赖、工具权限、最大步数和失败回退；实际 Agent 顺序由 Scheduler 学习。

### 4.4 改造前后完整流程对照

改造前：

```text
Task
  ↓
GraphSetup 编译固定路径
  ↓
Analyst 固定串行
  ↓
Bull/Bear 固定轮转
  ↓
Research Manager → Trader
  ↓
Risk 固定轮转
  ↓
Portfolio Manager → END
```

改造后：

```text
Task
  ↓
GraphSetup 编译 Scheduler SuperGraph
  ↓
Scheduler 读取当前 AgentState
  ↓
Action Mask 过滤缺少必要输入的动作
  ↓
小模型输出一个 Agent Action Token
  ↓
Expert Agent 执行，内部 ToolNode 循环保持不变
  ↓
AgentState 更新，成本和 transition 写入 trajectory
  ↓
控制权返回 Scheduler
  ↓
重复，直到 PortfolioDecision + STOP
```

### 4.5 控制面与执行面

系统明确分成两个平面：

```text
控制面（新增）
├── Scheduler LM
├── Action Mask
├── Trajectory Recorder
├── Reward Scorer
└── SFT/GRPO Trainer

执行面（复用）
├── 12 个 Expert Agent
├── Quick/Deep LLM Provider
├── ToolNode 与数据 vendor
├── AgentState
├── Memory/Checkpoint
└── Reporting
```

控制面决定“调用谁”，执行面负责“把该 Agent 的专业任务完成”。这种分离使 Scheduler 训练失败时仍可回退到 Static Graph，也使 Static/Learned A/B 只改变 Agent 编排策略。

## 5. Scheduler 动作空间

### 5.1 特殊动作 Token

在 Scheduler tokenizer 中增加单 Token 动作：

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

模型每次只允许生成一个动作 Token，不生成自然语言解释。

### 5.2 最低数据依赖

Action Mask 不定义固定顺序，只阻止代码无法执行或缺少必要输入的动作。

| Action | 最低前置条件 |
|---|---|
| Market/Sentiment/News/Fundamentals | 无 |
| Bull/Bear | 无；报告可以为空，但低信息轨迹通常获得较低 Reward |
| Research Manager | 至少一条 Bull 或 Bear debate history |
| Trader | `investment_plan` 非空 |
| Aggressive/Conservative/Neutral | `trader_investment_plan` 非空 |
| Portfolio Manager | `trader_investment_plan` 非空且至少一条 risk history |
| STOP | `final_trade_decision` 非空 |

Action Mask 应直接作用于动作 Token logits，而不是生成非法文本后再解析：

```python
masked_logits = logits.masked_fill(~valid_action_mask, float("-inf"))
```

### 5.3 终止规则

- 正常终止：Portfolio Manager 已生成 `final_trade_decision`，Scheduler 选择 `<ACT_STOP>`。
- 最大步数：默认 `scheduler_max_steps=16`。
- 解析失败：只允许在有效 Token 集合内重试一次。
- 重试仍失败：终止本次 Learned trajectory，并从原任务输入重新运行真正的 `static` LangGraph，记录 `fallback_reason` 和重复成本。

## 6. Scheduler 状态表示

### 6.1 状态来源

Scheduler 输入来自全局 `AgentState`：

- ticker、日期、资产类型和 instrument context；
- 四类 Analyst report；
- Bull/Bear debate history；
- Research Manager investment plan；
- Trader proposal；
- Risk debate history；
- Portfolio Manager final decision；
- 已调用 Agent 序列；
- Agent/Tool 调用数、Token、延迟和剩余步数；
- 当前合法动作列表。

### 6.2 Prompt 模板

```text
<TASK>
ticker=NVDA
date=2024-05-10
asset_type=stock

<STATE_STATUS>
market_report=completed
sentiment_report=empty
news_report=empty
fundamentals_report=completed
investment_plan=empty
trader_plan=empty
risk_history=empty
final_decision=empty

<EXECUTION_HISTORY>
<ACT_MARKET> <ACT_FUNDAMENTALS>

<LATEST_CONTEXT>
Market: bullish momentum with elevated volatility...
Fundamentals: strong growth with expensive valuation...

<COST>
agent_calls=2
tool_calls=6
tokens=5200
remaining_steps=14

<VALID_ACTIONS>
<ACT_MARKET>
<ACT_SENTIMENT>
<ACT_NEWS>
<ACT_FUNDAMENTALS>
<ACT_BULL>
<ACT_BEAR>
```

目标输出：

```text
<ACT_NEWS>
```

### 6.3 上下文压缩

Scheduler 不需要读取所有原始 ToolMessage。状态序列化只保留：

1. 当前结构化字段是否存在；
2. 每份报告的限长摘要；
3. 最近若干 Scheduler action；
4. 累计成本和合法动作。

报告摘要由固定规则截断或现有报告的首要段落生成，不再引入额外可训练摘要模型。Scheduler 上下文最大长度建议从 2K–4K tokens 起步。

## 7. 轨迹数据设计

### 7.1 任务数据

最小任务记录：

```json
{
  "task_id": "NVDA_2024-05-10",
  "ticker": "NVDA",
  "trade_date": "2024-05-10",
  "asset_type": "stock",
  "split": "train"
}
```

任务数据来自历史 ticker/date 组合。训练、验证和 A/B 测试使用不重叠的任务清单。项目第一版不要求大规模样本，规模由实际 API 和算力预算决定。

任务集需要覆盖不同的信息需求，而不是只随机抽取相似股票：

| 任务类型 | 主要特征 | 希望覆盖的调度行为 |
|---|---|---|
| 技术面主导 | 趋势或波动信号明显，新闻较少 | Scheduler 能识别 Market 的高价值，避免无效新闻调用 |
| 新闻事件主导 | 财报、产品、监管、并购或宏观事件 | Scheduler 能优先调用 News/Sentiment |
| 基本面主导 | 估值、收入、利润、现金流变化突出 | Scheduler 能优先调用 Fundamentals |
| 证据冲突 | 技术面、新闻、基本面方向不一致 | Scheduler 能增加 Bull/Bear 或相关补充分析 |
| 信息稀缺 | 新闻、社交或某些 vendor 数据不可用 | Scheduler 能绕过低信息 Agent，而不是重复调用 |
| 风险敏感 | 高波动、快速涨跌或 Trader plan 较激进 | Scheduler 能选择合适的 Risk Agent |

数据划分原则：

```text
train：用于 Static Teacher、强模型候选、SFT 和 RL rollout
validation：用于选择 checkpoint 和 Reward 权重
test：只用于最终 Static/Learned A/B，不参与蒸馏和 RL
```

为了避免同一 ticker/date 的近似副本同时出现在训练和测试中，任务清单应按 ticker 或时间块分组后再切分。该要求用于保证 A/B 结果可解释，不要求论文级市场泛化实验。

### 7.2 Scheduler Transition

```python
@dataclass
class SchedulerTransition:
    task_id: str
    step_index: int
    scheduler_input: str
    valid_action_ids: list[int]
    action_id: int
    old_logprob: float
    agent_name: str
    observation_ref: str
    agent_calls: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    latency_ms: float
    done: bool
```

### 7.3 Scheduler Trajectory

```python
@dataclass
class SchedulerTrajectory:
    trajectory_id: str
    task_id: str
    policy_version: str
    transitions: list[SchedulerTransition]
    trader_action: str | None
    portfolio_rating: str | None
    completed: bool
    fallback_reason: str | None
    total_agent_calls: int
    total_tool_calls: int
    total_tokens: int
    total_latency_ms: float
    reward_components: dict[str, float]
    total_reward: float
```

### 7.4 Tool trace 边界

Tool trace 保存：

- 工具名称；
- 参数的可序列化摘要；
- 成功/失败；
- 延迟；
- observation 文件引用。

Tool trace 不保存为 Scheduler action，也不对工具调用 Token 计算策略 Loss。

### 7.5 数据目录

```text
data/scheduler/
├── tasks/
│   ├── train.jsonl
│   ├── validation.jsonl
│   └── test.jsonl
├── teacher/
│   ├── static_results.jsonl
│   ├── route_candidates.jsonl
│   ├── accepted_trajectories.jsonl
│   └── rejected_trajectories.jsonl
├── sft/
│   └── scheduler_sft.jsonl
├── rollouts/
│   └── <policy_version>/*.jsonl
└── observations/
    └── <task_id>/*
```

Tool observation 可按 `(tool_name, normalized_args, data_snapshot_id)` 缓存，保证同一任务的 Static/Learned 轨迹读取相同数据。不同调度路径中的 Agent report 不跨上下文复用，因为其输入状态不同。

## 8. 数据生成与蒸馏流程

### 8.1 三个数据角色

本方案不把任何单一大模型输出直接视为正确答案，而是组合三个角色：

| 角色 | 作用 | 不承担的职责 |
|---|---|---|
| Static Teacher | 提供原完整流程、Trader action、Portfolio rating 和成本基线 | 不规定 Learned Scheduler 必须复制固定路径 |
| Strong Route Teacher | 在部分 AgentState 上提出较合理、较多样的下一 Agent 候选 | 不直接决定候选轨迹是否合格 |
| TradingAgents Environment Verifier | 真实执行候选调用路径，检查合法性、完成度、结构化结果和成本 | 不生成 Scheduler 训练参数 |

核心原则：

```text
强模型负责提出候选
真实 Harness 负责执行候选
Reward 与过滤规则负责判断候选
小模型负责蒸馏和后续 RL 学习
```

### 8.2 Static Teacher 数据

对每个训练任务运行 Static Graph，记录：

- 固定 Agent 序列；
- 各 Expert Agent 输出；
- Trader 三分类动作；
- Portfolio Manager 五级评级；
- Agent/Tool 调用数；
- Token 和延迟；
- 完整数据快照标识。

Static Teacher 的作用：

1. 提供至少一条保证可完成的基础 trajectory；
2. 生成动作语法和基本依赖的 SFT 冷启动样本；
3. 为候选和 RL trajectory 提供稳定的结构化质量参考；
4. 提供 Static/Learned A/B 基线。

Static Teacher 的调用路径不是唯一正确路径，也不是 RL next-action 标签。

### 8.3 Strong Route Teacher 的上下文包

Strong Route Teacher 没有经过本项目的参数训练。它通过版本化 Prompt 在单次推理上下文中获得这套系统的局部知识。上下文包必须由现有代码、配置和已验证轨迹提炼，不能只给一句“选择下一个 Agent”。

上下文包由五个版本化资源构成：

```text
teacher/
├── agent_catalog.yaml          # Agent 能力、读写状态、前置条件和成本
├── state_schema.yaml           # AgentState 字段、完成标志和摘要规则
├── orchestration_rules.yaml    # 硬约束、预算和 STOP 条件
├── positive_examples.jsonl     # Static 与已验证动态示例
└── failure_examples.jsonl      # 错误动作、失败原因和纠正动作
```

这些资源的事实来源如下：

| Teacher 资料 | 当前代码事实来源 | 提炼内容 |
|---|---|---|
| Agent 清单 | `graph/setup.py`、各 `agents/**` factory | Agent 名称、职责、调用入口 |
| Analyst 注册 | `graph/analyst_execution.py` | `market/social/news/fundamentals` 与报告字段映射 |
| Tool 归属 | `TradingAgentsGraph._create_tool_nodes()` | 哪个 Analyst 内部能够调用哪些工具 |
| 状态说明 | `agents/utils/agent_states.py` | 报告、辩论、Trader、风险和最终决策字段 |
| 固定顺序 | `graph/setup.py` | Static 演示轨迹和回归基线 |
| 硬条件 | `graph/conditional_logic.py` | 工具循环、辩论轮数和终止边界 |
| 结果 Schema | `agents/schemas.py` | TraderProposal 与 PortfolioDecision 完成标准 |

`agent_catalog.yaml` 中每个 Agent 至少声明：

```yaml
id: news
action_token: <ACT_NEWS>
purpose: 分析公司新闻、全球新闻、宏观事件和内幕交易
reads:
  - company_of_interest
  - trade_date
  - market_report
writes:
  - news_report
prerequisites: []
use_when:
  - 缺少事件证据
  - 价格变化可能由新闻驱动
avoid_when:
  - 已有同一数据快照的充分新闻报告且没有新增事件
internal_tools:
  - get_news
  - get_global_news
  - get_insider_transactions
tool_policy_owner: expert_agent
```

该目录是 Teacher Prompt、Scheduler Action Registry、Action Mask、轨迹标签和测试的共同事实来源。不能分别维护多套互相漂移的 Agent 名称和能力说明。

### 8.4 固定的 Teacher Prompt 协议

Prompt 以 `teacher_prompt_version=v1` 冻结，分为固定部分和每步动态部分。

固定部分：

```text
<ROLE_AND_OBJECTIVE>
你是 TradingAgents 的 Agent 编排候选生成器。
在保证流程合法、证据充分和最终决策完整的前提下，
从 VALID_ACTIONS 中选择信息价值最高的下一位 Expert Agent。
你不调用工具、不生成交易结论，也不修改 AgentState。

<AGENT_CATALOG>
精简后的 Agent Card 列表。

<STATE_SCHEMA>
状态字段、完成标志和缺失含义。

<HARD_RULES>
只包含绝对不可违反的前置条件、STOP 条件和预算规则。

<EXAMPLES>
少量 Static 安全示例、已验证动态示例和显式纠正的失败示例。

<OUTPUT_SCHEMA>
只返回 next_agent、reason_code 和简短 reason。
```

每一步动态部分：

```text
<TASK>
ticker/date/asset_type/task_category/data_snapshot_id

<STATE_STATUS>
哪些报告、辩论、Trader 计划和最终决策已完成。

<EVIDENCE_SUMMARY>
各报告的限长结论、冲突和缺失信息。

<EXECUTION_HISTORY>
已执行 Agent、是否产生新信息、失败和重试。

<BUDGET>
剩余 Agent 步数、Token、Tool 和时间预算。

<VALID_ACTIONS>
由程序计算的合法 Agent Action Token。
```

Teacher 每次只允许返回一个动作：

```json
{
  "next_agent": "<ACT_FUNDAMENTALS>",
  "reason_code": "MISSING_FUNDAMENTAL_EVIDENCE",
  "reason": "监管新闻可能影响长期估值，需要补充基本面证据。"
}
```

`next_agent` 是训练候选；`reason_code` 和 `reason` 只用于审计、错误分析和人工抽查。Teacher 不一次性生成完整路径，因为后续动作必须依赖前一个 Expert Agent 的真实 observation。

同一任务需要多条候选路径时，采用多次完整 rollout 或不同随机种子，而不是一次返回多个互不执行的 next-action 列表。MVP 每个任务先生成 2 条 Teacher trajectory；只有数据多样性不足时才提高到 3 条。

### 8.5 正例与失败示例规则

Teacher Few-shot 示例控制在少量代表性案例，避免上下文过长：

- 常规 Static 完整路径；
- 新闻驱动的较短动态路径；
- 基本面驱动路径；
- 证据冲突后触发 Bull/Bear 的路径；
- 数据缺失后补充证据的路径；
- 风险信息充分后的合法 STOP 边界。

失败示例必须使用“错误动作 + 失败原因 + 纠正动作”格式：

```json
{
  "state_case": "market_only_no_trade_plan",
  "invalid_action": "<ACT_STOP>",
  "failure_reason": "缺少 investment_plan、trader_investment_plan 和 final_trade_decision",
  "corrected_actions": ["<ACT_NEWS>", "<ACT_FUNDAMENTALS>", "<ACT_BULL>"]
}
```

首批失败集合至少覆盖：

- 证据不足时过早 STOP；
- Trader 之前缺少 `investment_plan`；
- Portfolio Manager 之前缺少 Trader 计划或 risk history；
- 同一 Agent 连续调用但没有新增状态；
- 超出研究或风险辩论轮数；
- Teacher 输出不存在的 Agent；
- Tool/Expert 失败后重复同一无效路线；
- 忽略互相冲突的关键证据；
- 超过最大步数或成本预算；
- 最终输出 Schema 不完整。

失败数据的使用边界：

| 用途 | 使用方式 |
|---|---|
| Teacher Prompt | 只放带明确纠正的少量反例 |
| 普通 SFT | 只把 accepted action 作为生成标签；错误动作不当正标签 |
| 单元测试 | 固定验证 mask、过滤器和 Reward 排序 |
| RL | 作为低 Reward 轨迹降低错误动作概率 |
| 错误分析 | 统计失败类别，决定是否补规则、数据或状态信息 |

### 8.6 Strong Route Teacher 候选生成

从 Static Graph 执行中抽取不同深度的部分状态，或通过合法顺序扰动构造部分状态，然后把以下信息交给较强的大模型：

```text
任务信息
当前已有报告和辩论历史
已执行 Agent 序列
累计调用成本
当前合法 Agent actions
```

Strong Route Teacher 每次输出一个候选动作：

```json
{
  "next_agent": "<ACT_NEWS>",
  "reason_code": "MISSING_EVENT_EVIDENCE",
  "reason": "当前价格异常尚无事件解释。"
}
```

Strong Route Teacher 的自然语言理由只用于调试和人工阅读，不进入 Scheduler SFT target。SFT target 仍然只是单个 Agent Action Token。

为了避免蒸馏退化为固定顺序，候选状态需要包含：

- 四类 Analyst 的不同完成组合；
- 技术、新闻、基本面相互一致和冲突的状态；
- Bull/Bear 已进行不同轮数的状态；
- Trader plan 已产生但风险信息不同的状态；
- 数据缺失、工具失败和重复调用后的状态；
- 可以继续调用和可以结束之间的边界状态。

### 8.7 候选路径真实执行与过滤

Strong Route Teacher 提议的动作不能直接进入 SFT 数据。必须放回真实 TradingAgents Learned Graph 执行：

```text
部分 AgentState
  ↓
执行候选 Agent action
  ↓
Expert Agent 内部完成 LLM/ToolNode 流程
  ↓
得到新 AgentState
  ↓
继续由 Strong Route Teacher 或候选策略生成后续动作
  ↓
形成完整候选 trajectory
```

候选轨迹的过滤顺序：

1. 所有 Action Token 均合法；
2. Expert Agent 执行没有未处理异常；
3. 在最大步数内产生 TraderProposal 和 PortfolioDecision；
4. 最终评级与 Static Teacher 的距离在设定阈值内；
5. 没有无进展循环；
6. Agent/Tool/Token 成本没有超过上限；
7. 与已有 SFT 轨迹相比具有新的状态或调用顺序。

每条候选标记为：

```text
accepted：进入 SFT 冷启动集
rejected：保留失败原因，用于边界/负例构造或调试
```

这一步使蒸馏数据代表“强模型提出且真实 Harness 验证通过的调度行为”，而不是未经执行验证的大模型偏好。

### 8.8 SFT 冷启动数据

SFT 样本由 Scheduler 输入状态和单个目标 Action Token 组成：

```json
{
  "messages": [
    {"role": "system", "content": "Select exactly one valid agent action token."},
    {"role": "user", "content": "<TASK>...<STATE>...<VALID_ACTIONS>..."},
    {"role": "assistant", "content": "<ACT_MARKET>"}
  ],
  "source": "static|strong_teacher_verified",
  "trajectory_id": "..."
}
```

SFT 数据由以下部分组成：

- Static Graph 的可完成基础轨迹；
- Strong Route Teacher 提议且 Environment Verifier 接受的多样轨迹；
- STOP、重复 Agent、缺少必要字段等边界状态；
- 少量 rejection-derived 边界状态；只监督经过规则确认的纠正动作，绝不把失败动作作为 SFT 正标签。

SFT 只学习：

- Scheduler 输入格式；
- Action Token 语法；
- 基本数据依赖；
- 常见状态下的合理初始动作分布；
- STOP 和回退边界。

SFT 不承担最终成本优化，也不把 Strong Route Teacher 视为永远正确。

### 8.9 RL Rollout 数据

SFT 完成后，RL 阶段不再提供 next-action 标签。对每个训练任务，用同一个 Scheduler policy 采样 `group_size=4` 条轨迹。每一步：

1. 序列化当前 `AgentState`；
2. 计算合法 Action Mask；
3. Scheduler 从动作分布采样一个 Agent action；
4. LangGraph 调用对应冻结 Expert Agent；
5. Expert Agent 内部按原逻辑调用 ToolNode；
6. Expert Agent 输出写回 `AgentState`；
7. 保存 transition、logprob、Tool trace 和成本；
8. 直到 STOP、最大步数或失败回退。

RL 数据完全由当前 Scheduler 与冻结 TradingAgents 环境交互产生。Static Teacher 只在轨迹结束后提供结构化质量参考，不提供中间动作标签。

## 9. Reward 与打分方案

### 9.1 设计目标

Reward 同时鼓励：

1. 产生有效的结构化最终决策；
2. 接近完整 Static Teacher 的决策质量；
3. 减少不必要的 Agent/Tool 调用；
4. 减少 Token 和延迟；
5. 避免非法、循环或未完成轨迹。

第一版不直接使用未来股票收益作为训练 Reward。项目目标是学习高效编排，而不是证明交易策略盈利。

### 9.2 Portfolio Manager 质量分

五级评级映射：

```text
Buy          4
Overweight   3
Hold         2
Underweight  1
Sell         0
```

若 Learned 和 Static 分别为 `y_l`、`y_s`：

\[
Q_{PM}=1-\frac{|y_l-y_s|}{4}
\]

无有效 PortfolioDecision 时，`Q_PM=0`。

### 9.3 Trader 质量分

Trader 三分类映射：

```text
Buy   2
Hold  1
Sell  0
```

\[
Q_{Trader}=1-\frac{|a_l-a_s|}{2}
\]

无有效 TraderProposal 时，`Q_Trader=0`。

### 9.4 完成奖励

```text
有效 PortfolioDecision 且正常 STOP： +0.20
超过最大步数且未完成：               -1.00
动作 Token 无法解析：                -1.00
所有动作被 Mask：                    -1.00 并 fallback
```

### 9.5 成本惩罚

初始成本函数：

\[
C=
0.02N_{agent}
+0.005N_{tool}
+0.005\frac{N_{tokens}}{1000}
\]

延迟可先作为评估指标，避免 Token、调用量和延迟高度相关时重复惩罚。若后续确需纳入，新增独立权重。

### 9.6 重复与无进展惩罚

对调度前后的关键状态字段生成稳定摘要：

```text
market_report
sentiment_report
news_report
fundamentals_report
investment_debate_state.count
investment_plan
trader_investment_plan
risk_debate_state.count
final_trade_decision
```

若连续两次选择同一 Agent，且关键状态摘要没有变化：

```text
P_repeat = 0.10
```

### 9.7 总 Reward

\[
R(\tau)=
0.7Q_{PM}
+0.3Q_{Trader}
+R_{complete}
-C
-P_{repeat}
-P_{invalid}
\]

最终 Reward 建议裁剪到：

\[
R(\tau)\in[-1.5, 1.2]
\]

### 9.8 打分示例

假设：

```text
Learned PM rating 与 Static 完全一致：Q_PM=1.0
Learned Trader action 与 Static 一致： Q_Trader=1.0
正常完成：                            +0.20
Agent 调用 9 次：                     -0.18
Tool 调用 6 次：                      -0.03
总 Token 18000：                      -0.09
无非法动作、无无效重复：               0
```

则：

\[
R=0.7+0.3+0.2-0.18-0.03-0.09=0.90
\]

若 Scheduler 为了省成本过早调用 Portfolio Manager，导致最终评级与 Static Teacher 相反，则质量分下降，无法仅靠减少调用获得高 Reward。

## 10. 粗粒度信用分配

### 10.1 选择

本项目采用 trajectory-level group-relative credit，不构建细粒度 Agent contribution、价值网络、过程奖励模型或逐步反事实重放。

对同一任务采样 `G=4` 条轨迹：

\[
\tau_1,\tau_2,\tau_3,\tau_4
\]

得到：

\[
R_1,R_2,R_3,R_4
\]

组相对优势：

\[
A_i=\frac{R_i-\operatorname{mean}(R)}
{\operatorname{std}(R)+\epsilon}
\]

同一轨迹内所有 Scheduler action token 共享 `A_i`。

### 10.2 含义

- 高质量、低成本轨迹中的 Agent action 概率整体提高；
- 低质量、冗余或未完成轨迹中的 action 概率整体降低；
- Agent/Tool/Token 成本已经进入总 Reward，因此较短而有效的轨迹更容易获得正优势；
- 该设计不声称识别某个 Agent 的精确边际贡献。

这种信用分配粒度足以展示多轮轨迹、结果评分、组相对优势和策略更新的完整 Agentic RL 链路，同时保持项目可实现。

## 11. Scheduler 后训练

### 11.1 模型状态

训练维护三个策略状态：

- `pi_ref`：SFT 完成后冻结的参考 Scheduler；
- `pi_old`：生成当前 rollout 的旧策略；
- `pi_theta`：正在更新的 Scheduler。

只训练 Scheduler LoRA 参数。

### 11.2 Loss Mask

Scheduler 会读取任务、Agent report、Tool observation 和历史 action，但只有 Scheduler 生成的动作 Token 参与 Loss：

```text
Task / AgentState / reports / ToolMessage   loss_mask=0
Scheduler action token                      loss_mask=1
```

这与 Search-R1 中“环境 observation 作为上下文、模型 action 作为策略输出”的边界一致。

### 11.3 GRPO-style 目标

对动作 Token：

\[
\rho_{i,t}=
\frac{\pi_\theta(a_{i,t}\mid s_{i,t})}
{\pi_{old}(a_{i,t}\mid s_{i,t})}
\]

策略目标：

\[
L_{policy}=
-\frac{1}{G}
\sum_i
\sum_t m_{i,t}
\min\left(
\rho_{i,t}A_i,
\operatorname{clip}(\rho_{i,t},1-\varepsilon,1+\varepsilon)A_i
\right)
\]

加入参考策略 KL：

\[
L=L_{policy}+\beta D_{KL}(\pi_\theta\|\pi_{ref})
\]

其中 `m_{i,t}` 仅在 Scheduler action token 位置为 1。

### 11.4 初始训练配置

以下为实现起点，不是最终实验结论：

```yaml
scheduler:
  base_model: <set-on-autodl-before-training>
  base_model_revision: <freeze-before-training>
  tokenizer_revision: <freeze-before-training>
  max_context_tokens: 4096
  max_steps: 16
  action_temperature: 0.8

lora:
  rank: 16
  alpha: 32
  dropout: 0.05

sft:
  epochs: 1-3
  learning_rate: 2.0e-5

grpo:
  group_size: 4
  clip_epsilon: 0.2
  kl_beta: 0.01
  learning_rate: 1.0e-5
```

## 12. 完整训练循环

```text
1. 冻结 Agent Registry、Action Schema、State Schema 和 Teacher Prompt V1
2. 准备 ticker/date 任务清单并冻结 train/validation/test
3. 用 Static Graph 生成 Static Teacher 结果和基础 SFT 轨迹
4. 从 Static 轨迹提取正例，并编写带纠正动作的失败示例
5. 将当前状态、合法动作和上下文包发送给冻结 Strong Route Teacher API
6. Teacher 每次只提出一个 next-agent action
7. 在真实 TradingAgents Harness 中执行该 Expert Agent
8. 用更新后的 AgentState 继续 Teacher 决策，直到 STOP 或失败
9. 对每个任务重复两次，得到多样化 Teacher trajectory
10. 过滤非法、未完成、高成本、无进展和低质量候选
11. 将 accepted trajectory 拆成 step-level SFT action 数据
12. 合并 Static、verified Teacher 和合法边界状态
13. 用 SFT 教 Scheduler 状态格式、动作 Token 和终止边界
14. 冻结 SFT checkpoint 为 pi_ref
15. 复制当前策略为 pi_old
16. 每个训练任务采样 4 条 Learned Scheduler 轨迹
17. 每个 Scheduler action 调用真实冻结 Expert Agent
18. Expert Agent 内部照常调用 ToolNode
19. 收集 action logprob、Agent 输出、Tool trace、Token 和延迟
20. 使用完成性、结构化质量与成本计算 trajectory Reward
21. 在同任务轨迹组内计算 Group-Relative Advantage
22. 只在 Scheduler action Token 上计算 GRPO-style Loss
23. 更新 LoRA，保存新 policy version
24. 使用新策略继续下一轮 rollout
25. 在 held-out 任务上运行 Static/Learned-SFT/Learned-RL A/B
```

## 13. LangGraph 改造方案

### 13.1 双图模式

```python
def setup_graph(..., scheduler_mode: str = "static"):
    if scheduler_mode == "learned":
        return setup_learned_graph(...)
    return setup_static_graph(...)
```

`setup_static_graph()` 保留当前实现不变。

### 13.2 Learned Graph

```python
workflow.add_node("Scheduler", scheduler_node)
workflow.add_edge(START, "Scheduler")

workflow.add_conditional_edges(
    "Scheduler",
    route_scheduler_action,
    SCHEDULER_PATH_MAP,
)
```

路径表：

```python
SCHEDULER_PATH_MAP = {
    "market": "Market Analyst",
    "sentiment": "Sentiment Analyst",
    "news": "News Analyst",
    "fundamentals": "Fundamentals Analyst",
    "bull": "Bull Researcher",
    "bear": "Bear Researcher",
    "research_manager": "Research Manager",
    "trader": "Trader",
    "aggressive": "Aggressive Analyst",
    "conservative": "Conservative Analyst",
    "neutral": "Neutral Analyst",
    "portfolio_manager": "Portfolio Manager",
    "stop": END,
}
```

### 13.3 Agent 出口

Learned 模式下：

```text
Msg Clear Market       → Scheduler
Msg Clear Sentiment    → Scheduler
Msg Clear News         → Scheduler
Msg Clear Fundamentals → Scheduler
Bull Researcher        → Scheduler
Bear Researcher        → Scheduler
Research Manager       → Scheduler
Trader                 → Scheduler
Aggressive Analyst     → Scheduler
Conservative Analyst   → Scheduler
Neutral Analyst        → Scheduler
Portfolio Manager      → Scheduler
```

原有 Analyst → ToolNode → Analyst 循环保持不变。Learned 模式不注册原有 Bull/Bear 和 Risk 固定轮转条件边。

### 13.4 AgentState 扩展

```python
scheduler_action: str
scheduler_step: int
scheduler_history: list[str]
scheduler_trace: list[dict]
agent_call_count: int
tool_call_count: int
total_tokens: int
total_latency_ms: float
last_agent: str
fallback_reason: str
```

`Propagator.create_initial_state()` 还应显式初始化：

```text
investment_plan=""
trader_investment_plan=""
final_trade_decision=""
```

确保 Scheduler 探索不同顺序时不会因为缺失键而崩溃。

### 13.5 逐文件改造清单

| 文件 | 必要修改 | 保持不变的内容 | 主要验证 |
|---|---|---|---|
| `tradingagents/graph/setup.py` | 提取 Static Graph，增加 Scheduler node、Learned Graph 和 Agent→Scheduler 出口 | Agent factory 与 ToolNode 注册 | static 路径等价、learned 动态路由 |
| `tradingagents/graph/conditional_logic.py` | 暴露/复用 Analyst ToolNode 路由；Learned 模式不使用 debate/risk 固定轮转 | Tool call 检测 | ToolNode loop 无回归 |
| `tradingagents/agents/utils/agent_states.py` | 增加 Scheduler action、history、trace、成本和 fallback 字段 | 原 report/debate/risk 字段 | 状态合并与 checkpoint 序列化 |
| `tradingagents/graph/propagation.py` | 初始化所有 Scheduler 字段及可能提前访问的空输出字段 | recursion limit 与 graph args | 任意合法初始 action 不 KeyError |
| `tradingagents/graph/trading_graph.py` | 根据 config 加载 Scheduler policy，向 GraphSetup 注入 policy/recorder | Quick/Deep LLM、ToolNode、memory、signal processor | Python API static/learned 切换 |
| `tradingagents/default_config.py` | 增加 Scheduler 运行配置 | 原模型、数据 vendor、memory 配置 | 环境覆盖与类型转换 |
| `cli/main.py` | 暴露 scheduler mode/checkpoint 选择，显示调度动作与成本 | 现有状态面板和报告保存 | CLI 使用 compiled Learned Graph |
| `tradingagents/reporting.py` | 可选增加 scheduler trajectory/summary 报告入口 | 原五层业务报告树 | 原报告不回归 |
| `tradingagents/agents/**` | 不修改核心 Prompt 和角色行为 | 全部 Expert Agent 逻辑 | 现有 Agent 测试继续通过 |
| `tradingagents/dataflows/**` | 不修改 | 数据 vendor、缓存与错误处理 | Tool observation 可复用且无数据漂移 |

### 13.6 调用时序示例

```text
TradingAgentsGraph.propagate(task)
  ↓
Propagator.create_initial_state()
  ↓
Learned Graph: START → Scheduler
  ↓
SchedulerPolicy.select_action(state, valid_actions)
  ↓
TrajectoryRecorder.start_transition()
  ↓
LangGraph 路由到 Expert Agent
  ↓
Expert Agent → 可选 ToolNode 循环 → report/state update
  ↓
TrajectoryRecorder.finish_transition(cost, observation_ref)
  ↓
返回 Scheduler
  ↓
Portfolio Manager → Scheduler → STOP
  ↓
Reward.score(trajectory, static_teacher_result)
  ↓
保存 trajectory；训练模式下进入 rollout batch
```

## 14. 模块与接口

### 14.1 新增目录

```text
tradingagents/scheduler/
├── __init__.py
├── actions.py
├── action_mask.py
├── state_serializer.py
├── policy.py
├── scheduler_node.py
├── trajectory.py
├── trajectory_store.py
├── reward.py
├── static_teacher.py
├── route_teacher.py
├── candidate_filter.py
├── dataset.py
├── rollout.py
├── sft.py
├── grpo_trainer.py
└── evaluate.py
```

职责：

| 模块 | 职责 |
|---|---|
| `actions.py` | Action Token、Agent node 与稳定 action id 映射 |
| `action_mask.py` | 根据 AgentState 计算最低数据依赖和合法动作 |
| `state_serializer.py` | 将全局状态压缩为 Scheduler 输入文本 |
| `policy.py` | Static/Learned policy 加载、采样和 logprob 提取 |
| `scheduler_node.py` | LangGraph Scheduler node 与路由返回值 |
| `trajectory.py` | Transition/Trajectory 数据模型 |
| `trajectory_store.py` | JSONL/observation 引用的追加写入和回放 |
| `reward.py` | 结构化质量、完成、成本、重复和非法动作评分 |
| `static_teacher.py` | 运行 Static Graph 并生成结果参考 |
| `route_teacher.py` | 构造强模型请求并解析候选 Agent actions |
| `candidate_filter.py` | 执行结果合法性、完成度、质量、成本和去重过滤 |
| `dataset.py` | task manifest、SFT 样本和 split 加载 |
| `rollout.py` | 使用当前 policy 生成同任务多轨迹 |
| `sft.py` | Scheduler Action Token 冷启动训练 |
| `grpo_trainer.py` | Reward 归一化、组相对优势、Loss Mask、KL 和 LoRA 更新 |
| `evaluate.py` | Static/Learned A/B 和指标聚合 |

### 14.2 Policy 接口

```python
class SchedulerPolicy(Protocol):
    def select_action(
        self,
        scheduler_input: str,
        valid_action_ids: list[int],
        *,
        sample: bool,
    ) -> SchedulerDecision:
        ...
```

```python
@dataclass(frozen=True)
class SchedulerDecision:
    action: str
    action_id: int
    logprob: float
    policy_version: str
```

实现：

- `StaticSchedulerPolicy`：仅用于验证 Learned Graph 的 Scheduler 循环能否复现原路径，不作为产品第三模式或生产 fallback；
- `LearnedSchedulerPolicy`：加载 Scheduler 基模与 LoRA checkpoint。

### 14.3 Reward 接口

```python
class SchedulerReward:
    def score(
        self,
        trajectory: SchedulerTrajectory,
        teacher_result: StaticTeacherResult,
    ) -> RewardBreakdown:
        ...
```

```python
@dataclass(frozen=True)
class RewardBreakdown:
    pm_quality: float
    trader_quality: float
    completion_bonus: float
    agent_cost: float
    tool_cost: float
    token_cost: float
    repeat_penalty: float
    invalid_penalty: float
    total: float
```

## 15. 配置方案

在 `default_config.py` 增加：

```python
"scheduler_mode": "static",
"scheduler_base_model": None,
"scheduler_adapter_path": None,
"scheduler_max_steps": 16,
"scheduler_action_temperature": 0.0,
"scheduler_invalid_action_policy": "fallback_static",
"scheduler_fallback_enabled": True,
"scheduler_trace_enabled": True,
"scheduler_trace_dir": None,
"teacher_provider": None,
"teacher_model": None,
"teacher_prompt_version": "v1",
"teacher_trajectories_per_task": 2,
```

训练配置单独存放，避免运行配置和训练超参数混合。

## 16. Static/Learned A/B 评估

### 16.1 控制变量

两种模式必须使用相同：

- ticker/date 任务；
- Expert Agent 模型；
- Agent Prompt；
- Tool observation 快照；
- Tool vendor；
- 最大运行步数；
- 输出 Schema。

唯一变量是 Agent 编排策略。

### 16.2 指标

| 类别 | 指标 |
|---|---|
| 完成 | `completion_rate`、`invalid_action_rate`、`fallback_rate` |
| 质量 | PM 五级评级一致率、平均评级距离、Trader 三分类一致率 |
| 路径 | 平均轨迹长度、各 Agent 使用率、重复调用率 |
| 成本 | Agent 调用数、Tool 调用数、Token、延迟、估算费用 |

### 16.3 输出报告

```json
{
  "task_id": "NVDA_2024-05-10",
  "static": {
    "trajectory": ["market", "sentiment", "news"],
    "rating": "Buy",
    "agent_calls": 12,
    "tool_calls": 9,
    "tokens": 28000
  },
  "learned": {
    "trajectory": ["market", "fundamentals", "bull"],
    "rating": "Overweight",
    "agent_calls": 9,
    "tool_calls": 6,
    "tokens": 19000
  }
}
```

以上数值仅表示报告格式，不能作为项目结果使用。

## 17. 测试方案

### 17.1 单元测试

- Action Token 与 Agent node 一一映射；
- Action Mask 正确处理缺失前置字段；
- Reward 各组成项和总分计算正确；
- Static Scheduler 路径与 Static Graph 一致；
- Strong Route Teacher 输出只能映射到已注册 Action Token；
- Candidate Filter 能区分 accepted/rejected 并保留失败原因；
- Scheduler state serializer 输出稳定；
- Loss Mask 只覆盖 Scheduler action token；
- Trajectory JSONL 可序列化和回放。

### 17.2 图集成测试

- Learned Scheduler 可从 START 调用 Analyst；
- Analyst ToolNode 循环保持原行为；
- Agent 完成后返回 Scheduler；
- Portfolio Manager 后可正常 STOP；
- 最大步数触发 fallback；
- 缺少 checkpoint 时自动使用原 `static` LangGraph；
- CLI 和 Python API 都使用同一个 compiled learned graph。

### 17.3 训练 Smoke Test

使用 mock Expert Agent 和极小 Scheduler：

1. 构造两条质量相同但成本不同的轨迹；
2. 验证低成本轨迹 Reward 更高；
3. 完成一次 GRPO update；
4. 验证高 Reward action 的 log-prob 上升；
5. 验证 observation token 参数不产生策略 Loss。

## 18. 实施顺序

### Milestone 1：可切换的图编排

- 提取 Static Graph 为 `setup_static_graph()`；
- 增加 Scheduler node 和 `setup_learned_graph()`；
- 完成 Action Mask 和 Static fallback；
- 确保现有测试不回归。

### Milestone 2：轨迹与成本采集

- 记录 Scheduler transition；
- 统计 Agent/Tool 调用、Token 和延迟；
- 保存 Static Teacher 和 Learned trajectory；
- 支持 JSONL 回放。

### Milestone 3：蒸馏数据与候选过滤

- 运行 Static Teacher 任务集；
- 抽取和构造代表性部分状态；
- 调用 Strong Route Teacher 生成多样 Agent 候选；
- 在真实 Learned Graph 执行候选路径；
- 按合法性、完成度、质量、成本和去重规则过滤；
- 生成 accepted/rejected 数据清单。

### Milestone 4：小模型 SFT

- 注册 Action Token；
- 合并 Static 与 verified strong-teacher SFT 数据；
- 训练 LoRA；
- 验证动作格式和合法率。

### Milestone 5：GRPO-style 后训练

- 实现 group rollout；
- 实现 RewardBreakdown；
- 计算 Group-Relative Advantage；
- 实现 Action Token Loss Mask 和 KL；
- 保存 Scheduler checkpoint。

### Milestone 6：A/B 评估

- 冻结 Expert Agent 和数据快照；
- 在 held-out 任务运行 Static/Learned；
- 输出质量、路径和成本指标；
- 生成可用于简历的实测结果。

## 19. 失败模式与回退

| 失败模式 | 处理 |
|---|---|
| Scheduler checkpoint 不存在 | 启动时切换原 `static` LangGraph |
| 动作 Token 无法解析 | 在合法 Token 集合内重试一次 |
| 当前状态无合法动作 | 记录错误并使用 Static fallback |
| 超过最大步数 | 终止 Learned trajectory，从原任务输入重新运行 `static` LangGraph，并记录重复成本 |
| Expert Agent/Tool 失败 | 复用现有异常处理，不由 Scheduler 吞掉 |
| Reward 组内方差为零 | 跳过该任务组更新，避免除零和无效梯度 |
| Scheduler 总调用同一 Agent | 无进展惩罚 + 最大步数限制 |
| Learned 质量下降 | 保留 Static 模式，checkpoint 不自动替换当前最佳版本 |

## 20. 验收标准

### 功能验收

- `static` 模式保持 Static Graph 行为；
- `learned` 模式能生成不同 Agent 调度路径；
- Expert Agent 的工具调用仍由其原 LLM 和 ToolNode 完成；
- Scheduler trajectory 可完整保存和回放；
- 小模型能通过 SFT 生成合法 Action Token；
- 能完成至少一次 `rollout → reward → advantage → loss → update`；
- 能加载训练 checkpoint 运行 Learned Graph；
- 能输出 Static/Learned A/B 报告。

### 结果验收

实现完成后再设定和报告实测值。项目至少应证明：

1. Learned Scheduler 能稳定完成任务；
2. Agent 调度路径确实随任务状态变化；
3. 非法动作率受 Action Mask 控制；
4. 与 Static Graph 保持可解释的决策一致性；
5. Agent 调用、Tool 调用、Token 或延迟至少有一项可测变化。

## 21. 简历与项目说明边界

实现前只能表述为“设计方案”。完成训练闭环后，可写：

> 基于 LangGraph 将固定金融多 Agent 工作流重构为动态中央调度架构，使用轻量语言模型与 LoRA 构建 Scheduler Policy，将十二类 Expert Agent 选择建模为多轮决策过程；设计 Action Mask、轨迹采集、成本感知 Reward 和 GRPO-style 组相对优化，仅训练调度 Action Token，并保持 Expert Agent 与 ToolNode 执行解耦。

获得实测结果后，再增加：

> 构建 Static/Learned A/B 评估链路，在保持 X% 最终评级一致率的条件下，将平均 Agent 调用降低 Y%、Token 消耗降低 Z%。

`X/Y/Z` 必须来自真实 A/B 报告，不能预先填写。

## 22. 方法来源与借鉴边界

### 22.1 DeepSeek-R1 方法映射

本项目不复制 DeepSeek-R1 的模型规模，而是复用其“冷启动 → 结果验证 → 策略优化”的训练组织方式：

| DeepSeek-R1 训练角色 | 本项目对应物 |
|---|---|
| Base Model | 训练前在 AutoDL 配置中指定的 Scheduler 基模 |
| 冷启动高质量样本 | Static 安全轨迹 + verified Teacher 动态轨迹 |
| Few-shot/强模型生成 | Teacher Prompt V1 + 冻结强模型 API |
| Rejection Sampling | 真实 Harness 执行后按合法性、完成、质量和成本过滤 |
| Rule-based Reward | Action 合法、Schema、完成、循环和成本评分 |
| Reasoning-oriented RL | Scheduler trajectory GRPO-style 后训练 |
| 蒸馏到小模型 | accepted Agent 编排轨迹对 Scheduler LoRA 做 SFT |

R1-Zero 式“从未经冷启动的基模直接 RL”不作为 MVP 路线，因为随机 Scheduler 会产生大量非法或不完整 Agent 路径，并浪费冻结 Expert API 调用。Strong Teacher 自身不需要训练；它只提供候选。最终专门适配 TradingAgents 的能力沉淀在 Scheduler LoRA 中。

- DeepSeek-R1：借鉴 outcome-based RL、参考策略约束和 group-relative optimization 思路；不复现其完整推理模型训练。
  <https://arxiv.org/abs/2501.12948>
- Search-R1：借鉴多轮 Agent rollout、环境 observation masking 和结果奖励；不训练检索或工具策略。
  <https://arxiv.org/abs/2503.09516>
- Multi-Agent Collaboration via Evolving Orchestration（Puppeteer）：借鉴中央策略动态选择下一 Agent 的问题定义。
  <https://proceedings.neurips.cc/paper_files/paper/2025/hash/f1320d2e2842169c6fc89dcbd80e94d0-Abstract-Conference.html>
- Agent Lightning：借鉴 Agent execution、trajectory store 和训练后端解耦的系统边界。
  <https://arxiv.org/abs/2508.03680>

本项目的实现重点不是提出新 RL 算法，而是把固定 TradingAgents LangGraph 改造成可学习、可回放、可 A/B 验证的小模型 Agent 编排系统。
