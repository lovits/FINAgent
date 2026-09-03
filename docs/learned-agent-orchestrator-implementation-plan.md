# TradingAgents 双模式 Agent 编排器：分阶段实施技术文档

> 文档类型：实施方案，不代表代码已经完成
> 版本：v2.0
> 状态：WP0–WP9 离线代码框架已实现；真实 Teacher 调用和基模训练未执行
> 日期：2026-09-03
> 关联设计：[Learned Agent Orchestrator Design](./learned-agent-orchestrator-design.md)
> 当前轨迹、审核、SFT与GRPO可执行流程以[Scheduler轨迹数据流水线](./scheduler-trajectory-sft-grpo-pipeline.md)为准。

## 1. 这次改造到底要做什么

本项目当前已经有一套由 LangGraph 固定边连接起来的多 Agent 流程。此次改造不删除、不替换它，而是在同一个系统中增加第二种运行方式：由一个经过后训练的小语言模型，根据当前任务状态动态选择下一位专家 Agent。

最终必须同时保留两种一等运行模式：

| 模式 | 配置值 | 谁决定下一步 | 主要用途 |
|---|---|---|---|
| 原始固定模式 | `static` | 原项目规定的 LangGraph 边和条件边 | 稳定生产路径、回退路径、Teacher 数据来源、A/B 基线 |
| 学习编排模式 | `learned` | 小模型 Scheduler 在合法 Agent 集合中选择 | 动态协作、减少冗余 Agent 调用、展示强化学习优化能力 |

这里的“两个模式”不是“LangGraph 与 Scheduler 二选一”。两种模式的运行时底座都可以是 LangGraph：

- `static` 模式使用原有固定有向图；
- `learned` 模式使用“Scheduler 节点 + 专家 Agent 节点”的动态图；
- 两种模式共用现有 Expert Agents、工具节点、状态结构、记忆、数据源和最终报告；
- Scheduler 只决定“下一步调用哪个 Agent 或结束”，不负责选择 Agent 内部工具，也不直接生成交易结论。

一句话概括：**保留原图作为可靠基线，在它旁边增加一条可训练、可回退、可比较的动态编排路径。**

### 1.1 本轮定版结果

| 项目 | 定版选择 |
|---|---|
| 原模式 | `static`，完整保留当前固定 LangGraph |
| 新模式 | `learned`，Scheduler 节点动态选择 Expert Agent |
| Teacher | 强模型 API，只推理、不训练、逐步输出一个 action |
| Teacher 知识 | Agent Catalog + State Schema + Hard Rules + 正例 + 纠错反例 |
| 数据正确性 | Teacher 只提议，真实 Harness 执行和 Verifier 决定接收 |
| Scheduler 基模 | 当前不指定；在 AutoDL 正式训练前通过配置注入并冻结 revision |
| 参数更新 | 只训练 Scheduler LoRA |
| 冷启动 | Static 安全轨迹 + verified Teacher 动态轨迹 |
| 强化学习 | 同任务 4 条 rollout，轨迹级 GRPO-style 优化 |
| Reward | 完成/质量优先，Agent/Tool/Token/延迟成本次级 |
| 失败样本 | Teacher 纠错、单元测试和 RL 负轨迹；不作为 SFT 错误正标签 |
| 工具调用 | 继续由 Expert Agent 内部 LLM 和 ToolNode 控制 |

Teacher 具体使用哪家 API 是首次数据生成前必须填写的部署配置；没有该用户选择不妨碍先完成双模式、Registry、Prompt、轨迹和 Verifier 代码。

## 2. 改造前与改造后的整体结构

### 2.1 原系统

原项目的默认执行顺序是：

```text
Market Analyst
  -> Social/Sentiment Analyst
  -> News Analyst
  -> Fundamentals Analyst
  -> Bull Researcher <-> Bear Researcher
  -> Research Manager
  -> Trader
  -> Aggressive Debator
  -> Conservative Debator
  -> Neutral Debator
  -> Portfolio/Risk Manager
  -> END
```

其中研究辩论与风险辩论允许按计数循环，但“下一类角色是谁”主要由 `GraphSetup` 中预先写好的边和条件函数决定。分析师内部如需搜索行情、新闻或基本面数据，会进入各自的 ToolNode，再回到该分析师继续推理。

### 2.2 改造后的双模式入口

```text
                          +-----------------------------+
                          | TradingAgentsGraph          |
用户任务 + 配置 ---------->| scheduler_mode             |
                          +--------------+--------------+
                                         |
                     +-------------------+-------------------+
                     |                                       |
              scheduler_mode=static                   scheduler_mode=learned
                     |                                       |
          +----------v-----------+               +-----------v----------+
          | 原始固定 LangGraph    |               | Learned LangGraph    |
          | 固定边 + 条件边        |               | Scheduler 反复决策    |
          +----------+-----------+               +-----------+----------+
                     |                                       |
                     +-------------------+-------------------+
                                         |
                            共用 Expert Agents / ToolNodes
                                         |
                                  Trader + Risk Manager
                                         |
                                      最终报告
```

### 2.3 Learned 模式内部循环

```text
Scheduler 读取压缩状态
    -> 生成一个 Agent 动作
    -> 动作掩码检查是否合法
    -> 执行被选中的 Expert Agent
    -> Expert Agent 自己决定是否及如何调用工具
    -> 将结构化结果写回共享状态
    -> Scheduler 再次观察
    -> 继续选 Agent，或选择 <ACT_STOP>
```

Scheduler 能动态学习路径，但不是毫无限制地乱跳。系统通过状态机约束、动作掩码、最大步数和完成条件，将它限制在“业务合法的动态路径”内。

## 3. 本方案的边界

### 3.1 本次必须完成

- 原始 `static` 模式行为保持不变；
- 新增 `learned` 模式并接入同一运行入口；
- 建立统一 Agent 动作空间和合法动作掩码；
- 采集 Static Teacher 与强模型 Teacher 的编排轨迹；
- 对 Scheduler 小模型做 SFT；
- 使用轨迹级奖励进行 GRPO 风格强化学习；
- 实现轨迹记录、粗粒度信用分配、回退和离线回放；
- 用同一测试集完成 `static` 与 `learned` 的 A/B 对比；
- 形成可演示、可解释、可写进简历的工程闭环。

### 3.2 本次明确不做

- 不训练工具调度器；
- 不改变各 Expert Agent 内部的大模型工具调用逻辑；
- 不联合微调所有 Expert Agents；
- 不做真实资金交易；
- 不实现论文级 Process Reward Model；
- 不做每一个 token 或工具调用的精细边际信用估计；
- 不为了追求通用框架而重构与本任务无关的模块。

## 4. 当前代码与环境基线

### 4.1 已确认的代码入口

| 职责 | 当前文件 | 改造关系 |
|---|---|---|
| 总入口与运行 | `tradingagents/graph/trading_graph.py` | 增加模式选择，保持外部 API 基本不变 |
| 固定图构建 | `tradingagents/graph/setup.py` | 将现有逻辑明确命名为 Static Graph，再增加 Learned Graph 构建 |
| 条件跳转 | `tradingagents/graph/conditional_logic.py` | Static 模式继续使用；Learned 模式复用部分完成条件 |
| 分析师执行 | `tradingagents/graph/analyst_execution.py` | 两种模式共用，不改变工具责任边界 |
| 共享状态 | `tradingagents/agents/utils/agent_states.py` | 增加 Scheduler 所需的最小运行字段 |
| 结构化输出 | `tradingagents/agents/schemas.py` | 复用现有 Trader/Risk 输出；新增 Scheduler 动作 Schema |
| 图执行与日志 | `tradingagents/graph/propagation.py` | 接入轨迹采集和统一结果整理 |
| 默认配置 | `tradingagents/default_config.py` | 增加双模式和 Scheduler 配置段 |
| CLI | `cli/main.py` | 暴露模式选择并确保流式路径也经过目标图 |
| 报告 | `tradingagents/reporting.py` | 增加路径、成本、回退等可观测字段 |

### 4.2 当前环境状态

- 当前项目虚拟环境为 Python 3.12.9；
- 现有 CLI 可以正常显示帮助；
- 当前已安装的 113 个包依赖关系检查通过；
- 当前本地验证环境已安装 `torch`、`transformers`、`peft`、`accelerate`、`datasets`，并通过无下载的 Tiny Llama LoRA SFT/GRPO Smoke Test；
- 正式基模尚未指定或下载，真实 Strong Teacher 和 Expert API 数据生成尚未执行。

这部分只描述当前快照，不代表训练依赖必须一次性全部加入生产运行环境。推荐将推理依赖和训练依赖分开管理。

## 5. 开始开发前需要准备的内容

### 5.1 Scheduler 基础模型

当前阶段不指定、下载或训练真实基模。训练代码必须通过配置接收 `base_model`、model/tokenizer revision、dtype 和 device；到 AutoDL 正式训练前再选定并冻结。这个模型不需要具备专家级金融推理能力，它主要学习：

- 识别当前已经获得了哪些证据；
- 判断还缺哪类信息或观点；
- 判断是否需要继续辩论；
- 选择下一位 Agent；
- 判断什么时候已经可以结束。

正式训练前必须记录：

- `model_id`；
- 模型 revision/commit；
- tokenizer revision；
- 使用许可证；
- 最大上下文长度；
- 训练与推理精度，例如 BF16 或 FP16；
- LoRA 的目标层和预计显存。

模型选择原则：动作输出稳定、部署成本低、许可证清楚，比开放式长推理能力更重要。下载基模不是数据来源；它是后续接受 SFT 与 GRPO 更新的 Student Policy。

### 5.2 冻结 Expert Agent 执行条件

为了让 Static 与 Learned 的对比有效，需要冻结两套模式共同依赖的条件：

- `llm_provider`；
- quick thinking 与 deep thinking 模型；
- temperature；
- 最大重试次数；
- 分析师集合；
- 辩论轮数上限；
- 数据供应商与工具配置；
- 股票、交易日期与运行时间窗；
- Prompt 版本。

如果这些条件在 A/B 期间变化，就无法判断收益来自 Scheduler，还是来自专家模型或外部数据变化。

### 5.3 Strong Route Teacher

除了原固定流程产生的 Static Teacher 轨迹，还需要一个冻结的强模型 API 提出“下一步调用哪个 Agent”的候选。Teacher 不做参数训练，也不一次性臆造完整路径；每执行完一个真实 Expert Agent 后，它读取更新状态再做下一步。需要提前准备：

- Teacher 模型与 API 提供方；
- 鉴权方式，但不得将密钥写入轨迹；
- 固定、版本化的 Teacher Prompt 和输出 Schema；
- Agent Catalog、State Schema、Hard Rules、正例和纠错反例；
- 单任务完整候选轨迹数量，MVP 固定为 2 条；
- 预算上限；
- 超时和失败时退回 Static Teacher 的规则。

强模型只负责提出候选编排，候选必须在真实环境中执行并经过规则与结果验证，不能把 Teacher 的文字判断直接当成正确标签。Teacher 返回的理由用于审计，普通 SFT target 只保留单个 Agent Action Token。

### 5.4 代表性任务集

每个任务至少包含股票代码、决策日期、可用分析师集合、数据快照标识、任务类别和数据划分。建议覆盖以下六类：

| 类别 | 要验证的协作能力 |
|---|---|
| `technical` | 技术面证据已经足够时，能否减少无关分析 |
| `news_event` | 突发新闻场景下，能否优先调用新闻和情绪角色 |
| `fundamental` | 长期价值问题下，能否重视基本面与风险评估 |
| `conflict` | 多源结论冲突时，能否触发研究辩论或补充证据 |
| `sparse_data` | 数据不足时，能否避免直接结束并选择补证据 Agent |
| `risk_sensitive` | 高不确定性时，能否充分进行风险辩论和最终审查 |

任务应按股票与时间分割训练、验证和测试集合，尽量避免同一事件窗口同时进入训练集和测试集。

### 5.5 计算与存储预算

至少需要事先确认：

- 一张可进行 0.5B–1.5B LoRA 微调的 GPU，或等价云资源；
- Teacher 与 Expert Agent 的 API 预算；
- 轨迹、长文本观察、模型检查点和评估报告的存储目录；
- 每个任务的最大 Agent 步数、最大 Token、最大工具调用和最大耗时；
- 数据失败或供应商限流时的重跑策略。

### 5.6 统一工程约定

开发开始前要冻结以下约定，避免采集完数据后再改 Schema：

- Agent 动作名称与特殊 token；
- 每个状态下的动作掩码规则；
- 轨迹 Schema 版本；
- 状态序列化版本；
- Reward 配置版本；
- 模式配置键：`static` / `learned`；
- 所有训练产物目录；
- 随机种子和运行 ID 格式；
- Prompt 与模型版本追踪方式。

## 6. 总体阶段与依赖关系

| 阶段 | 核心目标 | 主要产物 | 必须通过后才能进入 |
|---|---|---|---|
| Phase 0 | 冻结原系统基线 | 基线清单、静态轨迹、测试报告 | Phase 1 |
| Phase 1 | 建立独立训练环境 | 可复现训练环境、模型加载 Smoke Test | Phase 2/5 |
| Phase 2 | 明确保留 Static Graph | `setup_static_graph`、回归结果 | Phase 3 |
| Phase 3 | 建立双模式与 Learned 骨架 | 模式开关、Scheduler 节点、动作掩码 | Phase 4 |
| Phase 4 | 建立轨迹与回放系统 | JSONL 轨迹、成本统计、Replay | Phase 5 |
| Phase 5 | 构造训练数据 | Static/Strong Teacher 数据、筛选报告 | Phase 6 |
| Phase 6 | Scheduler SFT | LoRA 检查点、SFT 评估 | Phase 7 |
| Phase 7 | 奖励与 Rollout | Reward 组件、候选组轨迹 | Phase 8 |
| Phase 8 | GRPO 风格强化学习 | RL 检查点、训练曲线、Smoke Test | Phase 9 |
| Phase 9 | 集成、A/B 与交付 | 双模式报告、演示材料、简历证据 | 完成 |

核心依赖链是：

```text
先证明 Static 没被改坏
  -> 再让 Learned 骨架跑通
  -> 再保证每一步都能记录和回放
  -> 再生成并筛选数据
  -> 先 SFT 学会合法编排
  -> 再用 Reward 做 RL 优化
  -> 最后进行公平 A/B
```

从 Phase 2 开始，`static` 与 `learned` 必须一直同时存在并进入测试，不允许在训练过程中把原模式临时删除。

## 7. Phase 0：冻结原系统基线

### 7.1 目标

在任何结构修改前，留下一个可复现的原始基线。后续如果 Static 模式结果发生变化，可以立刻定位是行为回归，而不是误以为是 Scheduler 的效果。

### 7.2 本阶段不改业务代码

只执行检查并保存结果：

```bash
git rev-parse HEAD
.venv/bin/python -m pytest -q
.venv/bin/tradingagents --help
uv pip check --python .venv/bin/python
```

再选择少量固定任务运行原图，至少覆盖：

- 一个常规任务；
- 一个会触发研究辩论的任务；
- 一个会触发风险辩论的任务；
- 一个工具失败或数据缺失任务。

### 7.3 必须保存的产物

```text
artifacts/baseline/
  baseline_manifest.json
  environment.txt
  config.redacted.json
  tests.txt
  runs/
    <run_id>/
      final_state.json
      report.md
      execution_summary.json
```

`baseline_manifest.json` 至少记录：

- Git commit；
- Python 版本；
- 依赖锁文件摘要；
- Expert 模型、Prompt 和数据源版本；
- 任务清单；
- 原始 Agent 顺序；
- 辩论轮数与最大循环数；
- 每个样例的最终决策、调用数、Token、延迟和错误。

### 7.4 验收标准

- 原 CLI 与 Python API 均能启动；
- 固定任务可生成最终 Trader/Risk 输出；
- 已记录原始路径和成本；
- 配置文件不包含 API Key 等秘密；
- 基线产物可由另一次运行复核。

### 7.5 退出门槛

没有基线快照，不进入 Phase 2。原系统本身存在的失败应单独记录，不能在双模式改造中顺手重构。

## 8. Phase 1：建立独立训练环境

### 8.1 目标

让项目同时支持轻量运行环境与 Scheduler 训练环境，避免部署原 TradingAgents 时被迫安装整套训练栈。

### 8.2 依赖组织

在 `pyproject.toml` 中增加可选训练依赖组，初始建议为：

```toml
[project.optional-dependencies]
scheduler-train = [
  "torch",
  "transformers",
  "peft",
  "accelerate",
  "datasets",
]
```

具体版本不能凭空填写，应在基础模型选定并完成 Python 3.12 兼容性验证后锁定。MVP 不强制引入 TRL：可以在项目内实现一个只处理离散 Scheduler 动作的最小 GRPO 风格训练器，从而减少第三方训练 API 变动带来的复杂度。

### 8.3 需要完成的工作

1. 选定基础模型与 tokenizer revision；
2. 验证模型可加载、可增加动作特殊 token；
3. 验证 LoRA 参数能够挂载并只更新预期层；
4. 验证 BF16/FP16 精度与目标 GPU 兼容；
5. 固定训练随机种子；
6. 输出完整环境清单与最小模型加载测试；
7. 将训练产物目录加入合理的忽略规则，但不忽略必要的小型配置和测试数据。

### 8.4 训练环境 Smoke Test

Smoke Test 只验证基础设施，不训练真实策略：

- 加载 base model 与 tokenizer；
- 添加全部 Agent 动作 token；
- 输入一条假的压缩状态；
- 约束输出为一个动作；
- 完成一次前向和反向传播；
- 保存并重新加载一个 LoRA adapter；
- 验证重新加载后动作词表一致。

### 8.5 验收标准

- 核心运行环境不安装训练依赖也能启动 Static 模式；
- 训练环境可以完成上述 Smoke Test；
- 训练参数量明显小于基础模型总参数量；
- 模型、tokenizer、依赖和硬件版本全部可追溯。

## 9. Phase 2：提取并永久保留 Static Graph

### 9.1 目标

把现有固定 LangGraph 明确固化为一个命名清晰的运行模式，但不改变它的业务行为。

### 9.2 主要修改点

在 `tradingagents/graph/setup.py` 中，将当前 `setup_graph()` 的固定图构建逻辑抽取为：

```python
setup_static_graph(...)
```

保留原有内容：

- 分析师节点及其 ToolNode；
- 分析师到 ToolNode、ToolNode 回分析师的循环；
- 分析师之间的固定顺序；
- Bull/Bear 研究辩论；
- Research Manager；
- Trader；
- 三位 Risk Debator；
- Portfolio/Risk Manager；
- 原有条件函数和结束条件。

外部统一入口可以暂时保持 `setup_graph()`，但它最终应根据模式分派：

```text
setup_graph(mode="static")  -> setup_static_graph()
setup_graph(mode="learned") -> setup_learned_graph()
```

### 9.3 为什么先做这一步

如果直接在现有 `setup_graph()` 上混入 Scheduler 条件边，原始图会逐步变成“看似还能运行、实际上无法证明等价”的混合图。先把 Static Graph 命名并冻结，才能保证：

- 原有功能有独立入口；
- Learned 失败时能安全回退；
- Static Teacher 轨迹始终可生成；
- A/B 的基线不是重新模拟出来的近似路径。

### 9.4 回归测试

对 Phase 0 的固定任务，比较抽取前后：

- Agent 节点序列；
- ToolNode 往返序列；
- 各辩论轮数；
- 状态关键字段；
- 最终 Trader 与 Risk 输出 Schema；
- 失败类型和回退行为。

外部模型可能造成文本细节波动，因此测试分两层：

1. 使用 mock/deterministic 模型做严格结构等价测试；
2. 使用真实模型做 Schema、路径与完成性回归，不要求自然语言逐字相同。

### 9.5 验收标准

- `static` 模式明确可选；
- 原固定图拓扑没有被 Scheduler 改写；
- CLI 和 Python API 的默认行为仍是 `static`；
- Phase 0 的结构回归通过；
- Static 模式可以独立生成完整报告。

## 10. Phase 3：建立双模式入口与 Learned Graph 骨架

### 10.1 目标

先不加载训练模型，用一个确定性的 `StaticSchedulerPolicy` 驱动 Learned Graph，证明“动态图运行机制”是正确的。

### 10.2 新增的最小 Scheduler 模块

```text
tradingagents/scheduler/
  __init__.py
  actions.py
  action_mask.py
  state_serializer.py
  policy.py
  scheduler_node.py
```

职责如下：

| 文件 | 职责 |
|---|---|
| `actions.py` | 定义稳定的 Agent 动作枚举和 token 映射 |
| `action_mask.py` | 根据当前状态返回合法动作集合 |
| `state_serializer.py` | 将庞大共享状态压缩为 Scheduler 输入 |
| `policy.py` | 定义统一策略接口及 `StaticSchedulerPolicy` / `LearnedSchedulerPolicy` |
| `scheduler_node.py` | 在 LangGraph 中调用策略、校验动作、记录决策并路由 |

### 10.3 Agent 动作空间

初始动作空间只包含 Agent 级动作：

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

动作 token 一旦进入数据采集就不得随意改名。如确需变更，必须提升动作 Schema 版本并迁移旧轨迹。

### 10.4 动作掩码的原则

动态不等于任意跳转。动作掩码只处理业务合法性，不替 Scheduler 规定唯一顺序。例如：

- 已达到最大总步数时，只允许进入安全收尾或回退；
- 没有任何分析证据时，不能直接 `<ACT_STOP>`；
- Trader 尚未给出投资计划时，不能调用最终 Risk Manager；
- 最终风险判断已完成时，只允许 `<ACT_STOP>`；
- 某 Agent 连续调用但状态没有新增信息时，暂时屏蔽重复动作；
- 已超过某类辩论最大轮数时，屏蔽该类继续辩论动作；
- 工具动作永远不出现在 Scheduler 动作集合中。

这些约束定义“什么绝对不合法”，而不是手写“下一步必须是谁”。

### 10.5 Scheduler 输入

小模型不应直接读取全部长报告和全部工具原文。`state_serializer.py` 生成一个稳定、可控长度的状态摘要，包含：

- 当前任务标识和日期；
- 已调用 Agent 集合与调用顺序；
- 各类报告是否存在；
- 关键结论的短摘要；
- Bull/Bear 与 Risk 辩论状态；
- Trader/Risk Manager 是否已输出；
- 剩余步数和成本预算；
- 上一步 Agent、状态是否产生新信息；
- 当前合法动作列表。

状态摘要使用固定模板和版本号。训练与推理必须调用同一个序列化器，不能各写一套 Prompt。

### 10.6 两个策略实现

`SchedulerPolicy` 接口接收压缩状态与合法动作，返回一个动作和可选元信息。

- `StaticSchedulerPolicy`：按原图顺序和原条件函数选择下一 Agent，用于测试 Learned Graph 运行机制；
- `LearnedSchedulerPolicy`：加载 SFT/RL 后的小模型，在动作掩码内选择下一 Agent。

这里的 `StaticSchedulerPolicy` 不是第三个产品模式，它只是开发和测试 Learned Graph 的确定性策略。用户可见的长期模式仍然只有 `static` 与 `learned`。

### 10.7 Learned Graph 拓扑

Learned Graph 的宏观拓扑是：

```text
START -> Scheduler
Scheduler -> 任一合法 Expert Agent
Expert Agent -> Scheduler
Scheduler -> <ACT_STOP>/END
```

但分析师调用工具时仍保留局部子循环：

```text
Scheduler -> Market Analyst
Market Analyst -> Market ToolNode -> Market Analyst
Market Analyst 完成报告 -> Scheduler
```

因此 Scheduler 只在一个 Agent 完成本轮工作后重新决策，不会插入 Agent 与其工具调用的中间过程。

### 10.8 共享状态增量

在现有 `AgentState` 上只增加 Scheduler 所需字段，例如：

- `scheduler_mode`；
- `scheduler_step`；
- `scheduler_action`；
- `scheduler_history`；
- `scheduler_trace_id`；
- `last_agent`；
- `no_progress_count`；
- `agent_call_count`；
- `tool_call_count`；
- `estimated_tokens`；
- `fallback_reason`。

长轨迹不全部塞进 LangGraph State。State 只放运行所需摘要，完整事件写入独立轨迹存储，避免 checkpoint 体积失控。

### 10.9 配置与入口

默认配置增加类似以下语义：

```python
"scheduler_mode": "static",
"scheduler_adapter_path": None,
"scheduler_max_steps": 16,
"scheduler_invalid_action_policy": "fallback_static",
"scheduler_trace_enabled": True,
```

CLI 与 Python API 都必须将这个配置传给图构建层。尤其要验证 CLI 的流式执行路径，因为它直接消费编译后的图，不能只在 `propagate()` 外层偷偷切换模式。

### 10.10 回退规则

Learned 模式在以下情况回退到真正的原 `static` LangGraph：

- checkpoint 缺失或加载失败；
- 模型没有输出可解析动作；
- 动作被掩码判为非法且重试仍失败；
- 连续无进展；
- 达到最大 Scheduler 步数仍未完成；
- Expert Agent 关键输出缺失，无法安全结束。

启动前发现 checkpoint 缺失时，直接选择 `setup_static_graph()`。运行中失败时，终止并保存该 Learned trajectory，再从原始 task 输入重新运行 `setup_static_graph()`；不尝试把任意 Learned 中间状态塞入固定图继续执行。回退必须记录 `requested_mode=learned`、`actual_mode=static`、原因、发生步骤和重复成本，不能静默假装 Learned 成功。

### 10.11 验收标准

- `static` 仍直接运行原固定图；
- `learned + StaticSchedulerPolicy` 能通过 Scheduler 循环复现原主要 Agent 顺序；
- ToolNode 仍只由对应 Expert Agent 控制；
- 两种模式均能产出兼容的最终结果；
- checkpoint/thread 标识包含模式和策略版本，避免两种图复用错误状态；
- 非法动作、最大步数和模型缺失回退测试通过。

## 11. Phase 4：轨迹、成本与离线回放

### 11.1 目标

把每一次 Scheduler 决策变成可训练、可审计、可回放的数据。没有可靠轨迹，就无法做 SFT、Reward、信用分配和 A/B。

### 11.2 建议新增模块

```text
tradingagents/scheduler/
  trajectory.py
  trajectory_store.py
  replay.py
  cost_tracker.py
```

### 11.3 一条完整轨迹包含什么

一条轨迹对应“一个任务的一次完整多 Agent 运行”，至少记录：

- 轨迹 Schema 版本；
- task ID、run ID、模式和策略版本；
- Expert 模型与 Prompt 版本；
- 数据快照标识；
- 每个 Scheduler step 的压缩状态；
- 当前合法动作；
- 被选择动作；
- 动作 log probability；
- 实际执行的 Agent；
- Agent 产出的结构化摘要；
- 工具调用名称、次数、耗时与成功状态；
- Token 和延迟估计；
- 是否产生新信息；
- 是否回退；
- 最终 Trader/Risk 输出；
- 各 Reward 分量及总 Reward。

### 11.4 存储格式

推荐：

- 决策和元数据使用 append-only JSONL；
- 很长的 Agent 输出或工具响应单独存储，通过相对引用关联；
- 每条轨迹先写临时文件，任务完成后原子提交；
- 中断轨迹也保存，并明确标记 `incomplete`；
- API Key、Cookie、Authorization Header 等永不写入轨迹；
- 股票公开数据可以记录快照引用，但需遵守对应数据源条款。

### 11.5 回放的两种层次

1. **逻辑回放**：不重新调用外部模型和工具，只把已记录观察依次交给策略，验证动作和掩码；
2. **环境重放**：重新执行 Agent 和工具，验证新策略的完整效果。

逻辑回放用于快速训练验证；环境重放用于最终评估。两者必须区分，不能把离线日志回放结果冒充真实环境运行效果。

### 11.6 成本统计

成本至少拆成：

- Agent 调用次数；
- Tool 调用次数；
- 输入/输出 Token；
- 总延迟；
- 失败重试次数；
- Scheduler 自身推理成本。

这既用于 Reward，也用于解释 Learned 模式是否真的减少冗余协作。

### 11.7 验收标准

- Static 与 Learned 使用同一轨迹 Schema；
- 每个 Scheduler 动作都能追溯到当时状态和合法动作；
- 完成、失败、回退和中断轨迹都能区分；
- 任意一条轨迹可进行逻辑回放；
- 轨迹中不包含秘密；
- Reward 可以仅凭轨迹及最终结果重算。

## 12. Phase 5：构造代表性训练数据

### 12.1 目标

数据不是简单收集“模型说应该调用谁”。正确的数据链是：提出候选路径，放进真实多 Agent 环境执行，再依据完成性、结果质量、合法性和成本筛选。

### 12.2 数据目录建议

```text
data/scheduler/
  tasks/
    train.jsonl
    validation.jsonl
    test.jsonl
  static_teacher/
  strong_teacher_candidates/
  executed_candidates/
  accepted/
  rejected/
  sft/
  manifests/
```

测试集在训练前冻结，不能根据 RL 结果反复挑选有利样例。

### 12.3 先冻结 Teacher 上下文包

正式调用 API 前，先从当前代码和配置生成五项版本化资源：

```text
configs/scheduler/teacher/v1/
  agent_catalog.yaml
  state_schema.yaml
  orchestration_rules.yaml
  positive_examples.jsonl
  failure_examples.jsonl
```

具体工作：

1. 从 `graph/setup.py` 和各 Agent factory 核对十二个 Expert Agent；
2. 从 `graph/analyst_execution.py` 核对 Analyst key、节点名和报告字段；
3. 从 `_create_tool_nodes()` 核对工具归属，并明确 `tool_policy_owner=expert_agent`；
4. 从 `agent_states.py` 提炼 Teacher 真正需要的完成状态和摘要字段；
5. 从 `conditional_logic.py` 提炼不可违反的前置条件与循环上限；
6. 从 `schemas.py` 提炼 Trader、Research Manager 和 Portfolio Manager 的完成标准；
7. 为每个 Agent 写出 use/avoid/prerequisites/reads/writes；
8. 给整个上下文包生成唯一 `teacher_context_version`。

不要把代码中所有长 Prompt 原样复制给 Teacher。Agent Catalog 只描述编排所需能力，不重复 Expert Agent 的专业推理指令。

### 12.4 数据来源一：Static Teacher

对全部训练任务运行原 `static` 模式，得到：

- 一定合法；
- 通常能够完成；
- 与现有业务流程一致；
- 但路径可能较长、存在冗余。

Static Teacher 的作用是提供“安全完整路径”和边界条件，而不是证明固定路径是最优路径。

### 12.5 数据来源二：Strong Route Teacher

强模型读取与 Scheduler 相同或稍丰富的状态摘要，每一步只提出一个下一 Agent；对应 Expert Agent 真实执行后，再读取新状态继续选择。MVP 对每个任务独立生成 2 条完整候选 trajectory，鼓励覆盖：

- 更短但完整的路径；
- 面对冲突时增加辩论的路径；
- 面对数据缺失时主动补证据的路径；
- 不同成本与质量取舍的路径。

Teacher 输出必须受相同动作 Schema 约束，不允许虚构不存在的 Agent。

### 12.6 Teacher Prompt V1 与示例集

实现 `teacher_prompt_version=v1`，固定段落顺序：

```text
ROLE_AND_OBJECTIVE
AGENT_CATALOG
STATE_SCHEMA
HARD_RULES
FEW_SHOT_EXAMPLES
TASK
STATE_STATUS
EVIDENCE_SUMMARY
EXECUTION_HISTORY
BUDGET
VALID_ACTIONS
OUTPUT_SCHEMA
```

固定输出：

```json
{
  "next_agent": "<ACT_NEWS>",
  "reason_code": "MISSING_EVENT_EVIDENCE",
  "reason": "当前价格异常缺少事件解释。"
}
```

Prompt 示例集至少包含：

- 一条常规 Static 完整示例；
- 一条新闻驱动动态示例；
- 一条基本面驱动动态示例；
- 一条关键证据冲突示例；
- 一条数据缺失补证据示例；
- 一条合法 STOP 示例；
- 过早 STOP、越级调用、无进展重复和不存在 Agent 的纠错反例。

失败示例必须明确给出错误动作、失败原因和纠正动作。没有纠正标签的失败文本不能直接放进 Few-shot。

### 12.7 候选必须真实执行

每条 Teacher 候选都放入 `learned` 图骨架执行。执行后才决定是否接收，验证项包括：

- 所有动作均合法；
- 未出现无限循环或连续无进展；
- Trader 给出合法结构化交易计划；
- Risk Manager 给出合法最终决策；
- 结果与 Static 基线的差异在允许范围内；
- Agent/Tool/Token/延迟没有超预算；
- 路径确实提供了与 Static 不完全相同的有效决策样本。

### 12.8 接收与拒绝规则

建议将候选分为：

- `accepted`：合法、完成、质量达标，可作为正样本；
- `rejected_quality`：完成但结果明显偏离基线或缺乏证据；
- `rejected_invalid`：动作非法或 Schema 错误；
- `rejected_incomplete`：没有形成完整最终输出；
- `rejected_loop`：重复调用且没有新信息；
- `rejected_budget`：成本或步数超限。

拒绝样本不用于普通 SFT 正标签，但可以用于构造边界测试、RL 低奖励样本和错误分析。

### 12.9 结果“正确”如何判断

金融决策往往不存在单一即时真值，因此 MVP 不把未来收益率直接当作唯一标签。采用分层代理判定：

1. **硬正确性**：流程完成、动作合法、结构化输出可解析；
2. **任务正确性**：该调用的证据类别被覆盖，关键冲突得到处理；
3. **决策一致性**：与同条件 Static 结果及强 Judge 的差异没有超过阈值；
4. **效率**：在质量达标后，路径更短、Token/工具/延迟更低；
5. **可选事后指标**：历史回测窗口中的结果只作为附加评估，不在 MVP 中独占 Reward。

这意味着蒸馏解决“先学会像一个合格编排器”，真实执行和 Reward 再解决“在保持质量的前提下学得更省、更灵活”。

### 12.10 SFT 数据组合

最终 SFT 正样本由三部分组成：

- Static Teacher 的完整安全路径；
- 经过真实执行和筛选的 Strong Teacher 路径；
- 人工或规则补充的边界状态，例如该结束、不能结束、数据不足、循环上限和回退状态。

每个训练样本只监督 Scheduler 的 Agent 动作。Expert Agent 的自然语言输出、工具调用内容和最终报告不作为 Scheduler 要模仿生成的标签。

SFT 转换器必须将一条 accepted trajectory 拆成多个 step-level 样本：

```text
(task + state_0 + history_0 + valid_actions_0) -> action_0
(task + state_1 + history_1 + valid_actions_1) -> action_1
...
```

每个样本还保存 `trajectory_id`、`step_id`、`source`、`teacher_model`、`teacher_prompt_version`、`state_schema_version`、`action_schema_version` 和 `verifier_version`，但这些审计元数据不作为模型生成目标。

### 12.11 首批 Pilot 数据规模

第一轮只验证管线，不追求最终效果：

| 项目 | Pilot 数量 |
|---|---:|
| 任务 | 12–18 个 |
| 任务类型 | 6 类，每类 2–3 个 |
| Static trajectory | 每任务 1 条 |
| Teacher trajectory | 每任务 2 条 |
| API 调用策略 | 每个 Agent 完成后调用 Teacher 一次 |
| 人工抽查 | accepted 与 rejected 各抽查至少 10 条决策 |

Pilot 退出条件：能够稳定完成 `task → Teacher逐步选择 → Expert执行 → trajectory → verifier → SFT JSONL`。在此之前不批量生成上百任务，也不开始正式 SFT。

### 12.12 数据质量报告

每次数据构造必须输出：

- 各任务类型数量；
- Static 与 Strong Teacher 样本比例；
- 候选接收率及拒绝原因；
- Agent 动作分布；
- 路径长度分布；
- 完成率；
- 成本分布；
- 重复任务和时间泄漏检查；
- train/validation/test 的股票与事件时间隔离情况。

### 12.13 验收标准

- 三个数据划分固定且无明显时间泄漏；
- 所有正样本均在真实图中完成过；
- 每条样本可追溯到任务、Teacher、环境和筛选规则；
- 动作类别没有严重缺失；
- 拒绝样本和拒绝原因可审计；
- 测试集未进入 SFT 或 RL 更新。
- Teacher 每次只输出一个合法 next-agent action；
- 所有 accepted 样本都能追溯到 Prompt、Teacher、状态和 Verifier 版本；
- 失败动作没有被误写为普通 SFT 正标签。

## 13. Phase 6：Scheduler 监督微调（SFT）

### 13.1 目标

先让小模型稳定学会“看状态，只输出一个合法 Agent 动作”，再进行强化学习。SFT 的重点不是寻找最优路径，而是建立可靠的动作语法、基本流程意识和结束判断。

### 13.2 训练对象

只训练 Scheduler 的 LoRA 参数：

```text
输入：当前任务 + 压缩状态 + 历史 Agent 路径 + 合法动作
输出：一个 Agent 动作 token
```

不要求模型生成解释。解释可由日志系统根据状态、掩码和动作补充，避免小模型同时学习长篇语言生成而分散容量。

### 13.3 标签掩码

每个训练序列中：

- 任务说明、状态摘要、历史、合法动作列表：只作为条件，loss mask 为 0；
- Teacher 给出的那个 Agent 动作 token：作为监督目标，loss mask 为 1；
- Expert Agent 输出和工具响应：不作为 Scheduler 生成目标。

这样训练出来的是“编排动作策略”，不是一个模仿所有专家文本的混合模型。

### 13.4 建议训练文件

```text
training/scheduler/
  configs/
    sft.yaml
  dataset.py
  collator.py
  model.py
  train_sft.py
  evaluate_policy.py
```

### 13.5 SFT 训练流程

```text
加载冻结的 base model
  -> 扩展/确认 Agent 动作 token
  -> 挂载 LoRA
  -> 加载 SFT train/validation
  -> 对动作位置计算交叉熵
  -> 按 validation 指标保存最佳 adapter
  -> 冻结一份作为后续 RL 的 reference policy
```

### 13.6 核心评估指标

| 指标 | 含义 |
|---|---|
| `exact_action_accuracy` | 预测动作与验证标签完全一致的比例 |
| `valid_action_rate` | 输出位于当前合法动作集合内的比例 |
| `parse_success_rate` | 输出能被稳定解析为一个动作的比例 |
| `extra_text_rate` | 除动作外额外生成文本的比例，越低越好 |
| `finish_validity` | 选择结束时，任务是否真的满足完成条件 |
| `masked_action_violation` | 选择被掩码动作的比例，目标接近 0 |
| `path_completion_rate` | 把模型放入完整 Learned Graph 后的完成率 |

动作准确率不是唯一目标。Teacher 路径可能有多个同样合理的下一步，因此更重要的是合法率、完整运行成功率和最终结果质量。

### 13.7 SFT 验收标准

- 模型稳定输出单一动作；
- 非法动作率达到预设上限以内；
- 在未见验证任务中可以完成 Learned Graph；
- `<ACT_STOP>` 不会明显过早或过晚；
- adapter 保存、加载后结果一致；
- 训练日志记录数据版本、模型 revision、seed 和超参数；
- 产生一份冻结的 `pi_ref`，供 RL 约束策略漂移。

## 14. Phase 7：Reward、打分与组内 Rollout

### 14.1 Reward 的目标

Reward 不是奖励“少调用 Agent”本身，而是奖励：

> **先把分析和最终决策做完整、做得可信，再在质量达标的候选中偏好更低成本、更少冗余的协作路径。**

因此质量约束优先于效率。一个只调用 Trader 就结束的极短路径，应因缺少证据或最终输出不完整而获得明显负分。

### 14.2 一次如何产生可比较候选

对同一个任务，从相同初始状态让当前 Scheduler 采样多条轨迹，MVP 建议每组 4 条：

```text
同一个 task
  -> rollout A：一条 Agent 路径
  -> rollout B：另一条 Agent 路径
  -> rollout C：另一条 Agent 路径
  -> rollout D：另一条 Agent 路径
  -> 分别真实执行 Expert Agents
  -> 计算各自 Reward
  -> 只比较同一 task 组内谁更好
```

同组候选使用相同 Expert 配置、数据快照、预算和最大步数。这样分数差主要反映编排路径，而不是任务难度不同。

### 14.3 Reward 组成

总分由五类信号组成。

#### A. 完成性

判断轨迹是否真正形成可用结果：

- 必要状态是否存在；
- Trader 输出是否符合 Schema；
- Risk Manager 最终输出是否符合 Schema；
- `<ACT_STOP>` 是否在合法时机发生。

完整完成获得正分；缺少关键结果或过早结束获得大额负分。

#### B. 决策质量

用多个弱但可审计的信号组合，而不是依赖一个神秘 Judge：

- Learned 的最终风险评级与同任务 Static 基线的距离；
- Learned 的 Trader 买入/持有/卖出方向与 Static 的一致或可接受差异；
- 必要证据是否覆盖；
- 强 Judge 对证据充分性、冲突处理和结论一致性的结构化评分；
- 可选历史数据上的事后指标。

Static 不是绝对真值，所以“与 Static 不同”不自动等于错误。若 Strong Teacher/Judge 能证明证据充分、输出合法，且差异可解释，候选仍可获得质量分。

#### C. Agent 编排成本

惩罚不必要的 Expert Agent 调用：

- 总 Agent 调用次数；
- 同一 Agent 重复调用次数；
- 没有带来新状态信息的调用。

#### D. 执行成本

惩罚：

- Tool 调用次数；
- Token 数；
- 总延迟；
- 失败重试。

注意：Scheduler 不直接选择工具，但它选择哪个 Agent 会间接影响工具成本，所以这些成本可以用于评价整条编排轨迹。

#### E. 安全与失败惩罚

以下情况给出显著负分：

- 非法动作；
- 解析失败；
- 超过最大步数；
- 连续无进展；
- 无限辩论倾向；
- 没有最终结论；
- 触发回退。

回退后的业务结果可能仍然成功，但 Scheduler 本身应收到失败惩罚，否则它可能学会依赖 Static 兜底。

### 14.4 初始 Reward 配置

以下是实现和调参起点，不是已经验证的实验结果：

| 分量 | 初始权重或范围 |
|---|---:|
| Risk Manager 评级接近度 | `0.70` |
| Trader 动作接近度 | `0.30` |
| 完整完成奖励 | `+0.20` |
| 每次 Agent 调用成本 | `-0.02` |
| 每次 Tool 调用成本 | `-0.005` |
| 每 1k Token 成本 | `-0.005` |
| 无进展重复 | `-0.10` |
| 非法动作 | `-1.00` |
| 不完整结束 | `-1.00` |
| 总 Reward 截断 | `[-1.50, 1.20]` |

这些值需要先在一小批固定任务上做 sanity check：完整但较长的路径应高于不完整短路径；质量相近时，较短路径应略高；单个成本项不应压过完整性。

### 14.5 Reward 计算顺序

```text
先做硬合法性和完成性检查
  -> 不合法/不完整：给予失败惩罚
  -> 合法且完整：计算质量分
  -> 质量达标：再扣 Agent、Tool、Token、延迟成本
  -> 加入循环/回退等惩罚
  -> 截断总分
  -> 保存每个分量，不只保存总分
```

### 14.6 Strong Judge 的约束

如果使用大模型 Judge，必须：

- 输出固定 JSON Schema；
- 只读取本次任务证据，不能获知策略名称；
- 不告诉它结果来自 Static 还是 Learned；
- 对一部分样本重复评分以检查稳定性；
- Judge 分只占质量的一部分；
- 保存模型版本和 Prompt 版本；
- Judge 失败时使用可计算的硬指标，而不是丢弃整批轨迹。

### 14.7 Reward 单元测试

必须人工构造至少三条 mock 轨迹：

- A：结果正确、完整、调用少；
- B：结果正确、完整、调用明显更多；
- C：调用最少但没有完成。

预期排序必须是：A 高于 B，B 高于 C。再增加非法动作、重复循环、回退、质量分歧等边界样例。

### 14.8 验收标准

- 同一轨迹重复计算得到相同 Reward；
- 每个 Reward 分量均可追溯；
- 完整性优先于省成本；
- Static 与 Learned 可使用同一打分器；
- Judge 被禁用时仍能计算基础 Reward；
- 四条同任务 Rollout 可以稳定完成组内排序。

## 15. Phase 8：GRPO 风格的 Scheduler 强化学习

### 15.1 为什么使用 GRPO 风格方法

Scheduler 的输出是少量离散 Agent 动作，且完整路径执行后才能较可靠地判断好坏。对每个任务采样多条路径、做组内相对比较，可以：

- 不需要额外训练一个大型价值模型；
- 降低不同股票和日期之间 Reward 尺度差异；
- 直接学习“同一个任务下，哪种协作路径更好”；
- 与 SFT 小模型 + 真实环境 Rollout 的工程链条自然衔接。

### 15.2 三份策略参数

训练时区分：

- `pi_theta`：当前正在更新的 Scheduler；
- `pi_old`：本轮采样时使用的旧策略快照，用于限制单次更新幅度；
- `pi_ref`：SFT 后冻结的参考策略，用于 KL 约束，防止策略为追求成本奖励而崩坏。

### 15.3 粗粒度信用分配

MVP 采用轨迹级信用分配：

1. 一条完整轨迹得到一个总 Reward；
2. 在同一任务的候选组中，将每条轨迹的分数转换为相对优势；
3. 该轨迹中由 Scheduler 选择的所有动作共享这个优势方向；
4. 高于组内平均的轨迹，其动作概率整体被提高；
5. 低于组内平均的轨迹，其动作概率整体被降低；
6. KL 约束阻止策略一次偏离 SFT 太远。

直观例子：

```text
同一任务有四条路径：
A 完整且 8 次 Agent 调用，得分 0.75
B 完整但 13 次调用，得分 0.52
C 不完整，得分 -1.00
D 完整但结论质量较差，得分 0.10

训练会增强 A 路径中 Scheduler 动作的概率，
适度减弱 B 和 D，明显减弱 C。
```

这不能精确证明 A 中每一个动作贡献相同，但对简历型 MVP 足够清晰、可实现、可解释。后续才考虑步骤级价值或反事实删 Agent 实验。

### 15.4 目标函数的宏观含义

训练目标由三件事组成：

- **奖励项**：让高相对优势路径里的动作更可能出现；
- **裁剪项**：限制新策略相对 `pi_old` 单次变化过大；
- **KL 项**：限制策略相对 SFT 参考策略漂移过远。

实现文档和代码应保留数学公式，但对项目讲解时可以概括为：

> 同一任务多跑几条 Agent 路径，奖励完成且质量高、成本低的路径；训练时提高好路径动作的概率，降低差路径动作的概率，同时限制每次更新不要走得太远。

### 15.5 最重要的 Loss Mask

一条轨迹会包含任务、状态摘要、Expert 输出、工具观察和 Scheduler 动作。策略梯度只能落在 Scheduler 生成的动作 token 上：

```text
任务文本             mask = 0
状态摘要             mask = 0
Expert Agent 输出    mask = 0
工具观察             mask = 0
Scheduler 动作 token mask = 1
```

否则模型会错误地学习生成 Expert 或 Tool 的文本，偏离“Agent 编排器”的目标。

### 15.6 训练循环

```text
从训练任务池采样一批 task
  -> 对每个 task 用 pi_old 采样 G 条路径
  -> 在真实 Learned Graph 中执行全部路径
  -> 计算每条路径 Reward 分量与总分
  -> 在每个 task 内计算相对优势
  -> 只提取 Scheduler 动作 token 的 log probability
  -> 计算裁剪策略损失 + KL 约束
  -> 更新 LoRA 参数 pi_theta
  -> 周期性同步 pi_old
  -> 在冻结 validation 集上评估
  -> 保存最佳 checkpoint 和完整元数据
```

### 15.7 建议训练模块

```text
training/scheduler/
  configs/
    grpo.yaml
  rollout_runner.py
  reward.py
  advantage.py
  grpo_loss.py
  train_grpo.py
  evaluate_policy.py
```

### 15.8 训练监控

至少记录：

- 平均总 Reward 与每个分量；
- 组内 Reward 标准差；
- 完成率；
- 非法动作率；
- 回退率；
- 平均 Agent 调用数；
- 平均 Tool 调用数；
- 平均 Token 和延迟；
- 与 Static 结果的评级距离；
- KL 值；
- clip fraction；
- 动作熵；
- 每个 Agent 动作频率；
- 路径长度分布。

如果 Reward 上升但完成率下降、`<ACT_STOP>` 激增或某些 Agent 几乎消失，说明发生奖励投机，应停止训练并检查 Reward。

### 15.9 最小梯度 Smoke Test

正式训练前用 Phase 7 的 A/B/C mock 轨迹验证：

- A 的动作 log probability 更新后上升；
- C 的动作 log probability 更新后下降；
- 非 Scheduler token 的梯度为 0；
- LoRA 以外的基础模型参数不更新；
- KL 与裁剪项数值有限且没有 NaN。

### 15.10 RL 验收标准

- 完整训练可从配置和 manifest 复现；
- 只有 Scheduler LoRA 参数更新；
- Reward、优势、动作 log probability 能逐条审计；
- validation 完成率没有因省成本明显下降；
- 相比 SFT，至少观察到一个稳定的编排改进信号，例如更少冗余调用或更低回退率；
- 训练中断后可以从 checkpoint 恢复。

## 16. Phase 9：运行集成、公平 A/B 与项目交付

### 16.1 目标

将两种模式作为长期功能交付，而不是训练完成后只留下一个不可比较的模型文件。

### 16.2 最终运行配置

推荐外部接口保持统一：

```text
scheduler_mode = static
```

或：

```text
scheduler_mode = learned
scheduler_adapter_path = <checkpoint>
```

默认仍建议 `static`，直到 Learned 模式完成验收。`learned` 缺少有效 checkpoint 时，不允许随机初始化运行，应明确报错或按配置回退到 `static`。

### 16.3 双模式共同契约

两种模式必须接收同一种任务输入，最终返回兼容的结果结构：

- 最终投资建议；
- Trader 计划；
- Risk Manager 决策；
- 各分析报告；
- 运行模式；
- 实际 Agent 路径；
- 成本统计；
- 是否回退及原因；
- trace ID。

这样上层 CLI、报告和评估无需为两套图维护两套业务接口。

### 16.4 A/B 公平性

同一个测试任务分别运行：

```text
A = static：原固定 LangGraph
B = learned：小模型 Scheduler LangGraph
```

必须保持一致：

- 任务和数据快照；
- Expert 模型与 Prompt；
- temperature 和随机设置；
- 工具供应商；
- 最大辩论轮数和总预算；
- 输出 Schema；
- Reward/Judge 版本。

如果外部实时数据无法固定，报告中要明确标注非确定性，并尽可能在短时间内成对运行。

### 16.5 A/B 指标

| 维度 | 指标 |
|---|---|
| 可用性 | 完成率、Schema 通过率、异常率 |
| 质量 | Risk 评级距离、Trader 动作一致率、Judge 分 |
| 编排 | 路径长度、重复 Agent 次数、每类 Agent 使用率 |
| 成本 | Agent/Tool 调用数、Token、延迟、估算费用 |
| 稳定性 | 非法动作率、回退率、跨随机种子方差 |
| 可解释性 | 轨迹是否能还原每次选择及当时合法动作 |

不要只报告平均 Reward。简历和面试最有说服力的是“质量不明显下降的情况下，调用和成本如何变化”，并附上可复核的评估规模和条件。

### 16.6 必须保留的三组比较

最终至少报告：

1. `static` vs `learned-SFT`：证明蒸馏后能正常编排；
2. `learned-SFT` vs `learned-RL`：证明强化学习带来的增量；
3. `static` vs `learned-RL`：证明最终系统价值。

这不是论文式大规模基线矩阵，而是回答三个工程问题：新模式能否工作、RL 是否有用、最终是否优于原流程的某些维度。

### 16.7 交付产物

```text
artifacts/scheduler/
  manifests/
  datasets/
  checkpoints/
    sft/
    grpo/
  evaluations/
    static-vs-sft/
    sft-vs-rl/
    static-vs-rl/
  traces/
  demos/
```

最终需要形成：

- 双模式可运行代码；
- 一键选择模式的 CLI/API；
- 数据构建和筛选脚本；
- SFT 与 GRPO 训练配置；
- 冻结的测试集；
- A/B 评估报告；
- 2 到 3 条可视化轨迹示例；
- 一段演示视频或可重复 Demo；
- 模型卡和数据卡；
- 失败案例说明；
- 可写入简历的真实量化指标。

### 16.8 验收标准

- 用户可以显式选择 `static` 或 `learned`；
- 两个模式均通过相同的端到端测试；
- Learned 故障时能够追踪并安全回退；
- 所有最终数字来自冻结测试集，不使用训练集成绩冒充效果；
- 可以从某次报告定位到配置、代码、模型、数据和轨迹版本；
- 原固定模式仍能独立运行，不依赖训练栈。

## 17. 逐文件实施映射

下面是目标文件级拆分。实际提交时按阶段小步落地，不建议一次性创建全部空文件。

### 17.1 现有运行代码

| 文件 | 计划变更 | 所属阶段 |
|---|---|---|
| `tradingagents/default_config.py` | 增加 `scheduler_mode`、adapter、最大步数、回退和 trace 配置 | 3 |
| `tradingagents/graph/setup.py` | 提取 `setup_static_graph`，增加 `setup_learned_graph` 和模式分派 | 2–3 |
| `tradingagents/graph/trading_graph.py` | 接受模式配置、注入策略、保持统一对外入口 | 3 |
| `tradingagents/graph/propagation.py` | 统一 trace 生命周期、结果和成本汇总 | 4 |
| `tradingagents/agents/utils/agent_states.py` | 增加最小 Scheduler 运行状态 | 3 |
| `tradingagents/agents/schemas.py` | 增加 Scheduler 动作输出 Schema | 3 |
| `tradingagents/reporting.py` | 报告模式、路径、成本、回退与策略版本 | 4/9 |
| `cli/main.py` | 增加模式和 checkpoint 参数，传入同一编译图 | 3/9 |
| `pyproject.toml` | 增加独立训练 extra 和必要脚本入口 | 1 |

`conditional_logic.py` 与 `analyst_execution.py` 原则上优先复用。只有当 Learned Graph 需要暴露明确的“Agent 已完成”信号时才做最小修改，不能借机改写 Expert Agent 业务逻辑。

### 17.2 新增 Scheduler 运行代码

| 文件 | 作用 | 关键测试 |
|---|---|---|
| `tradingagents/scheduler/actions.py` | Agent 动作、token、版本映射 | 枚举唯一、序列化往返 |
| `tradingagents/scheduler/agent_registry.py` | Agent Card、状态读写、前置条件和工具归属的共同事实源 | 与实际节点/状态/工具注册一致 |
| `tradingagents/scheduler/action_mask.py` | 计算合法动作 | 各状态边界表驱动测试 |
| `tradingagents/scheduler/state_serializer.py` | 压缩共享状态 | 确定性、长度、缺失字段测试 |
| `tradingagents/scheduler/policy.py` | 策略接口与 Static/Learned 实现 | 统一输入输出、checkpoint 错误 |
| `tradingagents/scheduler/scheduler_node.py` | 图节点、动作校验、路由 | 合法、非法、重试、回退 |
| `tradingagents/scheduler/teacher_prompt.py` | 从 Registry、规则、示例和动态状态组装版本化 Prompt | Prompt 快照和秘密剔除 |
| `tradingagents/scheduler/teacher_gateway.py` | 调用冻结 Teacher API 并解析单动作 Schema | 超时、非法 JSON、未知动作 |
| `tradingagents/scheduler/verifier.py` | 对候选轨迹执行硬检查与接收/拒绝分类 | 完成、循环、质量、预算边界 |
| `tradingagents/scheduler/trajectory.py` | 轨迹事件与 Schema | Schema 验证、版本检查 |
| `tradingagents/scheduler/trajectory_store.py` | 追加写、提交、中断恢复 | 原子写入、并发 run ID |
| `tradingagents/scheduler/cost_tracker.py` | Agent/Tool/Token/延迟统计 | 汇总一致性 |
| `tradingagents/scheduler/replay.py` | 逻辑回放 | 动作与 mask 可复现 |

### 17.3 新增训练代码

| 文件 | 作用 | 所属阶段 |
|---|---|---|
| `training/scheduler/dataset.py` | 读取任务、轨迹和 SFT 样本 | 5–6 |
| `training/scheduler/build_sft_dataset.py` | 将 accepted trajectory 拆成 step-level action 样本 | 5 |
| `training/scheduler/collator.py` | 构造输入与动作标签 mask | 6 |
| `training/scheduler/model.py` | 加载 base、tokenizer 与 LoRA | 1/6 |
| `training/scheduler/train_sft.py` | SFT 入口 | 6 |
| `training/scheduler/rollout_runner.py` | 成组运行 Learned Graph | 7–8 |
| `training/scheduler/reward.py` | Reward 分量和配置化组合 | 7 |
| `training/scheduler/advantage.py` | 同任务组内标准化优势 | 8 |
| `training/scheduler/grpo_loss.py` | 裁剪目标和 KL | 8 |
| `training/scheduler/train_grpo.py` | RL 主循环、恢复与保存 | 8 |
| `training/scheduler/evaluate_policy.py` | SFT/RL/Static 统一评估 | 6–9 |

### 17.4 配置、数据和测试

```text
configs/scheduler/
  static.yaml
  learned_sft.yaml
  learned_grpo.yaml

training/scheduler/configs/
  sft.yaml
  grpo.yaml

tests/scheduler/
  test_actions.py
  test_action_mask.py
  test_state_serializer.py
  test_static_graph_regression.py
  test_learned_graph.py
  test_fallback.py
  test_trajectory.py
  test_replay.py
  test_reward.py
  test_grpo_loss.py
  test_dual_mode_e2e.py
```

## 18. 测试矩阵

### 18.1 单元测试

| 被测对象 | 必测行为 |
|---|---|
| 动作枚举 | token 唯一、未知动作拒绝、版本正确 |
| 动作掩码 | 无证据不能结束、完成后只能结束、循环上限、预算上限 |
| 状态压缩 | 同一状态输出确定、顺序稳定、长度可控、秘密剔除 |
| 策略解析 | 单动作、额外文本、空输出、未知 token、超时 |
| Reward | 完整优先、质量优先、成本次级、失败强惩罚 |
| 组内优势 | 同分、零方差、极端分数、组大小不足 |
| Loss mask | 只有 Scheduler 动作位置有梯度 |
| 轨迹存储 | 中断、重复 run ID、Schema 版本、原子提交 |

### 18.2 图集成测试

| 场景 | Static 预期 | Learned 预期 |
|---|---|---|
| 常规完整任务 | 按原图完成 | Scheduler 合法选择并完成 |
| 分析师需调用工具 | 原 ToolNode 循环 | 同样由分析师控制 ToolNode |
| 研究意见冲突 | 按固定辩论轮次 | 可选择继续辩论或交给 Manager |
| 风险不确定 | 按固定风险流程 | 可动态增加必要风险角色 |
| 模型输出非法动作 | 不适用 | 重试或回退 Static |
| checkpoint 缺失 | 不受影响 | 明确错误或回退 Static |
| 达到最大步数 | 原图条件结束 | 安全收尾或回退 |
| checkpoint 恢复 | 恢复同一 Static 图 | 按模式/策略版本恢复 Learned 图 |

### 18.3 端到端测试

- CLI 使用默认参数时运行 `static`；
- CLI 显式选择 `learned` 并加载测试策略；
- Python API 两种模式返回兼容结果；
- 同任务两种模式都生成 trace 和报告；
- Learned 故障回退后报告明确显示“请求模式、实际模式、回退原因”；
- 报告中的成本汇总与轨迹逐步统计一致；
- 训练环境未安装时，Static 模式仍可导入和运行。

### 18.4 数据与训练测试

- 数据划分无 task ID 重复；
- 同一股票同一事件窗口不会跨 train/test；
- 每个 SFT 标签都属于该状态合法动作；
- 每条正轨迹确实完成；
- tokenizer 动作 token 在保存/加载后 ID 不变；
- base model 参数被冻结；
- SFT adapter 可恢复；
- GRPO 一个小 batch 可完成前向、反向和更新；
- 固定 seed 的离线逻辑回放结果可重复。

## 19. 每阶段建议的提交与验收节奏

为了让问题容易定位，推荐每个里程碑形成独立、可回滚的提交：

| 里程碑 | 提交内容 | 不能夹带的内容 |
|---|---|---|
| M0 | 基线 manifest 与回归测试 | 业务逻辑变化 |
| M1 | 训练 extra 与加载 Smoke Test | Scheduler 图改造 |
| M2 | Static Graph 命名抽取 | Learned 策略 |
| M3 | 双模式、动作、mask、StaticSchedulerPolicy | 真实模型训练 |
| M4 | 轨迹、成本和回放 | Reward 调参 |
| M5 | 任务、Teacher、执行筛选与数据报告 | 用测试集训练 |
| M6 | SFT 与评估 | RL 代码 |
| M7 | Reward 与 group rollout | 大规模训练 |
| M8 | GRPO 风格训练与 checkpoint | 宣称最终效果 |
| M9 | A/B、报告、Demo | 未经验证的简历数字 |

推荐顺序强调的是可验证性，不要求每个里程碑间隔很久。只要当前门槛通过，可以连续推进。

## 20. 开工前准备清单

### 20.1 必须由项目事实确定

- [ ] 当前用于基线的 Git commit；
- [ ] 需要支持的分析师集合；
- [ ] Static 模式的固定路径与所有条件循环；
- [ ] Expert Agent 使用的模型、Prompt 和数据源；
- [ ] 当前状态和输出 Schema；
- [ ] CLI 与 Python API 的真实调用方式；
- [ ] 现有测试是否通过及已知失败。

### 20.2 必须在训练前选择

- [ ] Scheduler base model；
- [ ] tokenizer/model revision 与许可证；
- [ ] LoRA 配置和精度；
- [ ] GPU/云训练资源；
- [ ] Strong Route Teacher；
- [ ] Strong Judge 是否启用；
- [ ] API 与训练预算；
- [ ] 最大 Agent 步数和成本预算。

### 20.3 必须在采数据前冻结

- [ ] Agent 动作 token；
- [ ] Action Mask 规则；
- [ ] 状态压缩模板；
- [ ] 轨迹 Schema；
- [ ] task manifest Schema；
- [ ] train/validation/test 划分；
- [ ] Teacher Prompt 和筛选规则；
- [ ] 数据快照或时间窗口策略。

### 20.4 必须在 RL 前验证

- [ ] SFT 输出可解析；
- [ ] SFT 完整运行成功率达标；
- [ ] `pi_ref` 已冻结；
- [ ] Reward 排序符合人工常识；
- [ ] Reward 分量可重算；
- [ ] Group rollout 使用相同环境条件；
- [ ] Loss mask 只覆盖 Scheduler 动作；
- [ ] 小 batch 梯度方向 Smoke Test 通过。

### 20.5 必须在写简历前具备

- [ ] 最终代码确实包含 `static` 与 `learned` 两种模式；
- [ ] Static 回归测试通过；
- [ ] 至少有 SFT 和 RL 两个 checkpoint；
- [ ] 冻结测试集规模可说明；
- [ ] A/B 条件一致；
- [ ] 效果数字可从报告复核；
- [ ] 有典型成功轨迹和失败轨迹；
- [ ] 能解释 Reward、信用分配、回退和工具责任边界。

## 21. 关键风险与处理方式

### 21.1 Scheduler 学会过早结束

**表现**：路径很短、成本下降，但证据不足或没有完整 Risk 输出。
**处理**：完成性硬门槛、非法 `<ACT_STOP>` mask、不完整强惩罚、监控结束位置分布。

### 21.2 Scheduler 学会依赖 Static 回退

**表现**：表面完成率高，实际频繁触发回退。
**处理**：将回退视为 Scheduler 失败并单独惩罚；报告同时展示请求模式与实际执行模式。

### 21.3 Reward 被成本项支配

**表现**：模型少调用关键 Agent。
**处理**：先过质量门槛再扣成本；降低成本权重；用 A/B/C 人工轨迹做排序回归。

### 21.4 Teacher 错误被蒸馏

**表现**：SFT 合法但经常做出质量差的路径。
**处理**：Teacher 只提候选；候选必须真实执行；保留 Static 安全路径；低质量候选拒绝而非直接训练。

### 21.5 在线数据导致 A/B 不公平

**表现**：两种模式看到不同新闻或行情。
**处理**：优先使用数据快照；否则成对短间隔运行并记录时间、供应商和响应标识。

### 21.6 小模型读不下完整上下文

**表现**：输入过长、状态关键点丢失。
**处理**：固定结构化压缩；只保留编排决策需要的信息；对摘要长度和字段缺失做测试。

### 21.7 动作空间与旧轨迹不兼容

**表现**：新增或改名 Agent 后，历史数据无法训练。
**处理**：动作 Schema 版本化；模型卡记录 token 映射；必要时显式迁移而不是静默兼容。

### 21.8 两种模式的 checkpoint 混用

**表现**：恢复时图拓扑与旧状态不一致。
**处理**：checkpoint signature 包含 mode、graph version、action schema 和 policy version；不匹配时拒绝恢复。

## 22. 推荐的最小可交付版本

如果项目目标是形成一项真实、能讲清楚的简历工程，第一版控制在以下范围最合适：

1. 保留并回归验证原 `static` LangGraph；
2. 新增 `learned` LangGraph 与 Agent 级动作空间；
3. 用 StaticSchedulerPolicy 先跑通动态图；
4. 从 6 类代表性任务生成 Static 和 Strong Teacher 候选；
5. 候选真实执行后筛选；
6. 对 0.5B–1.5B 小模型做 LoRA SFT；
7. 使用四候选组、轨迹级 Reward 和动作 token mask 做 GRPO 风格后训练；
8. 完成 `static`、`learned-SFT`、`learned-RL` 三组比较；
9. 展示完成率、质量、Agent 调用数、Token、延迟和回退率；
10. 保留一条成功动态路径和一条失败回退路径用于演示。

这个版本已经包含完整技术闭环：

```text
原固定多 Agent 系统
  -> 双模式图改造
  -> 轨迹采集
  -> Teacher 蒸馏数据
  -> 小模型 SFT
  -> 轨迹 Reward
  -> 粗粒度信用分配
  -> GRPO 风格 RL
  -> A/B 评估
  -> 安全回退
```

不需要在第一版加入工具调度学习、专家联合训练、复杂 PRM 或大规模论文基线，才能证明核心工程成立。

## 23. 最终项目讲解口径

完成后，对这个项目可以用下面这条逻辑讲清楚：

> 原 TradingAgents 使用 LangGraph 将多个金融分析、研究辩论、交易和风险 Agent 按固定规则连接。我保留了原固定图作为 Static 模式、稳定回退和 A/B 基线，同时新增 Learned 模式：由经过 LoRA SFT 和 GRPO 风格后训练的小语言模型，根据共享状态动态选择下一位 Expert Agent。训练数据来自原图安全轨迹和强模型提出、经真实环境执行筛选后的候选路径；奖励先约束流程完成和决策质量，再优化 Agent/Tool/Token/延迟成本；轨迹级组内相对优势只更新 Scheduler 的动作 token。最终系统能在同一接口下切换固定编排与学习编排，并用冻结测试集比较质量、成本、稳定性和回退率。

这段口径只有在相应模块、训练和报告真实完成后才能作为已完成经历使用。在实施期间，应把“已完成”“正在实现”“计划加入”分开表述。

## 24. 第一轮实际开发建议

第一轮只做 Phase 0 到 Phase 3，成功标准是：

- 原 Static 模式被明确保留且回归通过；
- 统一入口能选择 `static` / `learned`；
- Learned Graph 暂时由 `StaticSchedulerPolicy` 驱动；
- Learned Graph 可以完成一次端到端任务；
- Expert Agent 内部 ToolNode 行为没有变化；
- 失败能回退并留下原因。

这一步完成后再开始大规模轨迹采集。否则训练数据很可能建立在一个还会频繁变化的图和状态 Schema 上，后续需要全部重做。

## 25. 可直接执行的工作包

下面是实际开发时的任务顺序。每个工作包都必须有独立产物和验证结果；上一个门槛没有通过，不进入依赖它的下一个工作包。

### WP0：基线冻结与训练环境 Smoke Test

**目标**：证明原项目当前可运行，并确认 0.5B 基模能在独立训练环境加载。
**修改范围**：基线产物、可选训练依赖，不改图逻辑。
**产物**：

- Static 固定任务清单；
- 当前图节点序列和结果 Schema 快照；
- 测试报告；
- 本地随机初始化 Tiny Llama 的加载、LoRA 保存/恢复与单步反向 Smoke Test；
- model/tokenizer revision manifest。

**验收**：Static 原测试通过；训练环境可以保存和重新加载空初始化 LoRA。

### WP1：原 Static Graph 无损提取

**目标**：将现有 `setup_graph()` 明确拆成 `setup_static_graph()`，默认行为不变。
**主要文件**：`graph/setup.py`、`graph/trading_graph.py`、Static topology tests。
**任务**：

1. 为当前图节点和边建立 deterministic topology snapshot；
2. 提取 Static 构建函数；
3. 保留 Analyst ToolNode 循环、Bull/Bear 和 Risk 条件逻辑；
4. 保持默认配置仍进入 Static；
5. 对 Phase 0 固定任务做结构回归。

**验收**：抽取前后节点、边、关键状态字段和最终 Schema 等价。当前 CodeGraph 显示 `setup_graph()` 没有直接覆盖测试，因此这项测试是第一个必要新增测试。

### WP2：Agent Registry 与协议冻结

**目标**：建立 Teacher、Action Mask、轨迹和测试共享的 Agent 事实源。
**主要文件**：

```text
tradingagents/scheduler/actions.py
tradingagents/scheduler/agent_registry.py
configs/scheduler/teacher/v1/*
```

**任务**：

1. 注册十二个 Agent ID 与 Action Token；
2. 声明每个 Agent 的 reads/writes/prerequisites；
3. 声明 Analyst 的内部工具，但标注工具策略归 Expert；
4. 固定 `action_schema_version=v1`；
5. 固定 `state_schema_version=v1`；
6. 固定 `teacher_context_version=v1`；
7. 根据当前源码写 Agent Catalog 和 State Schema；
8. 编写最小正例和带纠正动作的失败例。

**验收**：Registry 与当前图中的节点、状态字段和 ToolNode 注册一一对应；不存在同义但不同 ID 的 Agent。

### WP3：Learned Graph 骨架

**目标**：不加载训练模型，先让 Scheduler 循环能够正确驱动所有 Expert Agent。
**主要文件**：`policy.py`、`scheduler_node.py`、`action_mask.py`、`setup.py`。
**任务**：

1. 增加 `static` / `learned` 模式分派；
2. 实现 `StaticSchedulerPolicy`；
3. 建立 `START → Scheduler → Expert → Scheduler`；
4. 保留 Expert 内部 ToolNode 回路；
5. 实现 STOP、最大步数、无进展和回退；
6. 将 mode、graph version 和 policy version 加入 checkpoint signature。

**验收**：`learned + StaticSchedulerPolicy` 可以完成端到端任务并复现 Static 主要路径；两种模式均返回兼容结果。

### WP4：轨迹、状态压缩与成本记录

**目标**：让每一次 Scheduler 决策都可以重算、回放和训练。
**主要文件**：`state_serializer.py`、`trajectory.py`、`trajectory_store.py`、`cost_tracker.py`、`replay.py`。
**任务**：

1. 固定 Scheduler 状态模板；
2. 保存 state、valid actions、selected action 和 logprob；
3. 保存 Agent observation 摘要和长文本引用；
4. 统计 Agent/Tool/Token/延迟/重试；
5. 标记完成、中断、非法和回退；
6. 提供不调用外部 API 的逻辑回放。

**验收**：Static 与 Learned 使用同一轨迹 Schema；任意轨迹可以恢复当时的合法动作和选择。

### WP5：Static 冷启动数据

**目标**：在不调用 Strong Teacher 的情况下先获得安全数据。
**任务**：

1. 冻结 12–18 个 Pilot task；
2. 覆盖六类代表性场景；
3. 每任务运行一条 Static trajectory；
4. 拆出初始 step-level action 样本；
5. 从 Static 轨迹选择 Few-shot 正例；
6. 构造 STOP、前置缺失、循环和越级调用失败状态。

**验收**：每条 Static 轨迹都能完成；样本标签属于当时合法动作；失败状态的纠正动作可由规则验证。

### WP6：Strong Teacher Pilot

**前置用户输入**：Teacher provider、model ID、API Key 和预算。
**目标**：验证冻结 API 模型能通过 Prompt V1 产生可执行的动态编排候选。
**主要文件**：`teacher_prompt.py`、`teacher_gateway.py`、`verifier.py`。
**任务**：

1. 组装固定上下文和当前动态状态；
2. 每次只请求一个 next-agent action；
3. 校验结构化输出和合法动作；
4. 真实执行被选 Expert Agent；
5. 用新状态继续调用 Teacher；
6. 每任务生成 2 条完整候选轨迹；
7. 按失败类型写入 accepted/rejected；
8. 人工抽查代表性正例和失败例。

**验收**：完成 `Teacher action → real Expert execution → new state → next Teacher action` 闭环；未经执行的 Teacher 文本不能进入 accepted。

### WP7：SFT 数据与 Scheduler LoRA

**目标**：把 accepted trajectory 转为动作监督数据，并得到第一个可运行 Scheduler。
**主要文件**：`build_sft_dataset.py`、`dataset.py`、`collator.py`、`model.py`、`train_sft.py`。
**任务**：

1. 合并 Static 和 verified Teacher 正轨迹；
2. 按 step 拆分输入和单动作 target；
3. 只在动作 token 上设置 loss mask；
4. 训练配置中指定的 Scheduler 基模 LoRA；
5. 评估动作解析、合法率、STOP 和完整运行；
6. 冻结最佳 adapter 为 `pi_ref`。

**验收**：adapter 重载一致；基础模型参数冻结；验证任务能在 Learned Graph 中完成；不输出无关长文本。

### WP8：Reward 与 GRPO-style 后训练

**目标**：让 Scheduler 在保持完成和质量的前提下学习更合适的协作路径。
**主要文件**：`reward.py`、`rollout_runner.py`、`advantage.py`、`grpo_loss.py`、`train_grpo.py`。
**任务**：

1. 先用人工 A/B/C 轨迹验证 Reward 排序；
2. 每个 task 从同一初始状态采样 4 条轨迹；
3. 真实执行冻结 Expert Agents；
4. 计算完成、质量、成本、循环、非法和回退分量；
5. 计算组相对优势；
6. 只对 Scheduler action token 反向传播；
7. 加入裁剪和参考策略 KL；
8. 保存最佳 RL adapter。

**验收**：高分 mock 路径概率上升、失败路径下降、非动作 token 梯度为零；验证完成率没有为降低成本而明显退化。

### WP9：双模式 A/B 与交付

**目标**：形成可演示、可复核、可写简历的结果。
**任务**：

1. 冻结 held-out task 和数据快照；
2. 成对运行 Static、Learned-SFT 和 Learned-RL；
3. 比较完成、质量、路径、成本和稳定性；
4. 输出成功、失败和回退轨迹；
5. 生成模型卡、数据卡和评估报告；
6. 只把实测指标写入简历。

**验收**：任何结果数字都能追溯到代码、配置、task、数据快照、模型和 trajectory。

## 26. 工作包依赖与当前起点

```text
WP0 基线/环境
  ↓
WP1 Static 无损提取
  ↓
WP2 Agent Registry 与协议冻结
  ↓
WP3 Learned Graph 骨架
  ↓
WP4 轨迹与回放
  ↓
WP5 Static 冷启动数据
  ↓  此处才必须提供 Strong Teacher API 配置
WP6 Teacher Pilot
  ↓
WP7 Scheduler SFT
  ↓
WP8 GRPO-style RL
  ↓
WP9 A/B 与交付
```

当前应从 `WP0 → WP1 → WP2` 开始。Teacher API Key 不是开始改代码的前置条件；它只在 WP6 首次真实生成动态候选时需要。基模可以在 WP0 做加载 Smoke Test，但正式训练必须等 WP5/WP6 的 accepted 数据和 Schema 稳定后再开始。

开始前仍需用户最终指定的外部条件只有：

- Strong Teacher 的 provider 和 model ID；
- 对应 API Key 的环境变量名称；
- Teacher/Expert API 的 Pilot 预算；
- 正式训练使用本机 MPS 还是 NVIDIA GPU；
- 首批 12–18 个 task 的 ticker/date，或授权按六类场景从历史日期中选取。
