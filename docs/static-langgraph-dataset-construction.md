# TradingAgents Static LangGraph 数据集设计（简历项目版）

> 版本：v1.0
> 日期：2026-09-03
> 状态：18条近期任务种子已完成；真实LangGraph轨迹尚未生成

> 扩展版：300条跨行业任务的数据设计与文件见[`static-langgraph-300-dataset.md`](static-langgraph-300-dataset.md)。本文件继续保留为低成本Pilot说明。
> 轨迹审核、SFT转换和GRPO训练数据的完整衔接见[`scheduler-trajectory-sft-grpo-pipeline.md`](scheduler-trajectory-sft-grpo-pipeline.md)。

## 1. 数据集要解决什么问题

这套数据不是金融知识问答，也不是让另一个模型凭空编写轨迹，而是直接记录原固定LangGraph在不同金融状态下如何完成一次决策：

```text
股票 + 历史日期
→ 四类Analyst取证
→ Bull/Bear研究辩论
→ Research Manager汇总
→ Trader给出Buy/Hold/Sell
→ 三类Risk Agent讨论
→ Portfolio Manager给出最终评级
```

这些Static轨迹有两个用途：

1. 作为Scheduler SFT的安全冷启动数据；
2. 作为后续Static与Learned模式A/B的基线。

## 2. 主要参考的方法

### TradingAgents：决定数据主体

TradingAgents论文使用逐交易日、point-in-time的多模态环境。每次决策只能读取交易日及以前的信息。数据包含：

- OHLCV和复权价格；
- 公司新闻、全球新闻和宏观信息；
- 社交媒体情绪；
- 内幕交易；
- 财务报表和财报；
- 公司资料；
- 技术指标。

原论文在2024-01-01至2024-03-29运行多只科技股，并使用累计收益、年化收益、Sharpe Ratio和最大回撤进行评价。

来源：[TradingAgents](https://arxiv.org/html/2412.20138)

### 其他近期工作：只补三点

- [FinTradeBench 2026](https://arxiv.org/abs/2603.19225)：先用少量任务校准、进行数值检查，确认可靠后再扩量；
- [Finance Agent Benchmark](https://arxiv.org/abs/2508.00828)：任务要分类型，并同时记录准确性和调用成本；
- [From Tasks to Teams 2026](https://aclanthology.org/2026.findings-acl.1934/)：除最终答案外，还要记录API/Tool失败和多步流程失败。

本项目不照搬这些基准的金融问答，而只借鉴任务分类、数值校验和失败记录方法。

### 最终取舍

| 来源 | 在本项目中的占比和作用 |
|---|---|
| TradingAgents原论文与当前代码 | 约80%：决定数据模态、逐日截止、固定Agent流程、Trader/Portfolio输出和市场结果 |
| FinTradeBench 2026 | 小量借鉴：18条先校准，通过后再扩量；检查行情数值 |
| Finance Agent Benchmark | 小量借鉴：任务分类，同时记录质量与调用成本 |
| 风险优先Agent评测 | 小量借鉴：记录Tool失败、流程失败和风险标签 |

明确不引入：

- FinTradeBench的1400道金融问答规模；
- Finance Agent Benchmark的9类专家问题体系；
- 多模型投票和复杂Judge流水线；
- 论文级排行榜、复杂风险Taxonomy和人工专家团队。

因此，这仍然是TradingAgents Static Trajectory Dataset，不会变成另一个金融QA Benchmark。

## 3. 为什么按这六类任务划分

分类直接对应当前TradingAgents的Agent功能。

| 类别 | 选择依据 | 主要检验的Agent逻辑 |
|---|---|---|
| `earnings_window` | 财报日期 | Fundamentals、News、Research Manager |
| `positive_momentum` | 过去5日明显上涨 | Market、Bull、Risk |
| `negative_momentum` | 过去5日明显下跌 | Market、Bear、Risk |
| `high_volatility` | 过去20日波动较高 | Market、三类Risk Agent、Portfolio Manager |
| `volume_shock` | 成交量相对20日均值异常 | Market、News、Sentiment |
| `quiet_control` | 低动量、低波动、正常成交量 | 检查固定图是否存在不必要的完整调用 |

它们不是给Scheduler看的答案，只用于保证任务覆盖不同金融状态。

## 4. 为什么还要看Agent运行后的标签

下面几类情况不能只根据行情提前判断，需要固定LangGraph真实跑完后再标记：

| 后验标签 | 依据 |
|---|---|
| `completed` | 自动：Investment Plan、Trader动作和Portfolio评级均可解析 |
| `evidence_conflict` | 人工：Market、News、Sentiment、Fundamentals或Bull/Bear结论明显冲突；v1在复核前保存为`null` |
| `data_sparse` | 自动：选中的Analyst报告缺失、Tool失败或返回无数据标记 |
| `high_risk` | 自动：任务属于高波动层，或最终评级为Underweight/Sell |

一条任务可以同时拥有多个后验标签，例如：

```text
earnings_window + evidence_conflict + high_risk
```

不需要建立复杂的几十类标签体系。

## 5. 一条数据记录什么

每个任务只保留以下核心内容：

```text
1. task_id、ticker、trade_date、任务类别
2. 使用的Expert模型、代码提交（对应Prompt版本）和数据截止日期
3. 固定Agent调用顺序
4. 每个Agent执行前后的状态摘要
5. Tool名称、执行状态、是否拿到数据和结果摘要
6. 四类Analyst报告和两轮辩论内容
7. Trader动作和Portfolio最终评级
8. Agent数、Tool数、Token、延迟和失败原因
9. completed/conflict/sparse/high_risk标签
10. 可选的5日收益和基准Alpha
```

其中：

- Agent顺序、字段完整性和Tool状态属于硬事实；
- LLM生成的报告和交易判断属于Static基线，不叫绝对真值；
- 未来5日收益只能在决策后补充，不能出现在Agent输入中。

## 6. 当前18条近期任务

数据窗口为2026-04-01至2026-08-28，共11只股票。

| 类别 | 股票和日期 |
|---|---|
| Earnings | NVDA 2026-08-26；WMT 2026-08-20；AMD 2026-08-04 |
| Positive momentum | AMD 2026-05-11；MSFT 2026-08-03；AMZN 2026-08-03 |
| Negative momentum | TSLA 2026-07-29；WMT 2026-05-27；MSFT 2026-06-08 |
| High volatility | TSLA 2026-07-06；AMZN 2026-08-25；GOOGL 2026-08-11 |
| Volume shock | AAPL 2026-06-26；GOOGL 2026-04-30；XOM 2026-06-18 |
| Quiet control | JPM 2026-08-26；JNJ 2026-04-13；AAPL 2026-06-01 |

选择约束：

- 每类3条；
- 同一股票最多2条；
- 同一股票的两个日期至少相隔21天；
- 只使用交易日期及以前的指标选任务；
- 不根据未来涨跌挑选任务。

### 6.1 任务种子由代码可重复构建

构建程序是`training/scheduler/build_task_seeds.py`。它不是用LLM编问题，而是执行四步确定性流程：

```text
逐股票下载OHLCV和财报日历
→ 对每个交易日计算仅使用t日及以前数据的3个滚动特征
→ 按6个采样层分别排序并贪心选取3条
→ 写出task JSONL、来源快照和manifest
```

六类排序方法：

| 采样层 | 排序规则 |
|---|---|
| Earnings | 财报日历内的交易日，优先较近期日期 |
| Positive momentum | 5日历史收益从高到低 |
| Negative momentum | 5日历史收益从低到高 |
| High volatility | 20日历史年化波动从高到低 |
| Volume shock | 20日成交量Z-score从高到低 |
| Quiet control | `abs(5日收益) + 20日波动 + 0.05 × abs(成交量Z-score)`从低到高 |

程序按表格顺序固定选择，并同时施加四个约束：每类使用不同股票、同一`ticker/date`不重复、每只股票最多2条、同一股票两条任务至少相隔21天。若某类不足3条，程序明确报错，不会偷偷放宽规则。这里的六类只是**抽样层**，不是交易答案，也不是Scheduler的目标动作。

重建命令如下；它只访问yfinance，不使用OpenRouter Key：

```bash
python -m training.scheduler.build_task_seeds \
  --start 2026-04-01 \
  --end 2026-08-28 \
  --output-tasks data/scheduler/tasks/static_langgraph_pilot_v1.rebuild.jsonl \
  --output-manifest data/scheduler/tasks/static_langgraph_pilot_v1.rebuild.manifest.json
```

当前18条清单就是该程序在manifest所记录的来源时间生成的结果。行情供应商可能修订历史数据，因此正式实验使用已提交的任务清单作为冻结快照；日后重新构建的文件用于审计方法，不应静默覆盖原清单。

任务文件：

```text
data/scheduler/tasks/static_langgraph_pilot_v1.jsonl
```

机器清单：

```text
data/scheduler/tasks/static_langgraph_pilot_v1.manifest.json
```

## 7. 简单的数据生成流程

### 第一步：校验18条任务

```bash
python -m training.scheduler.validate_task_manifest \
  --tasks data/scheduler/tasks/static_langgraph_pilot_v1.jsonl \
  --manifest data/scheduler/tasks/static_langgraph_pilot_v1.manifest.json
```

### 第二步：先运行1条Smoke

只选择一条任务运行固定LangGraph，确认：

- API和数据工具可用；
- 固定流程可以完成；
- 轨迹能够保存；
- Trader和Portfolio输出合法；
- 没有未来数据进入输入。

### 这里不新增“轨迹生成Prompt”

四类Analyst、Bull/Bear、Research Manager、Trader、Risk Agent和Portfolio Manager继续使用项目原有Prompt。采集器直接运行：

```text
TradingAgentsGraph(scheduler_mode="static")
→ 原setup_static_graph
→ LangGraph updates + values事件流
→ 保存真实节点、工具事件、状态前后变化和最终输出
```

因此，报告内容仍由原Expert Agent生成；新代码只负责记录，不替Agent决定它应该说什么。若单条任务中途失败，也会保存到`rejected.jsonl`，而不是让整批数据无记录地终止。

### 第三步：运行18条Static Pilot

```bash
python -m training.scheduler.generate_data \
  --tasks data/scheduler/tasks/static_langgraph_pilot_v1.jsonl \
  --output-dir data/scheduler/static_langgraph_pilot_v1 \
  --policy static \
  --trajectories-per-task 1
```

输出：

```text
accepted.jsonl
rejected.jsonl
```

`accepted.jsonl`中的每条记录都同时包含两层轨迹：

- `node_steps`：完整静态图事件，包括Agent、Tool和消息清理节点；
- `scheduler_examples`：从真实静态路径抽出的`状态 → 合法动作 → 实际Agent`监督样本，供小模型SFT使用。

### 第四步：人工检查

18条数量很小，可以全部查看：

- Agent顺序是否符合固定图；
- 工具数据是否对应正确日期；
- 报告是否发生公司或时间错配；
- 冲突和数据缺失标签是否合理；
- 最终结果和轨迹是否完整。

### 第五步：审核并转换为Scheduler SFT数据

Static轨迹审核通过后，再执行：

```bash
python -m training.scheduler.prepare_sft_dataset \
  --input data/scheduler/static_langgraph_pilot_v1/accepted.jsonl \
  --output-dir data/scheduler/sft/static_langgraph_pilot_v1 \
  --dataset-id tradingagents-scheduler-pilot-sft-v1
```

命令先运行确定性轨迹审核，再生成`pilot.jsonl`、`audit_report.json`和`manifest.json`。转换结果只训练中央Scheduler选择“下一位Agent”；各Agent内部如何调用工具仍由原Agent逻辑负责。

### 第六步：再决定是否扩量

如果18条能够稳定生成，就扩展到约60条任务，并简单切分：

```text
训练集：42条
验证集：9条
测试集：9条
```

按时间段和股票划分，避免同一股票的相邻日期同时进入训练集和测试集。

## 8. 与Teacher数据的关系

当前阶段只生成固定LangGraph数据：

```text
18个任务
→ 每个任务1条Static轨迹
→ 作为安全冷启动与A/B基线
```

Static数据确认没有问题后，下一阶段才让Strong Teacher针对相同或训练集任务生成动态路径。两类数据不能混在第一次调试中，否则无法判断错误来自原图、工具还是Teacher。

## 9. 项目边界

这个简历项目不需要：

- 建设上千条金融QA；
- 人工标注复杂金融结论；
- 设计几十种互斥类别；
- 完整复现TradingAgents论文的全部回测；
- 建设论文级实时排行榜。

第一版只要证明：

```text
近期金融任务可复现选择
→ 固定LangGraph生成完整可审计轨迹
→ 轨迹可转成SFT数据
→ 后续可训练并与动态Scheduler做A/B
```

这已经足够形成一个逻辑完整、面试时能讲清楚的数据工程与Agentic RL项目。

## 10. 数据可用性与复现边界

- 仓库保存任务种子、构建方法、来源manifest、校验器和轨迹采集代码；
- yfinance原始行情不重新分发，重建时由使用者按供应商条件重新获取；
- 每条Static轨迹记录代码提交、模型/provider、数据vendor和信息截止日期；
- 当前尚未调用Expert模型生成18条真实轨迹，因此`accepted.jsonl`和`rejected.jsonl`还不存在；
- 未来5日收益或Alpha只能作为运行后的评估标签追加，永远不进入任务筛选或Scheduler输入。

这一区分保证“已经实现的数据构建代码”“已冻结的任务清单”和“尚待付费API生成的轨迹”不会混为一谈。
