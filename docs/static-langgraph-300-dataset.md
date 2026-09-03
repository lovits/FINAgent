# TradingAgents Static LangGraph 300任务数据集设计

> 版本：v1
> 日期：2026-09-03
> 当前状态：300条任务种子已生成并通过校验；完整LLM轨迹尚未生成

后续轨迹审核、SFT和GRPO流程见[`scheduler-trajectory-sft-grpo-pipeline.md`](scheduler-trajectory-sft-grpo-pipeline.md)。

## 1. 扩大什么

本次把18条Pilot扩展为300个`股票 + 历史交易日`任务。扩大的是Static LangGraph的运行任务池，不是用LLM编写300条答案，也不是已经得到300条可训练轨迹。

```text
300条任务种子
→ 原Static LangGraph逐条运行
→ accepted/rejected完整轨迹
→ action-only SFT数据
→ Scheduler SFT与后续RL
```

## 2. 与TradingAgents论文的关系

TradingAgents原论文在2024-01-01至2024-03-29对Apple、Nvidia、Microsoft、Meta、Google等股票进行逐交易日决策，每次只能使用当日及以前的数据；输入覆盖价格、新闻、情绪、内部人交易、财务报表和技术指标。

本数据集保留它的三个核心方法：

1. 一个任务就是一个股票在一个历史交易日的完整决策；
2. 任务选择和Agent输入都遵守point-in-time边界；
3. 每个任务最终由同一套TradingAgents多Agent流程执行。

本项目为了训练通用Scheduler，将原论文偏科技股的范围扩展为跨行业股票池。这是工程扩展，不宣称复现原论文的收益结果。

来源：[TradingAgents论文](https://arxiv.org/html/2412.20138)

## 3. 冻结后的规模

| 项目 | 设计 |
|---|---|
| 总任务数 | 300 |
| 市场窗口 | 2025-01-02至2026-08-28 |
| 股票候选池 | 66只美股 |
| 行业 | 11个，每行业6只 |
| 任务类型 | 6类，每类50条 |
| Train | 210条，44只股票 |
| Validation | 42条，11只股票 |
| Test | 48条，11只股票 |
| 股票重叠 | 三个集合之间为0 |
| 单股票上限 | 6条 |
| 同股票日期间隔 | 至少21个自然日 |

六类采样层为：

- `earnings_window`：财报日期；
- `positive_momentum`：5日历史收益较高；
- `negative_momentum`：5日历史收益较低；
- `high_volatility`：20日历史波动较高；
- `volume_shock`：成交量相对20日窗口异常；
- `quiet_control`：低动量、低波动、成交量正常的对照任务。

每个采样层50条，不代表每类有唯一正确的Agent顺序。它们只保证Scheduler看到不同市场状态。

## 4. 股票池与拆分方法

股票池按11个行业组织：信息技术、通信服务、可选消费、日常消费、金融、医疗、工业、能源、公用事业、房地产和材料。

每个行业固定分配：

```text
4只股票 → Train
1只股票 → Validation
1只股票 → Test
```

因此，同一公司不会同时出现在训练集和测试集。任务选择在三个股票集合内分别执行：Train每类35条、Validation每类7条、Test每类8条，合计每类50条。

股票和拆分的唯一事实来源是：

```text
training/scheduler/configs/datasets/static_langgraph_300_v1.json
```

## 5. 数据构建流程

```text
冻结股票池和split
→ yfinance逐股票下载OHLCV与财报日历
→ 仅用t日及以前数据计算5日收益、20日波动、成交量Z-score
→ 在每个split内部按六类规则排序
→ 施加ticker/date去重、单股票上限和21日间隔
→ 写出300条JSONL和manifest
→ 校验数量、checksum、类别、split和未来字段
```

任何一个类别无法满足冻结约束时，程序直接失败，不会静默减少数量、换股票或放宽日期间隔。

## 6. 代码对应关系

| 文件 | 作用 |
|---|---|
| `build_task_seeds.py` | 计算point-in-time市场特征和六类候选排序 |
| `scaled_task_seeds.py` | 读取66只股票计划，按ticker-disjoint split生成300条任务 |
| `validate_task_manifest.py` | 校验数量、类别、split隔离、日期间隔和SHA-256 |
| `static_trace.py` | 运行原Static LangGraph并采集Agent/Tool/状态事件 |
| `generate_data.py` | 按split和每类数量分阶段生成可续跑的完整轨迹 |
| `trajectory_audit.py` | 审核动作、节点、时间截止、终止、输出和重复ID |
| `prepare_sft_dataset.py` | 审核后按split生成SFT数据、统计和checksum |

## 7. 重建与校验

重新获取公开市场数据并构建任务：

```bash
python -m training.scheduler.scaled_task_seeds \
  --plan training/scheduler/configs/datasets/static_langgraph_300_v1.json \
  --output-tasks data/scheduler/tasks/static_langgraph_300_v1.rebuild.jsonl \
  --output-manifest data/scheduler/tasks/static_langgraph_300_v1.rebuild.manifest.json
```

校验冻结版本：

```bash
python -m training.scheduler.validate_task_manifest \
  --tasks data/scheduler/tasks/static_langgraph_300_v1.jsonl \
  --manifest data/scheduler/tasks/static_langgraph_300_v1.manifest.json
```

yfinance可能修订历史行情或财报日期，因此重新构建结果用于审计，不应静默覆盖冻结文件。

## 8. 分阶段生成完整轨迹

先从Train中每类10条生成60条平衡轨迹：

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

60条通过人工与自动检查后，保留同一Train输出目录并去掉`--tasks-per-family`再次运行；`--resume`会跳过已有60条并补齐Train。随后单独生成Validation。Strong Teacher与RL只能使用Train；Validation用于选择配置；Test只用于最终Static/Learned A/B。

原Static路径每条至少包含12次宏观Agent调用，因此300条完整运行至少约3600次Agent调用，尚未包含Tool循环。正式运行前必须先估算模型费用并设置批次恢复策略。

## 9. 当前验收结果

- 300条任务；
- 66只唯一股票；
- 每类50条；
- Train/Validation/Test为210/42/48；
- 每个split覆盖11个行业；
- 同一股票不跨split；
- 单只股票最多6条；
- 实际任务日期为2025-01-06至2026-08-26；
- 未来收益、答案和目标动作字段数量为0；
- 任务文件SHA-256已写入manifest并由校验器核对。

## 10. 当前边界

- 已完成：任务设计、股票池、300条任务文件、manifest、重建代码和校验代码；
- 未完成：300条Static LLM轨迹、人工冲突标签、SFT文件和模型训练；
- 未使用：OpenRouter Key；
- 不重新分发：yfinance原始行情与第三方新闻内容；
- 不声称：这些任务包含真实交易真值，或扩大数量后已经获得收益提升。
