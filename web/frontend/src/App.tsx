import {
  Activity,
  ArrowLeft,
  BarChart3,
  Bot,
  Check,
  ChevronRight,
  CircleDot,
  FileText,
  GitBranch,
  Play,
  RefreshCw,
  Settings2,
  Sparkles,
  Wrench,
} from "lucide-react";
import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";

import { createRun, getRun } from "./api";
import type {
  Analyst,
  NodeEvent,
  OrchestrationMode,
  ReportEvent,
  ResearchDepth,
  RunRequest,
  RunSnapshot,
} from "./types";

const MODES: Array<{
  value: OrchestrationMode;
  title: string;
  label: string;
  description: string;
}> = [
  {
    value: "static",
    title: "Static LangGraph",
    label: "固定编排",
    description: "保留原版TradingAgents调用路线，适合作为稳定基线。",
  },
  {
    value: "teacher",
    title: "Teacher Scheduler",
    label: "云端动态编排",
    description: "由Teacher模型根据当前报告与合法Agent动态选择下一步。",
  },
  {
    value: "learned",
    title: "Learned Scheduler",
    label: "本地模型编排",
    description: "使用SFT与GRPO训练后的本地Qwen调度器，仅支持Shallow。",
  },
];

const ANALYSTS: Array<{ value: Analyst; label: string; detail: string }> = [
  { value: "market", label: "市场分析", detail: "行情与技术指标" },
  { value: "social", label: "情绪分析", detail: "新闻、StockTwits与Reddit" },
  { value: "news", label: "新闻分析", detail: "公司、宏观与事件信息" },
  { value: "fundamentals", label: "基本面分析", detail: "财务报表与公司信息" },
];

const REPORT_ORDER = [
  "market_report",
  "sentiment_report",
  "news_report",
  "fundamentals_report",
  "investment_plan",
  "trader_investment_plan",
  "final_trade_decision",
];

const REPORT_TITLES: Record<string, string> = {
  market_report: "市场分析",
  sentiment_report: "市场情绪",
  news_report: "新闻分析",
  fundamentals_report: "基本面分析",
  investment_plan: "研究团队结论",
  trader_investment_plan: "交易计划",
  final_trade_decision: "投资组合决策",
};

function today(): string {
  const now = new Date();
  const local = new Date(now.getTime() - now.getTimezoneOffset() * 60_000);
  return local.toISOString().slice(0, 10);
}

const INITIAL_FORM: RunRequest = {
  ticker: "NVDA",
  analysis_date: today(),
  output_language: "Chinese",
  analysts: ["market", "social", "news", "fundamentals"],
  research_depth: "shallow",
  orchestration_mode: "static",
};

export default function App() {
  const [form, setForm] = useState<RunRequest>(INITIAL_FORM);
  const [run, setRun] = useState<RunSnapshot | null>(null);
  const [nodes, setNodes] = useState<NodeEvent[]>([]);
  const [reports, setReports] = useState<Record<string, string>>({});
  const [activeReport, setActiveReport] = useState("market_report");
  const [formError, setFormError] = useState("");
  const eventSourceRef = useRef<EventSource | null>(null);

  const isBusy = run?.status === "queued" || run?.status === "running";
  const visibleReports = useMemo(
    () => REPORT_ORDER.filter((key) => reports[key]),
    [reports],
  );

  useEffect(() => () => eventSourceRef.current?.close(), []);

  function updateMode(mode: OrchestrationMode) {
    setForm((current) => ({
      ...current,
      orchestration_mode: mode,
      research_depth: mode === "learned" ? "shallow" : current.research_depth,
    }));
  }

  function toggleAnalyst(analyst: Analyst) {
    setForm((current) => {
      const selected = current.analysts.includes(analyst)
        ? current.analysts.filter((value) => value !== analyst)
        : [...current.analysts, analyst];
      return { ...current, analysts: selected };
    });
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!form.analysts.length) {
      setFormError("请至少选择一名分析师。");
      return;
    }
    setFormError("");
    setNodes([]);
    setReports({});
    try {
      const created = await createRun(form);
      setRun(created);
      connectEvents(created.run_id);
    } catch (error) {
      setFormError(error instanceof Error ? error.message : "任务创建失败。");
    }
  }

  function connectEvents(runId: string) {
    eventSourceRef.current?.close();
    const source = new EventSource(`/api/runs/${runId}/events`);
    eventSourceRef.current = source;
    source.onopen = () => setFormError("");
    source.addEventListener("node.progress", (event) => {
      setNodes((current) => [...current, JSON.parse(event.data) as NodeEvent]);
    });
    source.addEventListener("report.updated", (event) => {
      const report = JSON.parse(event.data) as ReportEvent;
      setReports((current) => {
        if (Object.keys(current).length === 0) setActiveReport(report.section);
        return { ...current, [report.section]: report.content };
      });
    });
    source.addEventListener("run.completed", () => finishRun(runId, source));
    source.addEventListener("run.failed", () => finishRun(runId, source));
    source.onerror = () => {
      if (source.readyState !== EventSource.CLOSED) {
        setFormError("实时连接暂时中断，正在自动重连。");
      }
    };
  }

  async function finishRun(runId: string, source: EventSource) {
    source.close();
    const snapshot = await getRun(runId);
    setRun(snapshot);
    setReports(snapshot.report_sections);
    const first = REPORT_ORDER.find((key) => snapshot.report_sections[key]);
    if (first) setActiveReport(first);
  }

  function reset() {
    eventSourceRef.current?.close();
    setRun(null);
    setNodes([]);
    setReports({});
    setFormError("");
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-mark" aria-hidden="true"><BarChart3 size={20} /></div>
        <div>
          <p className="eyebrow">MULTI-AGENT RESEARCH CONSOLE</p>
          <h1>TradingAgents <span>RL</span></h1>
        </div>
        <nav className="page-nav" aria-label="分析流程">
          <span className={!run ? "active" : "complete"}>1&nbsp; 任务配置</span>
          <ChevronRight size={14} aria-hidden="true" />
          <span className={run ? "active" : ""}>2&nbsp; 分析报告</span>
        </nav>
        <div className={`run-status ${run?.status ?? "idle"}`} role="status" aria-live="polite">
          <CircleDot size={14} />
          {statusLabel(run?.status)}
        </div>
      </header>

      {!run ? <main className="configure-page">
        <section className="control-panel configure-card" aria-labelledby="task-title">
          <div className="section-heading">
            <div><p className="step-label">01 / CONFIGURE</p><h2 id="task-title">新建分析任务</h2></div>
            <Settings2 size={20} aria-hidden="true" />
          </div>
          <form onSubmit={submit}>
            <div className="field-grid">
              <label className="field">
                <span>股票代码</span>
                <input
                  value={form.ticker}
                  onChange={(event) => setForm({ ...form, ticker: event.target.value.toUpperCase() })}
                  placeholder="例如 NVDA"
                  disabled={isBusy}
                  required
                />
              </label>
              <label className="field">
                <span>分析日期</span>
                <input
                  type="date"
                  value={form.analysis_date}
                  onChange={(event) => setForm({ ...form, analysis_date: event.target.value })}
                  disabled={isBusy}
                  required
                />
              </label>
            </div>

            <fieldset className="mode-fieldset" disabled={isBusy}>
              <legend>Agent编排模式</legend>
              <div className="mode-grid">
                {MODES.map((mode) => (
                  <label className={`mode-card ${form.orchestration_mode === mode.value ? "selected" : ""}`} key={mode.value}>
                    <input
                      type="radio"
                      name="mode"
                      value={mode.value}
                      checked={form.orchestration_mode === mode.value}
                      onChange={() => updateMode(mode.value)}
                    />
                    <span className="mode-card-top"><Bot size={18} />{mode.label}</span>
                    <strong>{mode.title}</strong>
                    <small>{mode.description}</small>
                  </label>
                ))}
              </div>
            </fieldset>

            <fieldset className="analyst-fieldset" disabled={isBusy}>
              <legend>分析师团队</legend>
              <div className="analyst-grid">
                {ANALYSTS.map((analyst) => {
                  const checked = form.analysts.includes(analyst.value);
                  return (
                    <label className={`analyst-option ${checked ? "selected" : ""}`} key={analyst.value}>
                      <input type="checkbox" checked={checked} onChange={() => toggleAnalyst(analyst.value)} />
                      <span className="check-box">{checked && <Check size={14} />}</span>
                      <span><strong>{analyst.label}</strong><small>{analyst.detail}</small></span>
                    </label>
                  );
                })}
              </div>
            </fieldset>

            <div className="field-grid compact-fields">
              <label className="field">
                <span>研究强度</span>
                <select
                  value={form.research_depth}
                  onChange={(event) => setForm({ ...form, research_depth: event.target.value as ResearchDepth })}
                  disabled={isBusy || form.orchestration_mode === "learned"}
                >
                  <option value="shallow">Shallow · 快速研究</option>
                  <option value="medium">Medium · 中等研究</option>
                  <option value="deep">Deep · 深度研究</option>
                </select>
                {form.orchestration_mode === "learned" && <small>本地调度器固定使用Shallow。</small>}
              </label>
              <label className="field">
                <span>报告语言</span>
                <select value={form.output_language} onChange={(event) => setForm({ ...form, output_language: event.target.value })} disabled={isBusy}>
                  <option value="Chinese">中文</option>
                  <option value="English">English</option>
                </select>
              </label>
            </div>

            {formError && <div className="form-error" role="alert">{formError}</div>}
            <div className="form-footer">
              <p>提交后进入分析工作台，实时查看Agent节点和报告生成进度。</p>
              <button className="primary-button" type="submit" disabled={isBusy}>
                {isBusy ? <><RefreshCw className="spin" size={18} />分析进行中</> : <><Play size={18} />开始多Agent分析</>}
              </button>
            </div>
          </form>
        </section>
      </main> : <main className="analysis-page">
        <section className="result-panel analysis-card" aria-labelledby="result-title">
              <div className="run-header">
                <div className="run-title-group">
                  <button
                    className="back-button"
                    onClick={reset}
                    disabled={isBusy}
                    aria-label={isBusy ? "分析运行中，暂时无法返回" : "返回任务配置"}
                  >
                    <ArrowLeft size={18} />
                  </button>
                  <div><p className="step-label">02 / EXECUTION & REPORT</p><h2 id="result-title">{run.request.ticker} 分析工作台</h2></div>
                </div>
                <div className="run-meta"><span>{modeTitle(run.request.orchestration_mode)}</span><span>{run.request.analysis_date}</span></div>
              </div>

              {formError && <div className="connection-warning" role="status">{formError}</div>}

              <ProcessGraph
                analysts={run.request.analysts}
                events={nodes}
                mode={run.request.orchestration_mode}
                finished={run.status === "completed"}
              />

              <div className="execution-grid">
                <aside className="timeline-panel" aria-label="Agent执行进度">
                  <div className="panel-title"><Activity size={17} /><h3>执行进度</h3><span>{nodes.length}</span></div>
                  <div className="timeline" aria-live="polite">
                    {nodes.length === 0 && <div className="waiting"><span className="pulse" />正在初始化Agent图…</div>}
                    {nodes.map((node, index) => <TimelineItem node={node} index={index} key={`${index}-${node.node}`} />)}
                    {isBusy && nodes.length > 0 && <div className="waiting"><span className="pulse" />等待下一节点…</div>}
                  </div>
                </aside>

                <article className="report-panel">
                  <div className="panel-title"><FileText size={17} /><h3>分析报告</h3></div>
                  {visibleReports.length ? (
                    <>
                      <nav className="report-tabs" aria-label="报告章节">
                        {visibleReports.map((key) => (
                          <button className={activeReport === key ? "active" : ""} onClick={() => setActiveReport(key)} key={key}>
                            {REPORT_TITLES[key]}
                          </button>
                        ))}
                      </nav>
                      <div className="markdown-body"><ReactMarkdown>{reports[activeReport] ?? ""}</ReactMarkdown></div>
                    </>
                  ) : <div className="report-placeholder"><Sparkles size={28} /><p>Agent完成分析后，报告会在这里逐段出现。</p></div>}
                </article>
              </div>

              {run.status === "completed" && <CompletionSummary run={run} onReset={reset} />}
              {run.status === "failed" && <div className="run-error" role="alert"><strong>任务未完成</strong><p>{run.error}</p><button onClick={reset}>重新创建任务</button></div>}
        </section>
      </main>}
    </div>
  );
}

function TimelineItem({ node, index }: { node: NodeEvent; index: number }) {
  const Icon = node.kind === "tool" ? Wrench : node.kind === "scheduler" ? Bot : Check;
  return <div className={`timeline-item ${node.kind} ${node.status}`}><div className="timeline-icon"><Icon size={14} /></div><div><small>STEP {String(index + 1).padStart(2, "0")}</small><strong>{node.node}</strong>{node.selected_action && <span>{actionLabel(node.selected_action)}</span>}</div></div>;
}

const ACTION_NODE: Record<string, string> = {
  ACT_MARKET: "Market Analyst",
  ACT_SENTIMENT: "Sentiment Analyst",
  ACT_NEWS: "News Analyst",
  ACT_FUNDAMENTALS: "Fundamentals Analyst",
  ACT_BULL: "Bull Researcher",
  ACT_BEAR: "Bear Researcher",
  ACT_RESEARCH_MANAGER: "Research Manager",
  ACT_TRADER: "Trader",
  ACT_AGGRESSIVE: "Aggressive Analyst",
  ACT_CONSERVATIVE: "Conservative Analyst",
  ACT_NEUTRAL: "Neutral Analyst",
  ACT_PORTFOLIO_MANAGER: "Portfolio Manager",
  ACT_STOP: "完成",
};

const ANALYST_NODE: Record<Analyst, string> = {
  market: "Market Analyst",
  social: "Sentiment Analyst",
  news: "News Analyst",
  fundamentals: "Fundamentals Analyst",
};

function actionLabel(action: string) {
  return action.replace(/[<>]/g, "");
}

function ProcessGraph({ analysts, events, mode, finished }: {
  analysts: Analyst[];
  events: NodeEvent[];
  mode: OrchestrationMode;
  finished: boolean;
}) {
  const completed = new Set(events.filter((event) => event.status === "completed").map((event) => event.node));
  const latestSchedulerIndex = events
    .map((event) => Boolean(event.kind === "scheduler" && event.selected_action))
    .lastIndexOf(true);
  const schedulerTarget = latestSchedulerIndex >= 0
    ? ACTION_NODE[actionLabel(events[latestSchedulerIndex].selected_action ?? "")]
    : undefined;
  const targetFinished = schedulerTarget
    ? events.slice(latestSchedulerIndex + 1).some((event) => event.node === schedulerTarget && event.status === "completed")
    : false;
  const running = new Set(events.filter((event) => event.status === "running").map((event) => event.node));
  if (schedulerTarget && !targetFinished) running.add(schedulerTarget);
  const path = events
    .filter((event) => event.kind !== "tool" && event.node !== "Scheduler")
    .map((event) => event.node)
    .filter((node, index, values) => index === 0 || node !== values[index - 1]);

  const nodeState = (node: string): GraphNodeState => {
    if (completed.has(node)) return "completed";
    if (running.has(node)) return "running";
    if (finished) return "skipped";
    return "pending";
  };
  const analystNodes = analysts.map((key, index) => ({
    name: ANALYST_NODE[key],
    x: 46 + (index % 2) * 134,
    y: 154 + Math.floor(index / 2) * 68,
  }));
  const hasStarted = (names: string[]) => names.some((name) => nodeState(name) !== "pending");
  const researchNodes = ["Bull Researcher", "Bear Researcher", "Research Manager"];
  const riskNodes = ["Aggressive Analyst", "Conservative Analyst", "Neutral Analyst"];

  return <section className="process-graph" aria-label="多Agent分析阶段图">
    <div className="process-heading">
      <div><GitBranch size={17} /><h3>Agent阶段流程</h3></div>
      <span>{mode === "static" ? "固定LangGraph" : "动态Scheduler路径"}</span>
    </div>
    <div className="workflow-canvas">
      <svg viewBox="0 0 1460 330" role="img" aria-labelledby="workflow-title workflow-description">
        <title id="workflow-title">TradingAgents实时分析流程</title>
        <desc id="workflow-description">从编排入口、分析师团队、研究辩论、交易、风险决策到最终报告的实时节点状态。</desc>
        <defs>
          <marker id="flow-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
            <path d="M0 0L8 4L0 8Z" />
          </marker>
          <filter id="active-glow" x="-40%" y="-40%" width="180%" height="180%">
            <feGaussianBlur stdDeviation="5" result="blur" />
            <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
        </defs>

        <text className="lane-label" x="28" y="31">编排策略</text>
        <text className="lane-label" x="28" y="101">AGENT EXECUTION</text>
        <path className="lane-divider" d="M28 48H1432M28 111H1432" />

        <g className={`router-node ${events.length ? "completed" : "running"}`}>
          <rect x="28" y="54" width="252" height="43" rx="12" />
          <circle cx="50" cy="75.5" r="7" />
          <text x="66" y="80">{mode === "static" ? "Static LangGraph Router" : "Dynamic Scheduler"}</text>
          <text className="node-state" x="263" y="80" textAnchor="end">{events.length ? "已启动" : "初始化"}</text>
        </g>

        <path className="flow-link active" d="M154 97V116" markerEnd="url(#flow-arrow)" />
        <FlowLink from={310} to={336} active={hasStarted(researchNodes)} />
        <FlowLink from={626} to={652} active={nodeState("Trader") !== "pending"} />
        <FlowLink from={914} to={940} active={hasStarted(riskNodes)} />
        <FlowLink from={1082} to={1108} active={nodeState("Portfolio Manager") !== "pending"} />
        <FlowLink from={1368} to={1392} active={finished} />

        <StageShell x={28} width={282} index="01" label="多源分析" />
        {analystNodes.map((node) => <SvgAgentNode {...node} state={nodeState(node.name)} key={node.name} />)}

        <StageShell x={336} width={290} index="02" label="研究辩论" />
        <SvgAgentNode x={354} y={154} name="Bull Researcher" state={nodeState("Bull Researcher")} />
        <SvgAgentNode x={498} y={154} name="Bear Researcher" state={nodeState("Bear Researcher")} />
        <SvgAgentNode x={402} y={222} name="Research Manager" state={nodeState("Research Manager")} wide />

        <StageShell x={652} width={262} index="03" label="交易计划" />
        <SvgAgentNode x={711} y={188} name="Trader" state={nodeState("Trader")} wide />

        <StageShell x={940} width={142} index="04" label="风险评估" />
        <SvgAgentNode x={950} y={142} name="Aggressive Analyst" state={nodeState("Aggressive Analyst")} compact />
        <SvgAgentNode x={950} y={197} name="Conservative Analyst" state={nodeState("Conservative Analyst")} compact />
        <SvgAgentNode x={950} y={252} name="Neutral Analyst" state={nodeState("Neutral Analyst")} compact />

        <StageShell x={1108} width={260} index="05" label="组合决策" />
        <SvgAgentNode x={1167} y={188} name="Portfolio Manager" state={nodeState("Portfolio Manager")} wide />

        <g className={`final-node ${finished ? "completed" : "pending"}`}>
          <rect x="1392" y="160" width="56" height="96" rx="14" />
          <text x="1420" y="194" textAnchor="middle">报告</text>
          <text x="1420" y="215" textAnchor="middle">输出</text>
          <text className="node-state" x="1420" y="239" textAnchor="middle">{finished ? "完成" : "等待"}</text>
        </g>
      </svg>
    </div>
    <div className="actual-path">
      <span>实际调用路径</span>
      <div>{path.length ? path.map((node, index) => <span key={`${index}-${node}`}>{index > 0 && <ChevronRight size={12} />}{node}</span>) : "等待第一个Agent节点"}</div>
    </div>
  </section>;
}

type GraphNodeState = "pending" | "running" | "completed" | "skipped";

function FlowLink({ from, to, active }: { from: number; to: number; active: boolean }) {
  return <path className={`flow-link ${active ? "active" : ""}`} d={`M${from} 205H${to}`} markerEnd="url(#flow-arrow)" />;
}

function StageShell({ x, width, index, label }: { x: number; width: number; index: string; label: string }) {
  return <g className="stage-shell"><rect x={x} y="122" width={width} height="188" rx="16" /><text className="stage-index" x={x + 16} y="145">{index}</text><text className="stage-label" x={x + 46} y="145">{label}</text></g>;
}

function SvgAgentNode({ x, y, name, state, wide = false, compact = false }: {
  x: number;
  y: number;
  name: string;
  state: GraphNodeState;
  wide?: boolean;
  compact?: boolean;
}) {
  const width = compact ? 122 : wide ? 172 : 128;
  const label = name.replace(" Analyst", "").replace(" Researcher", "").replace(" Manager", " Mgr");
  const status = { pending: "等待", running: "运行中", completed: "完成", skipped: "未调用" }[state];
  return <g className={`svg-agent-node ${state}`} filter={state === "running" ? "url(#active-glow)" : undefined}>
    <rect x={x} y={y} width={width} height="50" rx="10" />
    <circle cx={x + 14} cy={y + 16} r="5" />
    <text className="agent-name" x={x + 25} y={y + 20}>{label}</text>
    <text className="node-state" x={x + 14} y={y + 39}>{status}</text>
  </g>;
}

function CompletionSummary({ run, onReset }: { run: RunSnapshot; onReset: () => void }) {
  const totalTokens = (run.metrics.input_tokens ?? 0) + (run.metrics.output_tokens ?? 0);
  return <section className="completion-summary"><div><p className="step-label">03 / COMPLETE</p><h3>分析报告已完成</h3></div><dl><div><dt>最终信号</dt><dd>{run.signal ?? "已生成"}</dd></div><div><dt>LLM调用</dt><dd>{run.metrics.llm_calls ?? 0}</dd></div><div><dt>工具调用</dt><dd>{run.metrics.tool_calls ?? 0}</dd></div><div><dt>Token</dt><dd>{totalTokens.toLocaleString()}</dd></div></dl><button className="secondary-button" onClick={onReset}>新建任务</button></section>;
}

function statusLabel(status?: RunSnapshot["status"]) {
  if (!status) return "等待任务";
  return {
    queued: "等待执行",
    running: "分析运行中",
    completed: "报告已完成",
    failed: "执行失败",
  }[status];
}

function modeTitle(mode: OrchestrationMode) {
  return MODES.find((item) => item.value === mode)?.title ?? mode;
}
