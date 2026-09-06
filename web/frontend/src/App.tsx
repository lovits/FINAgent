import {
  Activity,
  BarChart3,
  Bot,
  Check,
  CircleDot,
  FileText,
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
    source.addEventListener("node.completed", (event) => {
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
        <div className={`run-status ${run?.status ?? "idle"}`} role="status" aria-live="polite">
          <CircleDot size={14} />
          {statusLabel(run?.status)}
        </div>
      </header>

      <main className={`workspace ${run ? "has-run" : ""}`}>
        <section className="control-panel" aria-labelledby="task-title">
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
            <button className="primary-button" type="submit" disabled={isBusy}>
              {isBusy ? <><RefreshCw className="spin" size={18} />分析进行中</> : <><Play size={18} />开始多Agent分析</>}
            </button>
          </form>
        </section>

        <section className="result-panel" aria-labelledby="result-title">
          {!run ? <EmptyState /> : (
            <>
              <div className="run-header">
                <div><p className="step-label">02 / EXECUTION</p><h2 id="result-title">{run.request.ticker} 分析工作台</h2></div>
                <div className="run-meta"><span>{modeTitle(run.request.orchestration_mode)}</span><span>{run.request.analysis_date}</span></div>
              </div>

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
            </>
          )}
        </section>
      </main>
    </div>
  );
}

function EmptyState() {
  return <div className="empty-state"><div className="orb"><Bot size={34} /></div><p className="step-label">READY FOR ANALYSIS</p><h2>让Agent团队开始研究</h2><p>配置股票、分析师和编排模式。运行过程中可以实时查看节点完成状态，结束后阅读完整报告。</p></div>;
}

function TimelineItem({ node, index }: { node: NodeEvent; index: number }) {
  const Icon = node.kind === "tool" ? Wrench : node.kind === "scheduler" ? Bot : Check;
  return <div className={`timeline-item ${node.kind}`}><div className="timeline-icon"><Icon size={14} /></div><div><small>STEP {String(index + 1).padStart(2, "0")}</small><strong>{node.node}</strong>{node.selected_action && <span>{node.selected_action.replace(/[<>]/g, "")}</span>}</div></div>;
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
