import {
  Activity,
  ArrowLeft,
  BarChart3,
  Bot,
  Check,
  ChevronRight,
  CircleDot,
  Download,
  FileText,
  GitBranch,
  KeyRound,
  Play,
  RefreshCw,
  Save,
  Settings2,
  Sparkles,
  Square,
  X,
} from "lucide-react";
import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";

import { cancelRun, createRun, getActiveRun, getRun, getSettings, updateSettings } from "./api";
import type {
  Analyst,
  NodeEvent,
  OrchestrationMode,
  ReportEvent,
  ResearchDepth,
  RunRequest,
  RunSnapshot,
  UsageMetrics,
  WebSettings,
} from "./types";

interface CompletedRun {
  snapshot: RunSnapshot;
  nodes: NodeEvent[];
  reports: Record<string, string>;
}

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
  { value: "social", label: "情绪分析", detail: "新闻与市场情绪信号" },
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
  ticker: "MOUTAI",
  analysis_date: today(),
  output_language: "Chinese",
  analysts: ["market", "social", "news", "fundamentals"],
  research_depth: "shallow",
  orchestration_mode: "static",
};

function isAShareInput(value: string): boolean {
  const aliases = new Set(["MOUTAI", "KWEICHOWMOUTAI", "PINGAN", "CATL", "BYD"]);
  return value
    .split(/[,，\s]+/)
    .filter(Boolean)
    .every((ticker) => aliases.has(ticker.toUpperCase()) || /^\d{6}(\.(SS|SH|SZ))?$/i.test(ticker));
}

const EMPTY_USAGE: UsageMetrics = {
  llm_calls: 0,
  tool_calls: 0,
  input_tokens: 0,
  output_tokens: 0,
};

function totalTokens(metrics: Partial<UsageMetrics> | undefined): number {
  return (metrics?.input_tokens ?? 0) + (metrics?.output_tokens ?? 0);
}

function formatNumber(value: number): string {
  return new Intl.NumberFormat("zh-CN").format(value);
}

function formatCompact(value: number): string {
  return new Intl.NumberFormat("zh-CN", {
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(value);
}

function nodeUsage(events: NodeEvent[], node: string): UsageMetrics {
  return events
    .filter((event) => event.node === node)
    .reduce(
      (total, event) => ({
        llm_calls: total.llm_calls + (event.usage?.llm_calls ?? 0),
        tool_calls: total.tool_calls + (event.usage?.tool_calls ?? 0),
        input_tokens: total.input_tokens + (event.usage?.input_tokens ?? 0),
        output_tokens: total.output_tokens + (event.usage?.output_tokens ?? 0),
      }),
      EMPTY_USAGE,
    );
}

export default function App() {
  const [form, setForm] = useState<RunRequest>(INITIAL_FORM);
  const [run, setRun] = useState<RunSnapshot | null>(null);
  const [nodes, setNodes] = useState<NodeEvent[]>([]);
  const [reports, setReports] = useState<Record<string, string>>({});
  const [completedRuns, setCompletedRuns] = useState<CompletedRun[]>([]);
  const [batchTotal, setBatchTotal] = useState(1);
  const [activeReport, setActiveReport] = useState("market_report");
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [formError, setFormError] = useState("");
  const [settings, setSettings] = useState<WebSettings | null>(null);
  const [keyInput, setKeyInput] = useState("");
  const [expertModel, setExpertModel] = useState("");
  const [teacherModel, setTeacherModel] = useState("");
  const [settingsMessage, setSettingsMessage] = useState("");
  const eventSourceRef = useRef<EventSource | null>(null);
  const queueRef = useRef<RunRequest[]>([]);
  const nodesRef = useRef<NodeEvent[]>([]);
  const reportsRef = useRef<Record<string, string>>({});
  const completedRunsRef = useRef<CompletedRun[]>([]);

  const isBusy = ["queued", "running", "cancelling"].includes(run?.status ?? "");
  const visibleNodes = useMemo(
    () => nodes.filter((node) => node.kind !== "tool"),
    [nodes],
  );
  const failedNode = run?.status === "failed" ? run.error_details?.node : undefined;
  const failedNodeIndex = failedNode
    ? visibleNodes.map((node) => node.node).lastIndexOf(failedNode)
    : -1;
  const visibleReports = useMemo(
    () => REPORT_ORDER.filter((key) => reports[key]),
    [reports],
  );
  const liveMetrics = useMemo<UsageMetrics>(() => {
    const latest = nodes.at(-1)?.cumulative_metrics;
    if (latest) return latest;
    return {
      llm_calls: run?.metrics.llm_calls ?? 0,
      tool_calls: run?.metrics.tool_calls ?? 0,
      input_tokens: run?.metrics.input_tokens ?? 0,
      output_tokens: run?.metrics.output_tokens ?? 0,
    };
  }, [nodes, run]);

  useEffect(() => {
    void getSettings()
      .then((value) => {
        setSettings(value);
        setExpertModel(value.expert_model);
        setTeacherModel(value.teacher_model);
      })
      .catch(() => setSettingsMessage("无法读取本地模型设置。"));
    void getActiveRun()
      .then((activeRun) => {
        if (!activeRun) return;
        const restoredReports = { ...activeRun.report_sections };
        nodesRef.current = [];
        reportsRef.current = restoredReports;
        setForm(activeRun.request);
        setRun(activeRun);
        setNodes([]);
        setReports(restoredReports);
        const first = REPORT_ORDER.find((key) => restoredReports[key]);
        if (first) setActiveReport(first);
        connectEvents(activeRun.run_id);
      })
      .catch(() => setFormError("无法恢复后台分析任务，请刷新页面重试。"));
    return () => eventSourceRef.current?.close();
  }, []);

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
    const tickers = [...new Set(form.ticker.split(/[,，\s]+/).map((value) => value.trim()).filter(Boolean))];
    if (!tickers.length) {
      setFormError("请至少输入一个股票代码。");
      return;
    }
    const requests = tickers.map((ticker) => ({ ...form, ticker }));
    queueRef.current = requests.slice(1);
    setBatchTotal(requests.length);
    completedRunsRef.current = [];
    setCompletedRuns([]);
    setFormError("");
    await startRun(requests[0]);
  }

  async function startRun(payload: RunRequest) {
    nodesRef.current = [];
    reportsRef.current = {};
    setNodes([]);
    setReports({});
    setSelectedNode(null);
    try {
      const created = await createRun(payload);
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
      const item = JSON.parse(event.data) as NodeEvent;
      nodesRef.current = [...nodesRef.current, item];
      setNodes(nodesRef.current);
    });
    source.addEventListener("report.updated", (event) => {
      const report = JSON.parse(event.data) as ReportEvent;
      if (Object.keys(reportsRef.current).length === 0) setActiveReport(report.section);
      reportsRef.current = { ...reportsRef.current, [report.section]: report.content };
      setReports(reportsRef.current);
    });
    source.addEventListener("run.completed", () => finishRun(runId, source));
    source.addEventListener("run.failed", () => finishRun(runId, source));
    source.addEventListener("run.cancelled", () => finishRun(runId, source));
    source.onerror = () => {
      if (source.readyState !== EventSource.CLOSED) {
        setFormError("实时连接暂时中断，正在自动重连。");
      }
    };
  }

  async function finishRun(runId: string, source: EventSource) {
    source.close();
    const snapshot = await getRun(runId);
    const completed = {
      snapshot,
      nodes: [...nodesRef.current],
      reports: { ...snapshot.report_sections },
    };
    completedRunsRef.current = [...completedRunsRef.current, completed];
    setCompletedRuns(completedRunsRef.current);
    const next = queueRef.current.shift();
    if (next && snapshot.status !== "cancelled") {
      await startRun(next);
      return;
    }
    showCompleted(completed);
  }

  function reset() {
    queueRef.current = [];
    eventSourceRef.current?.close();
    setRun(null);
    setNodes([]);
    setReports({});
    setCompletedRuns([]);
    completedRunsRef.current = [];
    setBatchTotal(1);
    setSelectedNode(null);
    setFormError("");
  }

  async function stopAndReturn() {
    queueRef.current = [];
    eventSourceRef.current?.close();
    if (run && isBusy) {
      try {
        await cancelRun(run.run_id);
      } catch (error) {
        setFormError(error instanceof Error ? error.message : "停止任务失败。");
      }
    }
    reset();
  }

  function returnToConfig() {
    if (isBusy) void stopAndReturn();
    else reset();
  }

  function showCompleted(value: CompletedRun) {
    setRun(value.snapshot);
    setNodes(value.nodes);
    setReports(value.reports);
    const first = REPORT_ORDER.find((key) => value.reports[key]);
    if (first) setActiveReport(first);
    setSelectedNode(null);
  }

  async function saveSettings() {
    setSettingsMessage("");
    try {
      const value = await updateSettings({
        ...(keyInput.trim() ? { openrouter_api_key: keyInput.trim() } : {}),
        expert_model: expertModel,
        teacher_model: teacherModel,
      });
      setSettings(value);
      setKeyInput("");
      setSettingsMessage("模型设置已保存到本机.env。完整Key不会显示在页面上。");
    } catch (error) {
      setSettingsMessage(error instanceof Error ? error.message : "设置保存失败。");
    }
  }

  async function exportReport() {
    if (!run?.complete_report) return;
    await saveMarkdown(
      run.complete_report,
      `${run.request.ticker}_${run.request.analysis_date}_complete-report.md`,
    );
  }

  async function exportStageReport() {
    const content = reports[activeReport];
    if (!run || !content) return;
    await saveMarkdown(
      content,
      `${run.request.ticker}_${run.request.analysis_date}_${activeReport}.md`,
    );
  }

  async function saveMarkdown(content: string, filename: string) {
    try {
      const picker = (window as SavePickerWindow).showSaveFilePicker;
      if (picker) {
        const handle = await picker({ suggestedName: filename });
        const writable = await handle.createWritable();
        await writable.write(content);
        await writable.close();
        return;
      }
      const url = URL.createObjectURL(new Blob([content], { type: "text/markdown" }));
      const link = document.createElement("a");
      link.href = url;
      link.download = filename;
      link.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") return;
      setFormError(error instanceof Error ? error.message : "报告导出失败。");
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-mark" aria-hidden="true"><BarChart3 size={20} /></div>
        <h1>FIN <span>Agents</span></h1>
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
                <span>股票代码（多个代码用逗号或空格分隔）</span>
                <input
                  value={form.ticker}
                  onChange={(event) => setForm({ ...form, ticker: event.target.value.toUpperCase() })}
                  placeholder="例如 MOUTAI, PINGAN, CATL, BYD"
                  disabled={isBusy}
                  required
                />
                <small className="source-hint">英文别名会解析为A股代码：MOUTAI、PINGAN、CATL、BYD</small>
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

            <details className="settings-panel">
              <summary><KeyRound size={16} />模型与API设置</summary>
              <div className="settings-grid">
                <label className="field">
                  <span>OpenRouter API Key</span>
                  <input
                    type="password"
                    value={keyInput}
                    onChange={(event) => setKeyInput(event.target.value)}
                    placeholder={settings?.masked_key ?? "输入OpenRouter Key"}
                    autoComplete="off"
                  />
                  <small>{settings?.key_configured ? `当前：${settings.masked_key}` : "尚未配置"}</small>
                </label>
                <label className="field">
                  <span>专家Agent模型</span>
                  <input value={expertModel} onChange={(event) => setExpertModel(event.target.value)} placeholder="z-ai/glm-5.3-flash" />
                </label>
                <label className="field">
                  <span>Teacher调度模型</span>
                  <input value={teacherModel} onChange={(event) => setTeacherModel(event.target.value)} placeholder="z-ai/glm-5.3-flash" />
                </label>
                <div className="local-model-summary">
                  <span>本地调度器</span>
                  <strong>{settings?.scheduler_base_model ?? "未读取"}</strong>
                  <small>{settings?.scheduler_adapter ?? "未配置LoRA"}</small>
                </div>
              </div>
              <div className="settings-footer">
                <span role="status">{settingsMessage}</span>
                <button type="button" onClick={() => void saveSettings()}><Save size={15} />保存到本机</button>
              </div>
            </details>

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
                    onClick={returnToConfig}
                    aria-label={isBusy ? "停止当前任务并返回配置" : "返回任务配置"}
                  >
                    {isBusy ? <Square size={16} /> : <ArrowLeft size={18} />}
                  </button>
                  <div><p className="step-label">02 / EXECUTION & REPORT</p><h2 id="result-title">{run.request.ticker} 分析工作台</h2></div>
                </div>
                <div className="run-meta">
                  {batchTotal > 1 && <span>{Math.min(completedRuns.length + 1, batchTotal)} / {batchTotal}</span>}
                  <span>{modeTitle(run.request.orchestration_mode)}</span><span>{run.request.analysis_date}</span>
                  {run.resolved_ticker && run.resolved_ticker !== run.request.ticker && <span>解析：{run.resolved_ticker}</span>}
                  {(Object.values(run.data_sources ?? {}).length > 0 || isAShareInput(run.request.ticker)) && <span>数据：{Object.values(run.data_sources ?? {}).filter((value, index, values) => values.indexOf(value) === index).join(" + ") || "BaoStock + AKShare / Eastmoney"}</span>}
                  <span>专家：{run.models.expert ?? expertModel}</span><span>调度：{run.models.scheduler ?? "初始化中"}</span>
                </div>
              </div>

              {completedRuns.length > 1 && !isBusy && <nav className="batch-tabs" aria-label="批量任务报告">
                {completedRuns.map((item) => <button className={run.run_id === item.snapshot.run_id ? "active" : ""} onClick={() => showCompleted(item)} key={item.snapshot.run_id}>{item.snapshot.request.ticker}</button>)}
              </nav>}

              {formError && <div className="connection-warning" role="status">{formError}</div>}

              <LiveUsageStrip metrics={liveMetrics} running={isBusy} />

              <ProcessGraph
                analysts={run.request.analysts}
                events={visibleNodes}
                mode={run.request.orchestration_mode}
                finished={run.status === "completed"}
                failedNode={failedNode}
                onSelectNode={setSelectedNode}
              />

              <div className="execution-grid">
                <aside className="timeline-panel" aria-label="Agent执行进度">
                  <div className="panel-title"><Activity size={17} /><h3>执行进度</h3><span>{visibleNodes.length}</span></div>
                  <div className="timeline" aria-live="polite">
                    {visibleNodes.length === 0 && <div className="waiting"><span className="pulse" />正在初始化Agent图…</div>}
                    {visibleNodes.map((node, index) => <TimelineItem node={node} index={index} failed={index === failedNodeIndex} onSelect={setSelectedNode} key={`${index}-${node.node}`} />)}
                    {isBusy && visibleNodes.length > 0 && <div className="waiting"><span className="pulse" />等待下一节点…</div>}
                  </div>
                </aside>

                <article className="report-panel">
                  <div className="panel-title"><FileText size={17} /><h3>分析报告</h3>{reports[activeReport] && <button className="panel-export-button" onClick={() => void exportStageReport()}><Download size={14} />导出当前阶段</button>}</div>
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

              {run.status === "completed" && <CompletionSummary run={run} onReset={reset} onExport={exportReport} />}
              {run.status === "failed" && <RunErrorPanel run={run} onReset={reset} />}
        </section>
        {selectedNode && <NodeDetailDrawer node={selectedNode} events={visibleNodes} onClose={() => setSelectedNode(null)} />}
      </main>}
    </div>
  );
}

function TimelineItem({ node, index, failed, onSelect }: { node: NodeEvent; index: number; failed: boolean; onSelect: (node: string) => void }) {
  const isFailed = failed || node.status === "failed";
  const Icon = isFailed ? X : node.kind === "scheduler" ? Bot : Check;
  const tokens = totalTokens(node.usage);
  return <button className={`timeline-item ${node.kind} ${isFailed ? "failed" : node.status}`} onClick={() => onSelect(node.node)}><span className="timeline-icon"><Icon size={14} /></span><span className="timeline-copy"><small>STEP {String(index + 1).padStart(2, "0")}</small><strong>{node.node}</strong>{isFailed && <span className="timeline-error">失败</span>}{node.selected_action && <span className="timeline-action">{actionLabel(node.selected_action)}</span>}{tokens > 0 && <span className="timeline-token">+{formatNumber(tokens)} TOK</span>}</span></button>;
}

function LiveUsageStrip({ metrics, running }: { metrics: UsageMetrics; running: boolean }) {
  const total = totalTokens(metrics);
  return <section className="usage-strip" aria-label="实时资源消耗" aria-live="polite">
    <div className="usage-strip-title"><Activity size={16} aria-hidden="true" /><span>{running ? "实时消耗" : "本次消耗"}</span>{running && <i />}</div>
    <dl>
      <div><dt>输入 Token</dt><dd>{formatNumber(metrics.input_tokens)}</dd></div>
      <div><dt>输出 Token</dt><dd>{formatNumber(metrics.output_tokens)}</dd></div>
      <div className="usage-total"><dt>总 Token</dt><dd>{formatNumber(total)}</dd></div>
      <div><dt>模型调用</dt><dd>{formatNumber(metrics.llm_calls)}</dd></div>
      <div><dt>工具调用</dt><dd>{formatNumber(metrics.tool_calls)}</dd></div>
    </dl>
  </section>;
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

function ProcessGraph({ analysts, events, mode, finished, failedNode, onSelectNode }: {
  analysts: Analyst[];
  events: NodeEvent[];
  mode: OrchestrationMode;
  finished: boolean;
  failedNode?: string;
  onSelectNode: (node: string) => void;
}) {
  const completed = new Set(events.filter((event) => event.status === "completed").map((event) => event.node));
  const failed = new Set(events.filter((event) => event.status === "failed").map((event) => event.node));
  if (failedNode) failed.add(failedNode);
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
    if (failed.has(node)) return "failed";
    if (completed.has(node)) return "completed";
    if (running.has(node)) return "running";
    if (finished) return "skipped";
    return "pending";
  };
  const analystNodes = analysts.map((key, index) => ({
    name: ANALYST_NODE[key],
    x: 50 + (index % 2) * 135,
    y: 142 + Math.floor(index / 2) * 72,
  }));
  const hasStarted = (names: string[]) => names.some((name) => nodeState(name) !== "pending");
  const researchNodes = ["Bull Researcher", "Bear Researcher", "Research Manager"];
  const riskNodes = ["Aggressive Analyst", "Conservative Analyst", "Neutral Analyst"];
  const canOpen = (node: string) => events.some((event) => event.node === node);
  const schedulerTokens = totalTokens(nodeUsage(events, "Scheduler"));

  return <section className="process-graph" aria-label="多Agent分析阶段图">
    <div className="process-heading">
      <div><GitBranch size={17} /><h3>Agent阶段流程</h3></div>
      <span>{mode === "static" ? "固定LangGraph" : "动态Scheduler路径"}</span>
    </div>
    <div className="workflow-canvas">
      <svg viewBox="0 0 1000 590" role="img" aria-labelledby="workflow-title workflow-description">
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

        <path className="lane-divider" d="M30 78H970M30 326H970" />

        <g
          className={`router-node ${failed.has("Scheduler") ? "failed" : events.length ? "completed" : "running"} ${canOpen("Scheduler") ? "clickable" : ""}`}
          role={canOpen("Scheduler") ? "button" : undefined}
          tabIndex={canOpen("Scheduler") ? 0 : undefined}
          onClick={() => { if (canOpen("Scheduler")) onSelectNode("Scheduler"); }}
          onKeyDown={(event) => { if (canOpen("Scheduler") && (event.key === "Enter" || event.key === " ")) onSelectNode("Scheduler"); }}
        >
          <rect x="30" y="18" width="300" height="48" rx="13" />
          <circle cx="54" cy="42" r="7" />
          <text x="72" y="47">{mode === "static" ? "Static LangGraph Router" : "Dynamic Scheduler"}</text>
          <text className="node-state" x="312" y="47" textAnchor="end">{schedulerTokens ? `${formatCompact(schedulerTokens)} tok` : events.length ? "已启动" : "初始化"}</text>
        </g>

        <FlowPath d="M180 66V92" active />
        <FlowPath d="M330 194H360" active={hasStarted(researchNodes)} />
        <FlowPath d="M660 194H690" active={nodeState("Trader") !== "pending"} />
        <FlowPath d="M830 294V316H16V458H30" active={hasStarted(riskNodes)} />
        <FlowPath d="M330 458H360" active={nodeState("Portfolio Manager") !== "pending"} />
        <FlowPath d="M660 458H690" active={finished} />

        <StageShell x={30} y={96} width={300} height={198} index="01" label="多源分析" />
        {analystNodes.map((node) => <SvgAgentNode {...node} state={nodeState(node.name)} tokens={totalTokens(nodeUsage(events, node.name))} onSelect={canOpen(node.name) ? onSelectNode : undefined} key={node.name} />)}

        <StageShell x={360} y={96} width={300} height={198} index="02" label="研究辩论" />
        <SvgAgentNode x={380} y={142} name="Bull Researcher" state={nodeState("Bull Researcher")} tokens={totalTokens(nodeUsage(events, "Bull Researcher"))} onSelect={canOpen("Bull Researcher") ? onSelectNode : undefined} />
        <SvgAgentNode x={515} y={142} name="Bear Researcher" state={nodeState("Bear Researcher")} tokens={totalTokens(nodeUsage(events, "Bear Researcher"))} onSelect={canOpen("Bear Researcher") ? onSelectNode : undefined} />
        <SvgAgentNode x={424} y={214} name="Research Manager" state={nodeState("Research Manager")} tokens={totalTokens(nodeUsage(events, "Research Manager"))} onSelect={canOpen("Research Manager") ? onSelectNode : undefined} wide />

        <StageShell x={690} y={96} width={280} height={198} index="03" label="交易计划" />
        <SvgAgentNode x={744} y={170} name="Trader" state={nodeState("Trader")} tokens={totalTokens(nodeUsage(events, "Trader"))} onSelect={canOpen("Trader") ? onSelectNode : undefined} wide />

        <text className="route-caption" x="30" y="344">进入风险评估与最终决策</text>
        <StageShell x={30} y={356} width={300} height={204} index="04" label="风险评估" />
        <SvgAgentNode x={50} y={402} name="Aggressive Analyst" state={nodeState("Aggressive Analyst")} tokens={totalTokens(nodeUsage(events, "Aggressive Analyst"))} onSelect={canOpen("Aggressive Analyst") ? onSelectNode : undefined} />
        <SvgAgentNode x={185} y={402} name="Conservative Analyst" state={nodeState("Conservative Analyst")} tokens={totalTokens(nodeUsage(events, "Conservative Analyst"))} onSelect={canOpen("Conservative Analyst") ? onSelectNode : undefined} />
        <SvgAgentNode x={94} y={474} name="Neutral Analyst" state={nodeState("Neutral Analyst")} tokens={totalTokens(nodeUsage(events, "Neutral Analyst"))} onSelect={canOpen("Neutral Analyst") ? onSelectNode : undefined} wide />

        <StageShell x={360} y={356} width={300} height={204} index="05" label="组合决策" />
        <SvgAgentNode x={424} y={432} name="Portfolio Manager" state={nodeState("Portfolio Manager")} tokens={totalTokens(nodeUsage(events, "Portfolio Manager"))} onSelect={canOpen("Portfolio Manager") ? onSelectNode : undefined} wide />

        <StageShell x={690} y={356} width={280} height={204} index="06" label="报告交付" />
        <g className={`final-node ${finished ? "completed" : "pending"}`}>
          <rect x="744" y="416" width="172" height="92" rx="14" />
          <text x="830" y="450" textAnchor="middle">完整分析报告</text>
          <text className="node-state" x="830" y="478" textAnchor="middle">{finished ? "已生成" : "等待汇总"}</text>
        </g>
      </svg>
    </div>
    <div className="actual-path">
      <span>实际调用路径</span>
      <div>{path.length ? path.map((node, index) => <span key={`${index}-${node}`}>{index > 0 && <ChevronRight size={12} />}<button onClick={() => onSelectNode(node)}>{node}</button></span>) : "等待第一个Agent节点"}</div>
    </div>
  </section>;
}

type GraphNodeState = "pending" | "running" | "completed" | "failed" | "skipped";

function FlowPath({ d, active }: { d: string; active: boolean }) {
  return <path className={`flow-link ${active ? "active" : ""}`} d={d} markerEnd="url(#flow-arrow)" />;
}

function StageShell({ x, y, width, height, index, label }: { x: number; y: number; width: number; height: number; index: string; label: string }) {
  return <g className="stage-shell"><rect x={x} y={y} width={width} height={height} rx="16" /><text className="stage-index" x={x + 18} y={y + 29}>{index}</text><text className="stage-label" x={x + 54} y={y + 29}>{label}</text></g>;
}

function SvgAgentNode({ x, y, name, state, tokens, wide = false, onSelect }: {
  x: number;
  y: number;
  name: string;
  state: GraphNodeState;
  tokens: number;
  wide?: boolean;
  onSelect?: (node: string) => void;
}) {
  const width = wide ? 172 : 128;
  const label = name.replace(" Analyst", "").replace(" Researcher", "").replace(" Manager", " Mgr");
  const stateLabel = { pending: "等待", running: "运行中", completed: "完成", failed: "失败", skipped: "未调用" }[state];
  const status = tokens > 0 ? `${stateLabel} · ${formatCompact(tokens)} tok` : stateLabel;
  return <g className={`svg-agent-node ${state} ${onSelect ? "clickable" : ""}`} filter={state === "running" ? "url(#active-glow)" : undefined} role={onSelect ? "button" : undefined} tabIndex={onSelect ? 0 : undefined} onClick={() => onSelect?.(name)} onKeyDown={(event) => { if (onSelect && (event.key === "Enter" || event.key === " ")) onSelect(name); }}>
    <rect x={x} y={y} width={width} height="56" rx="11" />
    <circle cx={x + 15} cy={y + 18} r="5" />
    <text className="agent-name" x={x + 27} y={y + 23}>{label}</text>
    <text className="node-state" x={x + 15} y={y + 45}>{status}</text>
  </g>;
}

function usageForEvents(items: Array<{ event: NodeEvent }>): UsageMetrics {
  return items.reduce(
    (total, item) => ({
      llm_calls: total.llm_calls + (item.event.usage?.llm_calls ?? 0),
      tool_calls: total.tool_calls + (item.event.usage?.tool_calls ?? 0),
      input_tokens: total.input_tokens + (item.event.usage?.input_tokens ?? 0),
      output_tokens: total.output_tokens + (item.event.usage?.output_tokens ?? 0),
    }),
    EMPTY_USAGE,
  );
}

function NodeDetailDrawer({ node, events, onClose }: {
  node: string;
  events: NodeEvent[];
  onClose: () => void;
}) {
  const occurrences = events
    .map((event, index) => ({ event, index }))
    .filter((item) => item.event.node === node);
  const latest = occurrences.at(-1)?.event;
  const usage = usageForEvents(occurrences);
  useEffect(() => {
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [onClose]);
  return <div className="drawer-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
    <aside className="node-drawer" role="dialog" aria-modal="true" aria-labelledby="node-detail-title">
      <header><div><p className="step-label">NODE TRACE</p><h2 id="node-detail-title">{node}</h2></div><button onClick={onClose} aria-label="关闭节点详情"><X size={18} /></button></header>
      <div className="drawer-usage" aria-label="节点Token消耗"><div><span>输入 Token</span><strong>{formatNumber(usage.input_tokens)}</strong></div><div><span>输出 Token</span><strong>{formatNumber(usage.output_tokens)}</strong></div><div className="drawer-usage-total"><span>总 Token</span><strong>{formatNumber(totalTokens(usage))}</strong></div></div>
      <div className="drawer-section trace-section"><h3>Agent 输出 · {occurrences.length} 条</h3>{occurrences.map(({ event, index }) => {
        const records = event.messages?.length
          ? event.messages
          : event.message
            ? [{ type: event.kind, content: event.message, content_length: event.message.length, truncated: false }]
            : [];
        const eventTokens = totalTokens(event.usage);
        return <article className={`trace-card ${event.kind}`} key={index}>
          <header className="trace-card-header"><div><span>STEP {String(index + 1).padStart(2, "0")}</span><strong>{event.node}</strong></div><div><small>{event.timestamp_ms ? new Date(event.timestamp_ms).toLocaleTimeString("zh-CN", { hour12: false }) : ""}</small><b>{event.status === "completed" ? "已完成" : event.status === "failed" ? "失败" : "运行中"}</b></div></header>
          <div className="trace-metrics"><span>{event.kind.toUpperCase()}</span>{eventTokens > 0 && <span>+{formatNumber(eventTokens)} Token</span>}{Boolean(event.usage?.llm_calls) && eventTokens === 0 && <span>Token 未上报</span>}</div>
          {event.policy_id && <div className="scheduler-detail"><span>策略</span><code>{event.policy_id}</code>{event.scheduler_step != null && <span>第 {event.scheduler_step} 次决策</span>}</div>}
          {event.selected_action && <div className="scheduler-choice"><span>选择动作</span><strong>{actionLabel(event.selected_action)}</strong></div>}
          {event.valid_actions?.length ? <div className="trace-subsection"><h4>合法动作</h4><div className="action-chips">{event.valid_actions.map((action) => <span key={action}>{actionLabel(action)}</span>)}</div></div> : null}
          {event.scheduler_history?.length ? <div className="trace-subsection"><h4>历史动作</h4><div className="action-chips history">{event.scheduler_history.map((action, actionIndex) => <span key={`${action}-${actionIndex}`}>{actionIndex + 1}. {actionLabel(action)}</span>)}</div></div> : null}
          {event.produced_fields?.length ? <div className="trace-subsection"><h4>产出字段</h4><div className="field-chips">{event.produced_fields.map((field) => <span key={field}>{field}</span>)}</div></div> : null}
          {records.map((message, messageIndex) => <div className={`message-record ${event.kind}`} key={messageIndex}><div><strong>{event.kind === "scheduler" ? "调度输出" : "Agent 输出"}</strong></div><pre>{message.content}</pre>{message.truncated && <small>内容共 {formatNumber(message.content_length)} 字符，此处展示前 20,000 字符。</small>}</div>)}
        </article>;
      })}</div>
      {!latest && <p className="drawer-empty">当前节点尚未产生可展示内容。</p>}
    </aside>
  </div>;
}

function RunErrorPanel({ run, onReset }: { run: RunSnapshot; onReset: () => void }) {
  const details = run.error_details ?? {};
  return <section className="run-error" role="alert">
    <div className="run-error-heading"><strong>任务未完成</strong><span>{details.type ?? "运行错误"}</span></div>
    <dl>
      <div><dt>失败节点</dt><dd>{details.node ?? "未确定"}</dd></div>
      <div><dt>错误原因</dt><dd>{details.message ?? run.error ?? "未返回错误信息"}</dd></div>
      <div><dt>处理建议</dt><dd>{details.suggestion ?? "请重新创建任务。"}</dd></div>
    </dl>
    <button onClick={onReset}>重新创建任务</button>
  </section>;
}

function CompletionSummary({ run, onReset, onExport }: { run: RunSnapshot; onReset: () => void; onExport: () => Promise<void> }) {
  return <section className="completion-summary"><div><p className="step-label">03 / COMPLETE</p><h3>分析报告已完成</h3></div><dl><div><dt>最终信号</dt><dd>{run.signal ?? "已生成"}</dd></div><div><dt>LLM调用</dt><dd>{run.metrics.llm_calls ?? 0}</dd></div><div><dt>工具调用</dt><dd>{run.metrics.tool_calls ?? 0}</dd></div><div><dt>Token</dt><dd>{formatNumber(totalTokens(run.metrics))}</dd></div></dl><div className="completion-actions"><button className="secondary-button" onClick={() => void onExport()}><Download size={15} />导出完整报告</button><button className="secondary-button" onClick={onReset}>新建任务</button></div></section>;
}

function statusLabel(status?: RunSnapshot["status"]) {
  if (!status) return "等待任务";
  return {
    queued: "等待执行",
    running: "分析运行中",
    cancelling: "正在停止",
    cancelled: "已停止",
    completed: "报告已完成",
    failed: "执行失败",
  }[status];
}

interface SavePickerWindow extends Window {
  showSaveFilePicker?: (options: { suggestedName: string }) => Promise<{
    createWritable: () => Promise<{
      write: (content: string) => Promise<void>;
      close: () => Promise<void>;
    }>;
  }>;
}

function modeTitle(mode: OrchestrationMode) {
  return MODES.find((item) => item.value === mode)?.title ?? mode;
}
