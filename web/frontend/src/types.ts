export type OrchestrationMode = "static" | "teacher" | "learned";
export type ResearchDepth = "shallow" | "medium" | "deep";
export type Analyst = "market" | "social" | "news" | "fundamentals";

export interface RunRequest {
  ticker: string;
  analysis_date: string;
  output_language: string;
  analysts: Analyst[];
  research_depth: ResearchDepth;
  orchestration_mode: OrchestrationMode;
}

export interface RunSnapshot {
  run_id: string;
  status: "queued" | "running" | "cancelling" | "cancelled" | "completed" | "failed";
  request: RunRequest;
  report_sections: Record<string, string>;
  complete_report: string | null;
  signal: string | null;
  error: string | null;
  error_details: {
    node?: string;
    type?: string;
    message?: string;
    suggestion?: string;
  };
  metrics: Record<string, number>;
  models: Record<string, string>;
  resolved_ticker: string | null;
  data_sources: Record<string, string>;
  event_count: number;
}

export interface UsageMetrics {
  llm_calls: number;
  tool_calls: number;
  input_tokens: number;
  output_tokens: number;
}

export interface TraceMessage {
  type: string;
  content: string;
  content_length: number;
  truncated: boolean;
  name?: string;
  tool_call_id?: string;
  source?: string;
  summarized?: boolean;
}

export interface NodeEvent {
  node: string;
  kind: "scheduler" | "tool" | "agent";
  status: "running" | "completed" | "failed";
  selected_action?: string;
  valid_actions?: string[];
  produced_fields?: string[];
  message?: string | null;
  messages?: TraceMessage[];
  tool_calls?: Array<{ id?: string; name: string; argument_keys?: string[] }>;
  timestamp_ms?: number;
  usage?: UsageMetrics;
  cumulative_metrics?: UsageMetrics;
  scheduler_step?: number;
  scheduler_history?: string[];
  policy_id?: string;
}

export interface ReportEvent {
  section: string;
  title: string;
  content: string;
}

export interface WebSettings {
  provider: "openrouter";
  key_configured: boolean;
  masked_key: string | null;
  expert_model: string;
  teacher_model: string;
  scheduler_base_model: string;
  scheduler_adapter: string | null;
}

export interface WebSettingsUpdate {
  openrouter_api_key?: string;
  expert_model: string;
  teacher_model: string;
}
