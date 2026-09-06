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
  metrics: Record<string, number>;
  models: Record<string, string>;
  event_count: number;
}

export interface NodeEvent {
  node: string;
  kind: "scheduler" | "tool" | "agent";
  status: "running" | "completed";
  selected_action?: string;
  valid_actions?: string[];
  produced_fields?: string[];
  message?: string | null;
  tool_calls?: Array<{ name: string; args: unknown }>;
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
