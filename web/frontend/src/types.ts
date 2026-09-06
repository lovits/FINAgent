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
  status: "queued" | "running" | "completed" | "failed";
  request: RunRequest;
  report_sections: Record<string, string>;
  complete_report: string | null;
  signal: string | null;
  error: string | null;
  metrics: Record<string, number>;
  event_count: number;
}

export interface NodeEvent {
  node: string;
  kind: "scheduler" | "tool" | "agent";
  selected_action?: string;
  valid_actions?: string[];
}

export interface ReportEvent {
  section: string;
  title: string;
  content: string;
}
