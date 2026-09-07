import type { RunRequest, RunSnapshot, WebSettings, WebSettingsUpdate } from "./types";

async function responseJson<T>(response: Response): Promise<T> {
  const value = await response.json();
  if (!response.ok) {
    throw new Error(value.detail ?? "请求失败，请稍后重试。");
  }
  return value as T;
}

export async function createRun(payload: RunRequest): Promise<RunSnapshot> {
  const response = await fetch("/api/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return responseJson<RunSnapshot>(response);
}

export async function getRun(runId: string): Promise<RunSnapshot> {
  const response = await fetch(`/api/runs/${runId}`);
  return responseJson<RunSnapshot>(response);
}

export async function getActiveRun(): Promise<RunSnapshot | null> {
  const response = await fetch("/api/runs/active");
  return responseJson<RunSnapshot | null>(response);
}

export async function cancelRun(runId: string): Promise<RunSnapshot> {
  const response = await fetch(`/api/runs/${runId}/cancel`, { method: "POST" });
  return responseJson<RunSnapshot>(response);
}

export async function getSettings(): Promise<WebSettings> {
  const response = await fetch("/api/settings");
  return responseJson<WebSettings>(response);
}

export async function updateSettings(payload: WebSettingsUpdate): Promise<WebSettings> {
  const response = await fetch("/api/settings", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return responseJson<WebSettings>(response);
}
