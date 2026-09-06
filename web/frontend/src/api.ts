import type { RunRequest, RunSnapshot } from "./types";

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
