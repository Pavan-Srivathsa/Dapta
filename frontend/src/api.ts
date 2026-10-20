import * as Sentry from "@sentry/react";
import type { Adapter, Benchmark, Directory, Job, RouteDecision, Run, RunpodConnection } from "./types";
import { base44 } from "./base44Client";
import { toProxyTrace } from "./tracing";

const API_URL = import.meta.env.VITE_DAPTA_API_URL;

type ProxyFile = { name: string; type: string; base64: string };

// One client span per controller call; its trace continues in dapta-proxy, the controller and the Qwen worker.
function traced<T>(method: string, path: string, run: () => Promise<T>): Promise<T> {
  const route = path.split("?")[0].replace(/\/[0-9a-f-]{8,}(?=\/|$)/gi, "/{id}");
  return Sentry.startSpan({ name: `${method} ${route}`, op: "dapta.api" }, run);
}

async function invokeProxy<T>(payload: Record<string, unknown>): Promise<T> {
  const method = String(payload.method || "GET");
  return traced(method, String(payload.path), async () => {
    const trace = toProxyTrace(Sentry.getTraceData());
    try {
      const response = await base44.functions.invoke("dapta-proxy", trace ? { ...payload, trace } : payload);
      return response.data as T;
    } catch (reason) {
      const error = reason as { response?: { status?: number; data?: { detail?: string } }; message?: string };
      if (error.response?.status === 401) throw new SignInRequiredError();
      throw new Error(error.response?.data?.detail || error.message || "Dapta request failed");
    }
  });
}

/** The Base44 site is public; the controller is not. Raised for a visitor who is not signed in. */
export class SignInRequiredError extends Error {
  constructor() {
    super("Sign in to use Dapta");
    this.name = "SignInRequiredError";
  }
}

// Base44's /api/apps/auth/login route, not redirectToLogin(): with no app base URL that sends the
// visitor to /login on this site, which serves this app again and reloaded it in a loop.
export function signIn() {
  base44.auth.loginWithProvider("google", window.location.href);
}

async function encodeFile(file: File): Promise<ProxyFile> {
  const bytes = new Uint8Array(await file.arrayBuffer());
  let binary = "";
  for (let index = 0; index < bytes.length; index += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(index, index + 0x8000));
  }
  return { name: file.name, type: file.type, base64: btoa(binary) };
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  if (API_URL) {
    // Same-origin in local development: Sentry's fetch instrumentation adds the trace headers.
    return traced(init?.method || "GET", path, async () => {
      const response = await fetch(`${API_URL}${path}`, {
        ...init,
        headers: {
          ...(init?.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
          ...(init?.headers || {}),
        },
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(payload.detail || `Request failed: ${response.status}`);
      }
      return response.json();
    });
  }

  const method = init?.method || "GET";
  const headers = new Headers(init?.headers);
  const body = typeof init?.body === "string" ? JSON.parse(init.body) : undefined;
  return invokeProxy<T>({
    path,
    method,
    body,
    idempotencyKey: headers.get("Idempotency-Key"),
  });
}

export const api = {
  adapters: () => request<Adapter[]>("/v1/adapters"),
  directory: () => request<Directory>("/v1/directory"),
  runpodConnection: () => request<RunpodConnection>("/v1/users/me/runpod-connection"),
  connectRunpod: (apiKey: string) =>
    request<RunpodConnection>("/v1/users/me/runpod-connection", {
      method: "POST",
      body: JSON.stringify({ api_key: apiKey }),
    }),
  disconnectRunpod: () =>
    request<RunpodConnection>("/v1/users/me/runpod-connection/disconnect", { method: "POST", body: "{}" }),
  benchmarks: () => request<Benchmark[]>("/v1/benchmarks"),
  runs: () => request<Run[]>("/v1/runs"),
  jobs: () => request<Job[]>("/v1/jobs"),
  run: (id: string) => request<Run>(`/v1/runs/${id}`),
  extract: async (file: File) => {
    if (!API_URL) {
      return invokeProxy<{ filename: string; text: string; sha256: string }>({
        path: "/v1/documents/extract",
        method: "POST",
        file: await encodeFile(file),
      });
    }
    const form = new FormData();
    form.append("file", file);
    return request<{ filename: string; text: string; sha256: string }>("/v1/documents/extract", {
      method: "POST",
      body: form,
    });
  },
  route: (workflow: string) =>
    request<RouteDecision>(
      "/v1/route",
      { method: "POST", body: JSON.stringify({ workflow }) },
    ),
  compare: (workflow: string, documentText: string, adapterId: string, attachmentName?: string) =>
    request<Run>("/v1/comparisons", {
      method: "POST",
      headers: { "Idempotency-Key": crypto.randomUUID() },
      body: JSON.stringify({
        workflow,
        document_text: documentText,
        adapter_id: adapterId,
        attachment_name: attachmentName,
      }),
    }),
  createJob: (workflow: string, route: RouteDecision) =>
    request<Job>("/v1/jobs", {
      method: "POST",
      headers: { "Idempotency-Key": crypto.randomUUID() },
      body: JSON.stringify({ workflow, route }),
    }),
  retryJob: (jobId: string) =>
    request<Job>(`/v1/jobs/${jobId}/retry`, { method: "POST", body: "{}" }),
  cancelJob: (jobId: string) =>
    request<Job>(`/v1/jobs/${jobId}/cancel`, { method: "POST", body: "{}" }),
  repairEvaluation: (jobId: string) =>
    request<Job>(`/v1/jobs/${jobId}/repair-evaluation`, { method: "POST", body: "{}" }),
};
