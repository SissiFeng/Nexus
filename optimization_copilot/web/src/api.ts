const BASE_URL = "/api";

class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${BASE_URL}${path}`;
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(options.headers as Record<string, string>),
  };

  const response = await fetch(url, { ...options, headers });

  if (!response.ok) {
    const body = await response.text();
    throw new ApiError(
      body || `Request failed: ${response.statusText}`,
      response.status
    );
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return response.json();
}

/* ── Campaign CRUD ── */

export interface Campaign {
  id: string;
  name: string;
  status: string;
  iteration: number;
  total_trials: number;
  best_kpi: number | null;
  best_parameters: Record<string, number> | null;
  created_at: string;
  tags: string[];
  phases: { name: string; start: number; end: number }[];
  kpi_history: { iterations: number[]; values: number[] };
}

export interface CampaignSummary {
  id: string;
  name: string;
  status: string;
  iteration: number;
  best_kpi: number | null;
  tags: string[];
}

export interface Trial {
  id: string;
  parameters: Record<string, number>;
  kpis: Record<string, number>;
  status: string;
}

export interface Batch {
  campaign_id: string;
  iteration: number;
  trials: Trial[];
}

export interface AuditEntry {
  hash: string;
  prev_hash: string;
  event: string;
  timestamp: number;
  details: string;
}

export interface CompareResult {
  campaigns: CampaignSummary[];
  winner_id: string | null;
}

export interface StoreSummary {
  campaign_id: string;
  total_trials: number;
  best_kpi: number | null;
  parameter_ranges: Record<string, { min: number; max: number }>;
}

/* ── API Functions ── */

export function fetchCampaigns(): Promise<CampaignSummary[]> {
  return request<CampaignSummary[]>("/campaigns");
}

export function fetchCampaign(id: string): Promise<Campaign> {
  return request<Campaign>(`/campaigns/${id}`);
}

export function createCampaign(
  spec: Record<string, unknown>,
  name: string,
  tags: string[]
): Promise<Campaign> {
  return request<Campaign>("/campaigns", {
    method: "POST",
    body: JSON.stringify({ spec, name, tags }),
  });
}

export function startCampaign(id: string): Promise<Campaign> {
  return request<Campaign>(`/campaigns/${id}/start`, { method: "POST" });
}

export function stopCampaign(id: string): Promise<Campaign> {
  return request<Campaign>(`/campaigns/${id}/stop`, { method: "POST" });
}

export function pauseCampaign(id: string): Promise<Campaign> {
  return request<Campaign>(`/campaigns/${id}/pause`, { method: "POST" });
}

export function resumeCampaign(id: string): Promise<Campaign> {
  return request<Campaign>(`/campaigns/${id}/resume`, { method: "POST" });
}

export function deleteCampaign(id: string): Promise<void> {
  return request<void>(`/campaigns/${id}`, { method: "DELETE" });
}

export function fetchBatch(id: string): Promise<Batch> {
  return request<Batch>(`/campaigns/${id}/batch`);
}

export function submitTrials(
  id: string,
  results: Trial[]
): Promise<{ accepted: number }> {
  return request<{ accepted: number }>(`/campaigns/${id}/trials`, {
    method: "POST",
    body: JSON.stringify({ results }),
  });
}

export function fetchResult(id: string): Promise<Campaign> {
  return request<Campaign>(`/campaigns/${id}/result`);
}

export function fetchStoreSummary(id: string): Promise<StoreSummary> {
  return request<StoreSummary>(`/store/${id}/summary`);
}

export function fetchAuditLog(id: string): Promise<AuditEntry[]> {
  return request<AuditEntry[]>(`/reports/${id}/audit`);
}

export function compareCampaigns(ids: string[]): Promise<CompareResult> {
  return request<CompareResult>("/reports/compare", {
    method: "POST",
    body: JSON.stringify({ campaign_ids: ids }),
  });
}

export function searchCampaigns(q: string): Promise<CampaignSummary[]> {
  return request<CampaignSummary[]>(`/search?q=${encodeURIComponent(q)}`);
}
