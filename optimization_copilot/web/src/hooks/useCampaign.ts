import { useEffect, useState, useCallback } from "react";
import { fetchCampaign, type Campaign } from "../api";

interface UseCampaignResult {
  campaign: Campaign | null;
  loading: boolean;
  error: string | null;
  refresh: () => void;
  lastUpdated: number | null;
}

export function useCampaign(
  id: string | undefined,
  autoRefreshMs: number = 5000
): UseCampaignResult {
  const [campaign, setCampaign] = useState<Campaign | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<number | null>(null);

  const load = useCallback(async () => {
    if (!id) return;
    try {
      const data = await fetchCampaign(id);
      setCampaign(data);
      setError(null);
      setLastUpdated(Date.now());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch campaign");
    } finally {
      setLoading(false);
    }
  }, [id]);

  useEffect(() => {
    setLoading(true);
    load();

    const interval = setInterval(load, autoRefreshMs);
    return () => clearInterval(interval);
  }, [load, autoRefreshMs]);

  return { campaign, loading, error, refresh: load, lastUpdated };
}
