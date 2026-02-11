import { useEffect, useState, useCallback } from "react";
import { Link } from "react-router-dom";
import {
  fetchCampaigns,
  searchCampaigns,
  startCampaign,
  stopCampaign,
  deleteCampaign,
  type CampaignSummary,
} from "../api";

function StatusBadge({ status }: { status: string }) {
  return <span className={`badge badge-${status}`}>{status}</span>;
}

function truncateId(id: string): string {
  return id.slice(0, 8);
}

export default function Dashboard() {
  const [campaigns, setCampaigns] = useState<CampaignSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");

  const loadCampaigns = useCallback(async () => {
    try {
      const data = searchQuery
        ? await searchCampaigns(searchQuery)
        : await fetchCampaigns();
      setCampaigns(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load campaigns");
    } finally {
      setLoading(false);
    }
  }, [searchQuery]);

  useEffect(() => {
    setLoading(true);
    loadCampaigns();
  }, [loadCampaigns]);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    loadCampaigns();
  };

  const handleAction = async (
    action: (id: string) => Promise<unknown>,
    id: string
  ) => {
    try {
      await action(id);
      await loadCampaigns();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Action failed");
    }
  };

  return (
    <div className="page">
      <div className="page-header">
        <h1>Campaigns</h1>
        <form className="search-form" onSubmit={handleSearch}>
          <input
            type="text"
            className="search-input"
            placeholder="Search campaigns..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
          <button type="submit" className="btn btn-secondary">
            Search
          </button>
        </form>
      </div>

      {error && <div className="error-banner">{error}</div>}

      {loading ? (
        <div className="loading">Loading campaigns...</div>
      ) : campaigns.length === 0 ? (
        <div className="empty-state">
          <p>No campaigns found.</p>
        </div>
      ) : (
        <div className="table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Name</th>
                <th>Status</th>
                <th>Iteration</th>
                <th>Best KPI</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {campaigns.map((c) => (
                <tr key={c.id}>
                  <td className="mono">
                    <Link to={`/campaigns/${c.id}`}>
                      {truncateId(c.id)}
                    </Link>
                  </td>
                  <td>
                    <Link to={`/campaigns/${c.id}`}>{c.name}</Link>
                  </td>
                  <td>
                    <StatusBadge status={c.status} />
                  </td>
                  <td>{c.iteration}</td>
                  <td className="mono">
                    {c.best_kpi !== null ? c.best_kpi.toFixed(4) : "-"}
                  </td>
                  <td className="actions">
                    {(c.status === "draft" || c.status === "paused") && (
                      <button
                        className="btn btn-sm btn-primary"
                        onClick={() => handleAction(startCampaign, c.id)}
                      >
                        Start
                      </button>
                    )}
                    {c.status === "running" && (
                      <button
                        className="btn btn-sm btn-warning"
                        onClick={() => handleAction(stopCampaign, c.id)}
                      >
                        Stop
                      </button>
                    )}
                    <button
                      className="btn btn-sm btn-danger"
                      onClick={() => handleAction(deleteCampaign, c.id)}
                    >
                      Delete
                    </button>
                    <Link
                      to={`/reports/${c.id}`}
                      className="btn btn-sm btn-secondary"
                    >
                      Report
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
