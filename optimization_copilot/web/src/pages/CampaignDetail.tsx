import { useParams } from "react-router-dom";
import { useCampaign } from "../hooks/useCampaign";
import { useWebSocket } from "../hooks/useWebSocket";
import {
  startCampaign,
  stopCampaign,
  pauseCampaign,
  resumeCampaign,
} from "../api";
import KpiChart from "../components/KpiChart";
import PhaseTimeline from "../components/PhaseTimeline";
import TrialTable from "../components/TrialTable";
import { useState, useEffect } from "react";
import { fetchBatch, type Trial } from "../api";

function StatusBadge({ status }: { status: string }) {
  return <span className={`badge badge-${status}`}>{status}</span>;
}

function WsBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    connected: "var(--color-green)",
    connecting: "var(--color-yellow)",
    disconnected: "var(--color-gray)",
    error: "var(--color-red)",
  };
  return (
    <span
      className="ws-badge"
      style={{ backgroundColor: colors[status] || colors.disconnected }}
    >
      WS: {status}
    </span>
  );
}

export default function CampaignDetail() {
  const { id } = useParams<{ id: string }>();
  const { campaign, loading, error, refresh } = useCampaign(id);
  const { messages, status: wsStatus } = useWebSocket(id);
  const [trials, setTrials] = useState<Trial[]>([]);
  const [actionError, setActionError] = useState<string | null>(null);

  useEffect(() => {
    if (!id) return;
    fetchBatch(id)
      .then((batch) => setTrials(batch.trials))
      .catch(() => {
        /* batch may not exist yet */
      });
  }, [id, campaign?.iteration]);

  // Refresh campaign on websocket messages
  useEffect(() => {
    if (messages.length > 0) {
      refresh();
    }
  }, [messages.length, refresh]);

  const handleAction = async (action: (id: string) => Promise<unknown>) => {
    if (!id) return;
    setActionError(null);
    try {
      await action(id);
      refresh();
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Action failed");
    }
  };

  if (loading) return <div className="loading">Loading campaign...</div>;
  if (error) return <div className="error-banner">{error}</div>;
  if (!campaign) return <div className="error-banner">Campaign not found</div>;

  const status = campaign.status;

  return (
    <div className="page">
      <div className="campaign-header">
        <div>
          <h1>{campaign.name}</h1>
          <p className="campaign-id mono">{campaign.campaign_id}</p>
        </div>
        <div className="campaign-header-right">
          <StatusBadge status={status} />
          <WsBadge status={wsStatus} />
        </div>
      </div>

      {actionError && <div className="error-banner">{actionError}</div>}

      <div className="action-bar">
        {(status === "draft" || status === "stopped") && (
          <button
            className="btn btn-primary"
            onClick={() => handleAction(startCampaign)}
          >
            Start
          </button>
        )}
        {status === "running" && (
          <>
            <button
              className="btn btn-warning"
              onClick={() => handleAction(pauseCampaign)}
            >
              Pause
            </button>
            <button
              className="btn btn-danger"
              onClick={() => handleAction(stopCampaign)}
            >
              Stop
            </button>
          </>
        )}
        {status === "paused" && (
          <>
            <button
              className="btn btn-primary"
              onClick={() => handleAction(resumeCampaign)}
            >
              Resume
            </button>
            <button
              className="btn btn-danger"
              onClick={() => handleAction(stopCampaign)}
            >
              Stop
            </button>
          </>
        )}
      </div>

      <div className="stats-row">
        <div className="stat-card">
          <div className="stat-label">Iteration</div>
          <div className="stat-value">{campaign.iteration}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Total Trials</div>
          <div className="stat-value">{campaign.total_trials}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Best KPI</div>
          <div className="stat-value mono">
            {campaign.best_kpi !== null ? campaign.best_kpi.toFixed(6) : "-"}
          </div>
        </div>
      </div>

      {campaign.kpi_history.iterations.length > 0 && (
        <div className="card">
          <h2>KPI Convergence</h2>
          <KpiChart data={campaign.kpi_history} />
        </div>
      )}

      {campaign.phases.length > 0 && (
        <div className="card">
          <h2>Phase Timeline</h2>
          <PhaseTimeline phases={campaign.phases} />
        </div>
      )}

      {trials.length > 0 && (
        <div className="card">
          <h2>Recent Trials</h2>
          <TrialTable trials={trials} />
        </div>
      )}

      {messages.length > 0 && (
        <div className="card">
          <h2>Live Updates</h2>
          <div className="ws-messages">
            {messages.slice(-10).map((msg, i) => (
              <div key={i} className="ws-message">
                <span className="ws-type">{msg.type}</span>
                <span className="ws-time">
                  {new Date(msg.timestamp * 1000).toLocaleTimeString()}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
