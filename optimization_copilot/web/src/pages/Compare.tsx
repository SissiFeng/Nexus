import { useState } from "react";
import { compareCampaigns, type CompareResult } from "../api";

export default function Compare() {
  const [ids, setIds] = useState<string[]>(["", ""]);
  const [result, setResult] = useState<CompareResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const updateId = (index: number, value: string) => {
    setIds((prev) => {
      const next = [...prev];
      next[index] = value;
      return next;
    });
  };

  const addField = () => {
    if (ids.length < 4) {
      setIds((prev) => [...prev, ""]);
    }
  };

  const removeField = (index: number) => {
    if (ids.length > 2) {
      setIds((prev) => prev.filter((_, i) => i !== index));
    }
  };

  const handleCompare = async (e: React.FormEvent) => {
    e.preventDefault();
    const validIds = ids.map((id) => id.trim()).filter(Boolean);
    if (validIds.length < 2) {
      setError("Enter at least 2 campaign IDs to compare.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await compareCampaigns(validIds);
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Comparison failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <h1>Compare Campaigns</h1>

      <form className="compare-form" onSubmit={handleCompare}>
        {ids.map((id, i) => (
          <div key={i} className="compare-input-row">
            <label>Campaign {i + 1}</label>
            <input
              type="text"
              className="input"
              placeholder="Enter campaign ID..."
              value={id}
              onChange={(e) => updateId(i, e.target.value)}
            />
            {ids.length > 2 && (
              <button
                type="button"
                className="btn btn-sm btn-danger"
                onClick={() => removeField(i)}
              >
                Remove
              </button>
            )}
          </div>
        ))}
        <div className="compare-actions">
          {ids.length < 4 && (
            <button
              type="button"
              className="btn btn-secondary"
              onClick={addField}
            >
              Add Campaign
            </button>
          )}
          <button
            type="submit"
            className="btn btn-primary"
            disabled={loading}
          >
            {loading ? "Comparing..." : "Compare"}
          </button>
        </div>
      </form>

      {error && <div className="error-banner">{error}</div>}

      {result && (
        <div className="card">
          <h2>Results</h2>
          <div className="table-wrapper">
            <table className="data-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Name</th>
                  <th>Status</th>
                  <th>Iteration</th>
                  <th>Best KPI</th>
                  <th>Winner</th>
                </tr>
              </thead>
              <tbody>
                {result.campaigns.map((c) => (
                  <tr
                    key={c.campaign_id}
                    className={
                      c.campaign_id === result.winner_id ? "row-winner" : ""
                    }
                  >
                    <td className="mono">{c.campaign_id.slice(0, 8)}</td>
                    <td>{c.name}</td>
                    <td>
                      <span className={`badge badge-${c.status}`}>
                        {c.status}
                      </span>
                    </td>
                    <td>{c.iteration}</td>
                    <td className="mono">
                      {c.best_kpi !== null ? c.best_kpi.toFixed(4) : "-"}
                    </td>
                    <td>
                      {c.campaign_id === result.winner_id ? (
                        <span className="winner-tag">Winner</span>
                      ) : (
                        "-"
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
