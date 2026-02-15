import { useState } from "react";
import {
  createLoop,
  iterateLoop,
  deleteLoop,
  type Deliverable,
  type RankedCandidate,
  type LoopCreateRequest,
} from "../api";

function LoopView() {
  const [loopId, setLoopId] = useState<string | null>(null);
  const [deliverable, setDeliverable] = useState<Deliverable | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [jsonInput, setJsonInput] = useState(
    JSON.stringify(
      {
        campaign_id: "demo",
        observations: [
          { parameters: { smiles: "CCO", temp: 100 }, kpi_values: { yield: 0.8 }, iteration: 0 },
          { parameters: { smiles: "CCCO", temp: 120 }, kpi_values: { yield: 0.6 }, iteration: 0 },
          { parameters: { smiles: "CCCCO", temp: 90 }, kpi_values: { yield: 0.9 }, iteration: 0 },
        ],
        candidates: [
          { smiles: "C1CCCCC1", temp: 100, name: "Cyclohexane" },
          { smiles: "c1ccccc1", temp: 110, name: "Benzene" },
          { smiles: "CC(C)O", temp: 95, name: "Isopropanol" },
          { smiles: "CCOC", temp: 115, name: "Methyl-ethyl-ether" },
          { smiles: "CC=CC", temp: 100, name: "2-Butene" },
        ],
        parameter_specs: [
          { name: "smiles", type: "categorical" },
          { name: "temp", type: "continuous", lower: 80, upper: 130 },
        ],
        objectives: ["yield"],
        objective_directions: { yield: "maximize" },
        smiles_param: "smiles",
        batch_size: 3,
        acquisition_strategy: "ucb",
      },
      null,
      2
    )
  );

  const handleCreate = async () => {
    setError(null);
    setLoading(true);
    try {
      const req: LoopCreateRequest = JSON.parse(jsonInput);
      const result = await createLoop(req);
      setLoopId(result.loop_id);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const handleIterate = async () => {
    if (!loopId) return;
    setError(null);
    setLoading(true);
    try {
      const d = await iterateLoop(loopId);
      setDeliverable(d);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async () => {
    if (!loopId) return;
    try {
      await deleteLoop(loopId);
      setLoopId(null);
      setDeliverable(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <div className="page">
      <h1>Campaign Loop</h1>
      <p className="page-subtitle">Interactive closed-loop optimization with surrogate models</p>

      {error && <div className="error-banner">{error}</div>}

      {!loopId ? (
        <div className="card">
          <h2>Create Loop</h2>
          <p>Paste your campaign configuration (observations, candidates, objectives):</p>
          <textarea
            className="json-input"
            rows={20}
            value={jsonInput}
            onChange={(e) => setJsonInput(e.target.value)}
          />
          <button className="btn btn-primary" onClick={handleCreate} disabled={loading}>
            {loading ? "Creating..." : "Create Loop"}
          </button>
        </div>
      ) : (
        <>
          <div className="card">
            <div className="loop-header">
              <span className="badge badge-running">Loop: {loopId.slice(0, 8)}...</span>
              <div className="loop-actions">
                <button className="btn btn-primary" onClick={handleIterate} disabled={loading}>
                  {loading ? "Running..." : "Run Iteration"}
                </button>
                <button className="btn btn-danger" onClick={handleDelete}>
                  Delete Loop
                </button>
              </div>
            </div>
          </div>

          {deliverable && (
            <>
              <div className="card">
                <h2>Dashboard — Iteration {deliverable.iteration}</h2>
                <div className="stats-row">
                  <div className="stat-card">
                    <div className="stat-label">Batch Size</div>
                    <div className="stat-value">{deliverable.dashboard.batch_size}</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-label">Total Ranked</div>
                    <div className="stat-value">{deliverable.dashboard.ranked_candidates.length}</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-label">Strategy</div>
                    <div className="stat-value">{deliverable.dashboard.acquisition_strategy.toUpperCase()}</div>
                  </div>
                </div>

                <h3>Recommended Batch</h3>
                <div className="table-wrapper">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Rank</th>
                        <th>Name</th>
                        <th>Predicted</th>
                        <th>Uncertainty</th>
                        <th>Acq. Score</th>
                      </tr>
                    </thead>
                    <tbody>
                      {deliverable.dashboard.batch.map((c: RankedCandidate) => (
                        <tr key={c.rank}>
                          <td>{c.rank}</td>
                          <td>{c.name}</td>
                          <td>{c.predicted_mean.toFixed(4)}</td>
                          <td>{c.predicted_std.toFixed(4)}</td>
                          <td>{c.acquisition_score.toFixed(4)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              <div className="card">
                <h2>Intelligence</h2>
                {deliverable.intelligence.model_metrics.map((m) => (
                  <div key={m.objective_name} className="stats-row">
                    <div className="stat-card">
                      <div className="stat-label">Objective</div>
                      <div className="stat-value">{m.objective_name}</div>
                    </div>
                    <div className="stat-card">
                      <div className="stat-label">Training Points</div>
                      <div className="stat-value">{m.n_training_points}</div>
                    </div>
                    <div className="stat-card">
                      <div className="stat-label">Mean</div>
                      <div className="stat-value">{m.y_mean.toFixed(3)}</div>
                    </div>
                    <div className="stat-card">
                      <div className="stat-label">Fit Time</div>
                      <div className="stat-value">{m.fit_duration_ms.toFixed(1)} ms</div>
                    </div>
                  </div>
                ))}

                {deliverable.intelligence.learning_report && (
                  <div className="learning-report">
                    <h3>Learning Report</h3>
                    <p>MAE: {deliverable.intelligence.learning_report.mean_absolute_error.toFixed(4)}</p>
                    <p>{deliverable.intelligence.learning_report.summary}</p>
                  </div>
                )}
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
}

export default LoopView;
