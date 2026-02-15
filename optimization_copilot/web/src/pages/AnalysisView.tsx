import { useState } from "react";
import { runTopK, runCorrelation, runFanova, type TracedResponse } from "../api";

type AnalysisType = "top-k" | "correlation" | "fanova";

function AnalysisView() {
  const [analysisType, setAnalysisType] = useState<AnalysisType>("top-k");
  const [result, setResult] = useState<TracedResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Top-K state
  const [topKValues, setTopKValues] = useState("0.8, 0.3, 0.95, 0.6, 0.72");
  const [topKNames, setTopKNames] = useState("A, B, C, D, E");
  const [topKK, setTopKK] = useState(3);

  // Correlation state
  const [corrXs, setCorrXs] = useState("1, 2, 3, 4, 5, 6, 7, 8");
  const [corrYs, setCorrYs] = useState("2, 4, 5, 8, 10, 11, 14, 16");

  const handleRun = async () => {
    setError(null);
    setLoading(true);
    setResult(null);
    try {
      let r: TracedResponse;
      if (analysisType === "top-k") {
        const values = topKValues.split(",").map((s) => parseFloat(s.trim()));
        const names = topKNames.split(",").map((s) => s.trim());
        r = await runTopK(values, names, topKK);
      } else if (analysisType === "correlation") {
        const xs = corrXs.split(",").map((s) => parseFloat(s.trim()));
        const ys = corrYs.split(",").map((s) => parseFloat(s.trim()));
        r = await runCorrelation(xs, ys);
      } else {
        // fANOVA uses hardcoded demo data
        const X = Array.from({ length: 20 }, () =>
          Array.from({ length: 3 }, () => Math.random())
        );
        const y = X.map((row) => row[0] * 2 + row[1] * 3 + row[2]);
        r = await runFanova(X, y, ["x1", "x2", "x3"]);
      }
      setResult(r);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <h1>Analysis Pipeline</h1>
      <p className="page-subtitle">Traced computational analysis with execution provenance</p>

      {error && <div className="error-banner">{error}</div>}

      <div className="card">
        <h2>Select Analysis</h2>
        <div className="analysis-tabs">
          {(["top-k", "correlation", "fanova"] as AnalysisType[]).map((t) => (
            <button
              key={t}
              className={`btn ${analysisType === t ? "btn-primary" : "btn-secondary"}`}
              onClick={() => { setAnalysisType(t); setResult(null); }}
            >
              {t === "top-k" ? "Top-K" : t === "correlation" ? "Correlation" : "fANOVA"}
            </button>
          ))}
        </div>

        {analysisType === "top-k" && (
          <div className="analysis-form">
            <label>Values (comma-separated):</label>
            <input type="text" value={topKValues} onChange={(e) => setTopKValues(e.target.value)} />
            <label>Names (comma-separated):</label>
            <input type="text" value={topKNames} onChange={(e) => setTopKNames(e.target.value)} />
            <label>K:</label>
            <input type="number" value={topKK} onChange={(e) => setTopKK(parseInt(e.target.value))} />
          </div>
        )}

        {analysisType === "correlation" && (
          <div className="analysis-form">
            <label>X values (comma-separated):</label>
            <input type="text" value={corrXs} onChange={(e) => setCorrXs(e.target.value)} />
            <label>Y values (comma-separated):</label>
            <input type="text" value={corrYs} onChange={(e) => setCorrYs(e.target.value)} />
          </div>
        )}

        {analysisType === "fanova" && (
          <div className="analysis-form">
            <p>Runs fANOVA on 20 random samples with 3 features (demo data)</p>
          </div>
        )}

        <button className="btn btn-primary" onClick={handleRun} disabled={loading}>
          {loading ? "Running..." : "Run Analysis"}
        </button>
      </div>

      {result && (
        <div className="card">
          <h2>Result</h2>
          <div className="stats-row">
            <div className="stat-card">
              <div className="stat-label">Status</div>
              <div className="stat-value">
                <span className={`badge badge-${result.tag === "computed" ? "completed" : "error"}`}>
                  {result.tag.toUpperCase()}
                </span>
              </div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Traces</div>
              <div className="stat-value">{result.traces.length}</div>
            </div>
            {result.traces[0] && (
              <div className="stat-card">
                <div className="stat-label">Duration</div>
                <div className="stat-value">{result.traces[0].duration_ms.toFixed(1)} ms</div>
              </div>
            )}
          </div>

          <h3>Value</h3>
          <pre className="json-output">{JSON.stringify(result.value, null, 2)}</pre>

          {result.traces.length > 0 && (
            <>
              <h3>Execution Traces</h3>
              <div className="table-wrapper">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Module</th>
                      <th>Method</th>
                      <th>Duration (ms)</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.traces.map((t, i) => (
                      <tr key={i}>
                        <td><code>{t.module}</code></td>
                        <td><code>{t.method}</code></td>
                        <td>{t.duration_ms.toFixed(2)}</td>
                        <td>
                          <span className={`badge badge-${t.tag === "computed" ? "completed" : "error"}`}>
                            {t.tag}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default AnalysisView;
