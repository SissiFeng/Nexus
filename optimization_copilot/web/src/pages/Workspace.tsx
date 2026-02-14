import { useState, useEffect } from "react";
import { useParams } from "react-router-dom";
import {
  LayoutDashboard,
  Search,
  Beaker,
  Clock,
  Download,
  ChevronLeft,
  ChevronRight,
  Lightbulb,
  Sparkles,
  FlaskConical,
  Info,
  FileDown,
  Zap,
  AlertTriangle,
  CheckCircle,
  ArrowRight,
} from "lucide-react";
import { useCampaign } from "../hooks/useCampaign";
import { ChatPanel } from "../components/ChatPanel";
import PhaseTimeline from "../components/PhaseTimeline";
import RealConvergencePlot from "../components/ConvergencePlot";
import RealDiagnosticCards from "../components/DiagnosticCards";
import RealParameterImportance from "../components/ParameterImportance";
import RealScatterMatrix from "../components/ScatterMatrix";
import RealSuggestionCard from "../components/SuggestionCard";
import ParetoPlot from "../components/ParetoPlot";
import InsightsPanel from "../components/InsightsPanel";
import ErrorBoundary from "../components/ErrorBoundary";
import {
  fetchDiagnostics,
  fetchImportance,
  fetchSuggestions,
  fetchExport,
  type DiagnosticsData,
  type ParameterImportanceData,
  type SuggestionData,
} from "../api";

const DIAGNOSTIC_TOOLTIPS: Record<string, string> = {
  convergence_trend: "Rate of improvement per iteration. Positive = still improving.",
  exploration_coverage: "Fraction of parameter space explored. 30-80% is ideal.",
  failure_rate: "Proportion of failed experiments. Below 20% is normal.",
  noise_estimate: "Measurement noise level. Lower means more reliable data.",
  plateau_length: "Consecutive iterations without improvement. >30 may mean converged.",
  signal_to_noise: "Signal vs noise ratio. >3 means reliable trends.",
  best_kpi_value: "Best objective value achieved so far.",
  improvement_velocity: "Recent rate of improvement. Near 0 = diminishing returns.",
};

export default function Workspace() {
  const { id } = useParams<{ id: string }>();
  const { campaign, loading, error } = useCampaign(id);
  const [activeTab, setActiveTab] = useState<
    "overview" | "explore" | "suggestions" | "insights" | "history" | "export"
  >("overview");
  const [chatOpen, setChatOpen] = useState(true);

  // API data states
  const [diagnostics, setDiagnostics] = useState<DiagnosticsData | null>(null);
  const [importance, setImportance] = useState<ParameterImportanceData | null>(null);
  const [suggestions, setSuggestions] = useState<SuggestionData | null>(null);
  const [loadingSuggestions, setLoadingSuggestions] = useState(false);
  const [loadingDiag, setLoadingDiag] = useState(false);
  const [loadingImportance, setLoadingImportance] = useState(false);
  const [batchSize, setBatchSize] = useState(5);

  // Fetch diagnostics when overview tab is active
  useEffect(() => {
    if (!id || activeTab !== "overview") return;
    setLoadingDiag(true);
    fetchDiagnostics(id)
      .then(setDiagnostics)
      .catch(() => setDiagnostics(null))
      .finally(() => setLoadingDiag(false));
  }, [id, activeTab]);

  // Fetch importance when explore tab is active
  useEffect(() => {
    if (!id || activeTab !== "explore") return;
    setLoadingImportance(true);
    fetchImportance(id)
      .then(setImportance)
      .catch(() => setImportance(null))
      .finally(() => setLoadingImportance(false));
  }, [id, activeTab]);

  if (loading) {
    return (
      <div className="page">
        <div className="loading">Loading workspace...</div>
      </div>
    );
  }

  if (error || !campaign || !id) {
    return (
      <div className="page">
        <div className="error-banner">{error || "Campaign not found"}</div>
      </div>
    );
  }

  // Transform kpi_history → ConvergencePlot data format
  const convergenceData = campaign.kpi_history.iterations.map((iter, i) => {
    const values = campaign.kpi_history.values;
    const bestSoFar = Math.min(...values.slice(0, i + 1));
    return { iteration: iter, value: values[i], best: bestSoFar };
  });

  // Transform phases → ConvergencePlot phase format
  const phaseColors: Record<string, string> = {
    COLD_START: "#94a3b8",
    LEARNING: "#3b82f6",
    EXPLOITATION: "#22c55e",
    STAGNATION: "#eab308",
  };

  const convergencePhases = campaign.phases.map((p) => ({
    name: p.name,
    start: p.start,
    end: p.end,
    color: phaseColors[p.name] || "#94a3b8",
  }));

  const handleGenerateSuggestions = async () => {
    setLoadingSuggestions(true);
    try {
      const data = await fetchSuggestions(id, batchSize);
      setSuggestions(data);
    } catch (err) {
      console.error("Failed to generate suggestions:", err);
    } finally {
      setLoadingSuggestions(false);
    }
  };

  const handleExportSuggestionsCSV = () => {
    if (!suggestions || suggestions.suggestions.length === 0) return;
    const rows = suggestions.suggestions;
    const headers = Object.keys(rows[0]);
    const csvContent = [
      headers.join(","),
      ...rows.map((row) => headers.map((h) => row[h] ?? "").join(",")),
    ].join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `suggestions-${id.slice(0, 8)}-${new Date().toISOString().slice(0, 10)}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  /** Smart advisor: analyze diagnostics and suggest next steps */
  const getAdvisorInsights = () => {
    if (!diagnostics) return [];
    const insights: Array<{
      type: "success" | "warning" | "action";
      title: string;
      description: string;
    }> = [];

    // Check plateau
    if (diagnostics.plateau_length > 30) {
      insights.push({
        type: "warning",
        title: "Optimization has plateaued",
        description: `No improvement for ${diagnostics.plateau_length} iterations. Consider expanding your search space, adding new parameters, or switching to a different optimization strategy.`,
      });
    } else if (diagnostics.plateau_length > 15) {
      insights.push({
        type: "action",
        title: "Approaching a plateau",
        description: `No improvement for ${diagnostics.plateau_length} iterations. The optimizer may be converging — consider generating a diverse batch to explore alternative regions.`,
      });
    }

    // Check exploration coverage
    if (diagnostics.exploration_coverage < 0.3) {
      insights.push({
        type: "action",
        title: "Low exploration coverage",
        description: `Only ${(diagnostics.exploration_coverage * 100).toFixed(0)}% of the parameter space has been explored. Consider running more exploration trials before exploiting.`,
      });
    } else if (diagnostics.exploration_coverage > 0.8) {
      insights.push({
        type: "success",
        title: "Good exploration coverage",
        description: `${(diagnostics.exploration_coverage * 100).toFixed(0)}% of the space explored. The optimizer has a solid understanding of the landscape.`,
      });
    }

    // Check noise
    if (diagnostics.noise_estimate > 0.2) {
      insights.push({
        type: "warning",
        title: "High measurement noise",
        description: "Consider running replicate experiments to improve model accuracy, or check for systematic errors in your measurement process.",
      });
    }

    // Check convergence trend
    if (diagnostics.convergence_trend > 0.01) {
      insights.push({
        type: "success",
        title: "Still improving",
        description: `Convergence trend is ${diagnostics.convergence_trend.toFixed(3)} — the optimization is still finding better solutions. Keep going!`,
      });
    }

    // Check signal to noise
    if (diagnostics.signal_to_noise_ratio < 3) {
      insights.push({
        type: "warning",
        title: "Low signal-to-noise ratio",
        description: "The optimization signal is weak relative to noise. Consider increasing sample sizes or reducing experimental variability.",
      });
    }

    // If all good, add encouragement
    if (insights.length === 0) {
      insights.push({
        type: "success",
        title: "Campaign looks healthy",
        description: "All diagnostics are within normal ranges. Continue generating suggestions and running experiments.",
      });
    }

    return insights;
  };

  const handleExport = async (format: "csv" | "json" | "xlsx") => {
    try {
      const blob = await fetchExport(id, format);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `campaign-${id.slice(0, 8)}.${format}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Export failed:", err);
    }
  };

  const tabs = [
    { id: "overview", label: "Overview", icon: LayoutDashboard },
    { id: "explore", label: "Explore", icon: Search },
    { id: "suggestions", label: "Suggestions", icon: Beaker },
    { id: "insights", label: "Insights", icon: Lightbulb },
    { id: "history", label: "History", icon: Clock },
    { id: "export", label: "Export", icon: Download },
  ] as const;

  // Build trial data from real observations
  const trials = (campaign.observations ?? []).map((obs) => ({
    id: `trial-${obs.iteration}`,
    iteration: obs.iteration,
    parameters: obs.parameters,
    kpis: obs.kpi_values,
    status: "completed" as const,
  }));

  // Find best result
  const bestResult = trials.length > 0
    ? trials.reduce((best, trial) => {
        const bestVal = Object.values(best.kpis)[0] ?? 0;
        const trialVal = Object.values(trial.kpis)[0] ?? 0;
        return trialVal < bestVal ? trial : best; // minimize by default
      })
    : null;

  return (
    <ErrorBoundary>
    <div className="workspace-container">
      <div className={`workspace-main ${chatOpen ? "chat-open" : ""}`}>
        {/* Header */}
        <div className="workspace-header">
          <div>
            <h1>{campaign.name}</h1>
            <div className="workspace-meta">
              <span className="mono">ID: {id.slice(0, 8)}</span>
              <span className={`badge badge-${campaign.status}`}>
                {campaign.status}
              </span>
              <span>Iteration: {campaign.iteration}</span>
              <span>Trials: {campaign.total_trials}</span>
            </div>
          </div>
        </div>

        {/* Tabs */}
        <div className="workspace-tabs">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              className={`workspace-tab ${activeTab === tab.id ? "active" : ""}`}
              onClick={() => setActiveTab(tab.id)}
            >
              <tab.icon size={16} />
              <span>{tab.label}</span>
            </button>
          ))}
        </div>

        {/* Tab Content */}
        <div className="workspace-content">
          {activeTab === "overview" && (
            <div className="tab-panel">
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
                    {campaign.best_kpi?.toFixed(4) ?? "—"}
                  </div>
                </div>
                <div className="stat-card">
                  <div className="stat-label">Phase</div>
                  <div className="stat-value">
                    {campaign.phases[campaign.phases.length - 1]?.name || "—"}
                  </div>
                </div>
              </div>

              {/* Best Result Highlight */}
              {bestResult && (
                <div className="best-result-card">
                  <div className="best-result-header">
                    <div className="best-result-badge">
                      <CheckCircle size={16} /> Best Result Found
                    </div>
                    <span className="best-result-iter">Iteration {bestResult.iteration}</span>
                  </div>
                  <div className="best-result-kpi">
                    {Object.entries(bestResult.kpis).map(([name, value]) => (
                      <div key={name} className="best-result-kpi-item">
                        <span className="best-result-kpi-label">{name}</span>
                        <span className="best-result-kpi-value mono">{typeof value === 'number' ? value.toFixed(4) : String(value)}</span>
                      </div>
                    ))}
                  </div>
                  <div className="best-result-params">
                    {Object.entries(bestResult.parameters).map(([name, value]) => (
                      <div key={name} className="best-result-param">
                        <span className="best-result-param-name">{name}</span>
                        <span className="best-result-param-value mono">{typeof value === 'number' ? value.toFixed(3) : String(value)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <div className="card">
                <h2>Convergence</h2>
                {convergenceData.length > 0 ? (
                  <RealConvergencePlot
                    data={convergenceData}
                    objectiveName="Objective"
                    direction="minimize"
                    phases={convergencePhases}
                  />
                ) : (
                  <p className="empty-state">No convergence data yet.</p>
                )}
              </div>

              <div className="card">
                <h2>Diagnostics</h2>
                {loadingDiag ? (
                  <div className="loading">Loading diagnostics...</div>
                ) : diagnostics ? (
                  <RealDiagnosticCards
                    diagnostics={{
                      convergence_trend: diagnostics.convergence_trend,
                      exploration_coverage: diagnostics.exploration_coverage,
                      failure_rate: diagnostics.failure_rate,
                      noise_estimate: diagnostics.noise_estimate,
                      plateau_length: diagnostics.plateau_length,
                      signal_to_noise: diagnostics.signal_to_noise_ratio,
                      best_kpi_value: diagnostics.best_kpi_value,
                      improvement_velocity: diagnostics.improvement_velocity,
                    }}
                    tooltips={DIAGNOSTIC_TOOLTIPS}
                  />
                ) : (
                  <p className="empty-state">
                    Diagnostics unavailable. Start a campaign to see health metrics.
                  </p>
                )}
              </div>

              <div className="card">
                <h2>Phase Timeline</h2>
                <PhaseTimeline phases={campaign.phases} />
              </div>

              {/* Smart Advisor Panel */}
              {diagnostics && (
                <div className="card advisor-panel">
                  <div className="advisor-header">
                    <Zap size={18} />
                    <h2>What to Do Next</h2>
                  </div>
                  <div className="advisor-insights">
                    {getAdvisorInsights().map((insight, i) => (
                      <div key={i} className={`advisor-insight advisor-insight-${insight.type}`}>
                        <div className="advisor-insight-icon">
                          {insight.type === "success" ? (
                            <CheckCircle size={16} />
                          ) : insight.type === "warning" ? (
                            <AlertTriangle size={16} />
                          ) : (
                            <ArrowRight size={16} />
                          )}
                        </div>
                        <div className="advisor-insight-content">
                          <div className="advisor-insight-title">{insight.title}</div>
                          <div className="advisor-insight-desc">{insight.description}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                  <div className="advisor-cta">
                    <button
                      className="btn btn-primary btn-sm"
                      onClick={() => setActiveTab("suggestions")}
                    >
                      <Sparkles size={14} /> Generate Suggestions
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === "explore" && (
            <div className="tab-panel">
              <div className="card">
                <h2>Parameter Importance</h2>
                {loadingImportance ? (
                  <div className="loading">Loading importance data...</div>
                ) : importance ? (
                  <RealParameterImportance data={importance.importances} />
                ) : (
                  <p className="empty-state">
                    Parameter importance data unavailable.
                  </p>
                )}
              </div>

              <div className="card">
                <h2>Parameter Space</h2>
                {campaign.best_parameters ? (
                  <RealScatterMatrix
                    data={campaign.kpi_history.iterations.map((_, i) => ({
                      ...campaign.best_parameters!,
                      objective: campaign.kpi_history.values[i],
                    }))}
                    parameters={Object.keys(campaign.best_parameters)}
                    objectiveName="objective"
                    objectiveDirection="minimize"
                  />
                ) : (
                  <p className="empty-state">
                    No parameter data available for scatter plot.
                  </p>
                )}
              </div>

              {/* Pareto Front — shown when 2+ objectives are configured */}
              {(() => {
                const objNames = campaign.objective_names ?? [];
                const objDirs = campaign.objective_directions ?? {};
                const obs = campaign.observations ?? [];
                if (objNames.length >= 2) {
                  return (
                    <div className="card">
                      <h2>Pareto Front</h2>
                      {obs.length > 0 ? (
                        <ParetoPlot
                          observations={obs}
                          objectiveNames={objNames}
                          objectiveDirections={objNames.map(
                            (n) => objDirs[n] ?? "minimize"
                          )}
                        />
                      ) : (
                        <p className="empty-state">
                          No observation data yet. Run some experiments to see the Pareto front.
                        </p>
                      )}
                    </div>
                  );
                }
                return null;
              })()}
            </div>
          )}

          {activeTab === "suggestions" && (
            <div className="tab-panel">
              {/* Suggestions Controls */}
              <div className="suggestions-controls">
                <div className="suggestions-controls-left">
                  <button
                    className="btn btn-primary suggestions-generate-btn"
                    onClick={handleGenerateSuggestions}
                    disabled={loadingSuggestions}
                  >
                    {loadingSuggestions ? (
                      <>
                        <span className="suggestions-spinner" />
                        Generating...
                      </>
                    ) : (
                      <>
                        <Sparkles size={16} />
                        Generate Next Experiments
                      </>
                    )}
                  </button>
                  <label className="suggestions-batch-label">
                    Batch size
                    <select
                      className="suggestions-batch-select"
                      value={batchSize}
                      onChange={(e) => setBatchSize(Number(e.target.value))}
                    >
                      <option value="3">3</option>
                      <option value="5">5</option>
                      <option value="8">8</option>
                      <option value="10">10</option>
                    </select>
                  </label>
                </div>
                {suggestions && (
                  <div className="suggestions-meta">
                    <span className="suggestions-meta-pill">
                      <FlaskConical size={12} /> {suggestions.backend_used}
                    </span>
                    <span className="suggestions-meta-pill">
                      Phase: {suggestions.phase}
                    </span>
                    <button
                      className="btn btn-secondary btn-sm suggestions-export-btn"
                      onClick={handleExportSuggestionsCSV}
                      title="Download suggestions as CSV"
                    >
                      <FileDown size={14} /> Export CSV
                    </button>
                  </div>
                )}
              </div>

              {/* Empty State */}
              {!suggestions && !loadingSuggestions && (
                <div className="suggestions-empty">
                  <div className="suggestions-empty-icon">
                    <Beaker size={32} />
                  </div>
                  <h3>Ready to suggest your next experiments</h3>
                  <p>
                    The optimization engine will analyze your {campaign.total_trials} past experiments
                    and suggest the most promising parameter configurations to try next.
                  </p>
                  <div className="suggestions-empty-info">
                    <Info size={14} />
                    <span>
                      Each suggestion includes a confidence score and an explanation of why
                      it was chosen, so you can make informed decisions.
                    </span>
                  </div>
                </div>
              )}

              {/* Loading Skeleton */}
              {loadingSuggestions && !suggestions && (
                <div className="suggestions-grid">
                  {[1, 2, 3, 4, 5].map((i) => (
                    <div key={i} className="suggestion-skeleton">
                      <div className="skeleton-line skeleton-line-short" />
                      <div className="skeleton-line skeleton-line-medium" />
                      <div className="skeleton-line skeleton-line-long" />
                      <div className="skeleton-line skeleton-line-medium" />
                    </div>
                  ))}
                </div>
              )}

              {/* Suggestion Cards */}
              {suggestions && (
                <div className="suggestions-grid">
                  {suggestions.suggestions.map((sug, i) => (
                    <RealSuggestionCard
                      key={i}
                      index={i + 1}
                      suggestion={sug}
                      parameterSpecs={
                        campaign.spec?.parameters ??
                        Object.keys(sug).map((name) => ({
                          name,
                          type: "continuous" as const,
                          lower: 0,
                          upper: 1,
                        }))
                      }
                      objectiveName={campaign.objective_names?.[0] ?? "Objective"}
                      predictedValue={suggestions.predicted_values?.[i]}
                      predictedUncertainty={suggestions.predicted_uncertainties?.[i]}
                      phase={suggestions.phase}
                    />
                  ))}
                </div>
              )}
            </div>
          )}

          {activeTab === "insights" && (
            <div className="tab-panel">
              <InsightsPanel campaignId={id} />
            </div>
          )}

          {activeTab === "history" && (
            <div className="tab-panel">
              <div className="card">
                <div className="history-header">
                  <h2>Trial History ({trials.length} experiments)</h2>
                </div>
                {trials.length > 0 ? (
                  <div className="history-table-wrapper">
                    <table className="history-table">
                      <thead>
                        <tr>
                          <th>#</th>
                          {Object.keys(trials[0].parameters).map((p) => (
                            <th key={p}>{p}</th>
                          ))}
                          {Object.keys(trials[0].kpis).map((k) => (
                            <th key={k} className="history-kpi-col">{k}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {[...trials].reverse().map((trial) => {
                          const isBest = bestResult && trial.iteration === bestResult.iteration;
                          return (
                            <tr key={trial.id} className={isBest ? "history-row-best" : ""}>
                              <td className="history-iter">{trial.iteration}</td>
                              {Object.values(trial.parameters).map((v, j) => (
                                <td key={j} className="mono">{typeof v === "number" ? v.toFixed(3) : String(v)}</td>
                              ))}
                              {Object.values(trial.kpis).map((v, j) => (
                                <td key={j} className={`mono history-kpi-val ${isBest ? "history-best-val" : ""}`}>
                                  {typeof v === "number" ? v.toFixed(4) : String(v)}
                                  {isBest && <span className="history-best-badge">Best</span>}
                                </td>
                              ))}
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <p className="empty-state">No experiments recorded yet.</p>
                )}
              </div>
            </div>
          )}

          {activeTab === "export" && (
            <div className="tab-panel">
              <div className="card">
                <h2>Export Data</h2>
                <p style={{ color: "#718096", marginBottom: "16px", fontSize: "0.9rem" }}>
                  Export your campaign data and results in various formats.
                </p>
                <div style={{ display: "flex", gap: "12px", flexWrap: "wrap" }}>
                  <button
                    className="btn btn-secondary"
                    onClick={() => handleExport("csv")}
                  >
                    <Download size={16} /> Export CSV
                  </button>
                  <button
                    className="btn btn-secondary"
                    onClick={() => handleExport("json")}
                  >
                    <Download size={16} /> Export JSON
                  </button>
                  <button
                    className="btn btn-secondary"
                    onClick={() => handleExport("xlsx")}
                  >
                    <Download size={16} /> Export Excel
                  </button>
                </div>
              </div>

              <div className="card">
                <h2>Figure Export</h2>
                <div style={{ display: "grid", gap: "16px", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))" }}>
                  <label style={{ display: "flex", flexDirection: "column", gap: "6px", fontSize: "0.9rem", fontWeight: 500 }}>
                    Resolution
                    <select className="input">
                      <option>72 DPI (Screen)</option>
                      <option>150 DPI</option>
                      <option>300 DPI (Print)</option>
                      <option>600 DPI (Publication)</option>
                    </select>
                  </label>
                  <label style={{ display: "flex", flexDirection: "column", gap: "6px", fontSize: "0.9rem", fontWeight: 500 }}>
                    Format
                    <select className="input">
                      <option>PNG</option>
                      <option>SVG</option>
                      <option>PDF</option>
                    </select>
                  </label>
                  <label style={{ display: "flex", flexDirection: "column", gap: "6px", fontSize: "0.9rem", fontWeight: 500 }}>
                    Style
                    <select className="input">
                      <option>Default</option>
                      <option>ACS (Journals)</option>
                      <option>Nature</option>
                    </select>
                  </label>
                </div>
                <button className="btn btn-primary" style={{ marginTop: "16px" }}>
                  Export All Figures
                </button>
              </div>

              <div className="card">
                <h2>Generate Report</h2>
                <p style={{ color: "#718096", marginBottom: "16px", fontSize: "0.9rem" }}>
                  Generate a comprehensive PDF report including convergence plot,
                  parameter importance, best results, and full experiment history.
                </p>
                <button className="btn btn-primary">Generate PDF Report</button>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Chat Sidebar */}
      <div className={`workspace-chat ${chatOpen ? "open" : "closed"}`}>
        <button
          className="chat-toggle-btn"
          onClick={() => setChatOpen(!chatOpen)}
          aria-label={chatOpen ? "Close chat" : "Open chat"}
        >
          {chatOpen ? <ChevronRight size={20} /> : <ChevronLeft size={20} />}
        </button>
        {chatOpen && (
          <ChatPanel
            campaignId={id}
            isOpen={chatOpen}
            onToggle={() => setChatOpen(!chatOpen)}
          />
        )}
      </div>

      <style>{`
        .workspace-container {
          display: flex;
          height: calc(100vh - 56px);
          position: relative;
        }

        .workspace-main {
          flex: 1;
          display: flex;
          flex-direction: column;
          overflow: hidden;
          transition: width 0.3s ease;
        }

        .workspace-main.chat-open {
          width: 70%;
        }

        .workspace-header {
          padding: 24px 24px 16px;
          border-bottom: 1px solid var(--color-border);
          background: var(--color-surface);
        }

        .workspace-header h1 {
          font-size: 1.5rem;
          font-weight: 700;
          margin-bottom: 8px;
        }

        .workspace-meta {
          display: flex;
          gap: 12px;
          align-items: center;
          font-size: 0.85rem;
        }

        .workspace-tabs {
          display: flex;
          gap: 4px;
          padding: 0 24px;
          background: var(--color-surface);
          border-bottom: 2px solid var(--color-border);
          overflow-x: auto;
        }

        .workspace-tab {
          display: flex;
          align-items: center;
          gap: 6px;
          padding: 12px 16px;
          background: none;
          border: none;
          border-bottom: 2px solid transparent;
          margin-bottom: -2px;
          font-size: 0.9rem;
          font-weight: 500;
          color: var(--color-text-muted);
          cursor: pointer;
          transition: all 0.15s;
          white-space: nowrap;
        }

        .workspace-tab:hover {
          color: var(--color-text);
          background: var(--color-bg);
        }

        .workspace-tab.active {
          color: var(--color-primary);
          border-bottom-color: var(--color-primary);
        }

        .workspace-content {
          flex: 1;
          overflow-y: auto;
          padding: 24px;
          background: var(--color-bg);
        }

        .tab-panel {
          animation: fadeIn 0.2s ease;
        }

        .workspace-chat {
          width: 30%;
          border-left: 1px solid var(--color-border);
          background: var(--color-surface);
          position: relative;
          transition: width 0.3s ease;
          display: flex;
          flex-direction: column;
        }

        .workspace-chat.closed {
          width: 0;
          overflow: hidden;
        }

        .chat-toggle-btn {
          position: absolute;
          left: -12px;
          top: 50%;
          transform: translateY(-50%);
          width: 24px;
          height: 48px;
          background: var(--color-surface);
          border: 1px solid var(--color-border);
          border-radius: var(--radius) 0 0 var(--radius);
          display: flex;
          align-items: center;
          justify-content: center;
          cursor: pointer;
          transition: all 0.15s;
          z-index: 10;
        }

        .chat-toggle-btn:hover {
          background: var(--color-bg);
        }

        .suggestions-controls {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 24px;
          flex-wrap: wrap;
          gap: 12px;
        }
        .suggestions-controls-left {
          display: flex;
          align-items: center;
          gap: 16px;
        }
        .suggestions-generate-btn {
          display: flex;
          align-items: center;
          gap: 8px;
          font-weight: 600;
        }
        .suggestions-spinner {
          width: 16px;
          height: 16px;
          border: 2px solid rgba(255,255,255,0.3);
          border-top-color: #fff;
          border-radius: 50%;
          animation: spin 0.8s linear infinite;
        }
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
        .suggestions-batch-label {
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 0.82rem;
          color: var(--color-text-muted);
          font-weight: 500;
        }
        .suggestions-batch-select {
          padding: 4px 8px;
          border: 1px solid var(--color-border);
          border-radius: 6px;
          background: var(--color-surface);
          font-size: 0.82rem;
          font-family: inherit;
          color: var(--color-text);
        }
        .suggestions-meta {
          display: flex;
          gap: 8px;
        }
        .suggestions-meta-pill {
          display: inline-flex;
          align-items: center;
          gap: 4px;
          padding: 4px 10px;
          background: var(--color-bg);
          border: 1px solid var(--color-border);
          border-radius: 20px;
          font-size: 0.75rem;
          color: var(--color-text-muted);
          font-weight: 500;
        }
        .suggestions-empty {
          text-align: center;
          padding: 60px 24px;
          max-width: 480px;
          margin: 0 auto;
        }
        .suggestions-empty-icon {
          width: 72px;
          height: 72px;
          border-radius: 50%;
          background: var(--color-primary-subtle, rgba(79, 110, 247, 0.08));
          display: flex;
          align-items: center;
          justify-content: center;
          margin: 0 auto 20px;
          color: var(--color-primary);
        }
        .suggestions-empty h3 {
          font-size: 1.1rem;
          font-weight: 600;
          margin: 0 0 8px;
          color: var(--color-text);
        }
        .suggestions-empty p {
          font-size: 0.88rem;
          color: var(--color-text-muted);
          line-height: 1.6;
          margin: 0 0 16px;
        }
        .suggestions-empty-info {
          display: flex;
          align-items: flex-start;
          gap: 8px;
          padding: 12px 16px;
          background: var(--color-bg);
          border-radius: 8px;
          font-size: 0.82rem;
          color: var(--color-text-muted);
          line-height: 1.5;
          text-align: left;
        }
        .suggestions-empty-info svg {
          flex-shrink: 0;
          margin-top: 2px;
        }
        .suggestion-skeleton {
          background: var(--color-surface);
          border: 1px solid var(--color-border);
          border-radius: 12px;
          padding: 20px;
        }
        .skeleton-line {
          height: 12px;
          background: var(--color-bg);
          border-radius: 6px;
          margin-bottom: 12px;
          animation: pulse 1.5s ease-in-out infinite;
        }
        .skeleton-line-short { width: 40%; }
        .skeleton-line-medium { width: 70%; }
        .skeleton-line-long { width: 90%; }
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.4; }
        }

        .suggestions-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
          gap: 16px;
          margin-bottom: 24px;
        }

        @media (max-width: 1024px) {
          .workspace-main.chat-open {
            width: 100%;
          }

          .workspace-chat {
            position: fixed;
            right: 0;
            top: 56px;
            bottom: 0;
            width: 400px;
            max-width: 90vw;
            z-index: 100;
            box-shadow: -4px 0 12px rgba(0, 0, 0, 0.1);
          }

          .workspace-chat.closed {
            transform: translateX(100%);
          }
        }

        @media (max-width: 768px) {
          .workspace-header {
            padding: 16px;
          }

          .workspace-content {
            padding: 16px;
          }

          .workspace-tabs {
            padding: 0 16px;
          }

          .workspace-tab {
            padding: 10px 12px;
            font-size: 0.85rem;
          }

          .workspace-tab span {
            display: none;
          }

          .suggestions-grid {
            grid-template-columns: 1fr;
          }

          .workspace-chat {
            width: 100%;
          }
        }
      `}</style>
    </div>
    </ErrorBoundary>
  );
}
