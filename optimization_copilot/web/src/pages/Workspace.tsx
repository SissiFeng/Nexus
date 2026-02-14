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
} from "lucide-react";
import { useCampaign } from "../hooks/useCampaign";
import { ChatPanel } from "../components/ChatPanel";
import PhaseTimeline from "../components/PhaseTimeline";
import TrialTable from "../components/TrialTable";
import RealConvergencePlot from "../components/ConvergencePlot";
import RealDiagnosticCards from "../components/DiagnosticCards";
import RealParameterImportance from "../components/ParameterImportance";
import RealScatterMatrix from "../components/ScatterMatrix";
import RealSuggestionCard from "../components/SuggestionCard";
import InsightsPanel from "../components/InsightsPanel";
import {
  fetchDiagnostics,
  fetchImportance,
  fetchSuggestions,
  fetchExport,
  type DiagnosticsData,
  type ParameterImportanceData,
  type SuggestionData,
} from "../api";

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
      const data = await fetchSuggestions(id, 5);
      setSuggestions(data);
    } catch (err) {
      console.error("Failed to generate suggestions:", err);
    } finally {
      setLoadingSuggestions(false);
    }
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

  // Build trial table data from best_parameters (mock for now — real data from /batch)
  const mockTrials = campaign.kpi_history.iterations.map((iter, i) => ({
    id: `trial-${iter}`,
    parameters: campaign.best_parameters || {},
    kpis: { objective: campaign.kpi_history.values[i] },
    status: "completed",
  }));

  return (
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
            </div>
          )}

          {activeTab === "suggestions" && (
            <div className="tab-panel">
              <div className="suggestions-header">
                <button
                  className="btn btn-primary"
                  onClick={handleGenerateSuggestions}
                  disabled={loadingSuggestions}
                >
                  {loadingSuggestions
                    ? "Generating..."
                    : "Generate Suggestions"}
                </button>
                {suggestions && (
                  <span style={{ marginLeft: "12px", fontSize: "0.85rem", color: "#718096" }}>
                    Backend: {suggestions.backend_used} | Phase: {suggestions.phase}
                  </span>
                )}
              </div>

              {suggestions && (
                <div className="suggestions-grid">
                  {suggestions.suggestions.map((sug, i) => (
                    <RealSuggestionCard
                      key={i}
                      index={i + 1}
                      suggestion={sug}
                      parameterSpecs={
                        campaign.best_parameters
                          ? Object.keys(campaign.best_parameters).map((name) => ({
                              name,
                              type: "continuous" as const,
                            }))
                          : []
                      }
                      objectiveName="Objective"
                      predictedValue={suggestions.predicted_values?.[i]}
                      predictedUncertainty={suggestions.predicted_uncertainties?.[i]}
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
                <h2>Trial History</h2>
                <TrialTable trials={mockTrials} />
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

        .suggestions-header {
          margin-bottom: 24px;
          display: flex;
          align-items: center;
        }

        .suggestions-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
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
  );
}
