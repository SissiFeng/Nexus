import { useState, useEffect, useCallback, Fragment } from "react";
import { useParams, Link } from "react-router-dom";
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
  Copy,
  Check,
  ArrowUpDown,
  ArrowUp,
  ArrowDown,
  Pause,
  Play,
  Square,
  FileSpreadsheet,
  FileJson,
  FileText,
  Image,
  BarChart3,
  Table,
  ClipboardList,
  Filter,
  TrendingDown,
  Hash,
  Home,
  RefreshCw,
} from "lucide-react";
import { useCampaign } from "../hooks/useCampaign";
import { useToast } from "../components/Toast";
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
  pauseCampaign,
  resumeCampaign,
  stopCampaign,
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

function MiniSparkline({ values, highlightIdx }: { values: number[]; highlightIdx: number }) {
  if (values.length < 2) return null;
  const w = 48, h = 16, pad = 1;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const points = values.map((v, i) => {
    const x = pad + (i / (values.length - 1)) * (w - 2 * pad);
    const y = pad + (1 - (v - min) / range) * (h - 2 * pad);
    return { x, y };
  });
  const pathD = points.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
  const hl = points[highlightIdx];
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} style={{ verticalAlign: "middle", marginLeft: "6px", flexShrink: 0 }}>
      <path d={pathD} fill="none" stroke="var(--color-primary)" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" opacity="0.5" />
      {hl && <circle cx={hl.x} cy={hl.y} r="2" fill="var(--color-primary)" />}
    </svg>
  );
}

export default function Workspace() {
  const { id } = useParams<{ id: string }>();
  const { campaign, loading, error, refresh, lastUpdated } = useCampaign(id);
  const { toast } = useToast();
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
  const [copied, setCopied] = useState(false);
  const [historySortCol, setHistorySortCol] = useState<string | null>(null);
  const [historySortDir, setHistorySortDir] = useState<"asc" | "desc">("asc");
  const [historyPage, setHistoryPage] = useState(0);
  const [historyFilter, setHistoryFilter] = useState("");
  const [expandedTrialId, setExpandedTrialId] = useState<string | null>(null);
  const [showShortcuts, setShowShortcuts] = useState(false);
  const [dismissedWarnings, setDismissedWarnings] = useState<Set<string>>(new Set());
  const HISTORY_PAGE_SIZE = 25;

  const handleCopyBest = useCallback((params: Record<string, number>) => {
    navigator.clipboard.writeText(JSON.stringify(params, null, 2)).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
      toast("Parameters copied to clipboard");
    });
  }, [toast]);

  const handleHistorySort = useCallback((col: string) => {
    if (historySortCol === col) {
      setHistorySortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setHistorySortCol(col);
      setHistorySortDir("asc");
    }
  }, [historySortCol]);

  // Keyboard shortcuts for tab navigation
  useEffect(() => {
    const tabKeys: Record<string, typeof activeTab> = {
      "1": "overview", "2": "explore", "3": "suggestions",
      "4": "insights", "5": "history", "6": "export",
    };
    const handler = (e: KeyboardEvent) => {
      // Don't intercept when typing in inputs
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLSelectElement) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === "?" || (e.shiftKey && e.key === "/")) {
        setShowShortcuts((s) => !s);
        e.preventDefault();
        return;
      }
      if (e.key === "Escape") {
        setShowShortcuts(false);
        return;
      }
      const tab = tabKeys[e.key];
      if (tab) { setActiveTab(tab); e.preventDefault(); }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

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
      <div className="workspace-skeleton">
        <div className="skel-breadcrumb"><div className="skel-line" style={{ width: "180px" }} /></div>
        <div className="skel-header">
          <div className="skel-line" style={{ width: "280px", height: "24px" }} />
          <div className="skel-line" style={{ width: "220px", height: "14px", marginTop: "8px" }} />
        </div>
        <div className="skel-tabs">
          {[1,2,3,4,5,6].map((i) => <div key={i} className="skel-tab" />)}
        </div>
        <div className="skel-content">
          <div className="skel-stats">
            {[1,2,3,4].map((i) => <div key={i} className="skel-stat-card" />)}
          </div>
          <div className="skel-chart" />
          <div className="skel-chart" style={{ height: "120px" }} />
        </div>
        <style>{`
          .workspace-skeleton { padding: 0; background: var(--color-bg); min-height: calc(100vh - 56px); animation: fadeIn 0.2s ease; }
          .skel-breadcrumb { padding: 8px 24px; background: var(--color-surface); border-bottom: 1px solid var(--color-border-subtle); }
          .skel-header { padding: 24px 24px 16px; background: var(--color-surface); border-bottom: 1px solid var(--color-border); }
          .skel-tabs { display: flex; gap: 4px; padding: 12px 24px; background: var(--color-surface); border-bottom: 2px solid var(--color-border); }
          .skel-tab { width: 100px; height: 20px; background: var(--color-bg); border-radius: 6px; animation: pulse 1.5s ease-in-out infinite; }
          .skel-content { padding: 24px; }
          .skel-stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }
          .skel-stat-card { height: 80px; background: var(--color-surface); border: 1px solid var(--color-border); border-radius: 12px; animation: pulse 1.5s ease-in-out infinite; }
          .skel-chart { height: 240px; background: var(--color-surface); border: 1px solid var(--color-border); border-radius: 12px; margin-bottom: 16px; animation: pulse 1.5s ease-in-out infinite; }
          .skel-line { height: 14px; background: var(--color-bg); border-radius: 6px; animation: pulse 1.5s ease-in-out infinite; }
        `}</style>
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
      toast(`Generated ${data.suggestions.length} suggestions`);
    } catch (err) {
      console.error("Failed to generate suggestions:", err);
      toast("Failed to generate suggestions", "error");
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
    toast("Suggestions exported as CSV");
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
      toast(`Exported as ${format.toUpperCase()}`);
    } catch (err) {
      console.error("Export failed:", err);
      toast("Export failed", "error");
    }
  };

  const sugCount = suggestions?.suggestions?.length ?? 0;
  const trialCount = campaign.observations?.length ?? 0;

  const tabs = [
    { id: "overview", label: "Overview", icon: LayoutDashboard, badge: null as number | null },
    { id: "explore", label: "Explore", icon: Search, badge: null },
    { id: "suggestions", label: "Suggestions", icon: Beaker, badge: sugCount > 0 ? sugCount : null },
    { id: "insights", label: "Insights", icon: Lightbulb, badge: null },
    { id: "history", label: "History", icon: Clock, badge: trialCount > 0 ? trialCount : null },
    { id: "export", label: "Export", icon: Download, badge: null },
  ] as const;

  // Compute data quality warnings from diagnostics
  const dataWarnings: Array<{ id: string; level: "warning" | "error" | "info"; message: string }> = [];
  if (diagnostics) {
    if (diagnostics.failure_rate > 0.3) {
      dataWarnings.push({ id: "high-failure", level: "error", message: `High failure rate detected (${(diagnostics.failure_rate * 100).toFixed(0)}%). Check parameter bounds or experimental setup.` });
    } else if (diagnostics.failure_rate > 0.15) {
      dataWarnings.push({ id: "mod-failure", level: "warning", message: `Elevated failure rate (${(diagnostics.failure_rate * 100).toFixed(0)}%). Some experiments are failing — review constraints.` });
    }
    if (diagnostics.noise_estimate > 0.5) {
      dataWarnings.push({ id: "high-noise", level: "warning", message: `High measurement noise (${diagnostics.noise_estimate.toFixed(2)}). Consider adding replicates or smoothing.` });
    }
    if (diagnostics.signal_to_noise_ratio != null && diagnostics.signal_to_noise_ratio < 1.5) {
      dataWarnings.push({ id: "low-snr", level: "warning", message: `Low signal-to-noise ratio (${diagnostics.signal_to_noise_ratio.toFixed(2)}). Optimization trends may be unreliable.` });
    }
    if (diagnostics.plateau_length > 20) {
      dataWarnings.push({ id: "plateau", level: "info", message: `No improvement for ${diagnostics.plateau_length} iterations. The optimizer may have converged or needs a strategy change.` });
    }
  }
  if (campaign.observations && campaign.observations.length < 5) {
    dataWarnings.push({ id: "few-obs", level: "info", message: `Only ${campaign.observations.length} observation${campaign.observations.length === 1 ? "" : "s"} so far. Insights become more reliable after ~10 trials.` });
  }
  const visibleWarnings = dataWarnings.filter((w) => !dismissedWarnings.has(w.id));

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

  const handleExportHistoryCSV = () => {
    if (trials.length === 0) return;
    const paramKeys = Object.keys(trials[0].parameters);
    const kpiKeys = Object.keys(trials[0].kpis);
    const headers = ["iteration", ...paramKeys, ...kpiKeys];
    const csvContent = [
      headers.join(","),
      ...trials.map((t) => [
        t.iteration,
        ...paramKeys.map((p) => t.parameters[p] ?? ""),
        ...kpiKeys.map((k) => t.kpis[k] ?? ""),
      ].join(",")),
    ].join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `history-${id.slice(0, 8)}-${new Date().toISOString().slice(0, 10)}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    toast(`History exported (${trials.length} trials)`);
  };

  return (
    <ErrorBoundary>
    <div className="workspace-container">
      <div className={`workspace-main ${chatOpen ? "chat-open" : ""}`}>
        {/* Breadcrumb */}
        <div className="workspace-breadcrumb">
          <Link to="/" className="breadcrumb-link"><Home size={13} /> Dashboard</Link>
          <ChevronRight size={12} className="breadcrumb-sep" />
          <span className="breadcrumb-current">{campaign.name}</span>
        </div>

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
              {lastUpdated && (
                <span className="workspace-refresh-indicator" title="Auto-refreshes every 5s">
                  <RefreshCw size={11} className="refresh-spin" />
                  <span>Live</span>
                </span>
              )}
            </div>
          </div>
          <div className="workspace-actions">
            {campaign.status === "running" && (
              <>
                <button
                  className="btn btn-sm btn-secondary workspace-action-btn"
                  onClick={async () => { await pauseCampaign(id); refresh(); toast("Campaign paused"); }}
                  title="Pause campaign"
                >
                  <Pause size={14} /> Pause
                </button>
                <button
                  className="btn btn-sm btn-danger-outline workspace-action-btn"
                  onClick={async () => { await stopCampaign(id); refresh(); toast("Campaign stopped", "warning"); }}
                  title="Stop campaign"
                >
                  <Square size={14} /> Stop
                </button>
              </>
            )}
            {campaign.status === "paused" && (
              <button
                className="btn btn-sm btn-primary workspace-action-btn"
                onClick={async () => { await resumeCampaign(id); refresh(); toast("Campaign resumed"); }}
                title="Resume campaign"
              >
                <Play size={14} /> Resume
              </button>
            )}
          </div>
        </div>

        {/* Tabs */}
        <div className="workspace-tabs">
          {tabs.map((tab, idx) => (
            <button
              key={tab.id}
              className={`workspace-tab ${activeTab === tab.id ? "active" : ""}`}
              onClick={() => setActiveTab(tab.id)}
              title={`${tab.label} (${idx + 1})`}
            >
              <tab.icon size={16} />
              <span>{tab.label}</span>
              {tab.badge !== null && (
                <span className="tab-badge">{tab.badge > 99 ? "99+" : tab.badge}</span>
              )}
              <kbd className="tab-kbd">{idx + 1}</kbd>
            </button>
          ))}
        </div>

        {/* Data Quality Warnings */}
        {visibleWarnings.length > 0 && (
          <div className="quality-warnings">
            {visibleWarnings.map((w) => (
              <div key={w.id} className={`quality-warning quality-warning-${w.level}`}>
                <span className="quality-warning-icon">
                  {w.level === "error" ? <AlertTriangle size={14} /> : w.level === "warning" ? <AlertTriangle size={14} /> : <Info size={14} />}
                </span>
                <span className="quality-warning-msg">{w.message}</span>
                <button
                  className="quality-warning-dismiss"
                  onClick={() => setDismissedWarnings((prev) => new Set(prev).add(w.id))}
                  title="Dismiss"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        )}

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
                {/* Iterations since last improvement */}
                {convergenceData.length > 2 && (() => {
                  const bestVal = Math.min(...convergenceData.map((d) => d.best));
                  const lastImprovementIdx = convergenceData.reduce((acc, d, i) => d.best === bestVal ? i : acc, 0);
                  const itersSinceImprovement = convergenceData.length - 1 - lastImprovementIdx;
                  const isStale = itersSinceImprovement > 30;
                  return (
                    <div className={`stat-card ${isStale ? "stat-card-warning" : ""}`}>
                      <div className="stat-label">
                        <TrendingDown size={12} style={{ marginRight: 4, verticalAlign: "middle" }} />
                        Since Improvement
                      </div>
                      <div className="stat-value">
                        {itersSinceImprovement} <span style={{ fontSize: "0.7rem", fontWeight: 400, color: "var(--color-text-muted)" }}>iter</span>
                      </div>
                    </div>
                  );
                })()}
                <div className="stat-card">
                  <div className="stat-label">
                    <Hash size={12} style={{ marginRight: 4, verticalAlign: "middle" }} />
                    Unique Configs
                  </div>
                  <div className="stat-value">
                    {(() => {
                      const seen = new Set(trials.map((t) => JSON.stringify(t.parameters)));
                      return seen.size;
                    })()}
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
                    <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                      <span className="best-result-iter">Iteration {bestResult.iteration}</span>
                      <button
                        className="btn btn-sm btn-secondary best-result-copy"
                        onClick={() => handleCopyBest(bestResult.parameters)}
                        title="Copy parameters to clipboard"
                      >
                        {copied ? <Check size={13} /> : <Copy size={13} />}
                        {copied ? "Copied" : "Copy"}
                      </button>
                    </div>
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
                {trials.length > 0 ? (
                  <RealScatterMatrix
                    data={trials.map((t) => ({
                      ...t.parameters,
                      objective: Object.values(t.kpis)[0] ?? 0,
                    }))}
                    parameters={Object.keys(trials[0].parameters)}
                    objectiveName="objective"
                    objectiveDirection="minimize"
                  />
                ) : (
                  <p className="empty-state">
                    No parameter data available for scatter plot.
                  </p>
                )}
              </div>

              {/* Parameter Range Explorer */}
              {trials.length > 0 && campaign.spec?.parameters && (() => {
                const specs = campaign.spec.parameters;
                const rangeData = specs
                  .filter((s) => s.type === "continuous" && s.lower !== undefined && s.upper !== undefined)
                  .map((s) => {
                    const vals = trials.map((t) => Number(t.parameters[s.name]) || 0);
                    const sampledMin = Math.min(...vals);
                    const sampledMax = Math.max(...vals);
                    const fullRange = (s.upper! - s.lower!);
                    const coveragePct = fullRange > 0 ? ((sampledMax - sampledMin) / fullRange) * 100 : 0;
                    const uniqueVals = new Set(vals.map((v) => v.toFixed(4))).size;
                    return { name: s.name, lower: s.lower!, upper: s.upper!, sampledMin, sampledMax, coveragePct, uniqueVals };
                  });

                if (rangeData.length === 0) return null;
                return (
                  <div className="card">
                    <h2>Parameter Range Coverage</h2>
                    <p className="range-desc">
                      How much of each parameter's defined range has been sampled. Low coverage suggests unexplored regions.
                    </p>
                    <div className="range-list">
                      {rangeData.map((r) => (
                        <div key={r.name} className="range-row">
                          <span className="range-name mono">{r.name}</span>
                          <div className="range-track">
                            <div
                              className="range-fill"
                              style={{
                                left: `${((r.sampledMin - r.lower) / (r.upper - r.lower)) * 100}%`,
                                width: `${Math.max(((r.sampledMax - r.sampledMin) / (r.upper - r.lower)) * 100, 2)}%`,
                              }}
                            />
                          </div>
                          <span className="range-pct mono">{r.coveragePct.toFixed(0)}%</span>
                          <span className="range-bounds mono">{r.sampledMin.toFixed(2)}–{r.sampledMax.toFixed(2)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                );
              })()}

              {/* Parameter Correlation Summary */}
              {trials.length >= 5 && (() => {
                const paramNames = Object.keys(trials[0].parameters);
                const objValues = trials.map((t) => Object.values(t.kpis)[0] ?? 0);
                const correlations = paramNames.map((p) => {
                  const paramValues = trials.map((t) => Number(t.parameters[p]) || 0);
                  const n = paramValues.length;
                  const meanP = paramValues.reduce((a, b) => a + b, 0) / n;
                  const meanO = objValues.reduce((a, b) => a + b, 0) / n;
                  const stdP = Math.sqrt(paramValues.reduce((a, v) => a + (v - meanP) ** 2, 0) / n);
                  const stdO = Math.sqrt(objValues.reduce((a, v) => a + (v - meanO) ** 2, 0) / n);
                  if (stdP === 0 || stdO === 0) return { name: p, corr: 0 };
                  const cov = paramValues.reduce((a, v, i) => a + (v - meanP) * (objValues[i] - meanO), 0) / n;
                  return { name: p, corr: cov / (stdP * stdO) };
                }).sort((a, b) => Math.abs(b.corr) - Math.abs(a.corr));

                return (
                  <div className="card">
                    <h2>Parameter–Objective Correlation</h2>
                    <p style={{ fontSize: "0.82rem", color: "var(--color-text-muted)", marginBottom: "16px" }}>
                      Pearson correlation between each parameter and the primary objective.
                      Stronger values suggest higher influence.
                    </p>
                    <div className="correlation-list">
                      {correlations.map((c) => {
                        const absCorr = Math.abs(c.corr);
                        const color = absCorr > 0.5 ? "var(--color-primary)" : absCorr > 0.3 ? "var(--color-yellow)" : "var(--color-gray)";
                        const strength = absCorr > 0.5 ? "Strong" : absCorr > 0.3 ? "Moderate" : "Weak";
                        return (
                          <div key={c.name} className="correlation-row">
                            <span className="correlation-name mono">{c.name}</span>
                            <div className="correlation-bar-bg">
                              <div
                                className="correlation-bar-fill"
                                style={{
                                  width: `${absCorr * 100}%`,
                                  background: color,
                                  marginLeft: c.corr < 0 ? `${(1 - absCorr) * 50}%` : "50%",
                                }}
                              />
                            </div>
                            <span className="correlation-value mono" style={{ color }}>
                              {c.corr > 0 ? "+" : ""}{c.corr.toFixed(3)}
                            </span>
                            <span className="correlation-strength" style={{ color }}>{strength}</span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                );
              })()}

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
                      bestParams={bestResult?.parameters}
                      bestObjective={bestResult ? Object.values(bestResult.kpis)[0] : undefined}
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
                  <div className="history-toolbar">
                    <div className="history-search">
                      <Filter size={14} />
                      <input
                        type="text"
                        className="history-search-input"
                        placeholder="Filter trials..."
                        value={historyFilter}
                        onChange={(e) => { setHistoryFilter(e.target.value); setHistoryPage(0); }}
                      />
                      {historyFilter && (
                        <button className="history-search-clear" onClick={() => setHistoryFilter("")}>&times;</button>
                      )}
                    </div>
                    <button
                      className="btn btn-sm btn-secondary"
                      onClick={handleExportHistoryCSV}
                      title="Download all trials as CSV"
                    >
                      <Download size={13} /> CSV
                    </button>
                    <span className="history-sort-hint">Click headers to sort</span>
                  </div>
                </div>
                {trials.length > 0 ? (() => {
                  // Apply filter
                  const filterLower = historyFilter.toLowerCase();
                  let filtered = trials;
                  if (filterLower) {
                    filtered = trials.filter((t) => {
                      const iterStr = String(t.iteration);
                      const paramStr = Object.entries(t.parameters).map(([k, v]) => `${k}=${typeof v === "number" ? v.toFixed(4) : v}`).join(" ");
                      const kpiStr = Object.entries(t.kpis).map(([k, v]) => `${k}=${typeof v === "number" ? v.toFixed(4) : v}`).join(" ");
                      return `${iterStr} ${paramStr} ${kpiStr}`.toLowerCase().includes(filterLower);
                    });
                  }
                  // Sort
                  let sorted = [...filtered];
                  if (historySortCol) {
                    sorted.sort((a, b) => {
                      let va: number, vb: number;
                      if (historySortCol === "__iter__") {
                        va = a.iteration; vb = b.iteration;
                      } else if (historySortCol.startsWith("p:")) {
                        const key = historySortCol.slice(2);
                        va = Number(a.parameters[key]) || 0;
                        vb = Number(b.parameters[key]) || 0;
                      } else {
                        const key = historySortCol.slice(2);
                        va = Number(a.kpis[key]) || 0;
                        vb = Number(b.kpis[key]) || 0;
                      }
                      return historySortDir === "asc" ? va - vb : vb - va;
                    });
                  } else {
                    sorted.reverse();
                  }
                  // Paginate
                  const totalPages = Math.ceil(sorted.length / HISTORY_PAGE_SIZE);
                  const pageTrials = sorted.slice(historyPage * HISTORY_PAGE_SIZE, (historyPage + 1) * HISTORY_PAGE_SIZE);
                  const paramKeys = Object.keys(trials[0].parameters);
                  const kpiKeys = Object.keys(trials[0].kpis);

                  // Build chronological KPI arrays for sparklines
                  const chronoTrials = [...trials].sort((a, b) => a.iteration - b.iteration);
                  const SPARK_WINDOW = 10;
                  const getSparkData = (iteration: number, kpiKey: string) => {
                    const idx = chronoTrials.findIndex((t) => t.iteration === iteration);
                    if (idx < 0) return null;
                    const start = Math.max(0, idx - SPARK_WINDOW);
                    const end = Math.min(chronoTrials.length, idx + SPARK_WINDOW + 1);
                    const window = chronoTrials.slice(start, end);
                    const vals = window.map((t) => Number(t.kpis[kpiKey]) || 0);
                    const hlIdx = idx - start;
                    return { values: vals, highlightIdx: hlIdx };
                  };

                  return (
                    <>
                      {filterLower && (
                        <div className="history-filter-info">
                          Showing {filtered.length} of {trials.length} trials
                        </div>
                      )}
                      <div className="history-table-wrapper">
                        <table className="history-table">
                          <thead>
                            <tr>
                              <th className="history-th-sortable" onClick={() => handleHistorySort("__iter__")}>
                                # {historySortCol === "__iter__" && (historySortDir === "asc" ? <ArrowUp size={12} /> : <ArrowDown size={12} />)}
                                {historySortCol !== "__iter__" && <ArrowUpDown size={11} className="history-sort-idle" />}
                              </th>
                              {paramKeys.map((p) => (
                                <th key={p} className="history-th-sortable" onClick={() => handleHistorySort(`p:${p}`)}>
                                  {p}
                                  {historySortCol === `p:${p}` && (historySortDir === "asc" ? <ArrowUp size={12} /> : <ArrowDown size={12} />)}
                                  {historySortCol !== `p:${p}` && <ArrowUpDown size={11} className="history-sort-idle" />}
                                </th>
                              ))}
                              {kpiKeys.map((k) => (
                                <th key={k} className="history-kpi-col history-th-sortable" onClick={() => handleHistorySort(`k:${k}`)}>
                                  {k}
                                  {historySortCol === `k:${k}` && (historySortDir === "asc" ? <ArrowUp size={12} /> : <ArrowDown size={12} />)}
                                  {historySortCol !== `k:${k}` && <ArrowUpDown size={11} className="history-sort-idle" />}
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {pageTrials.map((trial) => {
                              const isBest = bestResult && trial.iteration === bestResult.iteration;
                              const isExpanded = expandedTrialId === trial.id;
                              return (
                                <Fragment key={trial.id}>
                                  <tr
                                    className={`history-row-clickable ${isBest ? "history-row-best" : ""} ${isExpanded ? "history-row-expanded" : ""}`}
                                    onClick={() => setExpandedTrialId(isExpanded ? null : trial.id)}
                                  >
                                    <td className="history-iter">{trial.iteration}</td>
                                    {paramKeys.map((p) => (
                                      <td key={p} className="mono">{typeof trial.parameters[p] === "number" ? (trial.parameters[p] as number).toFixed(3) : String(trial.parameters[p])}</td>
                                    ))}
                                    {kpiKeys.map((k) => {
                                      const spark = getSparkData(trial.iteration, k);
                                      return (
                                        <td key={k} className={`mono history-kpi-val ${isBest ? "history-best-val" : ""}`}>
                                          <span className="history-kpi-cell">
                                            <span>{typeof trial.kpis[k] === "number" ? (trial.kpis[k] as number).toFixed(4) : String(trial.kpis[k])}</span>
                                            {spark && <MiniSparkline values={spark.values} highlightIdx={spark.highlightIdx} />}
                                          </span>
                                          {isBest && <span className="history-best-badge">Best</span>}
                                        </td>
                                      );
                                    })}
                                  </tr>
                                  {isExpanded && (
                                    <tr className="history-detail-row">
                                      <td colSpan={1 + paramKeys.length + kpiKeys.length}>
                                        <div className="history-detail">
                                          <div className="history-detail-section">
                                            <span className="history-detail-label">Iteration</span>
                                            <span className="history-detail-value mono">{trial.iteration}</span>
                                          </div>
                                          {paramKeys.map((p) => (
                                            <div key={p} className="history-detail-section">
                                              <span className="history-detail-label">{p}</span>
                                              <span className="history-detail-value mono">
                                                {typeof trial.parameters[p] === "number" ? (trial.parameters[p] as number).toPrecision(6) : String(trial.parameters[p])}
                                              </span>
                                            </div>
                                          ))}
                                          {kpiKeys.map((k) => (
                                            <div key={k} className="history-detail-section">
                                              <span className="history-detail-label">{k}</span>
                                              <span className="history-detail-value mono" style={{ fontWeight: 600 }}>
                                                {typeof trial.kpis[k] === "number" ? (trial.kpis[k] as number).toPrecision(6) : String(trial.kpis[k])}
                                              </span>
                                            </div>
                                          ))}
                                          {bestResult && (
                                            <div className="history-detail-section">
                                              <span className="history-detail-label">vs Best</span>
                                              <span className="history-detail-value mono">
                                                {(() => {
                                                  const trialVal = Object.values(trial.kpis)[0] ?? 0;
                                                  const bestVal = Object.values(bestResult.kpis)[0] ?? 0;
                                                  const diff = trialVal - bestVal;
                                                  return diff === 0 ? "= Best" : `${diff > 0 ? "+" : ""}${diff.toFixed(4)}`;
                                                })()}
                                              </span>
                                            </div>
                                          )}
                                        </div>
                                      </td>
                                    </tr>
                                  )}
                                </Fragment>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                      {/* Pagination */}
                      {totalPages > 1 && (
                        <div className="history-pagination">
                          <button
                            className="btn btn-sm btn-secondary"
                            onClick={() => setHistoryPage((p) => Math.max(0, p - 1))}
                            disabled={historyPage === 0}
                          >
                            <ChevronLeft size={14} /> Prev
                          </button>
                          <span className="history-page-info">
                            Page {historyPage + 1} of {totalPages}
                            <span className="history-page-range">
                              ({historyPage * HISTORY_PAGE_SIZE + 1}–{Math.min((historyPage + 1) * HISTORY_PAGE_SIZE, sorted.length)} of {sorted.length})
                            </span>
                          </span>
                          <button
                            className="btn btn-sm btn-secondary"
                            onClick={() => setHistoryPage((p) => Math.min(totalPages - 1, p + 1))}
                            disabled={historyPage >= totalPages - 1}
                          >
                            Next <ChevronRight size={14} />
                          </button>
                        </div>
                      )}
                    </>
                  );
                })() : (
                  <p className="empty-state">No experiments recorded yet.</p>
                )}
              </div>
            </div>
          )}

          {activeTab === "export" && (
            <div className="tab-panel">
              {/* Campaign Summary */}
              <div className="export-summary">
                <div className="export-summary-item">
                  <span className="export-summary-label">Parameters</span>
                  <span className="export-summary-value">{trials.length > 0 ? Object.keys(trials[0].parameters).length : 0}</span>
                </div>
                <div className="export-summary-item">
                  <span className="export-summary-label">Objectives</span>
                  <span className="export-summary-value">{campaign.objective_names?.length ?? 1}</span>
                </div>
                <div className="export-summary-item">
                  <span className="export-summary-label">Experiments</span>
                  <span className="export-summary-value">{trials.length}</span>
                </div>
                <div className="export-summary-item">
                  <span className="export-summary-label">Best KPI</span>
                  <span className="export-summary-value mono">{campaign.best_kpi?.toFixed(4) ?? "—"}</span>
                </div>
              </div>

              {/* Data Export */}
              <div className="card">
                <h2><Table size={18} /> Data Export</h2>
                <p className="export-desc">
                  Download all experiment data including parameters, objectives, and iteration metadata.
                </p>
                <div className="export-grid">
                  <button className="export-card" onClick={() => handleExport("csv")}>
                    <div className="export-card-icon export-card-icon-csv"><FileSpreadsheet size={24} /></div>
                    <div className="export-card-info">
                      <span className="export-card-title">CSV</span>
                      <span className="export-card-desc">Spreadsheet-compatible format. Best for Excel, Google Sheets, or R/pandas.</span>
                    </div>
                  </button>
                  <button className="export-card" onClick={() => handleExport("json")}>
                    <div className="export-card-icon export-card-icon-json"><FileJson size={24} /></div>
                    <div className="export-card-info">
                      <span className="export-card-title">JSON</span>
                      <span className="export-card-desc">Structured data with full metadata. Best for programmatic consumption.</span>
                    </div>
                  </button>
                  <button className="export-card" onClick={() => handleExport("xlsx")}>
                    <div className="export-card-icon export-card-icon-xlsx"><FileSpreadsheet size={24} /></div>
                    <div className="export-card-info">
                      <span className="export-card-title">Excel</span>
                      <span className="export-card-desc">Native Excel workbook with formatted headers and data validation.</span>
                    </div>
                  </button>
                </div>
              </div>

              {/* Figure Export */}
              <div className="card">
                <h2><Image size={18} /> Figure Export</h2>
                <p className="export-desc">
                  Export publication-ready figures of your optimization results.
                </p>
                <div className="export-figure-options">
                  <label className="export-figure-label">
                    Resolution
                    <select className="input">
                      <option>72 DPI (Screen)</option>
                      <option>150 DPI</option>
                      <option>300 DPI (Print)</option>
                      <option>600 DPI (Publication)</option>
                    </select>
                  </label>
                  <label className="export-figure-label">
                    Format
                    <select className="input">
                      <option>PNG</option>
                      <option>SVG</option>
                      <option>PDF</option>
                    </select>
                  </label>
                  <label className="export-figure-label">
                    Style
                    <select className="input">
                      <option>Default</option>
                      <option>ACS (Journals)</option>
                      <option>Nature</option>
                    </select>
                  </label>
                </div>
                <div className="export-figure-charts">
                  <label className="export-figure-check">
                    <input type="checkbox" defaultChecked /> <BarChart3 size={14} /> Convergence Plot
                  </label>
                  <label className="export-figure-check">
                    <input type="checkbox" defaultChecked /> <BarChart3 size={14} /> Parameter Importance
                  </label>
                  <label className="export-figure-check">
                    <input type="checkbox" defaultChecked /> <BarChart3 size={14} /> Parameter Space
                  </label>
                  <label className="export-figure-check">
                    <input type="checkbox" /> <BarChart3 size={14} /> Phase Timeline
                  </label>
                </div>
                <button className="btn btn-primary export-figure-btn">
                  <Image size={14} /> Export Selected Figures
                </button>
              </div>

              {/* Report */}
              <div className="card">
                <h2><ClipboardList size={18} /> Campaign Report</h2>
                <p className="export-desc">
                  Generate a comprehensive PDF report with convergence analysis,
                  parameter importance, best results, diagnostics, and experiment history.
                </p>
                <div className="export-report-preview">
                  <div className="export-report-section"><FileText size={14} /> Executive Summary</div>
                  <div className="export-report-section"><BarChart3 size={14} /> Convergence Analysis</div>
                  <div className="export-report-section"><BarChart3 size={14} /> Parameter Importance</div>
                  <div className="export-report-section"><Table size={14} /> Full Experiment Log ({trials.length} rows)</div>
                  <div className="export-report-section"><CheckCircle size={14} /> Best Result + Recommendations</div>
                </div>
                <button className="btn btn-primary export-figure-btn">
                  <FileDown size={14} /> Generate PDF Report
                </button>
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

      {/* Keyboard Shortcut Help Modal */}
      {showShortcuts && (
        <div className="shortcut-overlay" onClick={() => setShowShortcuts(false)}>
          <div className="shortcut-modal" onClick={(e) => e.stopPropagation()}>
            <div className="shortcut-modal-header">
              <h3>Keyboard Shortcuts</h3>
              <button className="shortcut-close" onClick={() => setShowShortcuts(false)}>
                <span>&times;</span>
              </button>
            </div>
            <div className="shortcut-group">
              <div className="shortcut-group-title">Navigation</div>
              <div className="shortcut-item"><kbd>1</kbd> Overview</div>
              <div className="shortcut-item"><kbd>2</kbd> Explore</div>
              <div className="shortcut-item"><kbd>3</kbd> Suggestions</div>
              <div className="shortcut-item"><kbd>4</kbd> Insights</div>
              <div className="shortcut-item"><kbd>5</kbd> History</div>
              <div className="shortcut-item"><kbd>6</kbd> Export</div>
            </div>
            <div className="shortcut-group">
              <div className="shortcut-group-title">Actions</div>
              <div className="shortcut-item"><kbd>?</kbd> Toggle this help</div>
              <div className="shortcut-item"><kbd>Esc</kbd> Close modal / dialog</div>
            </div>
            <div className="shortcut-hint">Press <kbd>?</kbd> to dismiss</div>
          </div>
        </div>
      )}

      <style>{`
        .workspace-container {
          display: flex;
          height: calc(100vh - 56px);
          position: relative;
        }

        .workspace-breadcrumb {
          display: flex;
          align-items: center;
          gap: 6px;
          padding: 8px 24px;
          background: var(--color-surface);
          border-bottom: 1px solid var(--color-border-subtle);
          font-size: 0.78rem;
        }

        .breadcrumb-link {
          display: flex;
          align-items: center;
          gap: 4px;
          color: var(--color-text-muted);
          text-decoration: none;
          transition: color var(--transition-fast);
        }

        .breadcrumb-link:hover {
          color: var(--color-primary);
        }

        .breadcrumb-sep {
          color: var(--color-text-muted);
          opacity: 0.5;
        }

        .breadcrumb-current {
          color: var(--color-text);
          font-weight: 500;
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

        .workspace-header {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
        }

        .workspace-actions {
          display: flex;
          gap: 8px;
          flex-shrink: 0;
        }

        .workspace-action-btn {
          display: flex;
          align-items: center;
          gap: 5px;
          white-space: nowrap;
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

        .shortcut-overlay {
          position: fixed;
          inset: 0;
          background: rgba(0,0,0,0.4);
          z-index: 9000;
          display: flex;
          align-items: center;
          justify-content: center;
          animation: fadeIn 0.15s ease;
        }

        .shortcut-modal {
          background: var(--color-surface);
          border: 1px solid var(--color-border);
          border-radius: 16px;
          padding: 24px 28px;
          min-width: 320px;
          max-width: 400px;
          box-shadow: var(--shadow-lg);
        }

        .shortcut-modal-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 20px;
        }

        .shortcut-modal-header h3 {
          font-size: 1rem;
          font-weight: 700;
          margin: 0;
        }

        .shortcut-close {
          background: none;
          border: none;
          font-size: 1.3rem;
          cursor: pointer;
          color: var(--color-text-muted);
          line-height: 1;
          padding: 4px;
        }

        .shortcut-group {
          margin-bottom: 16px;
        }

        .shortcut-group-title {
          font-size: 0.72rem;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.06em;
          color: var(--color-text-muted);
          margin-bottom: 8px;
        }

        .shortcut-item {
          display: flex;
          align-items: center;
          gap: 10px;
          padding: 5px 0;
          font-size: 0.85rem;
          color: var(--color-text);
        }

        .shortcut-item kbd {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          min-width: 28px;
          height: 24px;
          padding: 0 6px;
          background: var(--color-bg);
          border: 1px solid var(--color-border);
          border-radius: 5px;
          font-size: 0.72rem;
          font-weight: 600;
          font-family: var(--font-mono);
          color: var(--color-text-muted);
        }

        .shortcut-hint {
          font-size: 0.75rem;
          color: var(--color-text-muted);
          text-align: center;
          margin-top: 12px;
          padding-top: 12px;
          border-top: 1px solid var(--color-border-subtle);
        }

        .shortcut-hint kbd {
          display: inline;
          padding: 1px 5px;
          background: var(--color-bg);
          border: 1px solid var(--color-border);
          border-radius: 3px;
          font-size: 0.7rem;
          font-family: var(--font-mono);
        }

        .workspace-refresh-indicator {
          display: inline-flex;
          align-items: center;
          gap: 4px;
          font-size: 0.72rem;
          font-weight: 600;
          color: var(--color-green);
          background: rgba(22, 179, 100, 0.08);
          padding: 2px 8px;
          border-radius: 10px;
          letter-spacing: 0.03em;
        }

        .refresh-spin {
          animation: refreshPulse 3s ease-in-out infinite;
        }

        @keyframes refreshPulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.3; }
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

        .tab-badge {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          min-width: 18px;
          height: 18px;
          padding: 0 5px;
          border-radius: 9px;
          font-size: 0.65rem;
          font-weight: 700;
          font-family: var(--font-mono);
          background: var(--color-primary-subtle);
          color: var(--color-primary);
          line-height: 1;
        }

        .workspace-tab.active .tab-badge {
          background: var(--color-primary);
          color: white;
        }

        .tab-kbd {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          width: 18px;
          height: 18px;
          border-radius: 3px;
          background: var(--color-bg);
          border: 1px solid var(--color-border);
          font-size: 0.65rem;
          font-weight: 600;
          color: var(--color-text-muted);
          margin-left: 4px;
          opacity: 0.5;
          transition: opacity var(--transition-fast);
        }

        .workspace-tab:hover .tab-kbd,
        .workspace-tab.active .tab-kbd {
          opacity: 0.8;
        }

        .quality-warnings {
          display: flex;
          flex-direction: column;
          gap: 6px;
          padding: 10px 24px;
          background: var(--color-surface);
          border-bottom: 1px solid var(--color-border-subtle);
        }

        .quality-warning {
          display: flex;
          align-items: center;
          gap: 10px;
          padding: 8px 12px;
          border-radius: var(--radius);
          font-size: 0.82rem;
          font-weight: 500;
          animation: slideDown 0.2s ease;
        }

        .quality-warning-error {
          background: #fef2f2;
          border: 1px solid #fca5a5;
          color: #991b1b;
        }

        .quality-warning-warning {
          background: #fffbeb;
          border: 1px solid #fcd34d;
          color: #92400e;
        }

        .quality-warning-info {
          background: #eff6ff;
          border: 1px solid #93c5fd;
          color: #1e40af;
        }

        [data-theme="dark"] .quality-warning-error {
          background: #450a0a;
          border-color: #991b1b;
          color: #fca5a5;
        }

        [data-theme="dark"] .quality-warning-warning {
          background: #451a03;
          border-color: #92400e;
          color: #fcd34d;
        }

        [data-theme="dark"] .quality-warning-info {
          background: #172554;
          border-color: #1e40af;
          color: #93c5fd;
        }

        .quality-warning-icon {
          flex-shrink: 0;
          display: flex;
          align-items: center;
        }

        .quality-warning-msg {
          flex: 1;
          line-height: 1.4;
        }

        .quality-warning-dismiss {
          flex-shrink: 0;
          background: none;
          border: none;
          color: inherit;
          opacity: 0.4;
          cursor: pointer;
          font-size: 1.1rem;
          padding: 0 4px;
          line-height: 1;
          transition: opacity var(--transition-fast);
        }

        .quality-warning-dismiss:hover {
          opacity: 1;
        }

        @keyframes slideDown {
          from { opacity: 0; transform: translateY(-6px); }
          to { opacity: 1; transform: translateY(0); }
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
            padding: 0 12px;
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
            scrollbar-width: none;
          }

          .workspace-tabs::-webkit-scrollbar {
            display: none;
          }

          .workspace-tab {
            padding: 10px 12px;
            font-size: 0.82rem;
            white-space: nowrap;
            flex-shrink: 0;
          }

          .tab-kbd {
            display: none;
          }

          .suggestions-grid {
            grid-template-columns: 1fr;
          }

          .workspace-layout {
            flex-direction: column;
          }

          .workspace-chat {
            width: 100%;
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            height: 50vh;
            z-index: 50;
            border-top: 1px solid var(--color-border);
            border-left: none;
            box-shadow: 0 -4px 20px rgba(0, 0, 0, 0.1);
          }

          .workspace-main {
            padding-bottom: 60px;
          }

          .workspace-breadcrumb {
            font-size: 0.78rem;
            padding: 8px 16px;
          }

          .workspace-meta {
            flex-wrap: wrap;
            gap: 6px;
          }

          .workspace-title {
            font-size: 1.1rem;
          }

          .stats-row {
            grid-template-columns: repeat(2, 1fr);
          }

          .history-toolbar {
            flex-direction: column;
            gap: 8px;
          }
        }

        @media (max-width: 480px) {
          .stats-row {
            grid-template-columns: 1fr;
          }

          .workspace-tab span {
            display: none;
          }

          .workspace-tab .tab-badge {
            display: inline-flex;
          }
        }
      `}</style>
    </div>
    </ErrorBoundary>
  );
}
