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
  Flag,
  Trophy,
  Rocket,
  Target,
  Star,
  Undo2,
  Activity,
  GitCompare,
  Sliders,
  Crosshair,
  CheckSquare,
  Package,
  BookOpen,
  Tag,
  Hexagon,
  RotateCcw,
  X,
  Layers,
  Timer,
  Brain,
  BarChart2,
  Clipboard,
  TrendingUp,
  Grid,
  Eye,
  Shuffle,
  PieChart,
  Radar,
  GitBranch,
  BoxSelect,
  Compass,
  Volume2,
  Ruler,
  Circle,
  Minimize2,
  Gauge,
  MapPin,
  Scan,
  Waves,
  Grid3x3,
  Diamond,
  FlaskRound,
  Aperture,
  Orbit,
  Thermometer,
  ScatterChart,
  AlignVerticalJustifyStart,
  LayoutGrid,
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

function MiniRangeBar({ value, lower, upper }: { value: number; lower: number; upper: number }) {
  const range = upper - lower;
  if (range <= 0) return null;
  const pct = Math.max(0, Math.min(100, ((value - lower) / range) * 100));
  return (
    <div className="mini-range-bar" title={`${lower} – ${upper}`}>
      <div className="mini-range-track">
        <div className="mini-range-fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

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
  const [bookmarks, setBookmarks] = useState<Set<string>>(() => {
    if (!id) return new Set<string>();
    try { const s = localStorage.getItem(`opt-bm-${id}`); return s ? new Set(JSON.parse(s) as string[]) : new Set<string>(); }
    catch { return new Set<string>(); }
  });
  const [trialNotes, setTrialNotes] = useState<Record<string, string>>(() => {
    if (!id) return {};
    try { const s = localStorage.getItem(`opt-notes-${id}`); return s ? JSON.parse(s) : {}; }
    catch { return {}; }
  });
  const [showBookmarkedOnly, setShowBookmarkedOnly] = useState(false);

  // Rejected suggestion stack (max 5)
  const [rejectedSuggestions, setRejectedSuggestions] = useState<Array<{ suggestion: Record<string, number>; index: number; timestamp: number }>>([]);
  const [showRejectedStack, setShowRejectedStack] = useState(false);

  // Convergence checkpoints
  const [checkpoints, setCheckpoints] = useState<Array<{ id: string; title: string; iteration: number; bestKpi: number; phase: string; timestamp: number }>>(() => {
    if (!id) return [];
    try { const s = localStorage.getItem(`opt-cp-${id}`); return s ? JSON.parse(s) : []; }
    catch { return []; }
  });
  const [showCheckpointModal, setShowCheckpointModal] = useState(false);
  const [checkpointTitle, setCheckpointTitle] = useState("");

  // Parameter sentinel toggle
  const [sentinelOpen, setSentinelOpen] = useState(true);

  // Trial comparison panel (History tab)
  const [compareSet, setCompareSet] = useState<Set<string>>(new Set());
  const [showCompareModal, setShowCompareModal] = useState(false);

  // What-if analysis (Explore tab)
  const [whatIfParam, setWhatIfParam] = useState<string | null>(null);
  const [whatIfValue, setWhatIfValue] = useState(0.5);

  // Goal tracker (Overview tab)
  const [goals, setGoals] = useState<Array<{ id: string; name: string; target: number; direction: "minimize" | "maximize"; created: number }>>(() => {
    if (!id) return [];
    try { const s = localStorage.getItem(`opt-goals-${id}`); return s ? JSON.parse(s) : []; }
    catch { return []; }
  });
  const [showGoalModal, setShowGoalModal] = useState(false);
  const [goalName, setGoalName] = useState("");
  const [goalTarget, setGoalTarget] = useState("");
  const [goalDirection, setGoalDirection] = useState<"minimize" | "maximize">("minimize");

  // Batch selection (Suggestions tab)
  const [selectedSuggestions, setSelectedSuggestions] = useState<Set<number>>(new Set());

  // Trial annotations / tags
  const [trialTags, setTrialTags] = useState<Record<string, string[]>>(() => {
    if (!id) return {};
    try { const s = localStorage.getItem(`opt-tags-${id}`); return s ? JSON.parse(s) : {}; }
    catch { return {}; }
  });
  const TRIAL_TAG_OPTIONS = ["promising", "anomaly", "investigate", "baseline", "outlier", "equipment-issue"] as const;

  // Decision journal (Overview tab)
  const [journalEntries, setJournalEntries] = useState<Array<{ id: string; text: string; iteration: number; timestamp: number }>>(() => {
    if (!id) return [];
    try { const s = localStorage.getItem(`opt-journal-${id}`); return s ? JSON.parse(s) : []; }
    catch { return []; }
  });
  const [journalInput, setJournalInput] = useState("");
  const [showJournal, setShowJournal] = useState(true);

  // Replay animation (Overview convergence)
  const [replayIdx, setReplayIdx] = useState<number | null>(null);
  const [replayPlaying, setReplayPlaying] = useState(false);

  // Statistical quick-compare (History tab)
  const [showStatCompare, setShowStatCompare] = useState(false);

  const HISTORY_PAGE_SIZE = 25;

  // Persist bookmarks & notes to localStorage
  useEffect(() => {
    if (!id) return;
    localStorage.setItem(`opt-bm-${id}`, JSON.stringify([...bookmarks]));
  }, [id, bookmarks]);
  useEffect(() => {
    if (!id) return;
    localStorage.setItem(`opt-notes-${id}`, JSON.stringify(trialNotes));
  }, [id, trialNotes]);
  useEffect(() => {
    if (!id) return;
    localStorage.setItem(`opt-cp-${id}`, JSON.stringify(checkpoints));
  }, [id, checkpoints]);
  useEffect(() => {
    if (!id) return;
    localStorage.setItem(`opt-goals-${id}`, JSON.stringify(goals));
  }, [id, goals]);
  useEffect(() => {
    if (!id) return;
    localStorage.setItem(`opt-tags-${id}`, JSON.stringify(trialTags));
  }, [id, trialTags]);
  useEffect(() => {
    if (!id) return;
    localStorage.setItem(`opt-journal-${id}`, JSON.stringify(journalEntries));
  }, [id, journalEntries]);

  const toggleBookmark = useCallback((trialId: string) => {
    setBookmarks(prev => {
      const next = new Set(prev);
      if (next.has(trialId)) next.delete(trialId);
      else next.add(trialId);
      return next;
    });
  }, []);

  const setTrialNote = useCallback((trialId: string, note: string) => {
    setTrialNotes(prev => {
      if (!note.trim()) {
        const { [trialId]: _, ...rest } = prev;
        return rest;
      }
      return { ...prev, [trialId]: note };
    });
  }, []);

  const handleRejectSuggestion = useCallback((sug: Record<string, number>, index: number) => {
    setRejectedSuggestions(prev => {
      const next = [{ suggestion: sug, index, timestamp: Date.now() }, ...prev];
      return next.slice(0, 5);
    });
    toast(`Suggestion #${index} rejected`);
  }, [toast]);

  const handleReconsider = useCallback((idx: number) => {
    setRejectedSuggestions(prev => prev.filter((_, i) => i !== idx));
    toast("Suggestion reconsidered");
  }, [toast]);

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

  // Keyboard shortcuts for tab navigation + power user actions
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
        setExpandedTrialId(null);
        return;
      }
      const tab = tabKeys[e.key];
      if (tab) { setActiveTab(tab); e.preventDefault(); return; }

      // Power user shortcuts
      if (e.key === "b") {
        // Jump to best trial on history tab
        setActiveTab("history");
        setTimeout(() => {
          const btn = document.querySelector('.btn-accent') as HTMLButtonElement;
          if (btn) btn.click();
        }, 50);
        e.preventDefault();
        return;
      }
      if (e.key === "f" || e.key === "/") {
        // Focus filter input
        const el = document.querySelector('.history-search-input') as HTMLInputElement;
        if (el) { el.focus(); e.preventDefault(); }
        return;
      }
      if (e.key === "g") {
        // Generate suggestions
        setActiveTab("suggestions");
        setTimeout(() => {
          const btn = document.querySelector('.suggestions-generate-btn') as HTMLButtonElement;
          if (btn && !btn.disabled) btn.click();
        }, 50);
        e.preventDefault();
        return;
      }
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

  // Trial comparison toggle
  const toggleCompare = useCallback((trialId: string) => {
    setCompareSet(prev => {
      const next = new Set(prev);
      if (next.has(trialId)) { next.delete(trialId); }
      else if (next.size < 3) { next.add(trialId); }
      else { toast("Select up to 3 trials to compare"); }
      return next;
    });
  }, [toast]);

  // Goal management
  const addGoal = () => {
    const target = parseFloat(goalTarget);
    if (!goalName.trim() || isNaN(target)) return;
    setGoals(prev => [...prev, { id: `goal-${Date.now()}`, name: goalName.trim(), target, direction: goalDirection, created: Date.now() }]);
    setGoalName("");
    setGoalTarget("");
    setShowGoalModal(false);
    toast(`Goal "${goalName.trim()}" created`);
  };

  const removeGoal = (goalId: string) => {
    setGoals(prev => prev.filter(g => g.id !== goalId));
  };

  // Batch selection toggle
  const toggleSuggestionSelect = useCallback((idx: number) => {
    setSelectedSuggestions(prev => {
      const next = new Set(prev);
      if (next.has(idx)) next.delete(idx);
      else next.add(idx);
      return next;
    });
  }, []);

  const handleExportSelectedCSV = useCallback(() => {
    if (!suggestions || selectedSuggestions.size === 0) return;
    const selected = suggestions.suggestions.filter((_, i) => selectedSuggestions.has(i));
    const headers = Object.keys(selected[0]);
    const csvContent = [
      headers.join(","),
      ...selected.map((row) => headers.map((h) => row[h] ?? "").join(",")),
    ].join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `batch-${id?.slice(0, 8)}-${new Date().toISOString().slice(0, 10)}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    toast(`Exported ${selected.length} suggestions as CSV`);
  }, [suggestions, selectedSuggestions, id, toast]);

  // Trial tag management
  const toggleTrialTag = useCallback((trialId: string, tag: string) => {
    setTrialTags(prev => {
      const existing = prev[trialId] || [];
      const has = existing.includes(tag);
      return { ...prev, [trialId]: has ? existing.filter(t => t !== tag) : [...existing, tag] };
    });
  }, []);

  // Decision journal
  const addJournalEntry = useCallback(() => {
    if (!journalInput.trim()) return;
    setJournalEntries(prev => [{
      id: `j-${Date.now()}`,
      text: journalInput.trim(),
      iteration: campaign?.iteration ?? 0,
      timestamp: Date.now(),
    }, ...prev]);
    setJournalInput("");
    toast("Journal entry added");
  }, [journalInput, campaign?.iteration, toast]);

  const removeJournalEntry = useCallback((entryId: string) => {
    setJournalEntries(prev => prev.filter(e => e.id !== entryId));
  }, []);

  // Campaign summary clipboard export
  const copyCampaignSummary = useCallback(() => {
    if (!campaign) return;
    const obs = campaign.observations ?? [];
    const paramNames = obs.length > 0 ? Object.keys(obs[0].parameters) : [];
    const objNames = campaign.objective_names ?? ["objective"];
    const bestKpi = campaign.best_kpi;
    const bestObs = obs.find(o => {
      const kv = Object.values(o.kpi_values);
      return kv.length > 0 && Number(kv[0]) === bestKpi;
    });
    const lines = [
      `# ${campaign.name}`,
      "",
      `**Campaign ID:** ${id?.slice(0, 8) ?? "—"}`,
      `**Status:** ${campaign.status}  |  **Phase:** ${campaign.phases[campaign.phases.length - 1]?.name ?? "—"}`,
      `**Iterations:** ${campaign.iteration}  |  **Total Trials:** ${campaign.total_trials}`,
      "",
      "## Parameters",
      paramNames.map(p => `- \`${p}\``).join("\n"),
      "",
      "## Objectives",
      objNames.map(o => `- ${o} (${campaign.objective_directions?.[o] ?? "minimize"})`).join("\n"),
      "",
      "## Best Result",
      bestKpi != null ? `- **Value:** ${bestKpi.toFixed(4)}` : "- No results yet",
      bestObs ? `- **Iteration:** ${bestObs.iteration}` : "",
      bestObs ? `- **Parameters:** ${paramNames.map(p => `${p}=${Number(bestObs.parameters[p]).toFixed(3)}`).join(", ")}` : "",
      "",
      `*Generated by Optimization Copilot on ${new Date().toISOString().slice(0, 10)}*`,
    ];
    navigator.clipboard.writeText(lines.join("\n")).then(() => toast("Summary copied to clipboard"));
  }, [campaign, id, toast]);

  // Replay animation timer
  const replayMaxIdx = campaign?.kpi_history?.iterations?.length ?? 0;
  useEffect(() => {
    if (!replayPlaying || replayIdx === null || replayMaxIdx === 0) return;
    if (replayIdx >= replayMaxIdx - 1) { setReplayPlaying(false); return; }
    const timer = setTimeout(() => setReplayIdx(prev => (prev ?? 0) + 1), 120);
    return () => clearTimeout(timer);
  }, [replayPlaying, replayIdx, replayMaxIdx]);

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

  const addCheckpoint = () => {
    if (!campaign || !checkpointTitle.trim()) return;
    const cp = {
      id: `cp-${Date.now()}`,
      title: checkpointTitle.trim(),
      iteration: campaign.iteration,
      bestKpi: campaign.best_kpi ?? 0,
      phase: campaign.phases[campaign.phases.length - 1]?.name ?? "unknown",
      timestamp: Date.now(),
    };
    setCheckpoints(prev => [...prev, cp]);
    setCheckpointTitle("");
    setShowCheckpointModal(false);
    toast(`Checkpoint "${cp.title}" saved`);
  };

  const removeCheckpoint = (cpId: string) => {
    setCheckpoints(prev => prev.filter(cp => cp.id !== cpId));
  };

  // Compute diversity score for a suggestion (avg normalized distance to last 3 trials)
  const computeDiversityScore = (sug: Record<string, number>): number | null => {
    if (trials.length < 3 || !campaign.spec?.parameters) return null;
    const specs = campaign.spec.parameters.filter(s => s.type === "continuous" && s.lower != null && s.upper != null);
    if (specs.length === 0) return null;
    const recent = [...trials].sort((a, b) => b.iteration - a.iteration).slice(0, 3);
    let totalDist = 0;
    for (const trial of recent) {
      let sumSq = 0;
      for (const spec of specs) {
        const range = (spec.upper! - spec.lower!);
        if (range <= 0) continue;
        const normSug = ((sug[spec.name] ?? 0) - spec.lower!) / range;
        const normTrial = (Number(trial.parameters[spec.name]) - spec.lower!) / range;
        sumSq += (normSug - normTrial) ** 2;
      }
      totalDist += Math.sqrt(sumSq / specs.length);
    }
    return totalDist / recent.length;
  };

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

  const handleExportHistoryJSON = () => {
    if (trials.length === 0) return;
    const data = trials.map((t) => ({
      iteration: t.iteration,
      parameters: t.parameters,
      kpis: t.kpis,
      bookmarked: bookmarks.has(t.id),
      note: trialNotes[t.id] || undefined,
    }));
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `history-${id.slice(0, 8)}-${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    toast(`History exported as JSON (${trials.length} trials)`);
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
              <div className="overview-actions-row">
                <button className="btn btn-sm btn-secondary" onClick={copyCampaignSummary} title="Copy campaign summary as Markdown">
                  <Clipboard size={13} /> Copy Summary
                </button>
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

              {/* Milestone Timeline */}
              {trials.length > 0 && (() => {
                const milestones: Array<{ icon: React.ReactNode; label: string; detail: string; iteration: number; color: string }> = [];
                milestones.push({ icon: <Rocket size={14} />, label: "Campaign Started", detail: `${campaign.phases[0]?.name ?? "init"} phase`, iteration: 0, color: "var(--color-gray)" });
                if (trials.length >= 1) {
                  const firstTrial = trials.reduce((min, t) => t.iteration < min.iteration ? t : min, trials[0]);
                  milestones.push({ icon: <FlaskConical size={14} />, label: "First Trial", detail: `Iteration ${firstTrial.iteration}`, iteration: firstTrial.iteration, color: "var(--color-blue)" });
                }
                if (bestResult) {
                  milestones.push({ icon: <Trophy size={14} />, label: "Best Result", detail: `Iter ${bestResult.iteration} — ${Object.values(bestResult.kpis)[0]?.toFixed(4) ?? ""}`, iteration: bestResult.iteration, color: "var(--color-green)" });
                }
                if (diagnostics && diagnostics.plateau_length > 15) {
                  const plateauStart = campaign.iteration - diagnostics.plateau_length;
                  milestones.push({ icon: <Flag size={14} />, label: "Plateau Detected", detail: `Since iteration ${plateauStart}`, iteration: plateauStart, color: "var(--color-yellow)" });
                }
                if (campaign.status === "completed") {
                  milestones.push({ icon: <Target size={14} />, label: "Completed", detail: `${campaign.iteration} iterations`, iteration: campaign.iteration, color: "var(--color-green)" });
                } else {
                  milestones.push({ icon: <Target size={14} />, label: "Current", detail: `Iteration ${campaign.iteration}`, iteration: campaign.iteration, color: "var(--color-primary)" });
                }
                milestones.sort((a, b) => a.iteration - b.iteration);
                return (
                  <div className="milestone-timeline">
                    <div className="milestone-line" />
                    {milestones.map((m, i) => (
                      <div key={i} className="milestone-item">
                        <div className="milestone-dot" style={{ background: m.color, borderColor: m.color }}>{m.icon}</div>
                        <div className="milestone-content">
                          <div className="milestone-label">{m.label}</div>
                          <div className="milestone-detail">{m.detail}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                );
              })()}

              {/* Checkpoint Controls */}
              <div className="checkpoint-controls">
                <button className="btn btn-sm btn-secondary" onClick={() => setShowCheckpointModal(true)}>
                  <Flag size={13} /> Create Checkpoint
                </button>
                {checkpoints.length > 0 && (
                  <span className="checkpoint-count">{checkpoints.length} checkpoint{checkpoints.length !== 1 ? "s" : ""}</span>
                )}
              </div>

              {/* Checkpoint Modal */}
              {showCheckpointModal && (
                <div className="shortcut-overlay" onClick={() => setShowCheckpointModal(false)}>
                  <div className="shortcut-modal checkpoint-modal" onClick={(e) => e.stopPropagation()}>
                    <div className="shortcut-modal-header">
                      <h3>Create Checkpoint</h3>
                      <button className="shortcut-close" onClick={() => setShowCheckpointModal(false)}>
                        <span>&times;</span>
                      </button>
                    </div>
                    <p className="checkpoint-modal-desc">
                      Capture the current campaign state as a checkpoint for future reference.
                    </p>
                    <div className="checkpoint-modal-snapshot">
                      <div className="checkpoint-snapshot-row">
                        <span>Iteration</span><span className="mono">{campaign.iteration}</span>
                      </div>
                      <div className="checkpoint-snapshot-row">
                        <span>Best KPI</span><span className="mono">{campaign.best_kpi?.toFixed(4) ?? "—"}</span>
                      </div>
                      <div className="checkpoint-snapshot-row">
                        <span>Phase</span><span>{campaign.phases[campaign.phases.length - 1]?.name ?? "—"}</span>
                      </div>
                    </div>
                    <input
                      type="text"
                      className="history-note-input checkpoint-title-input"
                      placeholder="Checkpoint name (e.g., 'Tightened bounds on X')"
                      value={checkpointTitle}
                      onChange={(e) => setCheckpointTitle(e.target.value)}
                      onKeyDown={(e) => { if (e.key === "Enter" && checkpointTitle.trim()) addCheckpoint(); }}
                      autoFocus
                    />
                    <div className="checkpoint-modal-actions">
                      <button className="btn btn-sm btn-secondary" onClick={() => setShowCheckpointModal(false)}>Cancel</button>
                      <button className="btn btn-sm btn-primary" onClick={addCheckpoint} disabled={!checkpointTitle.trim()}>
                        <Flag size={13} /> Save Checkpoint
                      </button>
                    </div>
                  </div>
                </div>
              )}

              {/* Saved Checkpoints List */}
              {checkpoints.length > 0 && (
                <div className="checkpoint-list">
                  {checkpoints.map((cp) => (
                    <div key={cp.id} className="checkpoint-item">
                      <div className="checkpoint-dot" />
                      <div className="checkpoint-info">
                        <span className="checkpoint-title">{cp.title}</span>
                        <span className="checkpoint-meta">
                          Iter {cp.iteration} &middot; Best {cp.bestKpi.toFixed(4)} &middot; {cp.phase} &middot; {new Date(cp.timestamp).toLocaleDateString()}
                        </span>
                      </div>
                      <button className="checkpoint-remove" onClick={() => removeCheckpoint(cp.id)} title="Remove checkpoint">&times;</button>
                    </div>
                  ))}
                </div>
              )}

              {/* Goal Tracker */}
              <div className="card goal-tracker-card">
                <div className="goal-tracker-header">
                  <div className="goal-tracker-header-left">
                    <Crosshair size={16} />
                    <h2 style={{ margin: 0 }}>Optimization Goals</h2>
                  </div>
                  <button className="btn btn-sm btn-secondary" onClick={() => setShowGoalModal(true)}>
                    + Add Goal
                  </button>
                </div>
                {goals.length === 0 ? (
                  <p className="goal-empty">Define target KPI values to track progress toward your research objectives.</p>
                ) : (
                  <div className="goal-list">
                    {goals.map(goal => {
                      const currentBest = campaign.best_kpi ?? 0;
                      const isMinimize = goal.direction === "minimize";
                      const progress = isMinimize
                        ? currentBest <= goal.target ? 100 : Math.max(0, Math.min(100, (1 - (currentBest - goal.target) / Math.abs(goal.target || 1)) * 100))
                        : currentBest >= goal.target ? 100 : Math.max(0, Math.min(100, (currentBest / Math.abs(goal.target || 1)) * 100));
                      const reached = (isMinimize && currentBest <= goal.target) || (!isMinimize && currentBest >= goal.target);
                      const velocity = diagnostics?.improvement_velocity ?? 0;
                      const status = reached ? "reached" : velocity < -0.01 ? "on-track" : velocity < 0 ? "at-risk" : "behind";
                      const remaining = !reached && velocity < -0.001
                        ? Math.ceil(Math.abs(currentBest - goal.target) / Math.abs(velocity))
                        : null;
                      return (
                        <div key={goal.id} className={`goal-item goal-item-${status}`}>
                          <div className="goal-item-top">
                            <span className="goal-item-name">{goal.name}</span>
                            <span className={`goal-status-badge goal-status-${status}`}>
                              {status === "reached" ? "Reached" : status === "on-track" ? "On Track" : status === "at-risk" ? "At Risk" : "Behind"}
                            </span>
                            <button className="goal-remove" onClick={() => removeGoal(goal.id)} title="Remove goal">&times;</button>
                          </div>
                          <div className="goal-progress-row">
                            <span className="mono goal-current">{currentBest.toFixed(4)}</span>
                            <div className="goal-progress-bar">
                              <div className="goal-progress-fill" style={{ width: `${Math.min(progress, 100)}%` }} />
                            </div>
                            <span className="mono goal-target-val">{goal.target.toFixed(4)}</span>
                          </div>
                          <div className="goal-meta-row">
                            <span>{isMinimize ? "minimize" : "maximize"} to {goal.target.toFixed(4)}</span>
                            {remaining && <span className="goal-eta">~{remaining} iter remaining</span>}
                            {reached && <span className="goal-eta" style={{ color: "var(--color-green)" }}>Target achieved!</span>}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>

              {/* Goal Modal */}
              {showGoalModal && (
                <div className="shortcut-overlay" onClick={() => setShowGoalModal(false)}>
                  <div className="shortcut-modal checkpoint-modal" onClick={(e) => e.stopPropagation()}>
                    <div className="shortcut-modal-header">
                      <h3>Add Optimization Goal</h3>
                      <button className="shortcut-close" onClick={() => setShowGoalModal(false)}><span>&times;</span></button>
                    </div>
                    <p className="checkpoint-modal-desc">
                      Set a target KPI value. Progress will be tracked relative to your current best result.
                    </p>
                    <div className="goal-modal-fields">
                      <label className="goal-modal-label">
                        Goal name
                        <input
                          type="text"
                          className="history-note-input"
                          placeholder="e.g., Reach ≤ -0.6 objective"
                          value={goalName}
                          onChange={(e) => setGoalName(e.target.value)}
                          autoFocus
                        />
                      </label>
                      <div className="goal-modal-row">
                        <label className="goal-modal-label" style={{ flex: 1 }}>
                          Target value
                          <input
                            type="number"
                            className="history-note-input"
                            placeholder="-0.6"
                            step="any"
                            value={goalTarget}
                            onChange={(e) => setGoalTarget(e.target.value)}
                          />
                        </label>
                        <label className="goal-modal-label">
                          Direction
                          <select
                            className="suggestions-batch-select"
                            value={goalDirection}
                            onChange={(e) => setGoalDirection(e.target.value as "minimize" | "maximize")}
                          >
                            <option value="minimize">Minimize</option>
                            <option value="maximize">Maximize</option>
                          </select>
                        </label>
                      </div>
                    </div>
                    <div className="checkpoint-modal-actions">
                      <button className="btn btn-sm btn-secondary" onClick={() => setShowGoalModal(false)}>Cancel</button>
                      <button className="btn btn-sm btn-primary" onClick={addGoal} disabled={!goalName.trim() || !goalTarget}>
                        <Crosshair size={13} /> Create Goal
                      </button>
                    </div>
                  </div>
                </div>
              )}

              {/* Parameter Variance Sentinel */}
              {trials.length >= 5 && campaign.spec?.parameters && (() => {
                const specs = campaign.spec.parameters.filter(s => s.type === "continuous" && s.lower != null && s.upper != null);
                if (specs.length === 0) return null;
                const chrono = [...trials].sort((a, b) => a.iteration - b.iteration);
                const recentN = Math.min(15, chrono.length);
                const recent = chrono.slice(-recentN);

                const paramStats = specs.map(s => {
                  const vals = recent.map(t => Number(t.parameters[s.name]) || 0);
                  const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
                  const variance = vals.reduce((a, v) => a + (v - mean) ** 2, 0) / vals.length;
                  const range = (s.upper! - s.lower!);
                  const normalizedStd = range > 0 ? Math.sqrt(variance) / range : 0;
                  const isCollapsed = normalizedStd < 0.05;
                  return { name: s.name, normalizedStd, isCollapsed, vals: chrono.slice(-15).map(t => Number(t.parameters[s.name]) || 0), lower: s.lower!, upper: s.upper! };
                });

                const collapsedCount = paramStats.filter(p => p.isCollapsed).length;

                return (
                  <div className={`card sentinel-card ${collapsedCount > 0 ? "sentinel-alert" : ""}`}>
                    <div className="sentinel-header" onClick={() => setSentinelOpen(s => !s)} style={{ cursor: "pointer" }}>
                      <div className="sentinel-header-left">
                        <Activity size={16} />
                        <h2 style={{ margin: 0 }}>Parameter Variance Sentinel</h2>
                        {collapsedCount > 0 && (
                          <span className="sentinel-alert-badge">{collapsedCount} collapsed</span>
                        )}
                      </div>
                      {sentinelOpen ? <ArrowUp size={14} /> : <ArrowDown size={14} />}
                    </div>
                    {sentinelOpen && (
                      <div className="sentinel-body">
                        <p className="sentinel-desc">
                          Monitoring recent {recentN}-trial variance for each parameter. Low variance may indicate parameter collapse.
                        </p>
                        {paramStats.map(p => {
                          const sparkW = 56, sparkH = 18, pad = 1;
                          const min = Math.min(...p.vals);
                          const max = Math.max(...p.vals);
                          const range = max - min || 1;
                          const points = p.vals.map((v, i) => ({
                            x: pad + (i / Math.max(1, p.vals.length - 1)) * (sparkW - 2 * pad),
                            y: pad + (1 - (v - min) / range) * (sparkH - 2 * pad),
                          }));
                          const d = points.map((pt, i) => `${i === 0 ? "M" : "L"}${pt.x.toFixed(1)},${pt.y.toFixed(1)}`).join(" ");
                          const sparkColor = p.isCollapsed ? "#ef4444" : p.normalizedStd < 0.15 ? "#eab308" : "#22c55e";
                          return (
                            <div key={p.name} className={`sentinel-row ${p.isCollapsed ? "sentinel-row-alert" : ""}`}>
                              <span className="sentinel-param-name mono">{p.name}</span>
                              <svg width={sparkW} height={sparkH} viewBox={`0 0 ${sparkW} ${sparkH}`} style={{ flexShrink: 0 }}>
                                <path d={d} fill="none" stroke={sparkColor} strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" />
                              </svg>
                              <div className="sentinel-variance-bar">
                                <div className="sentinel-variance-fill" style={{ width: `${Math.min(p.normalizedStd * 100 / 0.5, 100)}%`, background: sparkColor }} />
                              </div>
                              <span className="sentinel-std mono" style={{ color: sparkColor }}>
                                {(p.normalizedStd * 100).toFixed(1)}%
                              </span>
                              {p.isCollapsed && <span className="sentinel-warn-tag">Collapsed</span>}
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>
                );
              })()}

              <div className="card">
                <div className="convergence-header">
                  <h2>Convergence</h2>
                  {convergenceData.length > 5 && (
                    <div className="replay-controls">
                      {replayIdx !== null && (
                        <span className="replay-counter">
                          {replayIdx + 1} / {convergenceData.length}
                        </span>
                      )}
                      <button
                        className="btn btn-sm btn-secondary"
                        onClick={() => {
                          if (replayPlaying) {
                            setReplayPlaying(false);
                          } else {
                            setReplayIdx(0);
                            setReplayPlaying(true);
                          }
                        }}
                        title={replayPlaying ? "Pause replay" : "Replay optimization journey"}
                      >
                        {replayPlaying ? <><Pause size={12} /> Pause</> : <><RotateCcw size={12} /> Replay</>}
                      </button>
                      {replayIdx !== null && !replayPlaying && (
                        <button className="btn btn-sm btn-secondary" onClick={() => { setReplayIdx(null); setReplayPlaying(false); }} title="Reset">
                          <X size={12} />
                        </button>
                      )}
                    </div>
                  )}
                </div>
                {convergenceData.length > 0 ? (
                  <>
                    <RealConvergencePlot
                      data={replayIdx !== null ? convergenceData.slice(0, replayIdx + 1) : convergenceData}
                      objectiveName="Objective"
                      direction="minimize"
                      phases={convergencePhases}
                    />
                    {replayIdx !== null && replayIdx < convergenceData.length && (
                      <div className="replay-info">
                        <span className="replay-dot" />
                        Iteration {convergenceData[replayIdx]?.iteration ?? replayIdx}
                        {convergenceData[replayIdx]?.best !== undefined && (
                          <span className="replay-best"> — Best: {convergenceData[replayIdx].best.toFixed(4)}</span>
                        )}
                      </div>
                    )}
                  </>
                ) : (
                  <p className="empty-state">No convergence data yet.</p>
                )}
              </div>

              <div className="card">
                <h2>Diagnostics</h2>
                {loadingDiag ? (
                  <div className="loading">Loading diagnostics...</div>
                ) : diagnostics ? (
                  (() => {
                    const trendData: Record<string, number[]> = {};
                    if (trials.length >= 5) {
                      const chrono = [...trials].sort((a, b) => a.iteration - b.iteration);
                      const windowSize = Math.max(5, Math.floor(chrono.length / 10));
                      const nPoints = Math.min(12, Math.ceil(chrono.length / windowSize));
                      // Rolling best KPI
                      const bestKpiTrend: number[] = [];
                      let runningBest = Infinity;
                      for (let i = 0; i < nPoints; i++) {
                        const end = Math.min(chrono.length, (i + 1) * windowSize);
                        for (let j = i * windowSize; j < end; j++) {
                          const val = Number(Object.values(chrono[j].kpis)[0]) || 0;
                          if (val < runningBest) runningBest = val;
                        }
                        bestKpiTrend.push(runningBest);
                      }
                      trendData.best_kpi_value = bestKpiTrend;
                      // Rolling failure rate (proportion of "bad" trials per window — use high KPI as proxy)
                      const kpiVals = chrono.map((t) => Number(Object.values(t.kpis)[0]) || 0);
                      const median = [...kpiVals].sort((a, b) => a - b)[Math.floor(kpiVals.length / 2)];
                      const failTrend: number[] = [];
                      for (let i = 0; i < nPoints; i++) {
                        const start = i * windowSize;
                        const end = Math.min(chrono.length, (i + 1) * windowSize);
                        const slice = kpiVals.slice(start, end);
                        const failRate = slice.filter((v) => v > median * 1.5).length / slice.length;
                        failTrend.push(failRate);
                      }
                      trendData.failure_rate = failTrend;
                      // Rolling improvement velocity
                      const velTrend: number[] = [];
                      for (let i = 0; i < nPoints; i++) {
                        const start = i * windowSize;
                        const end = Math.min(chrono.length, (i + 1) * windowSize);
                        const slice = kpiVals.slice(start, end);
                        if (slice.length >= 2) {
                          velTrend.push((slice[slice.length - 1] - slice[0]) / slice.length);
                        } else {
                          velTrend.push(0);
                        }
                      }
                      trendData.improvement_velocity = velTrend;
                    }
                    return (
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
                        trendData={trendData}
                      />
                    );
                  })()
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

              {/* Experiment Cost / Time Tracker */}
              {convergenceData.length >= 3 && (() => {
                const totalIter = campaign.iteration ?? convergenceData.length;
                const startTime = campaign.created_at ? new Date(campaign.created_at).getTime() : 0;
                const elapsed = startTime ? Date.now() - startTime : 0;
                const elapsedHrs = elapsed / (1000 * 60 * 60);
                const avgPerIter = totalIter > 0 ? elapsed / totalIter : 0;
                const avgPerIterMin = avgPerIter / (1000 * 60);
                // Estimate remaining: compute improvement velocity
                const recent = convergenceData.slice(-10);
                const bestSoFar = Math.min(...convergenceData.map(d => d.best));
                const recentBest = Math.min(...recent.map(d => d.best));
                const firstBest = convergenceData[0]?.best ?? 0;
                const totalImprovement = firstBest - bestSoFar;
                const recentImprovement = recent.length > 1 ? (recent[0]?.best ?? 0) - recentBest : 0;
                const velocitySlowing = recentImprovement < totalImprovement * 0.02;
                const budgetPct = Math.min((totalIter / Math.max(totalIter + 20, 100)) * 100, 100);
                const efficiencyScore = totalImprovement !== 0 && totalIter > 0 ? Math.abs(totalImprovement) / totalIter : 0;

                return (
                  <div className="card cost-tracker-card">
                    <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "12px" }}>
                      <Timer size={16} style={{ color: "var(--color-primary)" }} />
                      <h2 style={{ margin: 0 }}>Experiment Tracker</h2>
                    </div>
                    <div className="cost-tracker-grid">
                      <div className="cost-tracker-item">
                        <span className="cost-tracker-label">Total Iterations</span>
                        <span className="cost-tracker-value">{totalIter}</span>
                      </div>
                      <div className="cost-tracker-item">
                        <span className="cost-tracker-label">Elapsed Time</span>
                        <span className="cost-tracker-value">{elapsedHrs < 1 ? `${(elapsedHrs * 60).toFixed(0)}m` : `${elapsedHrs.toFixed(1)}h`}</span>
                      </div>
                      <div className="cost-tracker-item">
                        <span className="cost-tracker-label">Avg per Iteration</span>
                        <span className="cost-tracker-value">{avgPerIterMin < 1 ? `${(avgPerIterMin * 60).toFixed(0)}s` : `${avgPerIterMin.toFixed(1)}m`}</span>
                      </div>
                      <div className="cost-tracker-item">
                        <span className="cost-tracker-label">Efficiency</span>
                        <span className="cost-tracker-value">{efficiencyScore.toFixed(4)}/iter</span>
                      </div>
                    </div>
                    <div className="cost-tracker-bar-section">
                      <div className="cost-tracker-bar-header">
                        <span>Optimization Progress</span>
                        <span className="mono" style={{ fontSize: "0.78rem" }}>{budgetPct.toFixed(0)}%</span>
                      </div>
                      <div className="cost-tracker-bar">
                        <div className="cost-tracker-bar-fill" style={{ width: `${budgetPct}%`, background: velocitySlowing ? "var(--color-yellow, #eab308)" : "var(--color-primary)" }} />
                      </div>
                      {velocitySlowing && (
                        <div className="cost-tracker-warn">
                          <AlertTriangle size={12} /> Improvement velocity is slowing — consider changing strategy
                        </div>
                      )}
                    </div>
                  </div>
                );
              })()}

              {/* Convergence Confidence Band */}
              {convergenceData.length >= 10 && (() => {
                const W = 460, H = 140, padL = 50, padR = 12, padT = 10, padB = 28;
                const plotW = W - padL - padR;
                const plotH = H - padT - padB;
                const bestVals = convergenceData.map(d => d.best);
                const trialVals = convergenceData.map(d => d.value);
                const minY = Math.min(...bestVals);
                const maxY = Math.max(...trialVals, ...bestVals);
                const yRange = maxY - minY || 1;
                const scaleX = (i: number) => padL + (i / (convergenceData.length - 1)) * plotW;
                const scaleY = (v: number) => padT + (1 - (v - minY) / yRange) * plotH;

                // Rolling window confidence band (std dev around best-so-far)
                const windowSize = Math.max(5, Math.floor(convergenceData.length / 15));
                const upper: string[] = [];
                const lower: string[] = [];
                for (let i = 0; i < convergenceData.length; i++) {
                  const start = Math.max(0, i - windowSize);
                  const end = Math.min(convergenceData.length, i + windowSize + 1);
                  const windowVals = trialVals.slice(start, end);
                  const mean = windowVals.reduce((a, b) => a + b, 0) / windowVals.length;
                  const variance = windowVals.reduce((a, b) => a + (b - mean) ** 2, 0) / windowVals.length;
                  const std = Math.sqrt(variance);
                  const best = bestVals[i];
                  upper.push(`${scaleX(i).toFixed(1)},${scaleY(best + std).toFixed(1)}`);
                  lower.push(`${scaleX(i).toFixed(1)},${scaleY(best - std).toFixed(1)}`);
                }
                const bandPath = `M${upper.join(" L")} L${lower.reverse().join(" L")} Z`;
                const bestPath = bestVals.map((v, i) => `${i === 0 ? "M" : "L"}${scaleX(i).toFixed(1)},${scaleY(v).toFixed(1)}`).join(" ");

                // Confidence narrowing metric
                const earlyStd = (() => {
                  const earlyW = trialVals.slice(0, Math.min(windowSize * 3, Math.floor(convergenceData.length / 3)));
                  const m = earlyW.reduce((a, b) => a + b, 0) / earlyW.length;
                  return Math.sqrt(earlyW.reduce((a, b) => a + (b - m) ** 2, 0) / earlyW.length);
                })();
                const lateStd = (() => {
                  const lateW = trialVals.slice(Math.max(0, convergenceData.length - windowSize * 3));
                  const m = lateW.reduce((a, b) => a + b, 0) / lateW.length;
                  return Math.sqrt(lateW.reduce((a, b) => a + (b - m) ** 2, 0) / lateW.length);
                })();
                const narrowing = earlyStd > 0 ? ((earlyStd - lateStd) / earlyStd * 100).toFixed(0) : "0";
                const isConverging = lateStd < earlyStd;

                return (
                  <div className="card">
                    <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}>
                      <Eye size={16} style={{ color: "var(--color-primary)" }} />
                      <h2 style={{ margin: 0 }}>Convergence Confidence</h2>
                      <span className={`findings-badge findings-badge-${isConverging ? "success" : "warning"}`} style={{ marginLeft: "auto" }}>
                        {isConverging ? `${narrowing}% narrower` : "Widening"}
                      </span>
                    </div>
                    <p style={{ fontSize: "0.78rem", color: "var(--color-text-muted)", margin: "0 0 8px" }}>
                      Shaded band shows rolling uncertainty (±1 std dev). Narrowing bands indicate the model is converging.
                    </p>
                    <svg width={W} height={H} style={{ display: "block", width: "100%", maxWidth: `${W}px` }}>
                      {/* Grid lines */}
                      {[0, 0.25, 0.5, 0.75, 1].map(f => (
                        <line key={f} x1={padL} y1={padT + f * plotH} x2={padL + plotW} y2={padT + f * plotH}
                          stroke="var(--color-border)" strokeWidth="0.5" strokeDasharray="3,3" />
                      ))}
                      {/* Confidence band */}
                      <path d={bandPath} fill="var(--color-primary)" opacity="0.12" />
                      {/* Best-so-far line */}
                      <path d={bestPath} fill="none" stroke="var(--color-primary)" strokeWidth="2" strokeLinecap="round" />
                      {/* Trial scatter */}
                      {convergenceData.map((d, i) => (
                        <circle key={i} cx={scaleX(i)} cy={scaleY(d.value)} r="1.5" fill="var(--color-text-muted)" opacity="0.3" />
                      ))}
                      {/* Y axis labels */}
                      {[0, 0.5, 1].map(f => (
                        <text key={f} x={padL - 4} y={padT + (1 - f) * plotH + 3} textAnchor="end" fontSize="9" fontFamily="var(--font-mono)" fill="var(--color-text-muted)">
                          {(minY + f * yRange).toFixed(3)}
                        </text>
                      ))}
                      {/* X axis */}
                      <text x={padL} y={H - 4} textAnchor="start" fontSize="9" fontFamily="var(--font-mono)" fill="var(--color-text-muted)">0</text>
                      <text x={padL + plotW} y={H - 4} textAnchor="end" fontSize="9" fontFamily="var(--font-mono)" fill="var(--color-text-muted)">{convergenceData.length}</text>
                      <text x={padL + plotW / 2} y={H - 4} textAnchor="middle" fontSize="10" fill="var(--color-text-muted)">Iteration</text>
                    </svg>
                    <div className="efficiency-legend" style={{ maxWidth: `${W}px` }}>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: 14, height: 3, background: "var(--color-primary)", borderRadius: 2, marginRight: 4, verticalAlign: "middle" }} />Best So Far</span>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: 14, height: 8, background: "var(--color-primary)", opacity: 0.15, borderRadius: 2, marginRight: 4, verticalAlign: "middle" }} />±1σ Band</span>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: 5, height: 5, borderRadius: "50%", background: "var(--color-text-muted)", opacity: 0.4, marginRight: 4, verticalAlign: "middle" }} />Trials</span>
                    </div>
                  </div>
                );
              })()}

              {/* Objective Distribution by Window */}
              {trials.length >= 15 && (() => {
                const chrono = [...trials].sort((a, b) => a.iteration - b.iteration);
                const objKey = Object.keys(chrono[0].kpis)[0];
                const vals = chrono.map(t => Number(t.kpis[objKey]) || 0);
                const n = vals.length;

                // Split into 3 equal windows
                const windowSize = Math.ceil(n / 3);
                const windows = [
                  { label: "Early", sublabel: `1–${Math.min(windowSize, n)}`, vals: vals.slice(0, windowSize), color: "rgba(239,68,68,0.5)" },
                  { label: "Mid", sublabel: `${windowSize + 1}–${Math.min(windowSize * 2, n)}`, vals: vals.slice(windowSize, windowSize * 2), color: "rgba(234,179,8,0.5)" },
                  { label: "Late", sublabel: `${windowSize * 2 + 1}–${n}`, vals: vals.slice(windowSize * 2), color: "rgba(34,197,94,0.5)" },
                ].filter(w => w.vals.length > 0);

                // Compute stats per window
                const windowStats = windows.map(w => {
                  const sorted = [...w.vals].sort((a, b) => a - b);
                  return {
                    ...w,
                    min: sorted[0],
                    max: sorted[sorted.length - 1],
                    q1: sorted[Math.floor(sorted.length * 0.25)],
                    median: sorted[Math.floor(sorted.length * 0.5)],
                    q3: sorted[Math.floor(sorted.length * 0.75)],
                    mean: w.vals.reduce((a, b) => a + b, 0) / w.vals.length,
                  };
                });

                const allMin = Math.min(...windowStats.map(w => w.min));
                const allMax = Math.max(...windowStats.map(w => w.max));
                const valRange = allMax - allMin || 1;

                // Detect trend: compare early vs late median
                const earlyMed = windowStats[0]?.median ?? 0;
                const lateMed = windowStats[windowStats.length - 1]?.median ?? 0;
                const improving = lateMed < earlyMed;
                const improvePct = earlyMed !== 0 ? (((earlyMed - lateMed) / Math.abs(earlyMed)) * 100).toFixed(1) : "0";

                const W = 460, H = 140, padL = 50, padR = 12, padT = 10, padB = 28;
                const plotW = W - padL - padR;
                const plotH = H - padT - padB;
                const boxWidth = Math.min(60, plotW / (windows.length * 2));
                const scaleY = (v: number) => padT + (1 - (v - allMin) / valRange) * plotH;

                return (
                  <div className="card">
                    <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}>
                      <BoxSelect size={16} style={{ color: "var(--color-primary)" }} />
                      <h2 style={{ margin: 0 }}>Distribution by Window</h2>
                      <span className={`findings-badge findings-badge-${improving ? "success" : "warning"}`} style={{ marginLeft: "auto" }}>
                        {improving ? `${improvePct}% shift` : "Not improving"}
                      </span>
                    </div>
                    <p style={{ fontSize: "0.78rem", color: "var(--color-text-muted)", margin: "0 0 8px" }}>
                      Box plots of {objKey} for early, mid, and late trial windows. A leftward/downward shift indicates improving outcomes.
                    </p>
                    <svg width={W} height={H} style={{ display: "block", width: "100%", maxWidth: `${W}px` }}>
                      {/* Horizontal grid lines */}
                      {[0, 0.25, 0.5, 0.75, 1].map(f => (
                        <line key={f} x1={padL} y1={padT + f * plotH} x2={padL + plotW} y2={padT + f * plotH}
                          stroke="var(--color-border)" strokeWidth="0.5" strokeDasharray="3,3" />
                      ))}
                      {/* Y-axis labels */}
                      {[0, 0.5, 1].map(f => (
                        <text key={f} x={padL - 4} y={padT + (1 - f) * plotH + 3} textAnchor="end" fontSize="9" fontFamily="var(--font-mono)" fill="var(--color-text-muted)">
                          {(allMin + f * valRange).toFixed(3)}
                        </text>
                      ))}
                      {/* Box plots */}
                      {windowStats.map((ws, wi) => {
                        const cx = padL + ((wi + 0.5) / windows.length) * plotW;
                        const x1 = cx - boxWidth / 2;
                        return (
                          <g key={wi}>
                            {/* Whisker: min to max */}
                            <line x1={cx} y1={scaleY(ws.min)} x2={cx} y2={scaleY(ws.max)} stroke={ws.color.replace("0.5", "0.8")} strokeWidth="1" />
                            {/* Min and max caps */}
                            <line x1={cx - 8} y1={scaleY(ws.min)} x2={cx + 8} y2={scaleY(ws.min)} stroke={ws.color.replace("0.5", "0.8")} strokeWidth="1" />
                            <line x1={cx - 8} y1={scaleY(ws.max)} x2={cx + 8} y2={scaleY(ws.max)} stroke={ws.color.replace("0.5", "0.8")} strokeWidth="1" />
                            {/* Box: Q1 to Q3 */}
                            <rect x={x1} y={scaleY(ws.q3)} width={boxWidth} height={Math.max(1, scaleY(ws.q1) - scaleY(ws.q3))}
                              fill={ws.color} stroke={ws.color.replace("0.5", "0.9")} strokeWidth="1" rx="2" />
                            {/* Median line */}
                            <line x1={x1} y1={scaleY(ws.median)} x2={x1 + boxWidth} y2={scaleY(ws.median)}
                              stroke={ws.color.replace("0.5", "1")} strokeWidth="2" />
                            {/* Label */}
                            <text x={cx} y={H - 8} textAnchor="middle" fontSize="10" fontWeight="500" fill="var(--color-text-secondary)">
                              {ws.label}
                            </text>
                            <text x={cx} y={H - 0} textAnchor="middle" fontSize="8" fill="var(--color-text-muted)" fontFamily="var(--font-mono)">
                              {ws.sublabel}
                            </text>
                          </g>
                        );
                      })}
                    </svg>
                    <div className="efficiency-legend" style={{ maxWidth: `${W}px` }}>
                      {windowStats.map((ws, i) => (
                        <span key={i} className="efficiency-legend-item" style={{ fontFamily: "var(--font-mono)", fontSize: "0.72rem" }}>
                          <span style={{ display: "inline-block", width: 10, height: 10, borderRadius: 2, background: ws.color, marginRight: 4, verticalAlign: "middle" }} />
                          {ws.label}: med={ws.median.toFixed(4)}
                        </span>
                      ))}
                    </div>
                  </div>
                );
              })()}

              {/* Auto-generated Findings Summary */}
              {trials.length >= 5 && (() => {
                const findings: Array<{ icon: string; text: string; type: "success" | "info" | "warning" }> = [];
                const objKey = Object.keys(trials[0].kpis)[0];
                const objVals = trials.map(t => Number(t.kpis[objKey]) || 0);
                const bestVal = Math.min(...objVals);
                const worstVal = Math.max(...objVals);
                const meanVal = objVals.reduce((a, b) => a + b, 0) / objVals.length;
                const chrono = [...trials].sort((a, b) => a.iteration - b.iteration);
                const bestTrial = chrono.find(t => Number(t.kpis[objKey]) === bestVal);

                // Finding 1: Best result
                if (bestTrial) {
                  findings.push({ icon: "trophy", text: `Best result of ${bestVal.toFixed(4)} found at iteration ${bestTrial.iteration} out of ${trials.length} trials.`, type: "success" });
                }

                // Finding 2: Parameter importance
                if (importance && importance.importances.length > 0) {
                  const sorted = [...importance.importances].sort((a, b) => b.importance - a.importance);
                  const top = sorted[0];
                  const topPct = (top.importance * 100).toFixed(1);
                  findings.push({ icon: "target", text: `${top.name} is the most influential parameter (${topPct}% importance).`, type: "info" });
                  if (sorted.length > 1) {
                    const bottom = sorted[sorted.length - 1];
                    if (bottom.importance < 0.05) {
                      findings.push({ icon: "info", text: `${bottom.name} has minimal impact (${(bottom.importance * 100).toFixed(1)}%) — consider fixing it to reduce dimensionality.`, type: "warning" });
                    }
                  }
                }

                // Finding 3: Plateau detection
                const recent20 = chrono.slice(-20);
                if (recent20.length >= 10) {
                  const recentBests = recent20.map(t => Number(t.kpis[objKey]) || 0);
                  const recentBest = Math.min(...recentBests);
                  const recentWorst = Math.max(...recentBests);
                  const recentRange = Math.abs(recentWorst - recentBest);
                  const totalRange = Math.abs(worstVal - bestVal);
                  if (totalRange > 0 && recentRange / totalRange < 0.05) {
                    findings.push({ icon: "pause", text: `Optimization has plateaued — the last ${recent20.length} trials show <5% variation. Consider exploring new regions.`, type: "warning" });
                  }
                }

                // Finding 4: Improvement rate
                if (chrono.length >= 10) {
                  const firstHalf = chrono.slice(0, Math.floor(chrono.length / 2));
                  const secondHalf = chrono.slice(Math.floor(chrono.length / 2));
                  const firstBest = Math.min(...firstHalf.map(t => Number(t.kpis[objKey]) || 0));
                  const secondBest = Math.min(...secondHalf.map(t => Number(t.kpis[objKey]) || 0));
                  if (secondBest < firstBest) {
                    const improvPct = ((firstBest - secondBest) / Math.abs(firstBest) * 100).toFixed(1);
                    findings.push({ icon: "trending", text: `Second half of trials improved ${improvPct}% over the first half — optimization is progressing well.`, type: "success" });
                  }
                }

                // Finding 5: Spread
                const cv = meanVal !== 0 ? (Math.sqrt(objVals.reduce((a, v) => a + (v - meanVal) ** 2, 0) / objVals.length) / Math.abs(meanVal)) : 0;
                if (cv > 0.5) {
                  findings.push({ icon: "scatter", text: `High variability (CV=${(cv * 100).toFixed(0)}%) in objective values suggests the search space has diverse outcomes.`, type: "info" });
                }

                if (findings.length === 0) return null;

                const iconMap: Record<string, React.ReactNode> = {
                  trophy: <Trophy size={14} />,
                  target: <Target size={14} />,
                  info: <Info size={14} />,
                  pause: <Pause size={14} />,
                  trending: <TrendingDown size={14} />,
                  scatter: <Activity size={14} />,
                };

                return (
                  <div className="card findings-card">
                    <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "12px" }}>
                      <Brain size={16} style={{ color: "var(--color-primary)" }} />
                      <h2 style={{ margin: 0 }}>Key Findings</h2>
                      <span className="findings-badge">{findings.length}</span>
                    </div>
                    <div className="findings-list">
                      {findings.map((f, i) => (
                        <div key={i} className={`findings-item findings-${f.type}`}>
                          <span className="findings-icon">{iconMap[f.icon] || <Info size={14} />}</span>
                          <span className="findings-text">{f.text}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                );
              })()}

              {/* Diminishing Returns Detector */}
              {trials.length >= 15 && (() => {
                const chrono = [...trials].sort((a, b) => a.iteration - b.iteration);
                const objKey = Object.keys(chrono[0].kpis)[0];
                const vals = chrono.map(t => Number(t.kpis[objKey]) || 0);

                // Compute cumulative best at each point
                const cumBest: number[] = [];
                let runBest = Infinity;
                for (const v of vals) {
                  if (v < runBest) runBest = v;
                  cumBest.push(runBest);
                }

                // Marginal improvement in windows of different sizes
                const windows = [10, 25, 50].filter(w => w <= chrono.length);
                const marginalData: Array<{ window: number; improvements: Array<{ x: number; y: number }> }> = [];
                for (const w of windows) {
                  const improvements: Array<{ x: number; y: number }> = [];
                  for (let i = w; i < cumBest.length; i += Math.max(1, Math.floor(w / 3))) {
                    const improvementInWindow = cumBest[i - w] - cumBest[i]; // positive = improved
                    improvements.push({ x: i, y: improvementInWindow });
                  }
                  marginalData.push({ window: w, improvements });
                }

                const allY = marginalData.flatMap(d => d.improvements.map(p => p.y));
                const maxImprovement = Math.max(...allY, 0.001);
                const totalImprovement = cumBest[0] - cumBest[cumBest.length - 1];
                const recentImprovement = cumBest.length > 20 ? cumBest[cumBest.length - 20] - cumBest[cumBest.length - 1] : 0;
                const recentPct = totalImprovement > 0 ? (recentImprovement / totalImprovement * 100).toFixed(0) : "0";
                const diminishing = totalImprovement > 0 && recentImprovement < totalImprovement * 0.05;

                const W = 460, H = 130, padL = 50, padR = 12, padT = 10, padB = 28;
                const plotW = W - padL - padR;
                const plotH = H - padT - padB;
                const scaleX = (x: number) => padL + (x / (chrono.length - 1)) * plotW;
                const scaleY = (y: number) => padT + (1 - y / maxImprovement) * plotH;

                const colors = ["var(--color-primary)", "var(--color-blue, #3b82f6)", "var(--color-text-muted)"];
                const dashes = ["", "4,3", "2,2"];

                return (
                  <div className="card">
                    <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}>
                      <PieChart size={16} style={{ color: "var(--color-primary)" }} />
                      <h2 style={{ margin: 0 }}>Diminishing Returns</h2>
                      <span className={`findings-badge findings-badge-${diminishing ? "warning" : "success"}`} style={{ marginLeft: "auto" }}>
                        {diminishing ? "Low marginal gain" : `${recentPct}% recent`}
                      </span>
                    </div>
                    <p style={{ fontSize: "0.78rem", color: "var(--color-text-muted)", margin: "0 0 8px" }}>
                      Marginal improvement per window of trials. Declining curves suggest diminishing returns from additional experiments.
                    </p>
                    <svg width={W} height={H} style={{ display: "block", width: "100%", maxWidth: `${W}px` }}>
                      {/* Grid lines */}
                      {[0, 0.25, 0.5, 0.75, 1].map(f => (
                        <line key={f} x1={padL} y1={padT + f * plotH} x2={padL + plotW} y2={padT + f * plotH}
                          stroke="var(--color-border)" strokeWidth="0.5" strokeDasharray="3,3" />
                      ))}
                      {/* Lines for each window */}
                      {marginalData.map((d, di) => {
                        if (d.improvements.length < 2) return null;
                        const path = d.improvements.map((p, i) => `${i === 0 ? "M" : "L"}${scaleX(p.x).toFixed(1)},${scaleY(p.y).toFixed(1)}`).join(" ");
                        return (
                          <path key={di} d={path} fill="none" stroke={colors[di]} strokeWidth="1.8" strokeLinecap="round"
                            strokeDasharray={dashes[di]} opacity="0.8" />
                        );
                      })}
                      {/* Zero line */}
                      <line x1={padL} y1={scaleY(0)} x2={padL + plotW} y2={scaleY(0)} stroke="var(--color-text-muted)" strokeWidth="0.5" opacity="0.3" />
                      {/* Y axis labels */}
                      {[0, 0.5, 1].map(f => (
                        <text key={f} x={padL - 4} y={padT + (1 - f) * plotH + 3} textAnchor="end" fontSize="9" fontFamily="var(--font-mono)" fill="var(--color-text-muted)">
                          {(f * maxImprovement).toFixed(3)}
                        </text>
                      ))}
                      {/* X axis */}
                      <text x={padL} y={H - 4} textAnchor="start" fontSize="9" fontFamily="var(--font-mono)" fill="var(--color-text-muted)">0</text>
                      <text x={padL + plotW} y={H - 4} textAnchor="end" fontSize="9" fontFamily="var(--font-mono)" fill="var(--color-text-muted)">{chrono.length}</text>
                      <text x={padL + plotW / 2} y={H - 4} textAnchor="middle" fontSize="10" fill="var(--color-text-muted)">Iteration</text>
                    </svg>
                    <div className="efficiency-legend" style={{ maxWidth: `${W}px` }}>
                      {marginalData.map((d, di) => (
                        <span key={di} className="efficiency-legend-item">
                          <span style={{ display: "inline-block", width: 14, height: 2, background: colors[di], borderRadius: 1, marginRight: 4, verticalAlign: "middle" }} />
                          Last {d.window} trials
                        </span>
                      ))}
                    </div>
                  </div>
                );
              })()}

              {/* Cost-Adjusted Stopping Signal */}
              {trials.length >= 15 && (() => {
                const chrono = [...trials].sort((a, b) => a.iteration - b.iteration);
                const objKey = Object.keys(chrono[0].kpis)[0];
                const vals = chrono.map(t => Number(t.kpis[objKey]) || 0);

                // Cumulative best
                const cumBest: number[] = [];
                let rb = Infinity;
                for (const v of vals) { if (v < rb) rb = v; cumBest.push(rb); }

                // Cost-adjusted regret: improvement per unit cost (iteration)
                const totalImprovement = cumBest[0] - cumBest[cumBest.length - 1];
                const regretSeries: Array<{ iter: number; regret: number; marginal: number }> = [];
                for (let i = 1; i < cumBest.length; i++) {
                  const improvement = cumBest[0] - cumBest[i];
                  const regret = totalImprovement > 0 ? improvement / totalImprovement : 0; // normalized 0-1
                  const marginalWindow = Math.min(10, i);
                  const marginal = i >= marginalWindow ? (cumBest[i - marginalWindow] - cumBest[i]) / marginalWindow : 0;
                  regretSeries.push({ iter: i, regret, marginal });
                }

                // Stopping signal: compute efficiency ratio
                const recentMarginal = regretSeries.slice(-10).reduce((a, r) => a + r.marginal, 0) / 10;
                const avgMarginal = totalImprovement / chrono.length;
                const efficiencyRatio = avgMarginal > 0 ? recentMarginal / avgMarginal : 0;

                // Zone classification
                let zone: "green" | "yellow" | "red";
                let zoneLabel: string;
                if (efficiencyRatio > 0.3) { zone = "green"; zoneLabel = "Continue"; }
                else if (efficiencyRatio > 0.05) { zone = "yellow"; zoneLabel = "Caution"; }
                else { zone = "red"; zoneLabel = "Consider stopping"; }

                // Project future: exponential decay of marginal improvement
                const projectedIters = 30;
                const projections: Array<{ iter: number; regret: number }> = [];
                const lastRegret = regretSeries[regretSeries.length - 1]?.regret ?? 0;
                const decayRate = efficiencyRatio > 0 ? Math.min(0.95, 1 - efficiencyRatio * 0.1) : 0.98;
                let projMarginal = recentMarginal;
                let projRegret = lastRegret;
                for (let i = 1; i <= projectedIters; i++) {
                  projMarginal *= decayRate;
                  const projImprovement = projMarginal * (totalImprovement > 0 ? chrono.length / totalImprovement : 0);
                  projRegret = Math.min(1, projRegret + projImprovement * 0.01);
                  projections.push({ iter: chrono.length + i, regret: projRegret });
                }

                const W = 460, H = 140, padL = 50, padR = 12, padT = 10, padB = 28;
                const plotW = W - padL - padR;
                const plotH = H - padT - padB;
                const totalIters = chrono.length + projectedIters;
                const scaleX = (x: number) => padL + (x / totalIters) * plotW;
                const scaleY = (y: number) => padT + (1 - y) * plotH;

                const zoneColors = { green: "#22c55e", yellow: "#eab308", red: "#ef4444" };
                const regretPath = regretSeries.map((r, i) => `${i === 0 ? "M" : "L"}${scaleX(r.iter).toFixed(1)},${scaleY(r.regret).toFixed(1)}`).join(" ");
                const projPath = projections.length > 0
                  ? `M${scaleX(chrono.length).toFixed(1)},${scaleY(lastRegret).toFixed(1)} ${projections.map(p => `L${scaleX(p.iter).toFixed(1)},${scaleY(p.regret).toFixed(1)}`).join(" ")}`
                  : "";

                return (
                  <div className="card">
                    <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}>
                      <Flag size={16} style={{ color: zoneColors[zone] }} />
                      <h2 style={{ margin: 0 }}>Stopping Signal</h2>
                      <span className={`findings-badge findings-badge-${zone === "green" ? "success" : zone === "yellow" ? "warning" : "warning"}`} style={{ marginLeft: "auto", background: zone === "red" ? "rgba(239,68,68,0.1)" : undefined, color: zone === "red" ? "#ef4444" : undefined }}>
                        {zoneLabel}
                      </span>
                    </div>
                    <p style={{ fontSize: "0.78rem", color: "var(--color-text-muted)", margin: "0 0 8px" }}>
                      Normalized improvement progress over iterations. Flat curve = diminishing returns. Dashed line = projected trajectory.
                    </p>
                    <svg width={W} height={H} style={{ display: "block", width: "100%", maxWidth: `${W}px` }}>
                      {/* Zone background */}
                      <rect x={padL} y={padT} width={plotW} height={plotH * 0.3} fill="rgba(34,197,94,0.05)" />
                      <rect x={padL} y={padT + plotH * 0.3} width={plotW} height={plotH * 0.4} fill="rgba(234,179,8,0.05)" />
                      <rect x={padL} y={padT + plotH * 0.7} width={plotW} height={plotH * 0.3} fill="rgba(239,68,68,0.05)" />
                      {/* Grid */}
                      {[0, 0.25, 0.5, 0.75, 1].map(f => (
                        <line key={f} x1={padL} y1={padT + f * plotH} x2={padL + plotW} y2={padT + f * plotH}
                          stroke="var(--color-border)" strokeWidth="0.5" strokeDasharray="3,3" />
                      ))}
                      {/* Now boundary */}
                      <line x1={scaleX(chrono.length)} y1={padT} x2={scaleX(chrono.length)} y2={padT + plotH}
                        stroke="var(--color-text-muted)" strokeWidth="1" strokeDasharray="4,3" opacity="0.5" />
                      <text x={scaleX(chrono.length)} y={padT - 2} textAnchor="middle" fontSize="8" fill="var(--color-text-muted)">Now</text>
                      {/* Regret curve */}
                      <path d={regretPath} fill="none" stroke={zoneColors[zone]} strokeWidth="2" strokeLinecap="round" />
                      {/* Projection */}
                      {projPath && <path d={projPath} fill="none" stroke={zoneColors[zone]} strokeWidth="1.5" strokeDasharray="4,3" opacity="0.5" />}
                      {/* Y-axis */}
                      {[0, 0.5, 1].map(f => (
                        <text key={f} x={padL - 4} y={scaleY(f) + 3} textAnchor="end" fontSize="9" fontFamily="var(--font-mono)" fill="var(--color-text-muted)">
                          {(f * 100).toFixed(0)}%
                        </text>
                      ))}
                      {/* X-axis */}
                      <text x={padL} y={H - 4} textAnchor="start" fontSize="9" fontFamily="var(--font-mono)" fill="var(--color-text-muted)">0</text>
                      <text x={padL + plotW} y={H - 4} textAnchor="end" fontSize="9" fontFamily="var(--font-mono)" fill="var(--color-text-muted)">{totalIters}</text>
                      <text x={padL + plotW / 2} y={H - 4} textAnchor="middle" fontSize="10" fill="var(--color-text-muted)">Iteration</text>
                    </svg>
                    <div className="efficiency-legend" style={{ maxWidth: `${W}px` }}>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: 14, height: 3, background: zoneColors[zone], borderRadius: 1, marginRight: 4, verticalAlign: "middle" }} />Improvement</span>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: 14, height: 0, borderTop: `2px dashed ${zoneColors[zone]}`, opacity: 0.5, marginRight: 4, verticalAlign: "middle" }} />Projected</span>
                      <span className="efficiency-legend-item" style={{ fontFamily: "var(--font-mono)" }}>Efficiency: {(efficiencyRatio * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                );
              })()}

              {/* Cumulative Regret Curve */}
              {trials.length >= 10 && (() => {
                const objKey = Object.keys(trials[0].kpis)[0];
                const vals = trials.map(t => Number(t.kpis[objKey]) || 0);
                // Assume minimize: regret = value - bestPossible; use running best as proxy
                const globalBest = Math.min(...vals);
                // cumulative regret for BO
                let cumRegret = 0;
                const boRegret = vals.map(v => { cumRegret += Math.max(0, v - globalBest); return cumRegret; });
                // random baseline: average regret per trial * iteration
                const avgVal = vals.reduce((a, b) => a + b, 0) / vals.length;
                const avgRegretPerTrial = Math.max(0, avgVal - globalBest);
                const randomRegret = vals.map((_, i) => avgRegretPerTrial * (i + 1));

                const maxR = Math.max(boRegret[boRegret.length - 1], randomRegret[randomRegret.length - 1], 1);
                const W = 400, H = 180, padL = 52, padR = 16, padT = 28, padB = 32;
                const plotW = W - padL - padR, plotH = H - padT - padB;
                const n = vals.length;

                const toX = (i: number) => padL + (i / (n - 1)) * plotW;
                const toY = (v: number) => padT + (1 - v / maxR) * plotH;

                const boPath = boRegret.map((v, i) => `${i === 0 ? "M" : "L"}${toX(i).toFixed(1)},${toY(v).toFixed(1)}`).join(" ");
                const randPath = randomRegret.map((v, i) => `${i === 0 ? "M" : "L"}${toX(i).toFixed(1)},${toY(v).toFixed(1)}`).join(" ");

                // Find divergence point: where BO regret drops significantly below random
                let divergeIdx = -1;
                for (let i = 3; i < n; i++) {
                  if (randomRegret[i] > 0 && boRegret[i] / randomRegret[i] < 0.7) { divergeIdx = i; break; }
                }

                const savings = maxR > 0 ? ((1 - boRegret[n - 1] / randomRegret[n - 1]) * 100) : 0;

                return (
                  <div className="card">
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "4px" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                        <TrendingDown size={16} style={{ color: "var(--color-primary)" }} />
                        <h2 style={{ margin: 0 }}>Cumulative Regret</h2>
                      </div>
                      <span className="findings-badge" style={{ background: savings > 30 ? "var(--color-green-bg)" : savings > 10 ? "var(--color-yellow-bg)" : "var(--color-red-bg)", color: savings > 30 ? "var(--color-green)" : savings > 10 ? "var(--color-yellow)" : "var(--color-red)" }}>
                        {savings > 0 ? `${savings.toFixed(0)}% less regret` : "No improvement"}
                      </span>
                    </div>
                    <p style={{ fontSize: "0.78rem", color: "var(--color-text-muted)", margin: "0 0 8px" }}>
                      Cumulative distance from optimal. Lower is better. Dashed = random baseline.
                    </p>
                    <svg width={W} height={H} style={{ display: "block", width: "100%", maxWidth: `${W}px` }}>
                      {/* Y-axis labels */}
                      {[0, 0.25, 0.5, 0.75, 1].map(f => (
                        <Fragment key={f}>
                          <line x1={padL} y1={padT + (1 - f) * plotH} x2={padL + plotW} y2={padT + (1 - f) * plotH} stroke="var(--color-border)" strokeWidth="0.5" strokeDasharray="2,3" />
                          <text x={padL - 6} y={padT + (1 - f) * plotH + 3} textAnchor="end" fontSize="9" fill="var(--color-text-muted)" fontFamily="var(--font-mono)">
                            {(f * maxR).toFixed(1)}
                          </text>
                        </Fragment>
                      ))}
                      {/* X-axis label */}
                      <text x={padL + plotW / 2} y={H - 4} textAnchor="middle" fontSize="9" fill="var(--color-text-muted)">Trials</text>
                      {/* Random baseline (dashed) */}
                      <path d={randPath} fill="none" stroke="var(--color-text-muted)" strokeWidth="1.5" strokeDasharray="5,3" opacity="0.6" />
                      {/* BO regret (solid) */}
                      <path d={boPath} fill="none" stroke="var(--color-primary)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                      {/* Fill between random and BO */}
                      <path
                        d={`${boPath} L${toX(n - 1).toFixed(1)},${toY(randomRegret[n - 1]).toFixed(1)} ${randomRegret.slice().reverse().map((v, i) => `L${toX(n - 1 - i).toFixed(1)},${toY(v).toFixed(1)}`).join(" ")} Z`}
                        fill="var(--color-primary)"
                        opacity="0.07"
                      />
                      {/* Divergence marker */}
                      {divergeIdx >= 0 && (
                        <>
                          <line x1={toX(divergeIdx)} y1={padT} x2={toX(divergeIdx)} y2={padT + plotH} stroke="var(--color-green)" strokeWidth="1" strokeDasharray="3,2" opacity="0.6" />
                          <text x={toX(divergeIdx) + 4} y={padT + 10} fontSize="8" fill="var(--color-green)" fontFamily="var(--font-mono)">diverge</text>
                        </>
                      )}
                      {/* End markers */}
                      <circle cx={toX(n - 1)} cy={toY(boRegret[n - 1])} r="3" fill="var(--color-primary)" />
                      <circle cx={toX(n - 1)} cy={toY(randomRegret[n - 1])} r="3" fill="var(--color-text-muted)" opacity="0.6" />
                    </svg>
                    {/* Legend */}
                    <div style={{ display: "flex", gap: "16px", marginTop: "4px" }}>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: 14, height: 2, background: "var(--color-primary)", marginRight: 4, verticalAlign: "middle" }} />BO</span>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: 14, height: 2, background: "var(--color-text-muted)", marginRight: 4, verticalAlign: "middle", borderTop: "1px dashed var(--color-text-muted)" }} />Random</span>
                    </div>
                  </div>
                );
              })()}

              {/* Dominated Region / Best-so-far Area */}
              {trials.length >= 10 && (() => {
                const drObjKey = Object.keys(trials[0].kpis)[0];
                const drVals = trials.map(t => Number(t.kpis[drObjKey]) || 0);
                let drBest = drVals[0];
                const drBestLine = drVals.map(v => { drBest = Math.min(drBest, v); return drBest; });
                const drMin = Math.min(...drVals), drMax = Math.max(...drVals);
                const drRange = drMax - drMin || 1;

                const W = 400, H = 160, padL = 52, padR = 16, padT = 24, padB = 28;
                const plotW = W - padL - padR, plotH = H - padT - padB;
                const n = drVals.length;
                const toX = (i: number) => padL + (i / (n - 1)) * plotW;
                const toY = (v: number) => padT + (1 - (v - drMin) / drRange) * plotH;

                // Best-so-far line path
                const bsfPath = drBestLine.map((v, i) => `${i === 0 ? "M" : "L"}${toX(i).toFixed(1)},${toY(v).toFixed(1)}`).join(" ");
                // Area under best-so-far (dominated region)
                const areaPath = `${bsfPath} L${toX(n - 1).toFixed(1)},${(padT + plotH).toFixed(1)} L${padL},${(padT + plotH).toFixed(1)} Z`;

                // Dominated area ratio (how much of the objective range is dominated)
                const dominatedPct = drRange > 0 ? ((drMax - drBestLine[n - 1]) / drRange * 100) : 0;

                // Milestones: when best improved
                const milestones: { idx: number; val: number }[] = [];
                for (let i = 1; i < n; i++) {
                  if (drBestLine[i] < drBestLine[i - 1]) milestones.push({ idx: i, val: drBestLine[i] });
                }

                return (
                  <div className="card">
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "4px" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                        <Flag size={16} style={{ color: "var(--color-primary)" }} />
                        <h2 style={{ margin: 0 }}>Dominated Region</h2>
                      </div>
                      <span className="findings-badge" style={{ background: "var(--color-green-bg)", color: "var(--color-green)" }}>
                        {dominatedPct.toFixed(0)}% dominated
                      </span>
                    </div>
                    <p style={{ fontSize: "0.78rem", color: "var(--color-text-muted)", margin: "0 0 8px" }}>
                      Shaded area shows the dominated region below worst outcome. More shading = more progress.
                    </p>
                    <svg width={W} height={H} style={{ display: "block", width: "100%", maxWidth: `${W}px` }}>
                      {/* Y-axis */}
                      {[0, 0.5, 1].map(f => (
                        <Fragment key={f}>
                          <line x1={padL} y1={padT + (1 - f) * plotH} x2={padL + plotW} y2={padT + (1 - f) * plotH} stroke="var(--color-border)" strokeWidth="0.5" strokeDasharray="2,3" />
                          <text x={padL - 6} y={padT + (1 - f) * plotH + 3} textAnchor="end" fontSize="8" fill="var(--color-text-muted)" fontFamily="var(--font-mono)">
                            {(drMin + f * drRange).toFixed(3)}
                          </text>
                        </Fragment>
                      ))}
                      <text x={padL + plotW / 2} y={H - 4} textAnchor="middle" fontSize="9" fill="var(--color-text-muted)">Trials</text>
                      {/* Dominated area fill */}
                      <path d={areaPath} fill="var(--color-primary)" opacity="0.12" />
                      {/* Best-so-far line */}
                      <path d={bsfPath} fill="none" stroke="var(--color-primary)" strokeWidth="2" strokeLinecap="round" />
                      {/* Trial dots */}
                      {drVals.map((v, i) => (
                        <circle key={i} cx={toX(i)} cy={toY(v)} r="1.5" fill="var(--color-text-muted)" opacity="0.3" />
                      ))}
                      {/* Milestone markers */}
                      {milestones.slice(-5).map((m, i) => (
                        <Fragment key={i}>
                          <circle cx={toX(m.idx)} cy={toY(m.val)} r="3.5" fill="var(--color-primary)" stroke="white" strokeWidth="1" />
                        </Fragment>
                      ))}
                    </svg>
                    <div style={{ display: "flex", gap: "16px", marginTop: "4px", fontSize: "0.75rem", color: "var(--color-text-muted)" }}>
                      <span>{milestones.length} improvement{milestones.length !== 1 ? "s" : ""}</span>
                      <span>Best at trial #{milestones.length > 0 ? milestones[milestones.length - 1].idx : 0}</span>
                    </div>
                  </div>
                );
              })()}

              {/* Sample Efficiency Tracker */}
              {trials.length >= 15 && (() => {
                const seObjKey = Object.keys(trials[0].kpis)[0];
                const seVals = trials.map(t => Number(t.kpis[seObjKey]) || 0);
                // Best-so-far
                let seBest = seVals[0];
                const seBsf = seVals.map(v => { seBest = Math.min(seBest, v); return seBest; });
                // Improvement per window
                const seWinSize = Math.max(5, Math.floor(trials.length / 10));
                const seWindows: { start: number; end: number; improvement: number; cumImprove: number }[] = [];
                let seCumImprove = 0;
                for (let i = 0; i < trials.length; i += seWinSize) {
                  const end = Math.min(i + seWinSize - 1, trials.length - 1);
                  const winImprove = Math.abs(seBsf[end] - seBsf[i]);
                  seCumImprove += winImprove;
                  seWindows.push({ start: i, end, improvement: winImprove, cumImprove: seCumImprove });
                }
                if (seWindows.length < 2) return null;

                const seMaxImprove = Math.max(...seWindows.map(w => w.improvement), 0.001);
                const seMaxCum = seWindows[seWindows.length - 1].cumImprove || 1;

                const seW = 440, seH = 160, sePadL = 44, sePadR = 44, sePadT = 16, sePadB = 28;
                const sePlotW = seW - sePadL - sePadR;
                const sePlotH = seH - sePadT - sePadB;
                const barW = Math.max(8, sePlotW / seWindows.length - 4);

                // ROI threshold: average improvement
                const seAvgImprove = seWindows.reduce((s, w) => s + w.improvement, 0) / seWindows.length;
                const roiY = sePadT + sePlotH - (seAvgImprove / seMaxImprove) * sePlotH;

                // Recent efficiency ratio
                const lastThird = seWindows.slice(-Math.ceil(seWindows.length / 3));
                const firstThird = seWindows.slice(0, Math.ceil(seWindows.length / 3));
                const recentAvg = lastThird.reduce((s, w) => s + w.improvement, 0) / lastThird.length;
                const earlyAvg = firstThird.reduce((s, w) => s + w.improvement, 0) / firstThird.length;
                const efficiencyRatio = earlyAvg > 0 ? recentAvg / earlyAvg : 0;

                // Cumulative improvement line
                const cumPath = seWindows.map((w, i) => {
                  const x = sePadL + (i + 0.5) / seWindows.length * sePlotW;
                  const y = sePadT + sePlotH - (w.cumImprove / seMaxCum) * sePlotH;
                  return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
                }).join(" ");

                return (
                  <div className="card" style={{ marginBottom: 18 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
                      <Zap size={18} style={{ color: "var(--color-text-muted)" }} />
                      <div>
                        <h2 style={{ margin: 0 }}>Sample Efficiency</h2>
                        <p className="stat-label" style={{ margin: 0, textTransform: "none" }}>
                          Improvement per {seWinSize}-trial window. Declining bars = diminishing returns.
                        </p>
                      </div>
                      <span className="findings-badge" style={{ marginLeft: "auto", color: efficiencyRatio < 0.3 ? "var(--color-red, #ef4444)" : efficiencyRatio < 0.7 ? "var(--color-yellow, #eab308)" : "var(--color-green, #22c55e)" }}>
                        {(efficiencyRatio * 100).toFixed(0)}% recent efficiency
                      </span>
                    </div>
                    <div style={{ overflowX: "auto" }}>
                      <svg width={seW} height={seH} viewBox={`0 0 ${seW} ${seH}`} style={{ display: "block", margin: "0 auto" }}>
                        {/* Grid */}
                        {[0, 0.25, 0.5, 0.75, 1].map(f => (
                          <line key={f} x1={sePadL} y1={sePadT + (1 - f) * sePlotH} x2={sePadL + sePlotW} y2={sePadT + (1 - f) * sePlotH} stroke="var(--color-border)" strokeWidth={0.5} />
                        ))}
                        {/* Bars (improvement per window) */}
                        {seWindows.map((w, i) => {
                          const x = sePadL + (i + 0.5) / seWindows.length * sePlotW - barW / 2;
                          const h = (w.improvement / seMaxImprove) * sePlotH;
                          const barColor = w.improvement > seAvgImprove ? "rgba(34,197,94,0.5)" : "rgba(234,179,8,0.4)";
                          return (
                            <rect key={i} x={x} y={sePadT + sePlotH - h} width={barW} height={Math.max(h, 1)} fill={barColor} rx={2}>
                              <title>Window {i + 1}: trials {w.start}–{w.end}, improvement {w.improvement.toPrecision(3)}</title>
                            </rect>
                          );
                        })}
                        {/* ROI threshold line */}
                        <line x1={sePadL} y1={roiY} x2={sePadL + sePlotW} y2={roiY} stroke="var(--color-text-muted)" strokeWidth={1} strokeDasharray="4,3" opacity={0.5} />
                        <text x={sePadL + sePlotW + 3} y={roiY + 3} fontSize="7" fill="var(--color-text-muted)" fontFamily="var(--font-mono)">avg</text>
                        {/* Cumulative improvement line (right axis) */}
                        <path d={cumPath} fill="none" stroke="rgba(99,102,241,0.7)" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />
                        {/* Dots on cumulative line */}
                        {seWindows.map((w, i) => {
                          const x = sePadL + (i + 0.5) / seWindows.length * sePlotW;
                          const y = sePadT + sePlotH - (w.cumImprove / seMaxCum) * sePlotH;
                          return <circle key={i} cx={x} cy={y} r={2.5} fill="rgba(99,102,241,0.8)" />;
                        })}
                        {/* Left Y axis label */}
                        <text x={8} y={sePadT + sePlotH / 2} textAnchor="middle" fontSize="9" fill="var(--color-text-muted)" fontFamily="var(--font-mono)" transform={`rotate(-90,8,${sePadT + sePlotH / 2})`}>
                          Δ per window
                        </text>
                        {/* Right Y axis label */}
                        <text x={seW - 8} y={sePadT + sePlotH / 2} textAnchor="middle" fontSize="9" fill="rgba(99,102,241,0.7)" fontFamily="var(--font-mono)" transform={`rotate(90,${seW - 8},${sePadT + sePlotH / 2})`}>
                          Cumulative
                        </text>
                        {/* X axis */}
                        <text x={sePadL + sePlotW / 2} y={seH - 2} textAnchor="middle" fontSize="10" fill="var(--color-text-muted)" fontFamily="var(--font-mono)">
                          Trial Window
                        </text>
                      </svg>
                    </div>
                    <div style={{ display: "flex", gap: "16px", marginTop: "6px", flexWrap: "wrap" }}>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: 10, height: 10, borderRadius: 2, background: "rgba(34,197,94,0.5)", marginRight: 4, verticalAlign: "middle" }} />Above avg</span>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: 10, height: 10, borderRadius: 2, background: "rgba(234,179,8,0.4)", marginRight: 4, verticalAlign: "middle" }} />Below avg</span>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: 10, height: 3, background: "rgba(99,102,241,0.7)", marginRight: 4, verticalAlign: "middle" }} />Cumulative</span>
                    </div>
                  </div>
                );
              })()}

              {/* Noise-Signal Decomposition Gauge */}
              {trials.length >= 10 && (() => {
                const nsObjKey = Object.keys(trials[0].kpis)[0];
                const nsVals = trials.map(t => Number(t.kpis[nsObjKey]) || 0);
                const nsMean = nsVals.reduce((a, b) => a + b, 0) / nsVals.length;
                const nsGlobalVar = nsVals.reduce((a, v) => a + (v - nsMean) ** 2, 0) / nsVals.length;
                // Estimate noise: average local variance from k nearest neighbors
                const nsParamKeys = Object.keys(trials[0].parameters);
                const nsNormTrials = trials.map(t => nsParamKeys.map(k => Number(t.parameters[k]) || 0));
                const nsK = Math.min(5, Math.floor(trials.length / 3));
                let nsNoiseSum = 0;
                for (let i = 0; i < trials.length; i++) {
                  const nsDists = nsNormTrials.map((nt, j) => ({
                    j,
                    d: nt.reduce((s, v, dim) => s + (v - nsNormTrials[i][dim]) ** 2, 0),
                  })).filter(x => x.j !== i).sort((a, b) => a.d - b.d).slice(0, nsK);
                  const nsNeighVals = nsDists.map(x => nsVals[x.j]);
                  const nsLocalMean = nsNeighVals.reduce((a, b) => a + b, 0) / nsNeighVals.length;
                  nsNoiseSum += nsNeighVals.reduce((a, v) => a + (v - nsLocalMean) ** 2, 0) / nsNeighVals.length;
                }
                const nsNoiseVar = nsNoiseSum / trials.length;
                const nsSignalVar = Math.max(0, nsGlobalVar - nsNoiseVar);
                const nsRatio = nsGlobalVar > 0 ? nsNoiseVar / nsGlobalVar : 0;
                const nsSnr = nsNoiseVar > 0 ? nsSignalVar / nsNoiseVar : 99;
                const nsStatus = nsRatio < 0.3 ? "Signal-dominant" : nsRatio < 0.6 ? "Moderate noise" : "Noise-dominant";
                const nsColor = nsRatio < 0.3 ? "#22c55e" : nsRatio < 0.6 ? "#eab308" : "#ef4444";
                const nsW = 300, nsH = 120, nsPad = 20;
                const nsBarW = 30, nsBarH = nsH - 2 * nsPad;
                const nsBarX = nsPad + 40;
                const nsBarY = nsPad;
                const nsFillH = nsBarH * Math.min(nsRatio, 1);
                return (
                  <div className="card">
                    <div style={{ display: "flex", alignItems: "flex-start", gap: "10px", marginBottom: "8px" }}>
                      <Volume2 size={16} style={{ color: "var(--color-primary)", marginTop: 2 }} />
                      <div style={{ flex: 1 }}>
                        <h2 style={{ margin: 0 }}>Noise-Signal Decomposition</h2>
                        <p style={{ margin: "2px 0 0", fontSize: "0.78rem", color: "var(--color-text-muted)" }}>
                          Ratio of measurement noise to total objective variance.
                        </p>
                      </div>
                      <span className="findings-badge" style={{ background: nsColor + "22", color: nsColor, border: `1px solid ${nsColor}44` }}>
                        SNR: {nsSnr.toFixed(1)}
                      </span>
                    </div>
                    <svg width={nsW} height={nsH} viewBox={`0 0 ${nsW} ${nsH}`} role="img" aria-label="Noise-signal gauge" style={{ display: "block", margin: "0 auto" }}>
                      {/* Gauge background */}
                      <rect x={nsBarX} y={nsBarY} width={nsBarW} height={nsBarH} rx={4} fill="var(--color-border)" />
                      {/* Noise fill (from bottom) */}
                      <rect x={nsBarX} y={nsBarY + nsBarH - nsFillH} width={nsBarW} height={nsFillH} rx={4} fill={nsColor} opacity={0.7} />
                      {/* Zone lines */}
                      <line x1={nsBarX - 4} y1={nsBarY + nsBarH * 0.4} x2={nsBarX + nsBarW + 4} y2={nsBarY + nsBarH * 0.4} stroke="var(--color-text-muted)" strokeWidth={0.5} strokeDasharray="3,2" />
                      <line x1={nsBarX - 4} y1={nsBarY + nsBarH * 0.7} x2={nsBarX + nsBarW + 4} y2={nsBarY + nsBarH * 0.7} stroke="var(--color-text-muted)" strokeWidth={0.5} strokeDasharray="3,2" />
                      {/* Zone labels */}
                      <text x={nsBarX + nsBarW + 10} y={nsBarY + nsBarH * 0.2} fontSize="9" fill="#ef4444" fontFamily="var(--font-mono)">Noise-heavy</text>
                      <text x={nsBarX + nsBarW + 10} y={nsBarY + nsBarH * 0.55} fontSize="9" fill="#eab308" fontFamily="var(--font-mono)">Moderate</text>
                      <text x={nsBarX + nsBarW + 10} y={nsBarY + nsBarH * 0.85} fontSize="9" fill="#22c55e" fontFamily="var(--font-mono)">Signal-clear</text>
                      {/* Percentage labels */}
                      <text x={nsBarX - 6} y={nsBarY + 4} textAnchor="end" fontSize="8" fill="var(--color-text-muted)" fontFamily="var(--font-mono)">100%</text>
                      <text x={nsBarX - 6} y={nsBarY + nsBarH} textAnchor="end" fontSize="8" fill="var(--color-text-muted)" fontFamily="var(--font-mono)">0%</text>
                      {/* Current level marker */}
                      <line x1={nsBarX - 2} y1={nsBarY + nsBarH - nsFillH} x2={nsBarX + nsBarW + 2} y2={nsBarY + nsBarH - nsFillH} stroke={nsColor} strokeWidth={2} />
                      <text x={nsBarX - 6} y={nsBarY + nsBarH - nsFillH + 3} textAnchor="end" fontSize="9" fill={nsColor} fontWeight="600" fontFamily="var(--font-mono)">{(nsRatio * 100).toFixed(0)}%</text>
                      {/* Stats on right */}
                      <text x={nsBarX + nsBarW + 10} y={nsH - 6} fontSize="9" fill="var(--color-text-muted)" fontFamily="var(--font-mono)">
                        σ²noise: {nsNoiseVar.toPrecision(3)} | σ²signal: {nsSignalVar.toPrecision(3)}
                      </text>
                    </svg>
                    <div style={{ textAlign: "center", marginTop: "4px", fontSize: "0.82rem", fontWeight: 600, color: nsColor }}>
                      {nsStatus}
                    </div>
                  </div>
                );
              })()}

              {/* Budget Utilization Curve */}
              {trials.length >= 3 && (() => {
                const buObs = [...trials].sort((a, b) => a.iteration - b.iteration);
                const buObjKey = campaign.objective_names?.[0] || Object.keys(buObs[0].kpis)[0];
                const buDir = (campaign.objective_directions?.[buObjKey] || "minimize") === "minimize" ? -1 : 1;
                const buVals = buObs.map(o => (o.kpis[buObjKey] ?? 0) * buDir);
                const buN = buVals.length;
                const buBest: number[] = [];
                let buRunBest = -Infinity;
                for (let i = 0; i < buN; i++) { buRunBest = Math.max(buRunBest, buVals[i]); buBest.push(buRunBest); }
                const buStart = buBest[0];
                const buEnd = buBest[buN - 1];
                const buRange = buEnd - buStart || 1;
                const buFrac = buBest.map(v => Math.max(0, Math.min(1, (v - buStart) / buRange)));
                // AUC via trapezoidal rule
                let buAUC = 0;
                for (let i = 1; i < buN; i++) {
                  const dx = 1 / (buN - 1);
                  buAUC += (buFrac[i - 1] + buFrac[i]) * 0.5 * dx;
                }
                const buAF = buAUC > 0 ? buAUC / 0.5 : 1.0;
                const buW = 200, buH = 80, buPad = 20;
                const buPts = buFrac.map((f, i) => ({
                  x: buPad + (i / (buN - 1)) * (buW - 2 * buPad),
                  y: buH - buPad - f * (buH - 2 * buPad),
                }));
                const buDiagStart = { x: buPad, y: buH - buPad };
                const buDiagEnd = { x: buW - buPad, y: buPad };
                // Build fill path between curve and diagonal
                const buCurvePath = buPts.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
                const buDiagPath = buPts.map((_, i) => {
                  const t = i / (buN - 1);
                  const dx = buDiagStart.x + t * (buDiagEnd.x - buDiagStart.x);
                  const dy = buDiagStart.y + t * (buDiagEnd.y - buDiagStart.y);
                  return `${dx.toFixed(1)},${dy.toFixed(1)}`;
                }).reverse().map((s, i) => `${i === 0 ? "L" : "L"}${s}`).join(" ");
                const buFillPath = buCurvePath + " " + buDiagPath + " Z";
                const buFillColor = buAF >= 1.0 ? "rgba(34,197,94,0.18)" : "rgba(239,68,68,0.15)";
                const buAccColor = buAF >= 1.0 ? "#22c55e" : "#ef4444";
                return (
                  <div className="card" style={{ padding: "14px 16px" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                      <Gauge size={16} style={{ color: "var(--color-text-muted)" }} />
                      <h3 style={{ margin: 0, fontSize: "0.92rem" }}>Budget Utilization</h3>
                      <span className="findings-badge" style={{ background: buAccColor + "18", color: buAccColor, marginLeft: "auto" }}>
                        {buAF.toFixed(1)}x acceleration
                      </span>
                    </div>
                    <svg width="100%" viewBox={`0 0 ${buW} ${buH}`} style={{ display: "block" }}>
                      {/* Grid lines */}
                      {[0.25, 0.5, 0.75].map(t => (
                        <Fragment key={`bug${t}`}>
                          <line x1={buPad + t * (buW - 2 * buPad)} y1={buPad} x2={buPad + t * (buW - 2 * buPad)} y2={buH - buPad} stroke="var(--color-border)" strokeWidth="0.5" />
                          <line x1={buPad} y1={buH - buPad - t * (buH - 2 * buPad)} x2={buW - buPad} y2={buH - buPad - t * (buH - 2 * buPad)} stroke="var(--color-border)" strokeWidth="0.5" />
                        </Fragment>
                      ))}
                      {/* Axes */}
                      <line x1={buPad} y1={buH - buPad} x2={buW - buPad} y2={buH - buPad} stroke="var(--color-text-muted)" strokeWidth="0.8" />
                      <line x1={buPad} y1={buPad} x2={buPad} y2={buH - buPad} stroke="var(--color-text-muted)" strokeWidth="0.8" />
                      {/* Fill between curve and diagonal */}
                      <path d={buFillPath} fill={buFillColor} />
                      {/* Random baseline diagonal */}
                      <line x1={buDiagStart.x} y1={buDiagStart.y} x2={buDiagEnd.x} y2={buDiagEnd.y} stroke="var(--color-text-muted)" strokeWidth="1" strokeDasharray="4 3" opacity={0.6} />
                      {/* Campaign curve */}
                      <polyline points={buPts.map(p => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ")} fill="none" stroke="#3b82f6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                      {/* Endpoint dot */}
                      <circle cx={buPts[buPts.length - 1].x} cy={buPts[buPts.length - 1].y} r="3" fill="#3b82f6" />
                      {/* Labels */}
                      <text x={buPad - 2} y={buH - buPad + 11} fontSize="7" fill="var(--color-text-muted)" textAnchor="start">0%</text>
                      <text x={buW - buPad + 2} y={buH - buPad + 11} fontSize="7" fill="var(--color-text-muted)" textAnchor="end">100%</text>
                      <text x={buPad - 4} y={buH - buPad} fontSize="7" fill="var(--color-text-muted)" textAnchor="end">0</text>
                      <text x={buPad - 4} y={buPad + 3} fontSize="7" fill="var(--color-text-muted)" textAnchor="end">1</text>
                    </svg>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.72rem", color: "var(--color-text-muted)", marginTop: 4 }}>
                      <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
                        <span style={{ display: "inline-block", width: 12, height: 2, background: "#3b82f6", borderRadius: 1 }} /> Campaign
                      </span>
                      <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
                        <span style={{ display: "inline-block", width: 12, height: 0, borderTop: "1.5px dashed var(--color-text-muted)" }} /> Random baseline
                      </span>
                      <span>AUC: {buAUC.toFixed(3)}</span>
                    </div>
                  </div>
                );
              })()}

              {/* Information Gain Efficiency */}
              {trials.length >= 6 && (() => {
                const igObjKey = campaign.objective_names?.[0] || Object.keys(trials[0].kpis)[0];
                const igSorted = [...trials].sort((a, b) => a.iteration - b.iteration);
                const igVals = igSorted.map(t => t.kpis[igObjKey] ?? 0);
                const igN = igVals.length;
                const igWinSize = Math.max(3, Math.floor(igN / 12));
                // Compute rolling variance reduction (proxy for information gain)
                const igGains: { iter: number; gain: number }[] = [];
                for (let i = igWinSize; i < igN; i += Math.max(1, Math.floor(igWinSize / 2))) {
                  const before = igVals.slice(Math.max(0, i - igWinSize * 2), i - igWinSize);
                  const after = igVals.slice(i - igWinSize, i);
                  if (before.length < 2 || after.length < 2) continue;
                  const varBefore = (() => { const m = before.reduce((a, b) => a + b, 0) / before.length; return before.reduce((a, v) => a + (v - m) ** 2, 0) / before.length; })();
                  const varAfter = (() => { const m = after.reduce((a, b) => a + b, 0) / after.length; return after.reduce((a, v) => a + (v - m) ** 2, 0) / after.length; })();
                  const gain = varBefore > 1e-12 ? Math.max(0, 0.5 * Math.log(varBefore / Math.max(varAfter, 1e-15))) : 0;
                  igGains.push({ iter: igSorted[i].iteration, gain: Math.min(gain, 3) });
                }
                if (igGains.length < 3) return null;
                const igMax = Math.max(...igGains.map(g => g.gain), 0.01);
                const igAvg = igGains.reduce((a, g) => a + g.gain, 0) / igGains.length;
                const igRecent = igGains.slice(-3).reduce((a, g) => a + g.gain, 0) / 3;
                const igPhase = igRecent > igAvg * 0.75 ? "Active learning" : igRecent > igAvg * 0.3 ? "Diminishing" : "Saturated";
                const igPhaseColor = igPhase === "Active learning" ? "#22c55e" : igPhase === "Diminishing" ? "#f59e0b" : "#ef4444";
                const igW = 200, igH = 65, igPadX = 18, igPadY = 10;
                const igChartW = igW - 2 * igPadX, igChartH = igH - 2 * igPadY;
                const igPts = igGains.map((g, i) => ({
                  x: igPadX + (i / (igGains.length - 1)) * igChartW,
                  y: igH - igPadY - (g.gain / igMax) * igChartH,
                }));
                const igPath = igPts.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
                const igAreaPath = `M${igPts[0].x},${igH - igPadY} ${igPts.map(p => `L${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ")} L${igPts[igPts.length - 1].x},${igH - igPadY} Z`;
                return (
                  <div className="card" style={{ padding: "14px 16px" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                      <Waves size={16} style={{ color: "var(--color-text-muted)" }} />
                      <h3 style={{ margin: 0, fontSize: "0.92rem" }}>Information Gain</h3>
                      <span className="findings-badge" style={{ background: igPhaseColor + "18", color: igPhaseColor, marginLeft: "auto" }}>
                        {igPhase}
                      </span>
                    </div>
                    <svg width="100%" viewBox={`0 0 ${igW} ${igH}`} style={{ display: "block" }}>
                      <defs>
                        <linearGradient id="ig-fill" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor={igPhaseColor} stopOpacity="0.25" />
                          <stop offset="100%" stopColor={igPhaseColor} stopOpacity="0.02" />
                        </linearGradient>
                      </defs>
                      {/* Grid lines */}
                      {[0.25, 0.5, 0.75].map(t => (
                        <line key={`igg${t}`} x1={igPadX} y1={igH - igPadY - t * igChartH} x2={igW - igPadX} y2={igH - igPadY - t * igChartH} stroke="var(--color-border)" strokeWidth="0.5" />
                      ))}
                      {/* Fill */}
                      <path d={igAreaPath} fill="url(#ig-fill)" />
                      {/* Line */}
                      <path d={igPath} fill="none" stroke={igPhaseColor} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
                      {/* Endpoint */}
                      <circle cx={igPts[igPts.length - 1].x} cy={igPts[igPts.length - 1].y} r="2.5" fill={igPhaseColor} />
                      {/* Labels */}
                      <text x={igPadX} y={igH - 1} fontSize="5.5" fill="var(--color-text-muted)">early</text>
                      <text x={igW - igPadX} y={igH - 1} fontSize="5.5" fill="var(--color-text-muted)" textAnchor="end">recent</text>
                      <text x={igPadX - 2} y={igPadY + 3} fontSize="5" fill="var(--color-text-muted)" textAnchor="end">high</text>
                      <text x={igPadX - 2} y={igH - igPadY} fontSize="5" fill="var(--color-text-muted)" textAnchor="end">low</text>
                    </svg>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.72rem", color: "var(--color-text-muted)", marginTop: 2 }}>
                      <span>Avg gain: {igAvg.toFixed(3)} nats/window</span>
                      <span>Recent: {igRecent.toFixed(3)} nats/window</span>
                    </div>
                  </div>
                );
              })()}

              {/* Surrogate Fidelity Gauge */}
              {trials.length >= 10 && (() => {
                const sfK = 5; // k-NN neighbourhood size
                const sfTrials = trials.slice(-Math.min(trials.length, 60));
                const sfSpecs = campaign.spec?.parameters?.filter((s: { name: string; type: string; lower?: number; upper?: number }) => s.type === "continuous" && s.lower != null && s.upper != null) || [];
                if (sfSpecs.length === 0) return null;

                // Normalize parameters to [0,1]
                const sfNorm = sfTrials.map(t => sfSpecs.map((s: { name: string; lower?: number; upper?: number }) => {
                  const lo = s.lower ?? 0, hi = s.upper ?? 1;
                  return hi > lo ? (Number(t.parameters[s.name]) - lo) / (hi - lo) : 0.5;
                }));
                const sfKpi = Object.keys(sfTrials[0]?.kpis || {})[0];
                if (!sfKpi) return null;
                const sfYs = sfTrials.map(t => Number(t.kpis[sfKpi]) || 0);

                // For each point, compute LOO prediction variance using k-NN
                const sfVars: number[] = [];
                for (let i = 0; i < sfNorm.length; i++) {
                  const dists = sfNorm.map((pt, j) => ({
                    d: Math.sqrt(pt.reduce((s, v, k) => s + (v - sfNorm[i][k]) ** 2, 0)),
                    j,
                  })).filter(x => x.j !== i).sort((a, b) => a.d - b.d);
                  const neighbors = dists.slice(0, sfK);
                  const nMean = neighbors.reduce((s, n) => s + sfYs[n.j], 0) / sfK;
                  const nVar = neighbors.reduce((s, n) => s + (sfYs[n.j] - nMean) ** 2, 0) / sfK;
                  sfVars.push(nVar);
                }

                const sfGlobalVar = (() => {
                  const m = sfYs.reduce((a, b) => a + b, 0) / sfYs.length;
                  return sfYs.reduce((s, y) => s + (y - m) ** 2, 0) / sfYs.length;
                })();
                const sfAvgLocalVar = sfVars.reduce((a, b) => a + b, 0) / sfVars.length;
                // Fidelity = 1 - (avg local variance / global variance), clamped [0,1]
                const sfFidelity = sfGlobalVar > 0 ? Math.max(0, Math.min(1, 1 - sfAvgLocalVar / sfGlobalVar)) : 0.5;
                const sfPct = sfFidelity * 100;

                // Gauge geometry
                const sfW = 200, sfH = 115, sfCx = sfW / 2, sfCy = 95, sfR = 72;
                const sfAngleStart = Math.PI; // 180° arc (left to right)
                const sfNeedleAngle = sfAngleStart - sfFidelity * Math.PI;
                const sfNeedleLen = sfR - 8;
                const sfNx = sfCx + sfNeedleLen * Math.cos(sfNeedleAngle);
                const sfNy = sfCy - sfNeedleLen * Math.sin(sfNeedleAngle);

                // Arc segment helper
                const sfArc = (startFrac: number, endFrac: number) => {
                  const a1 = sfAngleStart - startFrac * Math.PI;
                  const a2 = sfAngleStart - endFrac * Math.PI;
                  const x1 = sfCx + sfR * Math.cos(a1), y1 = sfCy - sfR * Math.sin(a1);
                  const x2 = sfCx + sfR * Math.cos(a2), y2 = sfCy - sfR * Math.sin(a2);
                  return `M${x1.toFixed(1)},${y1.toFixed(1)} A${sfR},${sfR} 0 0,1 ${x2.toFixed(1)},${y2.toFixed(1)}`;
                };

                const sfColor = sfPct >= 70 ? "#22c55e" : sfPct >= 40 ? "#f59e0b" : "#ef4444";
                const sfLabel = sfPct >= 70 ? "High fidelity" : sfPct >= 40 ? "Moderate" : "Low fidelity";

                return (
                  <div className="card" style={{ padding: "14px 16px" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                      <Gauge size={16} style={{ color: "var(--color-text-muted)" }} />
                      <h3 style={{ margin: 0, fontSize: "0.92rem" }}>Surrogate Fidelity</h3>
                      <span className="findings-badge" style={{ background: sfColor + "18", color: sfColor, marginLeft: "auto" }}>
                        {sfPct.toFixed(0)}% fidelity
                      </span>
                    </div>
                    <svg width="100%" viewBox={`0 0 ${sfW} ${sfH}`} style={{ display: "block", margin: "0 auto" }}>
                      {/* Background arcs: red, yellow, green */}
                      <path d={sfArc(0, 0.4)} fill="none" stroke="rgba(239,68,68,0.2)" strokeWidth="14" strokeLinecap="round" />
                      <path d={sfArc(0.4, 0.7)} fill="none" stroke="rgba(245,158,11,0.2)" strokeWidth="14" strokeLinecap="round" />
                      <path d={sfArc(0.7, 1)} fill="none" stroke="rgba(34,197,94,0.2)" strokeWidth="14" strokeLinecap="round" />
                      {/* Active fill up to current fidelity */}
                      <path d={sfArc(0, sfFidelity)} fill="none" stroke={sfColor} strokeWidth="14" strokeLinecap="round" opacity={0.7} />
                      {/* Tick marks */}
                      {[0, 0.4, 0.7, 1].map((f, i) => {
                        const a = sfAngleStart - f * Math.PI;
                        const x1 = sfCx + (sfR + 10) * Math.cos(a), y1 = sfCy - (sfR + 10) * Math.sin(a);
                        const x2 = sfCx + (sfR + 15) * Math.cos(a), y2 = sfCy - (sfR + 15) * Math.sin(a);
                        return <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke="var(--color-text-muted)" strokeWidth="1" opacity={0.5} />;
                      })}
                      {/* Labels at thresholds */}
                      <text x={sfCx - sfR - 12} y={sfCy + 10} fontSize="7" fill="var(--color-text-muted)" textAnchor="middle">0%</text>
                      <text x={sfCx} y={sfCy - sfR - 8} fontSize="7" fill="var(--color-text-muted)" textAnchor="middle">50%</text>
                      <text x={sfCx + sfR + 12} y={sfCy + 10} fontSize="7" fill="var(--color-text-muted)" textAnchor="middle">100%</text>
                      {/* Needle */}
                      <line x1={sfCx} y1={sfCy} x2={sfNx} y2={sfNy} stroke={sfColor} strokeWidth="2.5" strokeLinecap="round" />
                      <circle cx={sfCx} cy={sfCy} r="4" fill={sfColor} />
                      {/* Center value */}
                      <text x={sfCx} y={sfCy - 18} textAnchor="middle" fontSize="18" fontWeight="700" fill={sfColor} fontFamily="var(--font-mono)">{sfPct.toFixed(0)}%</text>
                      <text x={sfCx} y={sfCy - 6} textAnchor="middle" fontSize="7.5" fill="var(--color-text-muted)">{sfLabel}</text>
                    </svg>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.72rem", color: "var(--color-text-muted)", marginTop: 2 }}>
                      <span>σ²_local: {sfAvgLocalVar.toFixed(5)}</span>
                      <span>σ²_global: {sfGlobalVar.toFixed(5)}</span>
                    </div>
                  </div>
                );
              })()}

              {/* Prediction Calibration Plot */}
              {trials.length >= 10 && (() => {
                const pc2Sorted = [...trials].sort((a, b) => a.iteration - b.iteration);
                const pc2KpiKey = Object.keys(pc2Sorted[0]?.kpis || {})[0];
                if (!pc2KpiKey) return null;
                const pc2Vals = pc2Sorted.map(t => t.kpis[pc2KpiKey] ?? 0);
                const pc2N = pc2Vals.length;

                // LOO prediction: for each point i, predict using k=5 nearest neighbors
                const pc2K = Math.min(5, pc2N - 1);
                const pc2Predictions: number[] = [];
                for (let i = 0; i < pc2N; i++) {
                  const pc2Dists: { d: number; v: number }[] = [];
                  for (let j = 0; j < pc2N; j++) {
                    if (j === i) continue;
                    const pKeys = Object.keys(pc2Sorted[i].parameters);
                    let dist = 0;
                    for (const pk of pKeys) {
                      const diff = (Number(pc2Sorted[i].parameters[pk]) || 0) - (Number(pc2Sorted[j].parameters[pk]) || 0);
                      dist += diff * diff;
                    }
                    pc2Dists.push({ d: Math.sqrt(dist), v: pc2Vals[j] });
                  }
                  pc2Dists.sort((a, b) => a.d - b.d);
                  const pc2Neighbors = pc2Dists.slice(0, pc2K);
                  const pc2Pred = pc2Neighbors.reduce((s, n) => s + n.v, 0) / pc2Neighbors.length;
                  pc2Predictions.push(pc2Pred);
                }

                // Compute residual std from LOO
                const pc2Residuals = pc2Vals.map((v, i) => v - pc2Predictions[i]);
                const pc2ResMean = pc2Residuals.reduce((a, b) => a + b, 0) / pc2N;
                const pc2ResStd = Math.sqrt(pc2Residuals.reduce((s, r) => s + (r - pc2ResMean) ** 2, 0) / pc2N) || 0.001;

                // For calibration: compute predicted percentile for each point
                // using Gaussian CDF approximation: P(x) ≈ 0.5 * (1 + erf(x / sqrt(2)))
                const pc2Erf = (x: number) => {
                  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429;
                  const p = 0.3275911;
                  const sign = x < 0 ? -1 : 1;
                  const t = 1 / (1 + p * Math.abs(x));
                  const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
                  return sign * y;
                };
                const pc2NormCdf = (x: number) => 0.5 * (1 + pc2Erf(x / Math.SQRT2));

                // Predicted percentile: where does observed fall in predicted distribution N(pred, std)?
                const pc2PredPercentiles = pc2Vals.map((v, i) => pc2NormCdf((v - pc2Predictions[i]) / pc2ResStd));

                // Observed percentile: rank among all observed values
                const pc2ObsPercentiles = pc2Vals.map(v => {
                  const rank = pc2Vals.filter(o => o <= v).length;
                  return rank / pc2N;
                });

                // Bin into 10 bins and compute average predicted & observed percentile
                const pc2NBins = 10;
                const pc2Bins: { pred: number; obs: number }[] = [];
                for (let b = 0; b < pc2NBins; b++) {
                  const lo = b / pc2NBins, hi = (b + 1) / pc2NBins;
                  const inBin = pc2PredPercentiles.map((pp, i) => ({ pp, op: pc2ObsPercentiles[i] })).filter(d => d.pp >= lo && d.pp < hi);
                  if (inBin.length > 0) {
                    pc2Bins.push({
                      pred: inBin.reduce((s, d) => s + d.pp, 0) / inBin.length,
                      obs: inBin.reduce((s, d) => s + d.op, 0) / inBin.length,
                    });
                  }
                }

                // Expected Calibration Error
                const pc2ECE = pc2Bins.length > 0
                  ? pc2Bins.reduce((s, b) => s + Math.abs(b.pred - b.obs), 0) / pc2Bins.length
                  : 0;
                const pc2Cal = pc2ECE < 0.08 ? "Well-calibrated" : pc2ECE < 0.15 ? "Moderate" : "Miscalibrated";
                const pc2CalColor = pc2ECE < 0.08 ? "#22c55e" : pc2ECE < 0.15 ? "#f59e0b" : "#ef4444";

                // SVG layout
                const pc2W = 280, pc2H = 180, pc2Pad = 36;
                const pc2PlotW = pc2W - 2 * pc2Pad, pc2PlotH = pc2H - 2 * pc2Pad;

                return (
                  <div className="card" style={{ marginBottom: "16px" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                      <ScatterChart size={16} style={{ color: "var(--color-primary)" }} />
                      <h3 style={{ margin: 0, fontSize: "0.92rem" }}>Prediction Calibration</h3>
                      <span className="findings-badge" style={{ background: pc2CalColor + "18", color: pc2CalColor, marginLeft: "auto" }}>
                        ECE: {(pc2ECE * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div style={{ fontSize: "0.78rem", color: "var(--color-text-muted)", marginBottom: 8 }}>
                      LOO predicted percentile vs observed — {pc2Cal}
                    </div>
                    <svg width={pc2W} height={pc2H} viewBox={`0 0 ${pc2W} ${pc2H}`} style={{ width: "100%", height: "auto" }}>
                      {/* Grid lines */}
                      {[0, 0.25, 0.5, 0.75, 1].map(v => (
                        <Fragment key={`pc2g${v}`}>
                          <line
                            x1={pc2Pad} y1={pc2Pad + pc2PlotH * (1 - v)}
                            x2={pc2Pad + pc2PlotW} y2={pc2Pad + pc2PlotH * (1 - v)}
                            stroke="var(--color-border)" strokeWidth="0.5" strokeDasharray="3,3"
                          />
                          <text x={pc2Pad - 4} y={pc2Pad + pc2PlotH * (1 - v) + 3} fontSize="6" fill="var(--color-text-muted)" textAnchor="end">{(v * 100).toFixed(0)}%</text>
                          <text x={pc2Pad + pc2PlotW * v} y={pc2H - pc2Pad + 12} fontSize="6" fill="var(--color-text-muted)" textAnchor="middle">{(v * 100).toFixed(0)}%</text>
                        </Fragment>
                      ))}
                      {/* Perfect calibration line (diagonal) */}
                      <line x1={pc2Pad} y1={pc2Pad + pc2PlotH} x2={pc2Pad + pc2PlotW} y2={pc2Pad} stroke="var(--color-text-muted)" strokeWidth="1" strokeDasharray="4,4" opacity="0.5" />
                      {/* Binned calibration points */}
                      {pc2Bins.map((b, i) => (
                        <circle
                          key={`pc2b${i}`}
                          cx={pc2Pad + b.pred * pc2PlotW}
                          cy={pc2Pad + (1 - b.obs) * pc2PlotH}
                          r="4"
                          fill={pc2CalColor}
                          opacity="0.8"
                          stroke="white"
                          strokeWidth="1"
                        />
                      ))}
                      {/* Calibration line connecting bins */}
                      {pc2Bins.length > 1 && (
                        <polyline
                          points={pc2Bins.map(b => `${pc2Pad + b.pred * pc2PlotW},${pc2Pad + (1 - b.obs) * pc2PlotH}`).join(" ")}
                          fill="none"
                          stroke={pc2CalColor}
                          strokeWidth="1.5"
                          strokeLinejoin="round"
                        />
                      )}
                      {/* Scatter of individual points (faint) */}
                      {pc2PredPercentiles.map((pp, i) => (
                        <circle
                          key={`pc2p${i}`}
                          cx={pc2Pad + pp * pc2PlotW}
                          cy={pc2Pad + (1 - pc2ObsPercentiles[i]) * pc2PlotH}
                          r="1.5"
                          fill="var(--color-primary)"
                          opacity="0.2"
                        />
                      ))}
                      {/* Axis labels */}
                      <text x={pc2Pad + pc2PlotW / 2} y={pc2H - 4} fontSize="7" fill="var(--color-text-muted)" textAnchor="middle">Predicted Percentile</text>
                      <text x={8} y={pc2Pad + pc2PlotH / 2} fontSize="7" fill="var(--color-text-muted)" textAnchor="middle" transform={`rotate(-90, 8, ${pc2Pad + pc2PlotH / 2})`}>Observed Percentile</text>
                    </svg>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.72rem", color: "var(--color-text-muted)", marginTop: 4 }}>
                      <span>σ_residual: {pc2ResStd.toFixed(4)}</span>
                      <span>{pc2Bins.length} bins from {pc2N} points</span>
                    </div>
                  </div>
                );
              })()}

              {/* Decision Journal */}
              <div className="card decision-journal-card">
                <div className="decision-journal-header" onClick={() => setShowJournal(p => !p)} style={{ cursor: "pointer" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                    <BookOpen size={16} style={{ color: "var(--color-primary)" }} />
                    <h2 style={{ margin: 0 }}>Decision Journal</h2>
                    {journalEntries.length > 0 && (
                      <span className="journal-count">{journalEntries.length}</span>
                    )}
                  </div>
                  <ChevronRight size={16} style={{ transform: showJournal ? "rotate(90deg)" : "none", transition: "transform 0.2s", color: "var(--color-text-muted)" }} />
                </div>
                {showJournal && (
                  <div className="decision-journal-body">
                    <div className="journal-input-row">
                      <input
                        type="text"
                        className="journal-input"
                        placeholder="Record your reasoning, strategy changes, or observations..."
                        value={journalInput}
                        onChange={(e) => setJournalInput(e.target.value)}
                        onKeyDown={(e) => { if (e.key === "Enter") addJournalEntry(); }}
                      />
                      <button className="btn btn-sm btn-primary" onClick={addJournalEntry} disabled={!journalInput.trim()}>
                        Add
                      </button>
                    </div>
                    {journalEntries.length === 0 ? (
                      <p className="journal-empty">No entries yet. Record your experimental reasoning here for future reference.</p>
                    ) : (
                      <div className="journal-entries">
                        {journalEntries.map((entry) => (
                          <div key={entry.id} className="journal-entry">
                            <div className="journal-entry-meta">
                              <span className="journal-iter">Iter {entry.iteration}</span>
                              <span className="journal-time">{new Date(entry.timestamp).toLocaleDateString()} {new Date(entry.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span>
                              <button className="journal-delete" onClick={() => removeJournalEntry(entry.id)} title="Delete entry">
                                <X size={12} />
                              </button>
                            </div>
                            <div className="journal-entry-text">{entry.text}</div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
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

              {/* Parameter Boundary Saturation */}
              {trials.length >= 8 && (() => {
                const specs = campaign.spec?.parameters || [];
                const bsData = specs
                  .filter((s: { name: string; lower?: number; upper?: number }) => s.lower != null && s.upper != null && s.upper! > s.lower!)
                  .map((s: { name: string; lower?: number; upper?: number }) => {
                    const vals = trials.map(t => Number(t.parameters[s.name]) || 0);
                    const range = s.upper! - s.lower!;
                    const margin = range * 0.05; // 5% of range = "at boundary"
                    const lowerHits = vals.filter(v => v <= s.lower! + margin).length;
                    const upperHits = vals.filter(v => v >= s.upper! - margin).length;
                    const lowerPct = lowerHits / trials.length;
                    const upperPct = upperHits / trials.length;
                    // Temporal trend: split into halves and check if recent has more boundary hits
                    const half = Math.floor(vals.length / 2);
                    const earlyLower = vals.slice(0, half).filter(v => v <= s.lower! + margin).length / Math.max(half, 1);
                    const lateLower = vals.slice(half).filter(v => v <= s.lower! + margin).length / Math.max(vals.length - half, 1);
                    const earlyUpper = vals.slice(0, half).filter(v => v >= s.upper! - margin).length / Math.max(half, 1);
                    const lateUpper = vals.slice(half).filter(v => v >= s.upper! - margin).length / Math.max(vals.length - half, 1);
                    return { name: s.name, lowerPct, upperPct, total: lowerPct + upperPct, lowerTrend: lateLower - earlyLower, upperTrend: lateUpper - earlyUpper };
                  })
                  .filter((d: { total: number }) => d.total > 0)
                  .sort((a: { total: number }, b: { total: number }) => b.total - a.total);

                if (bsData.length === 0) return null;

                type BsItem = { name: string; lowerPct: number; upperPct: number; total: number; lowerTrend: number; upperTrend: number };
                const bsTyped = bsData as BsItem[];
                const W = 380, rowH = 28, padL = 90, padR = 16, padT = 24, padB = 8;
                const barW = W - padL - padR;
                const H = padT + bsTyped.length * rowH + padB;
                const maxPct = Math.max(...bsTyped.map(d => Math.max(d.lowerPct, d.upperPct)), 0.05);

                const satColor = (pct: number) => {
                  if (pct > 0.2) return "rgba(239,68,68,0.7)";
                  if (pct > 0.1) return "rgba(234,179,8,0.6)";
                  if (pct > 0) return "rgba(59,130,246,0.4)";
                  return "transparent";
                };

                const warningCount = bsTyped.filter(d => d.lowerPct > 0.15 || d.upperPct > 0.15).length;

                return (
                  <div className="card">
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "4px" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                        <AlertTriangle size={16} style={{ color: warningCount > 0 ? "var(--color-yellow)" : "var(--color-text-muted)" }} />
                        <h2 style={{ margin: 0 }}>Boundary Saturation</h2>
                      </div>
                      {warningCount > 0 && (
                        <span className="findings-badge" style={{ background: "var(--color-yellow-bg)", color: "var(--color-yellow)" }}>
                          {warningCount} param{warningCount > 1 ? "s" : ""} at bounds
                        </span>
                      )}
                    </div>
                    <p style={{ fontSize: "0.78rem", color: "var(--color-text-muted)", margin: "0 0 8px" }}>
                      How often trials land near parameter boundaries. High saturation suggests expanding the search range.
                    </p>
                    <svg width={W} height={H} style={{ display: "block", width: "100%", maxWidth: `${W}px` }}>
                      {/* Column headers */}
                      <text x={padL + barW * 0.25} y={14} textAnchor="middle" fontSize="9" fill="var(--color-text-muted)" fontFamily="var(--font-mono)">Lower</text>
                      <text x={padL + barW * 0.75} y={14} textAnchor="middle" fontSize="9" fill="var(--color-text-muted)" fontFamily="var(--font-mono)">Upper</text>
                      {bsTyped.map((d, i) => {
                        const y = padT + i * rowH;
                        const lBarW = (d.lowerPct / maxPct) * (barW / 2 - 8);
                        const uBarW = (d.upperPct / maxPct) * (barW / 2 - 8);
                        return (
                          <Fragment key={d.name}>
                            {/* Param label */}
                            <text x={padL - 6} y={y + rowH / 2 + 3} textAnchor="end" fontSize="10" fontFamily="var(--font-mono)" fill="var(--color-text-primary)" style={{ fontWeight: 500 }}>
                              {d.name.length > 10 ? d.name.slice(0, 10) + "…" : d.name}
                            </text>
                            {/* Lower bar */}
                            <rect x={padL} y={y + 4} width={Math.max(lBarW, 0)} height={rowH - 8} rx="3" fill={satColor(d.lowerPct)} />
                            {d.lowerPct > 0 && (
                              <text x={padL + lBarW + 4} y={y + rowH / 2 + 3} fontSize="9" fontFamily="var(--font-mono)" fill="var(--color-text-muted)">
                                {(d.lowerPct * 100).toFixed(0)}%{d.lowerTrend > 0.05 ? " ↑" : d.lowerTrend < -0.05 ? " ↓" : ""}
                              </text>
                            )}
                            {/* Divider */}
                            <line x1={padL + barW / 2} y1={y + 2} x2={padL + barW / 2} y2={y + rowH - 2} stroke="var(--color-border)" strokeWidth="0.5" />
                            {/* Upper bar */}
                            <rect x={padL + barW / 2 + 4} y={y + 4} width={Math.max(uBarW, 0)} height={rowH - 8} rx="3" fill={satColor(d.upperPct)} />
                            {d.upperPct > 0 && (
                              <text x={padL + barW / 2 + 4 + uBarW + 4} y={y + rowH / 2 + 3} fontSize="9" fontFamily="var(--font-mono)" fill="var(--color-text-muted)">
                                {(d.upperPct * 100).toFixed(0)}%{d.upperTrend > 0.05 ? " ↑" : d.upperTrend < -0.05 ? " ↓" : ""}
                              </text>
                            )}
                          </Fragment>
                        );
                      })}
                    </svg>
                  </div>
                );
              })()}

              {/* Local Optima Basin Map */}
              {trials.length >= 12 && (() => {
                const loSpecs = campaign.spec?.parameters || [];
                const loContSpecs = loSpecs.filter((s: { name: string; type: string; lower?: number; upper?: number }) => s.type === "continuous" && s.lower != null && s.upper != null);
                if (loContSpecs.length < 2) return null;
                // Pick top 2 parameters by variance
                const loParamVars = loContSpecs.map((s: { name: string; lower?: number; upper?: number }) => {
                  const vals = trials.map(t => Number(t.parameters[s.name]) || 0);
                  const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
                  const variance = vals.reduce((a, v) => a + (v - mean) ** 2, 0) / vals.length;
                  return { ...s, variance };
                }).sort((a: { variance: number }, b: { variance: number }) => b.variance - a.variance);
                const loP1 = loParamVars[0] as { name: string; lower: number; upper: number };
                const loP2 = loParamVars[1] as { name: string; lower: number; upper: number };
                const loObjKey = Object.keys(trials[0].kpis)[0];
                const loObjVals = trials.map(t => Number(t.kpis[loObjKey]) || 0);
                const loMin = Math.min(...loObjVals);
                const loMax = Math.max(...loObjVals);
                const loRange = loMax - loMin || 1;

                // Grid-based landscape approximation using IDW interpolation
                const loW = 320, loH = 220, loPad = 40;
                const plotW = loW - 2 * loPad, plotH = loH - 2 * loPad;
                const gridN = 16;
                const loTrialPts = trials.map((t, i) => ({
                  x: (Number(t.parameters[loP1.name]) - loP1.lower) / (loP1.upper - loP1.lower),
                  y: (Number(t.parameters[loP2.name]) - loP2.lower) / (loP2.upper - loP2.lower),
                  z: loObjVals[i],
                }));

                // IDW interpolation on grid
                const loGrid: number[][] = [];
                for (let gy = 0; gy < gridN; gy++) {
                  loGrid[gy] = [];
                  for (let gx = 0; gx < gridN; gx++) {
                    const px = (gx + 0.5) / gridN;
                    const py = (gy + 0.5) / gridN;
                    let wSum = 0, vSum = 0;
                    for (const pt of loTrialPts) {
                      const dist = Math.sqrt((px - pt.x) ** 2 + (py - pt.y) ** 2) + 0.001;
                      const w = 1 / (dist * dist);
                      wSum += w;
                      vSum += w * pt.z;
                    }
                    loGrid[gy][gx] = vSum / wSum;
                  }
                }

                // Find local minima in grid
                const loBasins: { gx: number; gy: number; val: number }[] = [];
                for (let gy = 1; gy < gridN - 1; gy++) {
                  for (let gx = 1; gx < gridN - 1; gx++) {
                    const v = loGrid[gy][gx];
                    const neighbors = [loGrid[gy-1][gx], loGrid[gy+1][gx], loGrid[gy][gx-1], loGrid[gy][gx+1]];
                    if (neighbors.every(n => v <= n)) {
                      loBasins.push({ gx, gy, val: v });
                    }
                  }
                }
                loBasins.sort((a, b) => a.val - b.val);
                const topBasins = loBasins.slice(0, 5);

                // Color function: blue (good) to red (bad)
                const loColor = (v: number) => {
                  const t = loRange > 0 ? (v - loMin) / loRange : 0.5;
                  const r = Math.round(59 + t * 196);
                  const g = Math.round(130 - t * 80);
                  const b = Math.round(246 - t * 196);
                  return `rgb(${r},${g},${b})`;
                };

                const cellW = plotW / gridN;
                const cellH = plotH / gridN;

                return (
                  <div className="card" style={{ marginBottom: 18 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
                      <Layers size={18} style={{ color: "var(--color-text-muted)" }} />
                      <div>
                        <h2 style={{ margin: 0 }}>Local Optima Basin Map</h2>
                        <p className="stat-label" style={{ margin: 0, textTransform: "none" }}>
                          IDW-interpolated landscape showing {loP1.name} vs {loP2.name}. Cooler = better.
                        </p>
                      </div>
                      {topBasins.length > 0 && (
                        <span className="findings-badge" style={{ marginLeft: "auto" }}>
                          {topBasins.length} basin{topBasins.length !== 1 ? "s" : ""} found
                        </span>
                      )}
                    </div>
                    <div style={{ overflowX: "auto" }}>
                      <svg width={loW} height={loH} viewBox={`0 0 ${loW} ${loH}`} style={{ display: "block", margin: "0 auto" }}>
                        {/* Heatmap cells */}
                        {loGrid.flatMap((row, gy) =>
                          row.map((v, gx) => (
                            <rect
                              key={`${gy}-${gx}`}
                              x={loPad + gx * cellW}
                              y={loPad + gy * cellH}
                              width={cellW + 0.5}
                              height={cellH + 0.5}
                              fill={loColor(v)}
                              opacity={0.75}
                            />
                          ))
                        )}
                        {/* Trial points */}
                        {loTrialPts.map((pt, i) => (
                          <circle
                            key={i}
                            cx={loPad + pt.x * plotW}
                            cy={loPad + pt.y * plotH}
                            r={3}
                            fill="none"
                            stroke="rgba(255,255,255,0.7)"
                            strokeWidth={1}
                          />
                        ))}
                        {/* Basin markers */}
                        {topBasins.map((b, i) => {
                          const bx = loPad + (b.gx + 0.5) / gridN * plotW;
                          const by = loPad + (b.gy + 0.5) / gridN * plotH;
                          return (
                            <g key={i}>
                              <circle cx={bx} cy={by} r={8} fill="none" stroke="#fff" strokeWidth={2} strokeDasharray="3,2" />
                              <text x={bx} y={by + 4} textAnchor="middle" fontSize="9" fontWeight="700" fill="#fff">
                                {i + 1}
                              </text>
                            </g>
                          );
                        })}
                        {/* Axes labels */}
                        <text x={loPad + plotW / 2} y={loH - 4} textAnchor="middle" fontSize="10" fill="var(--color-text-muted)" fontFamily="var(--font-mono)">
                          {loP1.name}
                        </text>
                        <text x={10} y={loPad + plotH / 2} textAnchor="middle" fontSize="10" fill="var(--color-text-muted)" fontFamily="var(--font-mono)" transform={`rotate(-90,10,${loPad + plotH / 2})`}>
                          {loP2.name}
                        </text>
                        {/* Axis ticks */}
                        {[0, 0.5, 1].map(f => (
                          <g key={f}>
                            <text x={loPad + f * plotW} y={loPad + plotH + 14} textAnchor="middle" fontSize="8" fill="var(--color-text-muted)" fontFamily="var(--font-mono)">
                              {(loP1.lower + f * (loP1.upper - loP1.lower)).toPrecision(3)}
                            </text>
                            <text x={loPad - 4} y={loPad + (1 - f) * plotH + 3} textAnchor="end" fontSize="8" fill="var(--color-text-muted)" fontFamily="var(--font-mono)">
                              {(loP2.lower + f * (loP2.upper - loP2.lower)).toPrecision(3)}
                            </text>
                          </g>
                        ))}
                      </svg>
                    </div>
                    <div style={{ display: "flex", gap: "16px", marginTop: "6px", flexWrap: "wrap" }}>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: 10, height: 10, borderRadius: 2, background: "rgb(59,130,246)", marginRight: 4, verticalAlign: "middle" }} />Better (lower)</span>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: 10, height: 10, borderRadius: 2, background: "rgb(255,50,50)", marginRight: 4, verticalAlign: "middle" }} />Worse (higher)</span>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: 10, height: 10, borderRadius: "50%", border: "2px dashed #fff", marginRight: 4, verticalAlign: "middle" }} />Basin center</span>
                    </div>
                  </div>
                );
              })()}

              {/* Prediction Residual Map */}
              {trials.length >= 12 && (() => {
                const prSpecs = campaign.spec?.parameters || [];
                const prContSpecs = prSpecs.filter((s: { name: string; type: string; lower?: number; upper?: number }) => s.type === "continuous" && s.lower != null && s.upper != null);
                if (prContSpecs.length < 2) return null;
                const prObjKey = Object.keys(trials[0].kpis)[0];
                // Pick top 2 parameters by correlation with objective
                const prObjVals = trials.map(t => Number(t.kpis[prObjKey]) || 0);
                const prMeanObj = prObjVals.reduce((a, b) => a + b, 0) / prObjVals.length;
                const prParamCorrs = prContSpecs.map((s: { name: string }) => {
                  const vals = trials.map(t => Number(t.parameters[s.name]) || 0);
                  const meanP = vals.reduce((a, b) => a + b, 0) / vals.length;
                  let num = 0, denP = 0, denO = 0;
                  for (let i = 0; i < vals.length; i++) {
                    const dp = vals[i] - meanP;
                    const doVal = prObjVals[i] - prMeanObj;
                    num += dp * doVal;
                    denP += dp * dp;
                    denO += doVal * doVal;
                  }
                  return { name: s.name, corr: Math.abs(denP > 0 && denO > 0 ? num / Math.sqrt(denP * denO) : 0) };
                }).sort((a: { corr: number }, b: { corr: number }) => b.corr - a.corr);
                const prP1Name = prParamCorrs[0].name;
                const prP2Name = prParamCorrs.length > 1 ? prParamCorrs[1].name : prParamCorrs[0].name;
                const prSpec1 = prContSpecs.find((s: { name: string }) => s.name === prP1Name) as { name: string; lower: number; upper: number };
                const prSpec2 = prContSpecs.find((s: { name: string }) => s.name === prP2Name) as { name: string; lower: number; upper: number };
                if (!prSpec1 || !prSpec2) return null;

                // LOO nearest-neighbor residuals
                const pKeys = Object.keys(trials[0].parameters);
                const prResiduals = trials.map((t, idx) => {
                  const actual = prObjVals[idx];
                  let bestDist = Infinity, predicted = actual;
                  for (let j = 0; j < trials.length; j++) {
                    if (j === idx) continue;
                    let dist = 0;
                    for (const k of pKeys) {
                      const diff = (Number(t.parameters[k]) || 0) - (Number(trials[j].parameters[k]) || 0);
                      dist += diff * diff;
                    }
                    if (dist < bestDist) {
                      bestDist = dist;
                      predicted = prObjVals[j];
                    }
                  }
                  return {
                    x: (Number(t.parameters[prP1Name]) - prSpec1.lower) / (prSpec1.upper - prSpec1.lower),
                    y: (Number(t.parameters[prP2Name]) - prSpec2.lower) / (prSpec2.upper - prSpec2.lower),
                    residual: actual - predicted,
                  };
                });

                const prMaxRes = Math.max(...prResiduals.map(r => Math.abs(r.residual)), 0.001);
                const prW = 320, prH = 240, prPadL = 44, prPadR = 16, prPadT = 16, prPadB = 32;
                const prPlotW = prW - prPadL - prPadR;
                const prPlotH = prH - prPadT - prPadB;

                // Mean absolute residual
                const mae = prResiduals.reduce((s, r) => s + Math.abs(r.residual), 0) / prResiduals.length;

                // Bias: mean residual
                const bias = prResiduals.reduce((s, r) => s + r.residual, 0) / prResiduals.length;
                const biasLabel = Math.abs(bias) < mae * 0.3 ? "Balanced" : bias > 0 ? "Over-predicting" : "Under-predicting";

                return (
                  <div className="card" style={{ marginBottom: 18 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
                      <AlertTriangle size={18} style={{ color: "var(--color-text-muted)" }} />
                      <div>
                        <h2 style={{ margin: 0 }}>Prediction Residuals</h2>
                        <p className="stat-label" style={{ margin: 0, textTransform: "none" }}>
                          Where the surrogate over/under-predicts, mapped on {prP1Name} vs {prP2Name}.
                        </p>
                      </div>
                      <span className="findings-badge" style={{ marginLeft: "auto" }}>
                        {biasLabel}
                      </span>
                    </div>
                    <div style={{ overflowX: "auto" }}>
                      <svg width={prW} height={prH} viewBox={`0 0 ${prW} ${prH}`} style={{ display: "block", margin: "0 auto" }}>
                        {/* Grid */}
                        {[0, 0.5, 1].map(f => (
                          <g key={f}>
                            <line x1={prPadL + f * prPlotW} y1={prPadT} x2={prPadL + f * prPlotW} y2={prPadT + prPlotH} stroke="var(--color-border)" strokeWidth={0.5} />
                            <line x1={prPadL} y1={prPadT + f * prPlotH} x2={prPadL + prPlotW} y2={prPadT + f * prPlotH} stroke="var(--color-border)" strokeWidth={0.5} />
                          </g>
                        ))}
                        {/* Residual points */}
                        {prResiduals.map((r, i) => {
                          const cx = prPadL + r.x * prPlotW;
                          const cy = prPadT + (1 - r.y) * prPlotH;
                          const normRes = r.residual / prMaxRes;
                          // Red = over-prediction (actual > predicted, residual > 0), Blue = under-prediction
                          const color = normRes > 0
                            ? `rgba(239,68,68,${Math.min(Math.abs(normRes) * 0.8 + 0.2, 1)})`
                            : `rgba(59,130,246,${Math.min(Math.abs(normRes) * 0.8 + 0.2, 1)})`;
                          const radius = 3 + Math.abs(normRes) * 4;
                          return (
                            <circle key={i} cx={cx} cy={cy} r={radius} fill={color} stroke="rgba(255,255,255,0.3)" strokeWidth={0.5}>
                              <title>Trial {i}: {prP1Name}={Number(trials[i].parameters[prP1Name]).toPrecision(3)}, {prP2Name}={Number(trials[i].parameters[prP2Name]).toPrecision(3)}, residual={r.residual.toPrecision(3)}</title>
                            </circle>
                          );
                        })}
                        {/* Axis labels */}
                        <text x={prPadL + prPlotW / 2} y={prH - 2} textAnchor="middle" fontSize="10" fill="var(--color-text-muted)" fontFamily="var(--font-mono)">
                          {prP1Name}
                        </text>
                        <text x={10} y={prPadT + prPlotH / 2} textAnchor="middle" fontSize="10" fill="var(--color-text-muted)" fontFamily="var(--font-mono)" transform={`rotate(-90,10,${prPadT + prPlotH / 2})`}>
                          {prP2Name}
                        </text>
                        {/* Tick labels */}
                        {[0, 0.5, 1].map(f => (
                          <g key={f}>
                            <text x={prPadL + f * prPlotW} y={prPadT + prPlotH + 14} textAnchor="middle" fontSize="8" fill="var(--color-text-muted)" fontFamily="var(--font-mono)">
                              {(prSpec1.lower + f * (prSpec1.upper - prSpec1.lower)).toPrecision(3)}
                            </text>
                            <text x={prPadL - 4} y={prPadT + (1 - f) * prPlotH + 3} textAnchor="end" fontSize="8" fill="var(--color-text-muted)" fontFamily="var(--font-mono)">
                              {(prSpec2.lower + f * (prSpec2.upper - prSpec2.lower)).toPrecision(3)}
                            </text>
                          </g>
                        ))}
                      </svg>
                    </div>
                    <div style={{ display: "flex", gap: "16px", marginTop: "6px", flexWrap: "wrap", alignItems: "center" }}>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: 10, height: 10, borderRadius: "50%", background: "rgba(239,68,68,0.7)", marginRight: 4, verticalAlign: "middle" }} />Over-predicting</span>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: 10, height: 10, borderRadius: "50%", background: "rgba(59,130,246,0.7)", marginRight: 4, verticalAlign: "middle" }} />Under-predicting</span>
                      <span style={{ marginLeft: "auto", fontSize: "0.78rem", color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}>
                        MAE: {mae.toPrecision(3)}
                      </span>
                    </div>
                  </div>
                );
              })()}

              {/* Lengthscale Adequacy Indicator */}
              {trials.length >= 10 && (() => {
                const laSpecs = campaign.spec?.parameters || [];
                const laContSpecs = laSpecs.filter((s: { name: string; type: string; lower?: number; upper?: number }) => s.type === "continuous" && s.lower != null && s.upper != null);
                if (laContSpecs.length < 2) return null;
                // For each parameter, compute median nearest-neighbor distance
                const laResults = laContSpecs.map((spec: { name: string; lower?: number; upper?: number }) => {
                  const range = (spec.upper ?? 1) - (spec.lower ?? 0);
                  const vals = trials.map(t => Number(t.parameters[spec.name]) || 0);
                  const sorted = [...vals].sort((a, b) => a - b);
                  // Nearest neighbor distances
                  const nnDists: number[] = [];
                  for (let i = 0; i < sorted.length; i++) {
                    let minD = Infinity;
                    if (i > 0) minD = Math.min(minD, sorted[i] - sorted[i - 1]);
                    if (i < sorted.length - 1) minD = Math.min(minD, sorted[i + 1] - sorted[i]);
                    nnDists.push(minD);
                  }
                  nnDists.sort((a, b) => a - b);
                  const medianNN = nnDists[Math.floor(nnDists.length / 2)];
                  const resolution = range > 0 ? medianNN / range : 0;
                  // Critical threshold: ~1/sqrt(n) as rough heuristic for GP resolution
                  const critical = 1 / Math.sqrt(trials.length);
                  const adequate = resolution <= critical;
                  return { name: spec.name, resolution, critical, adequate, range };
                });
                const laAdequateCount = laResults.filter((r: { adequate: boolean }) => r.adequate).length;
                const laMaxRes = Math.max(...laResults.map((r: { resolution: number }) => r.resolution), 0.01);
                const laW = 320, laRowH = 22, laPadL = 80, laPadR = 60, laPadT = 8, laPadB = 20;
                const laH = laPadT + laResults.length * laRowH + laPadB;
                const laPlotW = laW - laPadL - laPadR;
                return (
                  <div className="card">
                    <div style={{ display: "flex", alignItems: "flex-start", gap: "10px", marginBottom: "8px" }}>
                      <Ruler size={16} style={{ color: "var(--color-primary)", marginTop: 2 }} />
                      <div style={{ flex: 1 }}>
                        <h2 style={{ margin: 0 }}>Sampling Resolution</h2>
                        <p style={{ margin: "2px 0 0", fontSize: "0.78rem", color: "var(--color-text-muted)" }}>
                          Median nearest-neighbor distance per parameter vs. critical threshold.
                        </p>
                      </div>
                      <span className="findings-badge" style={{ background: laAdequateCount === laResults.length ? "rgba(34,197,94,0.12)" : "rgba(234,179,8,0.12)", color: laAdequateCount === laResults.length ? "#22c55e" : "#eab308", border: `1px solid ${laAdequateCount === laResults.length ? "#22c55e44" : "#eab30844"}` }}>
                        {laAdequateCount}/{laResults.length} adequate
                      </span>
                    </div>
                    <svg width={laW} height={laH} viewBox={`0 0 ${laW} ${laH}`} role="img" aria-label="Sampling resolution per parameter" style={{ display: "block" }}>
                      {laResults.map((r: { name: string; resolution: number; critical: number; adequate: boolean }, i: number) => {
                        const y = laPadT + i * laRowH;
                        const barW = Math.max(2, (r.resolution / laMaxRes) * laPlotW);
                        const critX = laPadL + (r.critical / laMaxRes) * laPlotW;
                        const barColor = r.adequate ? "#22c55e" : r.resolution < r.critical * 2 ? "#eab308" : "#ef4444";
                        return (
                          <g key={r.name}>
                            <text x={laPadL - 6} y={y + laRowH / 2 + 3} textAnchor="end" fontSize="10" fontFamily="var(--font-mono)" fill="var(--color-text-primary)">{r.name}</text>
                            <rect x={laPadL} y={y + 3} width={laPlotW} height={laRowH - 6} rx={3} fill="var(--color-border)" />
                            <rect x={laPadL} y={y + 3} width={Math.min(barW, laPlotW)} height={laRowH - 6} rx={3} fill={barColor} opacity={0.6}>
                              <title>{r.name}: resolution {(r.resolution * 100).toFixed(1)}% of range (threshold: {(r.critical * 100).toFixed(1)}%)</title>
                            </rect>
                            {/* Critical threshold mark */}
                            <line x1={critX} y1={y + 1} x2={critX} y2={y + laRowH - 1} stroke="var(--color-text-muted)" strokeWidth={1.5} strokeDasharray="3,2" />
                            <text x={laW - laPadR + 6} y={y + laRowH / 2 + 3} fontSize="9" fontFamily="var(--font-mono)" fill={barColor} fontWeight="600">
                              {(r.resolution * 100).toFixed(1)}%
                            </text>
                          </g>
                        );
                      })}
                      {/* Threshold label */}
                      <text x={laPadL + (laResults[0].critical / laMaxRes) * laPlotW} y={laH - 4} textAnchor="middle" fontSize="8" fill="var(--color-text-muted)" fontFamily="var(--font-mono)">
                        threshold
                      </text>
                    </svg>
                    <div style={{ display: "flex", gap: "16px", marginTop: "4px", flexWrap: "wrap", alignItems: "center" }}>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: 10, height: 10, borderRadius: 2, background: "#22c55e", opacity: 0.6, marginRight: 4, verticalAlign: "middle" }} />Adequate</span>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: 10, height: 10, borderRadius: 2, background: "#eab308", opacity: 0.6, marginRight: 4, verticalAlign: "middle" }} />Marginal</span>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: 10, height: 10, borderRadius: 2, background: "#ef4444", opacity: 0.6, marginRight: 4, verticalAlign: "middle" }} />Too coarse</span>
                      <span style={{ marginLeft: "auto", fontSize: "0.72rem", color: "var(--color-text-muted)" }}>Dashed = 1/√n threshold</span>
                    </div>
                  </div>
                );
              })()}

              {/* Parameter Variance Decomposition */}
              {trials.length >= 8 && (() => {
                const vdObjKey = campaign.objective_names?.[0] || Object.keys(trials[0].kpis)[0];
                const vdParams = campaign.spec?.parameters?.filter((s: { name: string; type: string; lower?: number; upper?: number }) => s.type === "continuous" && s.lower != null && s.upper != null) || [];
                if (vdParams.length < 2) return null;
                const vdObjVals = trials.map(t => t.kpis[vdObjKey] ?? 0);
                const vdMean = vdObjVals.reduce((a: number, b: number) => a + b, 0) / vdObjVals.length;
                const vdTotalVar = vdObjVals.reduce((a: number, v: number) => a + (v - vdMean) ** 2, 0) / vdObjVals.length;
                if (vdTotalVar < 1e-15) return null;
                const vdK = 5;
                const vdColors = ["#3b82f6", "#14b8a6", "#f59e0b", "#f43f5e", "#8b5cf6", "#06b6d4", "#84cc16"];
                // Compute marginal variance for each parameter
                const vdMarginals: { name: string; variance: number; color: string }[] = [];
                for (let pi = 0; pi < Math.min(vdParams.length, 6); pi++) {
                  const sp = vdParams[pi] as { name: string; lower?: number; upper?: number };
                  const pVals = trials.map(t => t.parameters[sp.name] ?? 0);
                  const sorted = pVals.map((v: number, i: number) => ({ v, obj: vdObjVals[i] })).sort((a: { v: number }, b: { v: number }) => a.v - b.v);
                  const binSize = Math.max(1, Math.floor(sorted.length / vdK));
                  const binMeans: number[] = [];
                  for (let b = 0; b < vdK; b++) {
                    const start = b * binSize;
                    const end = b === vdK - 1 ? sorted.length : (b + 1) * binSize;
                    const slice = sorted.slice(start, end);
                    if (slice.length === 0) continue;
                    binMeans.push(slice.reduce((a: number, c: { obj: number }) => a + c.obj, 0) / slice.length);
                  }
                  const bmMean = binMeans.reduce((a, b) => a + b, 0) / binMeans.length;
                  const margVar = binMeans.reduce((a, v) => a + (v - bmMean) ** 2, 0) / binMeans.length;
                  vdMarginals.push({ name: sp.name, variance: margVar, color: vdColors[pi % vdColors.length] });
                }
                const vdMargSum = vdMarginals.reduce((a, m) => a + m.variance, 0);
                // Interaction: residual not explained by marginals
                const vdInteraction = Math.max(0, vdTotalVar - vdMargSum) * 0.4; // conservative estimate
                const vdResidual = Math.max(0, vdTotalVar - vdMargSum - vdInteraction);
                const vdAll = vdMarginals.reduce((a, m) => a + m.variance, 0) + vdInteraction + vdResidual;
                const vdBarW = 200, vdBarH = 22;
                let vdX = 0;
                const vdSegments = vdMarginals.map(m => {
                  const w = (m.variance / vdAll) * vdBarW;
                  const seg = { x: vdX, w, color: m.color, name: m.name, pct: (m.variance / vdAll) * 100 };
                  vdX += w;
                  return seg;
                });
                const vdIntW = (vdInteraction / vdAll) * vdBarW;
                const vdResW = (vdResidual / vdAll) * vdBarW;
                const vdDominant = vdMarginals.length > 0 && (vdMarginals[0].variance / vdAll) > 0.6 ?
                  vdMarginals.reduce((max, m) => m.variance > max.variance ? m : max) : null;
                return (
                  <div className="card" style={{ padding: "14px 16px" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
                      <PieChart size={16} style={{ color: "var(--color-text-muted)" }} />
                      <h3 style={{ margin: 0, fontSize: "0.92rem" }}>Variance Decomposition</h3>
                      {vdDominant && (
                        <span className="findings-badge" style={{ background: "#f59e0b18", color: "#f59e0b", marginLeft: "auto" }}>
                          {vdDominant.name} dominates ({(vdDominant.variance / vdAll * 100).toFixed(0)}%)
                        </span>
                      )}
                    </div>
                    <svg width="100%" viewBox={`0 0 ${vdBarW} ${vdBarH + 30}`} style={{ display: "block" }}>
                      <defs>
                        <pattern id="vd-hatch" patternUnits="userSpaceOnUse" width="5" height="5" patternTransform="rotate(45)">
                          <line x1="0" y1="0" x2="0" y2="5" stroke="#64748b" strokeWidth="1.5" />
                        </pattern>
                      </defs>
                      {/* Stacked bar */}
                      {vdSegments.map((seg, i) => (
                        <rect key={`vds${i}`} x={seg.x} y={0} width={Math.max(seg.w, 0.5)} height={vdBarH} fill={seg.color} rx={i === 0 ? 4 : 0} opacity={0.85} />
                      ))}
                      {/* Interaction segment */}
                      <rect x={vdX} y={0} width={Math.max(vdIntW, 0.5)} height={vdBarH} fill="url(#vd-hatch)" opacity={0.6} />
                      {/* Residual segment */}
                      <rect x={vdX + vdIntW} y={0} width={Math.max(vdResW, 0.5)} height={vdBarH} fill="#e2e8f0" rx={0} />
                      {/* Last segment rounded right */}
                      <rect x={vdBarW - 4} y={0} width={4} height={vdBarH} fill="#e2e8f0" rx={4} />
                      {/* Legend below */}
                      {vdSegments.slice(0, 4).map((seg, i) => (
                        <g key={`vdl${i}`} transform={`translate(${i * 50}, ${vdBarH + 6})`}>
                          <rect x={0} y={0} width={7} height={7} rx={1.5} fill={seg.color} opacity={0.85} />
                          <text x={10} y={6.5} fontSize="5.5" fill="var(--color-text-muted)" fontFamily="var(--font-mono)">{seg.name.length > 6 ? seg.name.slice(0, 6) + ".." : seg.name}</text>
                        </g>
                      ))}
                      {vdIntW > 2 && (
                        <g transform={`translate(${Math.min(vdSegments.length, 4) * 50}, ${vdBarH + 6})`}>
                          <rect x={0} y={0} width={7} height={7} rx={1.5} fill="url(#vd-hatch)" />
                          <text x={10} y={6.5} fontSize="5.5" fill="var(--color-text-muted)">Interactions</text>
                        </g>
                      )}
                    </svg>
                    <div style={{ display: "flex", gap: 12, flexWrap: "wrap", fontSize: "0.72rem", color: "var(--color-text-muted)", marginTop: 4 }}>
                      {vdSegments.slice(0, 4).map(seg => (
                        <span key={seg.name}><strong style={{ color: seg.color }}>{seg.pct.toFixed(0)}%</strong> {seg.name}</span>
                      ))}
                      {vdIntW > 2 && <span><strong style={{ color: "#64748b" }}>{((vdInteraction / vdAll) * 100).toFixed(0)}%</strong> interactions</span>}
                    </div>
                  </div>
                );
              })()}

              {/* Ensemble Disagreement Map */}
              {trials.length >= 10 && (() => {
                const edSpecs = campaign.spec?.parameters?.filter((s: { name: string; type: string; lower?: number; upper?: number }) => s.type === "continuous" && s.lower != null && s.upper != null) || [];
                if (edSpecs.length < 2) return null;
                const edObjKey = campaign.objective_names?.[0] || Object.keys(trials[0].kpis)[0];
                // Pick top-2 parameters by variance in objective across quantile bins
                const edVarScores = edSpecs.map((sp: { name: string; lower?: number; upper?: number }) => {
                  const vals = trials.map(t => t.parameters[sp.name] ?? 0);
                  const objs = trials.map(t => t.kpis[edObjKey] ?? 0);
                  const sorted = vals.map((v: number, i: number) => ({ v, o: objs[i] })).sort((a: { v: number }, b: { v: number }) => a.v - b.v);
                  const half = Math.floor(sorted.length / 2);
                  const lo = sorted.slice(0, half).map((s: { o: number }) => s.o);
                  const hi = sorted.slice(half).map((s: { o: number }) => s.o);
                  const loMean = lo.reduce((a: number, b: number) => a + b, 0) / lo.length;
                  const hiMean = hi.reduce((a: number, b: number) => a + b, 0) / hi.length;
                  return { name: sp.name, lo: sp.lower ?? 0, hi: sp.upper ?? 1, score: Math.abs(hiMean - loMean) };
                }).sort((a: { score: number }, b: { score: number }) => b.score - a.score);
                const edP1 = edVarScores[0], edP2 = edVarScores[1];
                // Build 10x10 grid of LOO disagreement
                const edG = 10;
                const edGrid: number[][] = Array.from({ length: edG }, () => Array(edG).fill(0));
                const edNorm = (v: number, lo: number, hi: number) => hi > lo ? (v - lo) / (hi - lo) : 0.5;
                // For each grid cell, compute LOO prediction disagreement (std of k-nearest preds)
                const edK = 5;
                for (let gi = 0; gi < edG; gi++) {
                  for (let gj = 0; gj < edG; gj++) {
                    const cx = (gi + 0.5) / edG;
                    const cy = (gj + 0.5) / edG;
                    // Find k nearest trials
                    const dists = trials.map((t, idx) => ({
                      idx,
                      d: Math.sqrt((edNorm(t.parameters[edP1.name], edP1.lo, edP1.hi) - cx) ** 2 + (edNorm(t.parameters[edP2.name], edP2.lo, edP2.hi) - cy) ** 2),
                      obj: t.kpis[edObjKey] ?? 0,
                    })).sort((a, b) => a.d - b.d).slice(0, edK);
                    // LOO predictions: each point predicts using remaining k-1
                    const preds: number[] = [];
                    for (let li = 0; li < dists.length; li++) {
                      const others = dists.filter((_, j) => j !== li);
                      const wSum = others.reduce((a, o) => a + 1 / Math.max(o.d, 0.01), 0);
                      const pred = others.reduce((a, o) => a + (o.obj / Math.max(o.d, 0.01)), 0) / Math.max(wSum, 1e-10);
                      preds.push(pred);
                    }
                    const predMean = preds.reduce((a, b) => a + b, 0) / preds.length;
                    const disagreement = Math.sqrt(preds.reduce((a, v) => a + (v - predMean) ** 2, 0) / preds.length);
                    edGrid[gi][gj] = disagreement;
                  }
                }
                const edMax = Math.max(...edGrid.flat(), 1e-10);
                const edAvg = edGrid.flat().reduce((a, b) => a + b, 0) / (edG * edG);
                const edCellW = 14, edCellH = 14;
                const edSvgW = edG * edCellW + 30, edSvgH = edG * edCellH + 25;
                const edOx = 25, edOy = 5;
                return (
                  <div className="card" style={{ padding: "14px 16px" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                      <Grid3x3 size={16} style={{ color: "var(--color-text-muted)" }} />
                      <h3 style={{ margin: 0, fontSize: "0.92rem" }}>Model Disagreement</h3>
                      <span className="findings-badge" style={{ background: edAvg / edMax > 0.5 ? "#8b5cf618" : "#22c55e18", color: edAvg / edMax > 0.5 ? "#8b5cf6" : "#22c55e", marginLeft: "auto" }}>
                        avg {((edAvg / edMax) * 100).toFixed(0)}% disagreement
                      </span>
                    </div>
                    <svg width="100%" viewBox={`0 0 ${edSvgW} ${edSvgH}`} style={{ display: "block" }}>
                      {/* Heatmap cells */}
                      {edGrid.map((row, gi) => row.map((val, gj) => {
                        const intensity = val / edMax;
                        const r = Math.round(255 * intensity);
                        const g = Math.round(255 * (1 - intensity * 0.7));
                        const b = Math.round(180 + 75 * intensity);
                        return (
                          <rect key={`ed${gi}-${gj}`} x={edOx + gi * edCellW} y={edOy + (edG - 1 - gj) * edCellH} width={edCellW - 1} height={edCellH - 1} rx={2} fill={`rgb(${r},${g},${b})`} opacity={0.8} />
                        );
                      }))}
                      {/* Observation dots */}
                      {trials.slice(-40).map((t, i) => {
                        const nx = edNorm(t.parameters[edP1.name], edP1.lo, edP1.hi);
                        const ny = edNorm(t.parameters[edP2.name], edP2.lo, edP2.hi);
                        return <circle key={`edo${i}`} cx={edOx + nx * edG * edCellW} cy={edOy + (1 - ny) * edG * edCellH} r="1.8" fill="white" stroke="#333" strokeWidth="0.5" />;
                      })}
                      {/* Axis labels */}
                      <text x={edOx + edG * edCellW / 2} y={edSvgH - 1} fontSize="6" fill="var(--color-text-muted)" textAnchor="middle">{edP1.name}</text>
                      <text x={3} y={edOy + edG * edCellH / 2} fontSize="6" fill="var(--color-text-muted)" textAnchor="middle" transform={`rotate(-90, 3, ${edOy + edG * edCellH / 2})`}>{edP2.name}</text>
                    </svg>
                    <div style={{ display: "flex", gap: 8, fontSize: "0.72rem", color: "var(--color-text-muted)", marginTop: 4 }}>
                      <span style={{ display: "flex", alignItems: "center", gap: 3 }}>
                        <span style={{ display: "inline-block", width: 10, height: 10, borderRadius: 2, background: "rgb(255,76,255)" }} /> High
                      </span>
                      <span style={{ display: "flex", alignItems: "center", gap: 3 }}>
                        <span style={{ display: "inline-block", width: 10, height: 10, borderRadius: 2, background: "rgb(128,191,218)" }} /> Medium
                      </span>
                      <span style={{ display: "flex", alignItems: "center", gap: 3 }}>
                        <span style={{ display: "inline-block", width: 10, height: 10, borderRadius: 2, background: "rgb(0,255,180)" }} /> Low
                      </span>
                      <span style={{ marginLeft: "auto" }}>LOO cross-validation on {edK}-NN</span>
                    </div>
                  </div>
                );
              })()}

              {/* Response Surface Curvature */}
              {trials.length >= 12 && (() => {
                const rcSpecs = campaign.spec?.parameters?.filter((s: { name: string; type: string; lower?: number; upper?: number }) => s.type === "continuous" && s.lower != null && s.upper != null) || [];
                if (rcSpecs.length === 0) return null;
                const rcKpi = Object.keys(trials[0]?.kpis || {})[0];
                if (!rcKpi) return null;

                // For each parameter, estimate curvature using binned 2nd-order finite differences
                const rcData = rcSpecs.slice(0, 8).map((s: { name: string; lower?: number; upper?: number }) => {
                  const lo = s.lower ?? 0, hi = s.upper ?? 1;
                  const range = hi - lo || 1;
                  const nBins = 5;
                  const binWidth = range / nBins;
                  // Bin trials by parameter value, compute mean kpi per bin
                  const bins: { sum: number; count: number }[] = Array.from({ length: nBins }, () => ({ sum: 0, count: 0 }));
                  for (const t of trials) {
                    const v = Number(t.parameters[s.name]) || 0;
                    const bi = Math.min(nBins - 1, Math.max(0, Math.floor((v - lo) / binWidth)));
                    bins[bi].sum += Number(t.kpis[rcKpi]) || 0;
                    bins[bi].count++;
                  }
                  const binMeans = bins.map(b => b.count > 0 ? b.sum / b.count : NaN);
                  // Compute average absolute 2nd derivative (curvature proxy)
                  let curvSum = 0, curvCount = 0;
                  for (let i = 1; i < nBins - 1; i++) {
                    if (!isNaN(binMeans[i - 1]) && !isNaN(binMeans[i]) && !isNaN(binMeans[i + 1])) {
                      const d2 = Math.abs(binMeans[i + 1] - 2 * binMeans[i] + binMeans[i - 1]);
                      curvSum += d2;
                      curvCount++;
                    }
                  }
                  const curvature = curvCount > 0 ? curvSum / curvCount : 0;
                  return { name: s.name, curvature, binMeans };
                });

                const rcMax = Math.max(...rcData.map(d => d.curvature), 0.001);
                const rcW = 260, rcH = 22 * rcData.length + 28;
                const rcPadL = 80, rcPadR = 50, rcPadT = 4;
                const rcBarH = 14;
                const rcPlotW = rcW - rcPadL - rcPadR;

                const rcHighCurv = rcData.filter(d => d.curvature / rcMax > 0.6).length;

                return (
                  <div className="card" style={{ marginBottom: 18 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
                      <Aperture size={18} style={{ color: "var(--color-text-muted)" }} />
                      <div>
                        <h2 style={{ margin: 0 }}>Response Surface Curvature</h2>
                        <p className="stat-label" style={{ margin: 0, textTransform: "none" }}>
                          Estimated 2nd-derivative magnitude per parameter. High curvature = nonlinear response.
                        </p>
                      </div>
                      {rcHighCurv > 0 && (
                        <span className="findings-badge" style={{ marginLeft: "auto", color: "#8b5cf6", borderColor: "rgba(139,92,246,0.3)" }}>
                          {rcHighCurv} nonlinear
                        </span>
                      )}
                    </div>
                    <div style={{ overflowX: "auto" }}>
                      <svg width={rcW} height={rcH} viewBox={`0 0 ${rcW} ${rcH}`} style={{ display: "block", margin: "0 auto" }}>
                        {rcData.map((d, i) => {
                          const y = rcPadT + i * 22;
                          const barW = (d.curvature / rcMax) * rcPlotW;
                          const color = d.curvature / rcMax > 0.6 ? "#8b5cf6" : d.curvature / rcMax > 0.3 ? "#3b82f6" : "var(--color-text-muted)";
                          return (
                            <g key={i}>
                              <text x={rcPadL - 6} y={y + rcBarH / 2 + 3} textAnchor="end" fontSize="9" fontFamily="var(--font-mono)" fontWeight={d.curvature / rcMax > 0.6 ? 600 : 400} fill={color}>
                                {d.name.length > 10 ? d.name.slice(0, 9) + "…" : d.name}
                              </text>
                              <rect x={rcPadL} y={y} width={rcPlotW} height={rcBarH} rx={3} fill="var(--color-border)" opacity={0.3} />
                              <rect x={rcPadL} y={y} width={Math.max(2, barW)} height={rcBarH} rx={3} fill={color} opacity={0.6} />
                              <text x={rcPadL + Math.max(2, barW) + 4} y={y + rcBarH / 2 + 3} fontSize="8" fill={color} fontFamily="var(--font-mono)" fontWeight={500}>
                                {d.curvature.toFixed(4)}
                              </text>
                            </g>
                          );
                        })}
                        {/* X axis label */}
                        <text x={rcPadL + rcPlotW / 2} y={rcH - 2} textAnchor="middle" fontSize="7" fill="var(--color-text-muted)">|d²f/dx²| (binned finite difference)</text>
                      </svg>
                    </div>
                    <div style={{ display: "flex", gap: "16px", marginTop: "4px", flexWrap: "wrap", alignItems: "center" }}>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: 14, height: 8, background: "#8b5cf6", marginRight: 4, verticalAlign: "middle", borderRadius: 2, opacity: 0.6 }} />Nonlinear</span>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: 14, height: 8, background: "#3b82f6", marginRight: 4, verticalAlign: "middle", borderRadius: 2, opacity: 0.6 }} />Moderate</span>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: 14, height: 8, background: "var(--color-text-muted)", marginRight: 4, verticalAlign: "middle", borderRadius: 2, opacity: 0.6 }} />Linear</span>
                    </div>
                  </div>
                );
              })()}

              {/* Marginal Response Profiles */}
              {trials.length >= 8 && (() => {
                const mrSpecs = campaign.spec?.parameters?.filter((s: { name: string; type: string; lower?: number; upper?: number }) => s.type === "continuous" && s.lower != null && s.upper != null) || [];
                if (mrSpecs.length === 0) return null;
                const mrKpiKey = Object.keys(trials[0]?.kpis || {})[0];
                if (!mrKpiKey) return null;
                const mrParams = mrSpecs.slice(0, 6); // Up to 6 parameters
                const mrNBins = 5;

                // For each parameter, bin trials and compute mean + stderr of KPI
                const mrProfiles = mrParams.map((spec: { name: string; lower?: number; upper?: number }) => {
                  const lo = spec.lower!;
                  const hi = spec.upper!;
                  const range = hi - lo || 1;
                  const bins: { center: number; mean: number; stderr: number; count: number }[] = [];
                  for (let b = 0; b < mrNBins; b++) {
                    const bLo = lo + (b / mrNBins) * range;
                    const bHi = lo + ((b + 1) / mrNBins) * range;
                    const bCenter = (bLo + bHi) / 2;
                    const inBin = trials.filter(t => {
                      const v = Number(t.parameters[spec.name]) ?? 0;
                      return v >= bLo && v < bHi;
                    }).map(t => t.kpis[mrKpiKey] ?? 0);
                    if (inBin.length > 0) {
                      const mean = inBin.reduce((a, b2) => a + b2, 0) / inBin.length;
                      const variance = inBin.length > 1 ? inBin.reduce((s, v) => s + (v - mean) ** 2, 0) / (inBin.length - 1) : 0;
                      bins.push({ center: bCenter, mean, stderr: Math.sqrt(variance / inBin.length), count: inBin.length });
                    }
                  }
                  // Compute monotonicity score
                  let mrMono = 0;
                  if (bins.length > 1) {
                    let ups = 0, downs = 0;
                    for (let i2 = 1; i2 < bins.length; i2++) {
                      if (bins[i2].mean > bins[i2 - 1].mean) ups++;
                      else if (bins[i2].mean < bins[i2 - 1].mean) downs++;
                    }
                    mrMono = Math.max(ups, downs) / (bins.length - 1);
                  }
                  return { name: spec.name, bins, lo, hi, monotonicity: mrMono };
                });

                // SVG layout: small multiples
                const mrColW = 130, mrColH = 80, mrPad = 20, mrPadB = 16;
                const mrCols = Math.min(mrProfiles.length, 3);
                const mrRows = Math.ceil(mrProfiles.length / mrCols);
                const mrW = mrCols * mrColW + mrPad;
                const mrH = mrRows * mrColH + mrPad;

                // Global KPI range for consistent y-axis
                const mrAllMeans = mrProfiles.flatMap(p => p.bins.map(b => b.mean));
                const mrAllErrs = mrProfiles.flatMap(p => p.bins.map(b => b.stderr));
                const mrYMin = Math.min(...mrAllMeans.map((m, i) => m - mrAllErrs[i]));
                const mrYMax = Math.max(...mrAllMeans.map((m, i) => m + mrAllErrs[i]));
                const mrYRange = mrYMax - mrYMin || 1;

                const mrMonotonic = mrProfiles.filter(p => p.monotonicity > 0.8).length;

                return (
                  <div className="card" style={{ marginBottom: 18 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
                      <AlignVerticalJustifyStart size={18} style={{ color: "var(--color-text-muted)" }} />
                      <div>
                        <h2 style={{ margin: 0 }}>Marginal Response Profiles</h2>
                        <div style={{ fontSize: "0.78rem", color: "var(--color-text-muted)" }}>1D binned response per parameter</div>
                      </div>
                      <span className="findings-badge" style={{ marginLeft: "auto" }}>
                        {mrMonotonic} monotonic
                      </span>
                    </div>
                    <svg width={mrW} height={mrH} viewBox={`0 0 ${mrW} ${mrH}`} style={{ width: "100%", height: "auto" }}>
                      {mrProfiles.map((prof, pi) => {
                        const col = pi % mrCols;
                        const row = Math.floor(pi / mrCols);
                        const ox = col * mrColW + mrPad;
                        const oy = row * mrColH + 6;
                        const pw = mrColW - mrPad - 10;
                        const ph = mrColH - mrPadB - 12;
                        const xScale = (v: number) => ox + ((v - prof.lo) / (prof.hi - prof.lo || 1)) * pw;
                        const yScale = (v: number) => oy + ph - ((v - mrYMin) / mrYRange) * ph;

                        return (
                          <g key={`mr${pi}`}>
                            {/* Param name */}
                            <text x={ox + pw / 2} y={oy + 6} fontSize="6.5" fill="var(--color-text)" textAnchor="middle" fontWeight="600">{prof.name}</text>
                            {/* Baseline grid */}
                            <line x1={ox} y1={oy + ph} x2={ox + pw} y2={oy + ph} stroke="var(--color-border)" strokeWidth="0.5" />
                            <line x1={ox} y1={oy + 10} x2={ox} y2={oy + ph} stroke="var(--color-border)" strokeWidth="0.5" />
                            {/* Error band (confidence whiskers) */}
                            {prof.bins.map((b, bi) => {
                              const bx = xScale(b.center);
                              return (
                                <g key={`mre${bi}`}>
                                  <line x1={bx} y1={yScale(b.mean + b.stderr)} x2={bx} y2={yScale(b.mean - b.stderr)} stroke="var(--color-primary)" strokeWidth="1.5" opacity="0.3" />
                                  <line x1={bx - 2} y1={yScale(b.mean + b.stderr)} x2={bx + 2} y2={yScale(b.mean + b.stderr)} stroke="var(--color-primary)" strokeWidth="0.8" opacity="0.4" />
                                  <line x1={bx - 2} y1={yScale(b.mean - b.stderr)} x2={bx + 2} y2={yScale(b.mean - b.stderr)} stroke="var(--color-primary)" strokeWidth="0.8" opacity="0.4" />
                                </g>
                              );
                            })}
                            {/* Mean line */}
                            {prof.bins.length > 1 && (
                              <polyline
                                points={prof.bins.map(b => `${xScale(b.center)},${yScale(b.mean)}`).join(" ")}
                                fill="none"
                                stroke={prof.monotonicity > 0.8 ? "#22c55e" : "var(--color-primary)"}
                                strokeWidth="1.5"
                                strokeLinejoin="round"
                              />
                            )}
                            {/* Mean dots */}
                            {prof.bins.map((b, bi) => (
                              <circle
                                key={`mrd${bi}`}
                                cx={xScale(b.center)}
                                cy={yScale(b.mean)}
                                r="2"
                                fill={prof.monotonicity > 0.8 ? "#22c55e" : "var(--color-primary)"}
                              />
                            ))}
                            {/* Monotonicity indicator */}
                            <text x={ox + pw} y={oy + 6} fontSize="5" fill={prof.monotonicity > 0.8 ? "#22c55e" : "var(--color-text-muted)"} textAnchor="end">
                              ρ={prof.monotonicity.toFixed(2)}
                            </text>
                          </g>
                        );
                      })}
                    </svg>
                    <div style={{ display: "flex", gap: "16px", marginTop: 4, flexWrap: "wrap" }}>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: 14, height: 2, background: "#22c55e", marginRight: 4, verticalAlign: "middle" }} />Monotonic (ρ&gt;0.8)</span>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: 14, height: 2, background: "var(--color-primary)", marginRight: 4, verticalAlign: "middle" }} />Non-monotonic</span>
                      <span className="efficiency-legend-item" style={{ opacity: 0.5 }}>|: ±1 SE</span>
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

              {/* What-If Analysis */}
              {trials.length >= 5 && campaign.spec?.parameters && (() => {
                const specs = campaign.spec.parameters.filter(s => s.type === "continuous" && s.lower != null && s.upper != null);
                if (specs.length === 0) return null;
                const paramNames = specs.map(s => s.name);
                const activeParam = whatIfParam ?? paramNames[0];
                const activeSpec = specs.find(s => s.name === activeParam);
                if (!activeSpec) return null;
                const lower = activeSpec.lower!;
                const upper = activeSpec.upper!;
                const range = upper - lower;
                const currentVal = lower + whatIfValue * range;

                // Find nearby trials and compute local weighted average
                const trialDistances = trials.map(t => {
                  const pVal = Number(t.parameters[activeParam]) || 0;
                  const normDist = Math.abs(pVal - currentVal) / range;
                  const weight = Math.exp(-normDist * normDist * 50); // Gaussian kernel
                  const objVal = Number(Object.values(t.kpis)[0]) || 0;
                  return { objVal, weight, pVal, iteration: t.iteration, normDist };
                });
                const nearbyTrials = trialDistances.filter(t => t.normDist < 0.15).sort((a, b) => a.normDist - b.normDist);
                const totalWeight = trialDistances.reduce((a, t) => a + t.weight, 0);
                const predictedObj = totalWeight > 0 ? trialDistances.reduce((a, t) => a + t.objVal * t.weight, 0) / totalWeight : 0;
                // Weighted std for confidence band
                const weightedVariance = totalWeight > 0
                  ? trialDistances.reduce((a, t) => a + t.weight * (t.objVal - predictedObj) ** 2, 0) / totalWeight
                  : 0;
                const confidence = Math.sqrt(weightedVariance);

                // Mini scatter data — parameter vs objective
                const objVals = trials.map(t => Number(Object.values(t.kpis)[0]) || 0);
                const minObj = Math.min(...objVals);
                const maxObj = Math.max(...objVals);
                const objRange = maxObj - minObj || 1;
                const scatterW = 300, scatterH = 100, pad = 4;

                return (
                  <div className="card whatif-card">
                    <div className="whatif-header">
                      <Sliders size={16} />
                      <h2 style={{ margin: 0 }}>What-If Analysis</h2>
                    </div>
                    <p style={{ fontSize: "0.82rem", color: "var(--color-text-muted)", marginBottom: "16px" }}>
                      Explore how changing a single parameter affects the predicted objective value, based on nearby trial data.
                    </p>
                    <div className="whatif-controls">
                      <label className="whatif-label">
                        Parameter
                        <select
                          className="suggestions-batch-select"
                          value={activeParam}
                          onChange={(e) => { setWhatIfParam(e.target.value); setWhatIfValue(0.5); }}
                        >
                          {paramNames.map(p => <option key={p} value={p}>{p}</option>)}
                        </select>
                      </label>
                      <div className="whatif-slider-group">
                        <div className="whatif-slider-labels">
                          <span className="mono">{lower.toFixed(3)}</span>
                          <span className="mono whatif-current-val">{currentVal.toFixed(4)}</span>
                          <span className="mono">{upper.toFixed(3)}</span>
                        </div>
                        <input
                          type="range"
                          min="0"
                          max="1"
                          step="0.005"
                          value={whatIfValue}
                          onChange={(e) => setWhatIfValue(Number(e.target.value))}
                          className="whatif-slider"
                        />
                      </div>
                    </div>
                    <div className="whatif-results">
                      <div className="whatif-result-box">
                        <div className="whatif-result-label">Predicted Objective</div>
                        <div className="whatif-result-value mono">{predictedObj.toFixed(4)}</div>
                        <div className="whatif-result-ci mono">&plusmn; {confidence.toFixed(4)}</div>
                      </div>
                      <div className="whatif-result-box">
                        <div className="whatif-result-label">Nearby Trials</div>
                        <div className="whatif-result-value">{nearbyTrials.length}</div>
                        <div className="whatif-result-ci">within 15% of slider</div>
                      </div>
                      {bestResult && (
                        <div className="whatif-result-box">
                          <div className="whatif-result-label">vs Current Best</div>
                          <div className={`whatif-result-value mono ${predictedObj < (Number(Object.values(bestResult.kpis)[0]) || 0) ? "whatif-better" : "whatif-worse"}`}>
                            {(predictedObj - (Number(Object.values(bestResult.kpis)[0]) || 0)).toFixed(4)}
                          </div>
                          <div className="whatif-result-ci">{predictedObj < (Number(Object.values(bestResult.kpis)[0]) || 0) ? "improvement" : "worse"}</div>
                        </div>
                      )}
                    </div>
                    {/* Mini scatter */}
                    <svg width={scatterW} height={scatterH} viewBox={`0 0 ${scatterW} ${scatterH}`} className="whatif-scatter">
                      {trials.map((t, i) => {
                        const px = pad + ((Number(t.parameters[activeParam]) || 0) - lower) / range * (scatterW - 2 * pad);
                        const py = pad + (1 - ((Number(Object.values(t.kpis)[0]) || 0) - minObj) / objRange) * (scatterH - 2 * pad);
                        const dist = trialDistances[i].normDist;
                        return (
                          <circle
                            key={i}
                            cx={px}
                            cy={py}
                            r={dist < 0.15 ? 3.5 : 2}
                            fill={dist < 0.15 ? "var(--color-primary)" : "var(--color-text-muted)"}
                            opacity={dist < 0.15 ? 0.8 : 0.2}
                          />
                        );
                      })}
                      {/* Slider position line */}
                      <line
                        x1={pad + whatIfValue * (scatterW - 2 * pad)}
                        y1={pad}
                        x2={pad + whatIfValue * (scatterW - 2 * pad)}
                        y2={scatterH - pad}
                        stroke="var(--color-primary)"
                        strokeWidth="1.5"
                        strokeDasharray="4,3"
                        opacity="0.6"
                      />
                      {/* Predicted point */}
                      <circle
                        cx={pad + whatIfValue * (scatterW - 2 * pad)}
                        cy={pad + (1 - (predictedObj - minObj) / objRange) * (scatterH - 2 * pad)}
                        r="5"
                        fill="var(--color-primary)"
                        stroke="white"
                        strokeWidth="2"
                      />
                    </svg>
                  </div>
                );
              })()}

              {/* Parameter Sensitivity Radar Chart */}
              {trials.length >= 5 && importance && importance.importances.length >= 3 && (() => {
                const items = [...importance.importances].sort((a, b) => b.importance - a.importance);
                const n = items.length;
                const cx = 130, cy = 130, R = 100;
                const angleStep = (2 * Math.PI) / n;
                const maxImp = Math.max(...items.map(i => i.importance), 0.01);
                // Compute objective correlation per parameter
                const objVals = trials.map(t => Number(Object.values(t.kpis)[0]) || 0);
                const meanObj = objVals.reduce((a, b) => a + b, 0) / objVals.length;
                const corrMap: Record<string, number> = {};
                items.forEach(item => {
                  const pVals = trials.map(t => Number(t.parameters[item.name]) || 0);
                  const meanP = pVals.reduce((a, b) => a + b, 0) / pVals.length;
                  let num = 0, denP = 0, denO = 0;
                  pVals.forEach((p, i) => { const dp = p - meanP; const dobj = objVals[i] - meanObj; num += dp * dobj; denP += dp * dp; denO += dobj * dobj; });
                  corrMap[item.name] = denP > 0 && denO > 0 ? num / Math.sqrt(denP * denO) : 0;
                });
                // Build radar polygon
                const points = items.map((item, i) => {
                  const angle = -Math.PI / 2 + i * angleStep;
                  const r = (item.importance / maxImp) * R;
                  return { x: cx + r * Math.cos(angle), y: cy + r * Math.sin(angle), name: item.name, imp: item.importance, corr: corrMap[item.name] || 0, angle };
                });
                const polyStr = points.map(p => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
                // Grid rings
                const rings = [0.25, 0.5, 0.75, 1.0];
                return (
                  <div className="card">
                    <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}>
                      <Hexagon size={16} style={{ color: "var(--color-primary)" }} />
                      <h2 style={{ margin: 0 }}>Parameter Sensitivity</h2>
                    </div>
                    <p className="range-desc">Radar view of parameter importance. Larger radius = higher importance. Color indicates objective correlation direction.</p>
                    <div style={{ display: "flex", justifyContent: "center" }}>
                      <svg width={260} height={260} viewBox="0 0 260 260" style={{ overflow: "visible" }}>
                        {/* Grid rings */}
                        {rings.map(r => (
                          <circle key={r} cx={cx} cy={cy} r={R * r} fill="none" stroke="var(--color-border)" strokeWidth="0.5" strokeDasharray={r < 1 ? "3,3" : "none"} />
                        ))}
                        {/* Axis lines */}
                        {points.map((p, i) => (
                          <line key={i} x1={cx} y1={cy} x2={cx + R * Math.cos(p.angle)} y2={cy + R * Math.sin(p.angle)} stroke="var(--color-border)" strokeWidth="0.5" />
                        ))}
                        {/* Filled polygon */}
                        <polygon points={polyStr} fill="var(--color-primary)" fillOpacity={0.15} stroke="var(--color-primary)" strokeWidth="1.5" />
                        {/* Data points */}
                        {points.map((p, i) => (
                          <g key={i}>
                            <circle cx={p.x} cy={p.y} r={5} fill={p.corr < -0.2 ? "#22c55e" : p.corr > 0.2 ? "#ef4444" : "var(--color-text-muted)"} stroke="var(--color-surface)" strokeWidth="1.5" />
                            <title>{p.name}: importance {(p.imp * 100).toFixed(1)}%, correlation {p.corr.toFixed(3)}</title>
                          </g>
                        ))}
                        {/* Labels */}
                        {points.map((p, i) => {
                          const labelR = R + 20;
                          const lx = cx + labelR * Math.cos(p.angle);
                          const ly = cy + labelR * Math.sin(p.angle);
                          const anchor = Math.abs(p.angle) < 0.1 || Math.abs(p.angle - Math.PI) < 0.1 ? "middle" : p.angle > -Math.PI / 2 && p.angle < Math.PI / 2 ? "start" : "end";
                          return (
                            <text key={i} x={lx} y={ly} textAnchor={anchor} dominantBaseline="central" fontSize="11" fontFamily="var(--font-mono)" fill="var(--color-text)" fontWeight={p.imp === maxImp ? 600 : 400}>
                              {p.name}
                            </text>
                          );
                        })}
                      </svg>
                    </div>
                    <div className="radar-legend">
                      <span className="radar-legend-item"><span className="radar-dot" style={{ background: "#22c55e" }} /> Negative corr (better)</span>
                      <span className="radar-legend-item"><span className="radar-dot" style={{ background: "#ef4444" }} /> Positive corr (worse)</span>
                      <span className="radar-legend-item"><span className="radar-dot" style={{ background: "var(--color-text-muted)" }} /> Neutral</span>
                    </div>
                  </div>
                );
              })()}

              {/* Parameter-to-Parameter Correlation Heatmap */}
              {trials.length >= 5 && (() => {
                const paramNames = Object.keys(trials[0].parameters);
                if (paramNames.length < 2) return null;
                // Compute full correlation matrix
                const n = trials.length;
                const means: Record<string, number> = {};
                const stds: Record<string, number> = {};
                for (const p of paramNames) {
                  const vals = trials.map((t) => Number(t.parameters[p]) || 0);
                  means[p] = vals.reduce((a, b) => a + b, 0) / n;
                  stds[p] = Math.sqrt(vals.reduce((a, v) => a + (v - means[p]) ** 2, 0) / n);
                }
                const corrMatrix: number[][] = paramNames.map((pi) =>
                  paramNames.map((pj) => {
                    if (pi === pj) return 1;
                    if (stds[pi] === 0 || stds[pj] === 0) return 0;
                    const cov = trials.reduce((a, t) =>
                      a + (Number(t.parameters[pi]) - means[pi]) * (Number(t.parameters[pj]) - means[pj]), 0) / n;
                    return cov / (stds[pi] * stds[pj]);
                  })
                );
                const cellSize = Math.min(44, Math.max(28, 300 / paramNames.length));
                const labelW = 70;
                const svgW = labelW + paramNames.length * cellSize;
                const svgH = labelW + paramNames.length * cellSize;
                const corrColor = (v: number) => {
                  const abs = Math.abs(v);
                  if (v > 0) return `rgba(37, 99, 235, ${abs * 0.8})`;
                  if (v < 0) return `rgba(220, 38, 38, ${abs * 0.8})`;
                  return "transparent";
                };
                return (
                  <div className="card">
                    <h2>Parameter Correlation Heatmap</h2>
                    <p style={{ fontSize: "0.82rem", color: "var(--color-text-muted)", marginBottom: "16px" }}>
                      Pearson correlation between parameter pairs. Blue = positive, Red = negative. Strong inter-parameter correlations may indicate redundancy.
                    </p>
                    <div style={{ overflowX: "auto" }}>
                      <svg width={svgW} height={svgH} style={{ display: "block" }}>
                        {/* Column labels (top) */}
                        {paramNames.map((p, j) => (
                          <text
                            key={`cl-${j}`}
                            x={labelW + j * cellSize + cellSize / 2}
                            y={labelW - 6}
                            textAnchor="end"
                            fontSize="10"
                            fontFamily="var(--font-mono)"
                            fill="var(--color-text-muted)"
                            transform={`rotate(-45 ${labelW + j * cellSize + cellSize / 2} ${labelW - 6})`}
                          >
                            {p.length > 8 ? p.slice(0, 7) + "…" : p}
                          </text>
                        ))}
                        {/* Row labels + cells */}
                        {paramNames.map((pi, i) => (
                          <g key={`row-${i}`}>
                            <text
                              x={labelW - 6}
                              y={labelW + i * cellSize + cellSize / 2 + 3}
                              textAnchor="end"
                              fontSize="10"
                              fontFamily="var(--font-mono)"
                              fill="var(--color-text-muted)"
                            >
                              {pi.length > 8 ? pi.slice(0, 7) + "…" : pi}
                            </text>
                            {paramNames.map((_pj, j) => {
                              const v = corrMatrix[i][j];
                              return (
                                <g key={`cell-${i}-${j}`}>
                                  <rect
                                    x={labelW + j * cellSize}
                                    y={labelW + i * cellSize}
                                    width={cellSize - 1}
                                    height={cellSize - 1}
                                    rx="3"
                                    fill={corrColor(v)}
                                    stroke="var(--color-border)"
                                    strokeWidth="0.5"
                                  >
                                    <title>{`${paramNames[i]} vs ${paramNames[j]}: ${v.toFixed(3)}`}</title>
                                  </rect>
                                  {cellSize >= 32 && (
                                    <text
                                      x={labelW + j * cellSize + (cellSize - 1) / 2}
                                      y={labelW + i * cellSize + (cellSize - 1) / 2 + 3}
                                      textAnchor="middle"
                                      fontSize="9"
                                      fontFamily="var(--font-mono)"
                                      fontWeight="600"
                                      fill={Math.abs(v) > 0.4 ? "white" : "var(--color-text-muted)"}
                                    >
                                      {i === j ? "1" : v.toFixed(2)}
                                    </text>
                                  )}
                                </g>
                              );
                            })}
                          </g>
                        ))}
                      </svg>
                    </div>
                    {/* Legend */}
                    <div style={{ display: "flex", alignItems: "center", gap: "8px", marginTop: "12px", fontSize: "0.75rem", color: "var(--color-text-muted)" }}>
                      <span style={{ display: "inline-block", width: "14px", height: "14px", borderRadius: "3px", background: "rgba(220, 38, 38, 0.7)" }} />
                      <span>Negative</span>
                      <span style={{ display: "inline-block", width: "14px", height: "14px", borderRadius: "3px", background: "var(--color-border)" }} />
                      <span>Zero</span>
                      <span style={{ display: "inline-block", width: "14px", height: "14px", borderRadius: "3px", background: "rgba(37, 99, 235, 0.7)" }} />
                      <span>Positive</span>
                    </div>
                  </div>
                );
              })()}

              {/* Parameter Interaction Network */}
              {trials.length >= 15 && (() => {
                const pNames = Object.keys(trials[0].parameters);
                if (pNames.length < 3) return null;
                const oKey = Object.keys(trials[0].kpis)[0];
                const oVals = trials.map(t => Number(t.kpis[oKey]) || 0);
                const oMean = oVals.reduce((a, b) => a + b, 0) / oVals.length;

                // Compute pairwise interaction scores
                const edges: Array<{ a: number; b: number; strength: number; synergy: boolean }> = [];
                for (let i = 0; i < pNames.length; i++) {
                  for (let j = i + 1; j < pNames.length; j++) {
                    const aVals = trials.map(t => Number(t.parameters[pNames[i]]) || 0);
                    const bVals = trials.map(t => Number(t.parameters[pNames[j]]) || 0);
                    const aMean = aVals.reduce((s, v) => s + v, 0) / aVals.length;
                    const bMean = bVals.reduce((s, v) => s + v, 0) / bVals.length;

                    // Interaction: correlation of (a*b) with objective residual
                    const abProduct = aVals.map((a, k) => (a - aMean) * (bVals[k] - bMean));
                    const oResidual = oVals.map(v => v - oMean);
                    const prodMean = abProduct.reduce((s, v) => s + v, 0) / abProduct.length;
                    const resMean = oResidual.reduce((s, v) => s + v, 0) / oResidual.length;
                    let num = 0, denA = 0, denB = 0;
                    for (let k = 0; k < abProduct.length; k++) {
                      const da = abProduct[k] - prodMean;
                      const db = oResidual[k] - resMean;
                      num += da * db;
                      denA += da * da;
                      denB += db * db;
                    }
                    const corr = denA > 0 && denB > 0 ? num / Math.sqrt(denA * denB) : 0;
                    const strength = Math.abs(corr);
                    if (strength > 0.1) {
                      edges.push({ a: i, b: j, strength, synergy: corr < 0 }); // negative corr with minimize = synergistic
                    }
                  }
                }

                // Layout: circular node placement
                const W = 350, H = 280;
                const cx = W / 2, cy = H / 2;
                const radius = Math.min(W, H) / 2 - 40;
                const nodePositions = pNames.map((_, i) => {
                  const angle = (2 * Math.PI * i) / pNames.length - Math.PI / 2;
                  return { x: cx + radius * Math.cos(angle), y: cy + radius * Math.sin(angle) };
                });

                const maxStrength = edges.length > 0 ? Math.max(...edges.map(e => e.strength)) : 1;
                const strongEdges = edges.filter(e => e.strength > 0.15);

                return (
                  <div className="card">
                    <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}>
                      <GitCompare size={16} style={{ color: "var(--color-primary)" }} />
                      <h2 style={{ margin: 0 }}>Interaction Network</h2>
                      <span style={{ fontSize: "0.72rem", color: "var(--color-text-muted)", marginLeft: "auto", fontFamily: "var(--font-mono)" }}>
                        {strongEdges.length} interactions
                      </span>
                    </div>
                    <p style={{ fontSize: "0.78rem", color: "var(--color-text-muted)", margin: "0 0 8px" }}>
                      Parameter interactions detected from joint effects on {oKey}. Thick edges = strong interaction. Green = synergistic, red = antagonistic.
                    </p>
                    <svg width={W} height={H} style={{ display: "block", width: "100%", maxWidth: `${W}px` }}>
                      {/* Edges */}
                      {strongEdges.map((e, i) => {
                        const pa = nodePositions[e.a];
                        const pb = nodePositions[e.b];
                        const thickness = 1 + (e.strength / maxStrength) * 4;
                        const color = e.synergy ? "rgba(34,197,94,0.6)" : "rgba(239,68,68,0.5)";
                        return (
                          <line key={i} x1={pa.x} y1={pa.y} x2={pb.x} y2={pb.y}
                            stroke={color} strokeWidth={thickness} strokeLinecap="round">
                            <title>{pNames[e.a]} × {pNames[e.b]}: {e.synergy ? "synergistic" : "antagonistic"} ({e.strength.toFixed(3)})</title>
                          </line>
                        );
                      })}
                      {/* Nodes */}
                      {pNames.map((name, i) => {
                        const pos = nodePositions[i];
                        const nodeEdges = strongEdges.filter(e => e.a === i || e.b === i);
                        const totalStr = nodeEdges.reduce((s, e) => s + e.strength, 0);
                        const nodeRadius = 16 + Math.min(totalStr * 8, 8);
                        return (
                          <g key={i}>
                            <circle cx={pos.x} cy={pos.y} r={nodeRadius} fill="var(--color-card-bg)" stroke="var(--color-primary)" strokeWidth="2">
                              <title>{name}: {nodeEdges.length} interaction(s), total strength={totalStr.toFixed(3)}</title>
                            </circle>
                            <text x={pos.x} y={pos.y + 3} textAnchor="middle" fontSize="9" fontFamily="var(--font-mono)" fontWeight="600" fill="var(--color-text-primary)">
                              {name.length > 6 ? name.slice(0, 5) + "…" : name}
                            </text>
                          </g>
                        );
                      })}
                    </svg>
                    <div className="efficiency-legend" style={{ maxWidth: `${W}px` }}>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: 14, height: 3, background: "rgba(34,197,94,0.8)", borderRadius: 1, marginRight: 4, verticalAlign: "middle" }} />Synergistic</span>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: 14, height: 3, background: "rgba(239,68,68,0.7)", borderRadius: 1, marginRight: 4, verticalAlign: "middle" }} />Antagonistic</span>
                      <span className="efficiency-legend-item" style={{ color: "var(--color-text-muted)", fontSize: "0.72rem" }}>Edge width ∝ effect size</span>
                    </div>
                  </div>
                );
              })()}

              {/* Parallel Coordinates Plot */}
              {trials.length >= 3 && (() => {
                const paramNames = Object.keys(trials[0].parameters);
                if (paramNames.length < 2) return null;
                const objKey = Object.keys(trials[0].kpis)[0];
                const objVals = trials.map(t => Number(t.kpis[objKey]) || 0);
                const objMin = Math.min(...objVals);
                const objMax = Math.max(...objVals);
                const objRange = objMax - objMin || 1;
                // Compute min/max for each param
                const paramBounds = paramNames.map(p => {
                  const vals = trials.map(t => Number(t.parameters[p]) || 0);
                  const min = Math.min(...vals);
                  const max = Math.max(...vals);
                  return { name: p, min, max, range: max - min || 1 };
                });
                const W = Math.max(500, paramNames.length * 100);
                const H = 260;
                const padL = 50, padR = 30, padT = 30, padB = 40;
                const plotW = W - padL - padR;
                const plotH = H - padT - padB;
                const axisSpacing = plotW / (paramNames.length - 1);
                // Color function: green(good) → yellow → red(bad) for minimization
                const trialColor = (objVal: number) => {
                  const t = (objVal - objMin) / objRange;
                  if (t < 0.33) return `rgba(34, 197, 94, ${0.35 + t * 0.5})`;
                  if (t < 0.67) return `rgba(234, 179, 8, ${0.35 + (t - 0.33) * 0.5})`;
                  return `rgba(239, 68, 68, ${0.35 + (t - 0.67) * 0.5})`;
                };
                return (
                  <div className="card">
                    <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}>
                      <Layers size={16} style={{ color: "var(--color-primary)" }} />
                      <h2 style={{ margin: 0 }}>Parallel Coordinates</h2>
                    </div>
                    <p className="range-desc">Each line represents a trial. Lines are colored by objective value (green = best, red = worst). Hover for details.</p>
                    <div style={{ overflowX: "auto" }}>
                      <svg width={W} height={H} style={{ display: "block" }}>
                        {/* Axis lines + labels */}
                        {paramBounds.map((pb, i) => {
                          const x = padL + i * axisSpacing;
                          return (
                            <g key={pb.name}>
                              <line x1={x} y1={padT} x2={x} y2={padT + plotH} stroke="var(--color-border)" strokeWidth="1" />
                              <text x={x} y={padT + plotH + 16} textAnchor="middle" fontSize="10" fontFamily="var(--font-mono)" fill="var(--color-text-muted)">
                                {pb.name.length > 10 ? pb.name.slice(0, 9) + "…" : pb.name}
                              </text>
                              <text x={x} y={padT - 6} textAnchor="middle" fontSize="9" fontFamily="var(--font-mono)" fill="var(--color-text-muted)">
                                {pb.max.toPrecision(3)}
                              </text>
                              <text x={x} y={padT + plotH + 30} textAnchor="middle" fontSize="9" fontFamily="var(--font-mono)" fill="var(--color-text-muted)">
                                {pb.min.toPrecision(3)}
                              </text>
                            </g>
                          );
                        })}
                        {/* Trial polylines */}
                        {trials.map((trial, ti) => {
                          const objVal = Number(trial.kpis[objKey]) || 0;
                          const pts = paramBounds.map((pb, i) => {
                            const val = Number(trial.parameters[pb.name]) || 0;
                            const y = padT + plotH - ((val - pb.min) / pb.range) * plotH;
                            const x = padL + i * axisSpacing;
                            return `${x.toFixed(1)},${y.toFixed(1)}`;
                          });
                          return (
                            <polyline
                              key={ti}
                              points={pts.join(" ")}
                              fill="none"
                              stroke={trialColor(objVal)}
                              strokeWidth="1.2"
                              strokeLinejoin="round"
                              className="pcoord-line"
                            >
                              <title>Trial #{trial.iteration} — {objKey}: {objVal.toFixed(4)}</title>
                            </polyline>
                          );
                        })}
                      </svg>
                    </div>
                    <div className="pcoord-legend">
                      <span className="pcoord-legend-item"><span className="pcoord-dot" style={{ background: "rgba(34,197,94,0.8)" }} /> Best</span>
                      <span className="pcoord-legend-item"><span className="pcoord-dot" style={{ background: "rgba(234,179,8,0.8)" }} /> Mid</span>
                      <span className="pcoord-legend-item"><span className="pcoord-dot" style={{ background: "rgba(239,68,68,0.8)" }} /> Worst</span>
                    </div>
                  </div>
                );
              })()}

              {/* Optimization Efficiency Curve */}
              {trials.length >= 10 && (() => {
                const objKey = Object.keys(trials[0].kpis)[0];
                const chrono = [...trials].sort((a, b) => a.iteration - b.iteration);
                // Actual cumulative best
                const actualBest: number[] = [];
                let runBest = Infinity;
                chrono.forEach(t => {
                  const v = Number(t.kpis[objKey]) || 0;
                  runBest = Math.min(runBest, v);
                  actualBest.push(runBest);
                });
                // Random baseline: sort values ascending (best possible random schedule)
                const randBest: number[] = [];
                // Actually random = shuffled, use sorted descending for WORST case, ascending for best
                // Realistic: assume random sees values in original order but shuffled. Use median expectation.
                // Simplification: sort all values and compute expected min after k draws (order statistics)
                // Just use: random line = average of 100 random shuffles? Too heavy. Use: random[k] = min of first k+1 sorted values
                // Actually the simplest useful comparison: the actual values re-randomized
                // Use deterministic "median random": sort values, running min of every N-th percentile
                const allVals = chrono.map(t => Number(t.kpis[objKey]) || 0);
                // Seed a pseudo-random shuffle deterministically
                const shuffled = [...allVals];
                let seed = 42;
                for (let i = shuffled.length - 1; i > 0; i--) {
                  seed = (seed * 1103515245 + 12345) & 0x7fffffff;
                  const j = seed % (i + 1);
                  [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
                }
                let rMin = Infinity;
                shuffled.forEach(v => {
                  rMin = Math.min(rMin, v);
                  randBest.push(rMin);
                });

                const allBestVals = [...actualBest, ...randBest];
                const yMin = Math.min(...allBestVals);
                const yMax = Math.max(...allBestVals);
                const yRange = yMax - yMin || 1;
                const n = chrono.length;
                const W = 460, H = 200, padL = 55, padR = 10, padT = 10, padB = 30;
                const plotW = W - padL - padR;
                const plotH = H - padT - padB;
                const px = (i: number) => padL + (i / (n - 1)) * plotW;
                const py = (v: number) => padT + (1 - (v - yMin) / yRange) * plotH;

                // Acceleration factor: at which iteration does random reach the final BO best?
                const finalBOBest = actualBest[actualBest.length - 1];
                const randReachIdx = randBest.findIndex(v => v <= finalBOBest);
                const actualPath = actualBest.map((v, i) => `${i === 0 ? "M" : "L"}${px(i).toFixed(1)},${py(v).toFixed(1)}`).join(" ");
                const randPath = randBest.map((v, i) => `${i === 0 ? "M" : "L"}${px(i).toFixed(1)},${py(v).toFixed(1)}`).join(" ");

                return (
                  <div className="card">
                    <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}>
                      <TrendingUp size={16} style={{ color: "var(--color-primary)" }} />
                      <h2 style={{ margin: 0 }}>Sample Efficiency</h2>
                      {randReachIdx > 0 && (
                        <span className="efficiency-badge">
                          {((n / randReachIdx) || 1).toFixed(1)}x faster than random
                        </span>
                      )}
                    </div>
                    <p className="range-desc">Cumulative best objective value vs iteration number. Compares actual optimization (blue) with a random baseline (gray).</p>
                    <svg width={W} height={H} style={{ display: "block", width: "100%", maxWidth: `${W}px` }}>
                      {/* Grid */}
                      {[0, 0.25, 0.5, 0.75, 1].map(f => {
                        const y = padT + (1 - f) * plotH;
                        const val = yMin + f * yRange;
                        return (
                          <g key={f}>
                            <line x1={padL} y1={y} x2={padL + plotW} y2={y} stroke="var(--color-border)" strokeWidth="0.5" strokeDasharray="3,3" />
                            <text x={padL - 6} y={y + 3} textAnchor="end" fontSize="9" fontFamily="var(--font-mono)" fill="var(--color-text-muted)">
                              {val.toFixed(3)}
                            </text>
                          </g>
                        );
                      })}
                      {/* Random baseline */}
                      <path d={randPath} fill="none" stroke="var(--color-text-muted)" strokeWidth="1.2" strokeDasharray="4,3" opacity={0.5} />
                      {/* Actual optimization */}
                      <path d={actualPath} fill="none" stroke="var(--color-primary)" strokeWidth="2" />
                      {/* Final markers */}
                      <circle cx={px(n - 1)} cy={py(actualBest[n - 1])} r="3.5" fill="var(--color-primary)" />
                      <circle cx={px(n - 1)} cy={py(randBest[n - 1])} r="2.5" fill="var(--color-text-muted)" />
                      {/* X-axis labels */}
                      <text x={padL} y={H - 4} textAnchor="start" fontSize="9" fontFamily="var(--font-mono)" fill="var(--color-text-muted)">0</text>
                      <text x={padL + plotW} y={H - 4} textAnchor="end" fontSize="9" fontFamily="var(--font-mono)" fill="var(--color-text-muted)">{n}</text>
                      <text x={padL + plotW / 2} y={H - 4} textAnchor="middle" fontSize="9" fill="var(--color-text-muted)">Iteration</text>
                    </svg>
                    <div className="efficiency-legend">
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: "16px", height: "2px", background: "var(--color-primary)", verticalAlign: "middle", marginRight: "4px" }} /> Optimization</span>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: "16px", height: "2px", background: "var(--color-text-muted)", verticalAlign: "middle", marginRight: "4px", borderTop: "1px dashed var(--color-text-muted)" }} /> Random Baseline</span>
                    </div>
                  </div>
                );
              })()}

              {/* Parameter Impact Waterfall */}
              {trials.length >= 10 && importance && importance.importances.length >= 2 && (() => {
                const objKey = Object.keys(trials[0].kpis)[0];
                const allVals = trials.map(t => Number(t.kpis[objKey]) || 0).sort((a, b) => a - b);
                const q25 = Math.floor(allVals.length * 0.25);
                const q75 = Math.ceil(allVals.length * 0.75);
                const bestTrials = trials.filter(t => Number(t.kpis[objKey]) <= allVals[q25]);
                const worstTrials = trials.filter(t => Number(t.kpis[objKey]) >= allVals[q75]);
                if (bestTrials.length === 0 || worstTrials.length === 0) return null;

                const paramNames = Object.keys(trials[0].parameters);
                const impacts = paramNames.map(p => {
                  const bestMean = bestTrials.reduce((a, t) => a + (Number(t.parameters[p]) || 0), 0) / bestTrials.length;
                  const worstMean = worstTrials.reduce((a, t) => a + (Number(t.parameters[p]) || 0), 0) / worstTrials.length;
                  const diff = bestMean - worstMean;
                  const imp = importance.importances.find(i => i.name === p);
                  return { name: p, diff, importance: imp?.importance ?? 0, bestMean, worstMean };
                }).sort((a, b) => Math.abs(b.diff * b.importance) - Math.abs(a.diff * a.importance));

                const maxAbsDiff = Math.max(...impacts.map(i => Math.abs(i.diff * i.importance)));
                if (maxAbsDiff === 0) return null;

                const W = 460, barH = 24, gap = 4;
                const padL = 80, padR = 60;
                const plotW = W - padL - padR;
                const H = impacts.length * (barH + gap) + 30;
                const centerX = padL + plotW / 2;

                return (
                  <div className="card">
                    <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}>
                      <BarChart2 size={16} style={{ color: "var(--color-primary)" }} />
                      <h2 style={{ margin: 0 }}>Parameter Impact</h2>
                    </div>
                    <p className="range-desc">How parameter values differ between best and worst 25% of trials, weighted by importance. Positive = higher in best trials.</p>
                    <svg width={W} height={H} style={{ display: "block", width: "100%", maxWidth: `${W}px` }}>
                      {/* Center line */}
                      <line x1={centerX} y1={8} x2={centerX} y2={H - 20} stroke="var(--color-border)" strokeWidth="1" />
                      {impacts.map((item, i) => {
                        const y = 8 + i * (barH + gap);
                        const weighted = item.diff * item.importance;
                        const w = (weighted / maxAbsDiff) * (plotW / 2);
                        const isPos = w >= 0;
                        const barX = isPos ? centerX : centerX + w;
                        const barWidth = Math.abs(w);
                        const color = isPos ? "rgba(34,197,94,0.6)" : "rgba(239,68,68,0.5)";
                        return (
                          <g key={item.name}>
                            <text x={padL - 4} y={y + barH / 2 + 3} textAnchor="end" fontSize="10" fontFamily="var(--font-mono)" fill="var(--color-text)">
                              {item.name.length > 10 ? item.name.slice(0, 9) + "…" : item.name}
                            </text>
                            <rect x={barX} y={y} width={Math.max(barWidth, 1)} height={barH} rx="3" fill={color}>
                              <title>{item.name}: best avg {item.bestMean.toFixed(3)} vs worst avg {item.worstMean.toFixed(3)} (diff: {item.diff > 0 ? "+" : ""}{item.diff.toFixed(3)})</title>
                            </rect>
                            <text x={isPos ? centerX + Math.abs(w) + 4 : centerX + w - 4} y={y + barH / 2 + 3} textAnchor={isPos ? "start" : "end"} fontSize="9" fontFamily="var(--font-mono)" fill="var(--color-text-muted)">
                              {weighted > 0 ? "+" : ""}{weighted.toFixed(3)}
                            </text>
                          </g>
                        );
                      })}
                      {/* Labels */}
                      <text x={padL} y={H - 4} fontSize="9" fill="var(--color-text-muted)">Lower in best</text>
                      <text x={padL + plotW} y={H - 4} textAnchor="end" fontSize="9" fill="var(--color-text-muted)">Higher in best</text>
                    </svg>
                  </div>
                );
              })()}

              {/* Acquisition Function Proxy */}
              {trials.length >= 12 && (() => {
                const paramNames = Object.keys(trials[0].parameters);
                if (paramNames.length < 2) return null;
                const objKey = Object.keys(trials[0].kpis)[0];
                const afXParam = paramNames[0];
                const afYParam = paramNames[1];
                const afXVals = trials.map(t => Number(t.parameters[afXParam]) || 0);
                const afYVals = trials.map(t => Number(t.parameters[afYParam]) || 0);
                const afObjVals = trials.map(t => Number(t.kpis[objKey]) || 0);
                const afXMin = Math.min(...afXVals), afXMax = Math.max(...afXVals);
                const afYMin = Math.min(...afYVals), afYMax = Math.max(...afYVals);
                const afXRange = afXMax - afXMin || 1;
                const afYRange = afYMax - afYMin || 1;
                const afObjMin = Math.min(...afObjVals);
                const afObjMax = Math.max(...afObjVals);
                const afObjRange = afObjMax - afObjMin || 1;

                const res = 16;
                const afGrid: Array<Array<{ exploit: number; explore: number; acq: number }>> = [];
                for (let gy = 0; gy < res; gy++) {
                  const row: Array<{ exploit: number; explore: number; acq: number }> = [];
                  for (let gx = 0; gx < res; gx++) {
                    const cx = afXMin + (gx + 0.5) * afXRange / res;
                    const cy = afYMin + (gy + 0.5) * afYRange / res;

                    // Exploitation: IDW-interpolated objective value (normalized, lower=better)
                    let wSum = 0, vSum = 0;
                    for (let i = 0; i < trials.length; i++) {
                      const dx = (afXVals[i] - cx) / afXRange;
                      const dy = (afYVals[i] - cy) / afYRange;
                      const d = Math.sqrt(dx * dx + dy * dy) + 0.001;
                      const w = 1 / (d * d);
                      wSum += w;
                      vSum += w * afObjVals[i];
                    }
                    const predicted = vSum / wSum;
                    const exploit = 1 - (predicted - afObjMin) / afObjRange; // Higher = better predicted value

                    // Exploration: inverse of local density (sparse = high exploration value)
                    let nearCount = 0;
                    const radius = 0.15;
                    for (let i = 0; i < trials.length; i++) {
                      const dx = (afXVals[i] - cx) / afXRange;
                      const dy = (afYVals[i] - cy) / afYRange;
                      if (Math.sqrt(dx * dx + dy * dy) < radius) nearCount++;
                    }
                    const explore = 1 - Math.min(nearCount / 5, 1); // fewer nearby = higher exploration

                    // Acquisition score: balanced combination
                    const acq = 0.4 * exploit + 0.6 * explore;
                    row.push({ exploit, explore, acq });
                  }
                  afGrid.push(row);
                }

                const maxAcq = Math.max(...afGrid.flat().map(c => c.acq));
                const minAcq = Math.min(...afGrid.flat().map(c => c.acq));
                const acqRange = maxAcq - minAcq || 1;

                // Find top suggested cell
                let topGx = 0, topGy = 0, topAcq = -Infinity;
                for (let gy = 0; gy < res; gy++) {
                  for (let gx = 0; gx < res; gx++) {
                    if (afGrid[gy][gx].acq > topAcq) {
                      topAcq = afGrid[gy][gx].acq;
                      topGx = gx;
                      topGy = gy;
                    }
                  }
                }

                const W = 420, H = 280, padL = 50, padR = 20, padT = 10, padB = 36;
                const plotW = W - padL - padR;
                const plotH = H - padT - padB;
                const cellW = plotW / res;
                const cellH = plotH / res;

                const acqColor = (v: number) => {
                  const t = (v - minAcq) / acqRange;
                  // Purple (low) → blue (mid) → cyan (high)
                  if (t < 0.5) {
                    const p = t / 0.5;
                    return `rgb(${Math.round(88 - p * 30)}, ${Math.round(28 + p * 72)}, ${Math.round(135 + p * 70)})`;
                  }
                  const p = (t - 0.5) / 0.5;
                  return `rgb(${Math.round(58 - p * 50)}, ${Math.round(100 + p * 155)}, ${Math.round(205 + p * 50)})`;
                };

                return (
                  <div className="card">
                    <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}>
                      <Crosshair size={16} style={{ color: "var(--color-primary)" }} />
                      <h2 style={{ margin: 0 }}>Acquisition Landscape</h2>
                      <span style={{ fontSize: "0.72rem", color: "var(--color-text-muted)", marginLeft: "auto", fontFamily: "var(--font-mono)" }}>
                        {afXParam} × {afYParam}
                      </span>
                    </div>
                    <p style={{ fontSize: "0.78rem", color: "var(--color-text-muted)", margin: "0 0 8px" }}>
                      Estimated acquisition score combining predicted value (40%) and exploration need (60%). Bright = high-priority regions for next experiments.
                    </p>
                    <svg width={W} height={H} style={{ display: "block", width: "100%", maxWidth: `${W}px` }}>
                      {afGrid.map((row, gy) =>
                        row.map((cell, gx) => (
                          <rect
                            key={`${gy}-${gx}`}
                            x={padL + gx * cellW}
                            y={padT + gy * cellH}
                            width={cellW + 0.5}
                            height={cellH + 0.5}
                            fill={acqColor(cell.acq)}
                            opacity={0.85}
                          >
                            <title>{afXParam}≈{(afXMin + (gx + 0.5) * afXRange / res).toFixed(2)}, {afYParam}≈{(afYMin + (gy + 0.5) * afYRange / res).toFixed(2)} | Acq={cell.acq.toFixed(3)} (exploit={cell.exploit.toFixed(2)}, explore={cell.explore.toFixed(2)})</title>
                          </rect>
                        ))
                      )}
                      {/* Best acquisition cell marker */}
                      <rect
                        x={padL + topGx * cellW + 1}
                        y={padT + topGy * cellH + 1}
                        width={cellW - 2}
                        height={cellH - 2}
                        fill="none"
                        stroke="white"
                        strokeWidth="2"
                        strokeDasharray="3,2"
                      />
                      <text
                        x={padL + (topGx + 0.5) * cellW}
                        y={padT + (topGy + 0.5) * cellH + 3}
                        textAnchor="middle"
                        fontSize="9"
                        fill="white"
                        fontWeight="bold"
                      >★</text>
                      {/* Trial positions */}
                      {trials.map((t, i) => {
                        const px = padL + ((afXVals[i] - afXMin) / afXRange) * plotW;
                        const py = padT + ((afYVals[i] - afYMin) / afYRange) * plotH;
                        return (
                          <circle key={i} cx={px} cy={py} r="2" fill="rgba(255,255,255,0.5)" stroke="rgba(0,0,0,0.3)" strokeWidth="0.5">
                            <title>Trial #{t.iteration}: {objKey}={afObjVals[i].toFixed(4)}</title>
                          </circle>
                        );
                      })}
                      {/* Axes */}
                      <line x1={padL} y1={padT} x2={padL} y2={padT + plotH} stroke="var(--color-border)" strokeWidth="1" />
                      <line x1={padL} y1={padT + plotH} x2={padL + plotW} y2={padT + plotH} stroke="var(--color-border)" strokeWidth="1" />
                      {[0, 0.5, 1].map(f => (
                        <Fragment key={`af${f}`}>
                          <text x={padL + f * plotW} y={H - 12} textAnchor="middle" fontSize="9" fontFamily="var(--font-mono)" fill="var(--color-text-muted)">
                            {(afXMin + f * afXRange).toFixed(2)}
                          </text>
                          <text x={padL - 4} y={padT + (1 - f) * plotH + 3} textAnchor="end" fontSize="9" fontFamily="var(--font-mono)" fill="var(--color-text-muted)">
                            {(afYMin + f * afYRange).toFixed(2)}
                          </text>
                        </Fragment>
                      ))}
                      <text x={padL + plotW / 2} y={H - 0} textAnchor="middle" fontSize="10" fill="var(--color-text-muted)">{afXParam}</text>
                      <text x={10} y={padT + plotH / 2} textAnchor="middle" fontSize="10" fill="var(--color-text-muted)" transform={`rotate(-90, 10, ${padT + plotH / 2})`}>{afYParam}</text>
                    </svg>
                    <div className="efficiency-legend" style={{ maxWidth: `${W}px` }}>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: 10, height: 10, borderRadius: 2, background: "rgb(88,28,135)", marginRight: 4, verticalAlign: "middle" }} />Low priority</span>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: 10, height: 10, borderRadius: 2, background: "rgb(8,255,255)", marginRight: 4, verticalAlign: "middle" }} />High priority</span>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: 8, height: 8, border: "1.5px dashed white", background: "transparent", marginRight: 4, verticalAlign: "middle" }} />★ Next suggestion</span>
                    </div>
                  </div>
                );
              })()}

              {/* Parameter Space Sampling Density */}
              {trials.length >= 10 && (() => {
                const paramNames = Object.keys(trials[0].parameters);
                if (paramNames.length < 2) return null;
                const xParam = paramNames[0];
                const yParam = paramNames[1];
                const xVals = trials.map(t => Number(t.parameters[xParam]) || 0);
                const yVals = trials.map(t => Number(t.parameters[yParam]) || 0);
                const xMin = Math.min(...xVals), xMax = Math.max(...xVals);
                const yMin = Math.min(...yVals), yMax = Math.max(...yVals);
                const xRange = xMax - xMin || 1;
                const yRange = yMax - yMin || 1;
                const res = 12;
                const W = 380, H = 320, padL = 48, padR = 14, padT = 10, padB = 36;
                const plotW = W - padL - padR;
                const plotH = H - padT - padB;
                const cellW = plotW / res;
                const cellH = plotH / res;

                // Count trials per cell
                const density: number[][] = Array.from({ length: res }, () => Array(res).fill(0));
                for (let i = 0; i < trials.length; i++) {
                  const gx = Math.min(Math.floor(((xVals[i] - xMin) / xRange) * res), res - 1);
                  const gy = Math.min(Math.floor(((yVals[i] - yMin) / yRange) * res), res - 1);
                  density[gy][gx]++;
                }
                const maxDensity = Math.max(...density.flat());
                const emptyCells = density.flat().filter(d => d === 0).length;
                const totalCells = res * res;
                const coveragePct = ((1 - emptyCells / totalCells) * 100).toFixed(0);

                return (
                  <div className="card">
                    <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}>
                      <Radar size={16} style={{ color: "var(--color-primary)" }} />
                      <h2 style={{ margin: 0 }}>Sampling Density</h2>
                      <span style={{ fontSize: "0.72rem", color: "var(--color-text-muted)", marginLeft: "auto", fontFamily: "var(--font-mono)" }}>
                        {coveragePct}% cells sampled
                      </span>
                    </div>
                    <p style={{ fontSize: "0.78rem", color: "var(--color-text-muted)", margin: "0 0 8px" }}>
                      Trial density across the {xParam} × {yParam} space. Dark cells = well-sampled, light/white = gaps to explore.
                    </p>
                    <svg width={W} height={H} style={{ display: "block", width: "100%", maxWidth: `${W}px` }}>
                      {density.map((row, gy) =>
                        row.map((count, gx) => {
                          const opacity = count > 0 ? 0.15 + (count / maxDensity) * 0.75 : 0;
                          return (
                            <rect
                              key={`${gy}-${gx}`}
                              x={padL + gx * cellW}
                              y={padT + gy * cellH}
                              width={cellW - 0.5}
                              height={cellH - 0.5}
                              rx="2"
                              fill={count > 0 ? "var(--color-primary)" : "var(--color-border)"}
                              opacity={count > 0 ? opacity : 0.3}
                              stroke="var(--color-bg)"
                              strokeWidth="0.5"
                            >
                              <title>{xParam}:[{(xMin + gx * xRange / res).toFixed(2)},{(xMin + (gx + 1) * xRange / res).toFixed(2)}] × {yParam}:[{(yMin + gy * yRange / res).toFixed(2)},{(yMin + (gy + 1) * yRange / res).toFixed(2)}] — {count} trial{count !== 1 ? "s" : ""}</title>
                            </rect>
                          );
                        })
                      )}
                      {/* Gap markers for empty cells */}
                      {density.flatMap((row, gy) =>
                        row.map((count, gx) =>
                          count === 0 ? (
                            <text key={`gap-${gy}-${gx}`} x={padL + (gx + 0.5) * cellW} y={padT + (gy + 0.5) * cellH + 3} textAnchor="middle" fontSize="7" fill="var(--color-text-muted)" opacity="0.4">?</text>
                          ) : null
                        )
                      )}
                      {/* Axes */}
                      <line x1={padL} y1={padT + plotH} x2={padL + plotW} y2={padT + plotH} stroke="var(--color-border)" strokeWidth="1" />
                      <line x1={padL} y1={padT} x2={padL} y2={padT + plotH} stroke="var(--color-border)" strokeWidth="1" />
                      {[0, 0.5, 1].map(f => (
                        <Fragment key={`xl-${f}`}>
                          <text x={padL + f * plotW} y={H - 18} textAnchor="middle" fontSize="9" fontFamily="var(--font-mono)" fill="var(--color-text-muted)">{(xMin + f * xRange).toFixed(2)}</text>
                          <text x={padL - 4} y={padT + f * plotH + 3} textAnchor="end" fontSize="9" fontFamily="var(--font-mono)" fill="var(--color-text-muted)">{(yMin + f * yRange).toFixed(2)}</text>
                        </Fragment>
                      ))}
                      <text x={padL + plotW / 2} y={H - 4} textAnchor="middle" fontSize="10" fill="var(--color-text-muted)">{xParam}</text>
                      <text x={12} y={padT + plotH / 2} textAnchor="middle" fontSize="10" fill="var(--color-text-muted)" transform={`rotate(-90, 12, ${padT + plotH / 2})`}>{yParam}</text>
                    </svg>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.72rem", color: "var(--color-text-muted)", maxWidth: `${W}px`, padding: "2px 0" }}>
                      <span>{emptyCells} empty cells (gaps)</span>
                      <span>Max density: {maxDensity} trials/cell</span>
                    </div>
                  </div>
                );
              })()}

              {/* Surrogate Landscape Heatmap */}
              {trials.length >= 10 && (() => {
                const paramNames = Object.keys(trials[0].parameters);
                if (paramNames.length < 2) return null;
                const objKey = Object.keys(trials[0].kpis)[0];
                const xParam = paramNames[0];
                const yParam = paramNames[1];
                const xVals = trials.map(t => Number(t.parameters[xParam]) || 0);
                const yVals = trials.map(t => Number(t.parameters[yParam]) || 0);
                const objVals = trials.map(t => Number(t.kpis[objKey]) || 0);
                const xMin = Math.min(...xVals), xMax = Math.max(...xVals);
                const yMin = Math.min(...yVals), yMax = Math.max(...yVals);
                const oMin = Math.min(...objVals), oMax = Math.max(...objVals);
                const xRange = xMax - xMin || 1;
                const yRange = yMax - yMin || 1;
                const oRange = oMax - oMin || 1;
                const res = 20; // grid resolution
                const W = 400, H = 340, padL = 52, padR = 16, padT = 12, padB = 38;
                const plotW = W - padL - padR;
                const plotH = H - padT - padB;
                const cellW = plotW / res;
                const cellH = plotH / res;

                // Inverse-distance-weighted interpolation
                const grid: number[][] = [];
                const countGrid: number[][] = [];
                for (let gy = 0; gy < res; gy++) {
                  grid[gy] = [];
                  countGrid[gy] = [];
                  for (let gx = 0; gx < res; gx++) {
                    const cx = xMin + (gx + 0.5) * xRange / res;
                    const cy = yMin + (gy + 0.5) * yRange / res;
                    let wSum = 0, vSum = 0;
                    for (let i = 0; i < trials.length; i++) {
                      const dx = (xVals[i] - cx) / xRange;
                      const dy = (yVals[i] - cy) / yRange;
                      const dist = Math.sqrt(dx * dx + dy * dy) + 0.01;
                      const w = 1 / (dist * dist);
                      wSum += w;
                      vSum += w * objVals[i];
                    }
                    grid[gy][gx] = wSum > 0 ? vSum / wSum : 0;
                    countGrid[gy][gx] = trials.filter((_t, i) => {
                      const dx = Math.abs(xVals[i] - cx) / xRange;
                      const dy = Math.abs(yVals[i] - cy) / yRange;
                      return dx < 0.5 / res * 3 && dy < 0.5 / res * 3;
                    }).length;
                  }
                }

                // Color: green (best/low) → yellow → red (worst/high) for minimize
                const cellColor = (val: number) => {
                  const t = (val - oMin) / oRange;
                  if (t < 0.33) {
                    const p = t / 0.33;
                    return `rgb(${Math.round(34 + p * (234 - 34))}, ${Math.round(197 - p * (197 - 179))}, ${Math.round(94 - p * (94 - 8))})`;
                  }
                  const p = (t - 0.33) / 0.67;
                  return `rgb(${Math.round(234 + p * (239 - 234))}, ${Math.round(179 - p * (179 - 68))}, ${Math.round(8 + p * (68 - 8))})`;
                };

                return (
                  <div className="card">
                    <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}>
                      <Grid size={16} style={{ color: "var(--color-primary)" }} />
                      <h2 style={{ margin: 0 }}>Surrogate Landscape</h2>
                    </div>
                    <p style={{ fontSize: "0.78rem", color: "var(--color-text-muted)", margin: "0 0 10px" }}>
                      Estimated objective surface using inverse-distance interpolation. Green = better regions. White dots = actual trials.
                    </p>
                    <svg width={W} height={H} style={{ display: "block", width: "100%", maxWidth: `${W}px` }}>
                      {/* Heatmap cells */}
                      {grid.map((row, gy) =>
                        row.map((val, gx) => (
                          <rect
                            key={`${gy}-${gx}`}
                            x={padL + gx * cellW}
                            y={padT + gy * cellH}
                            width={cellW + 0.5}
                            height={cellH + 0.5}
                            fill={cellColor(val)}
                            opacity={countGrid[gy][gx] > 0 ? 0.85 : 0.4}
                          >
                            <title>{xParam}≈{(xMin + (gx + 0.5) * xRange / res).toFixed(2)}, {yParam}≈{(yMin + (gy + 0.5) * yRange / res).toFixed(2)} → {val.toFixed(4)}</title>
                          </rect>
                        ))
                      )}
                      {/* Trial points */}
                      {trials.map((t, i) => {
                        const px = padL + ((xVals[i] - xMin) / xRange) * plotW;
                        const py = padT + ((yVals[i] - yMin) / yRange) * plotH;
                        return (
                          <circle key={i} cx={px} cy={py} r="2.5" fill="white" stroke="rgba(0,0,0,0.4)" strokeWidth="0.5">
                            <title>Trial #{t.iteration}: {objKey}={objVals[i].toFixed(4)}</title>
                          </circle>
                        );
                      })}
                      {/* Axes */}
                      <line x1={padL} y1={padT} x2={padL} y2={padT + plotH} stroke="var(--color-border)" strokeWidth="1" />
                      <line x1={padL} y1={padT + plotH} x2={padL + plotW} y2={padT + plotH} stroke="var(--color-border)" strokeWidth="1" />
                      {/* X-axis labels */}
                      {[0, 0.5, 1].map(f => (
                        <text key={f} x={padL + f * plotW} y={H - 18} textAnchor="middle" fontSize="9" fontFamily="var(--font-mono)" fill="var(--color-text-muted)">
                          {(xMin + f * xRange).toFixed(2)}
                        </text>
                      ))}
                      <text x={padL + plotW / 2} y={H - 4} textAnchor="middle" fontSize="10" fill="var(--color-text-muted)">{xParam}</text>
                      {/* Y-axis labels */}
                      {[0, 0.5, 1].map(f => (
                        <text key={f} x={padL - 4} y={padT + f * plotH + 3} textAnchor="end" fontSize="9" fontFamily="var(--font-mono)" fill="var(--color-text-muted)">
                          {(yMin + f * yRange).toFixed(2)}
                        </text>
                      ))}
                      <text x={12} y={padT + plotH / 2} textAnchor="middle" fontSize="10" fill="var(--color-text-muted)" transform={`rotate(-90, 12, ${padT + plotH / 2})`}>{yParam}</text>
                      {/* Color legend */}
                      <defs>
                        <linearGradient id="hm-legend" x1="0" x2="1" y1="0" y2="0">
                          <stop offset="0%" stopColor="rgb(34,197,94)" />
                          <stop offset="33%" stopColor="rgb(234,179,8)" />
                          <stop offset="100%" stopColor="rgb(239,68,68)" />
                        </linearGradient>
                      </defs>
                      <rect x={padL + plotW + 4} y={padT} width={8} height={plotH} fill="url(#hm-legend)" rx="2" transform={`rotate(180, ${padL + plotW + 8}, ${padT + plotH / 2})`} />
                    </svg>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.72rem", color: "var(--color-text-muted)", fontFamily: "var(--font-mono)", maxWidth: `${W}px`, padding: "2px 0" }}>
                      <span>Better ({oMin.toFixed(3)})</span>
                      <span>Worse ({oMax.toFixed(3)})</span>
                    </div>
                  </div>
                );
              })()}

              {/* Surrogate Confidence Map */}
              {trials.length >= 12 && (() => {
                const paramNames = Object.keys(trials[0].parameters);
                if (paramNames.length < 2) return null;
                const objKey = Object.keys(trials[0].kpis)[0];
                const scXParam = paramNames[0];
                const scYParam = paramNames[1];
                const scXVals = trials.map(t => Number(t.parameters[scXParam]) || 0);
                const scYVals = trials.map(t => Number(t.parameters[scYParam]) || 0);
                const scObjVals = trials.map(t => Number(t.kpis[objKey]) || 0);
                const scXMin = Math.min(...scXVals), scXMax = Math.max(...scXVals);
                const scYMin = Math.min(...scYVals), scYMax = Math.max(...scYVals);
                const scXRange = scXMax - scXMin || 1;
                const scYRange = scYMax - scYMin || 1;

                const res = 16;
                const confGrid: number[][] = [];
                for (let gy = 0; gy < res; gy++) {
                  const row: number[] = [];
                  for (let gx = 0; gx < res; gx++) {
                    const cx = scXMin + (gx + 0.5) * scXRange / res;
                    const cy = scYMin + (gy + 0.5) * scYRange / res;

                    // Find nearby trials within radius
                    const nearbyVals: number[] = [];
                    const nearbyDists: number[] = [];
                    for (let i = 0; i < trials.length; i++) {
                      const dx = (scXVals[i] - cx) / scXRange;
                      const dy = (scYVals[i] - cy) / scYRange;
                      const dist = Math.sqrt(dx * dx + dy * dy);
                      if (dist < 0.25) {
                        nearbyVals.push(scObjVals[i]);
                        nearbyDists.push(dist);
                      }
                    }

                    // Confidence: combination of density (more nearby = confident) and local consistency (low variance = confident)
                    const density = Math.min(nearbyVals.length / 8, 1); // saturates at 8 nearby trials
                    let consistency = 1;
                    if (nearbyVals.length >= 2) {
                      const mean = nearbyVals.reduce((a, b) => a + b, 0) / nearbyVals.length;
                      const variance = nearbyVals.reduce((a, v) => a + (v - mean) ** 2, 0) / nearbyVals.length;
                      const globalMean = scObjVals.reduce((a, b) => a + b, 0) / scObjVals.length;
                      const globalVar = scObjVals.reduce((a, v) => a + (v - globalMean) ** 2, 0) / scObjVals.length;
                      consistency = globalVar > 0 ? Math.max(0, 1 - variance / globalVar) : 1;
                    } else {
                      consistency = 0;
                    }

                    const confidence = 0.6 * density + 0.4 * consistency;
                    row.push(confidence);
                  }
                  confGrid.push(row);
                }

                const maxConf = Math.max(...confGrid.flat());
                const minConf = Math.min(...confGrid.flat());
                const confRange = maxConf - minConf || 1;
                const highConfPct = (confGrid.flat().filter(c => c > 0.5).length / (res * res) * 100).toFixed(0);

                const W = 420, H = 280, padL = 50, padR = 20, padT = 10, padB = 36;
                const plotW = W - padL - padR;
                const plotH = H - padT - padB;
                const cellW = plotW / res;
                const cellH = plotH / res;

                const confColor = (v: number) => {
                  const t = (v - minConf) / confRange;
                  // Red (uncertain) → yellow → blue (confident)
                  if (t < 0.5) {
                    const p = t / 0.5;
                    return `rgb(${Math.round(239 - p * 5)}, ${Math.round(68 + p * 111)}, ${Math.round(68 - p * 60)})`;
                  }
                  const p = (t - 0.5) / 0.5;
                  return `rgb(${Math.round(234 - p * 175)}, ${Math.round(179 - p * 49)}, ${Math.round(8 + p * 227)})`;
                };

                return (
                  <div className="card">
                    <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}>
                      <Radar size={16} style={{ color: "var(--color-primary)" }} />
                      <h2 style={{ margin: 0 }}>Model Confidence</h2>
                      <span style={{ fontSize: "0.72rem", color: "var(--color-text-muted)", marginLeft: "auto", fontFamily: "var(--font-mono)" }}>
                        {highConfPct}% confident
                      </span>
                    </div>
                    <p style={{ fontSize: "0.78rem", color: "var(--color-text-muted)", margin: "0 0 8px" }}>
                      Surrogate model confidence based on local trial density and prediction consistency. Blue = reliable predictions, red = uncertain.
                    </p>
                    <svg width={W} height={H} style={{ display: "block", width: "100%", maxWidth: `${W}px` }}>
                      {confGrid.map((row, gy) =>
                        row.map((conf, gx) => (
                          <rect
                            key={`${gy}-${gx}`}
                            x={padL + gx * cellW}
                            y={padT + gy * cellH}
                            width={cellW + 0.5}
                            height={cellH + 0.5}
                            fill={confColor(conf)}
                            opacity={0.75}
                          >
                            <title>{scXParam}≈{(scXMin + (gx + 0.5) * scXRange / res).toFixed(2)}, {scYParam}≈{(scYMin + (gy + 0.5) * scYRange / res).toFixed(2)} | Confidence={conf.toFixed(3)}</title>
                          </rect>
                        ))
                      )}
                      {/* Trial positions */}
                      {trials.map((t, i) => {
                        const px = padL + ((scXVals[i] - scXMin) / scXRange) * plotW;
                        const py = padT + ((scYVals[i] - scYMin) / scYRange) * plotH;
                        return (
                          <circle key={i} cx={px} cy={py} r="2.5" fill="white" stroke="rgba(0,0,0,0.4)" strokeWidth="0.5">
                            <title>Trial #{t.iteration}: {objKey}={scObjVals[i].toFixed(4)}</title>
                          </circle>
                        );
                      })}
                      {/* Axes */}
                      <line x1={padL} y1={padT} x2={padL} y2={padT + plotH} stroke="var(--color-border)" strokeWidth="1" />
                      <line x1={padL} y1={padT + plotH} x2={padL + plotW} y2={padT + plotH} stroke="var(--color-border)" strokeWidth="1" />
                      {[0, 0.5, 1].map(f => (
                        <Fragment key={`sc${f}`}>
                          <text x={padL + f * plotW} y={H - 12} textAnchor="middle" fontSize="9" fontFamily="var(--font-mono)" fill="var(--color-text-muted)">
                            {(scXMin + f * scXRange).toFixed(2)}
                          </text>
                          <text x={padL - 4} y={padT + (1 - f) * plotH + 3} textAnchor="end" fontSize="9" fontFamily="var(--font-mono)" fill="var(--color-text-muted)">
                            {(scYMin + f * scYRange).toFixed(2)}
                          </text>
                        </Fragment>
                      ))}
                      <text x={padL + plotW / 2} y={H - 0} textAnchor="middle" fontSize="10" fill="var(--color-text-muted)">{scXParam}</text>
                      <text x={10} y={padT + plotH / 2} textAnchor="middle" fontSize="10" fill="var(--color-text-muted)" transform={`rotate(-90, 10, ${padT + plotH / 2})`}>{scYParam}</text>
                    </svg>
                    <div className="efficiency-legend" style={{ maxWidth: `${W}px` }}>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: 10, height: 10, borderRadius: 2, background: "rgb(239,68,68)", marginRight: 4, verticalAlign: "middle" }} />Uncertain</span>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: 10, height: 10, borderRadius: 2, background: "rgb(234,179,8)", marginRight: 4, verticalAlign: "middle" }} />Moderate</span>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: 10, height: 10, borderRadius: 2, background: "rgb(59,130,235)", marginRight: 4, verticalAlign: "middle" }} />Confident</span>
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

              {/* Suggestion Context Cards */}
              {trials.length >= 3 && (() => {
                const objKey = Object.keys(trials[0].kpis)[0];
                const objVals = trials.map(t => Number(t.kpis[objKey]) || 0);
                const bestVal = Math.min(...objVals);
                const meanVal = objVals.reduce((a, b) => a + b, 0) / objVals.length;
                const chrono = [...trials].sort((a, b) => a.iteration - b.iteration);
                const recentN = Math.min(10, chrono.length);
                const recentTrials = chrono.slice(-recentN);
                const recentBest = Math.min(...recentTrials.map(t => Number(t.kpis[objKey]) || 0));
                const recentImproving = recentBest < meanVal;

                // Parameter diversity: coefficient of variation across parameters
                const paramNames = Object.keys(trials[0].parameters);
                const paramDiversity = paramNames.length > 0 ? (() => {
                  let totalCv = 0;
                  for (const p of paramNames) {
                    const pVals = recentTrials.map(t => Number(t.parameters[p]) || 0);
                    const pMean = pVals.reduce((a, b) => a + b, 0) / pVals.length;
                    const pStd = Math.sqrt(pVals.reduce((a, v) => a + (v - pMean) ** 2, 0) / pVals.length);
                    totalCv += pMean !== 0 ? Math.abs(pStd / pMean) : 0;
                  }
                  return Math.min((totalCv / paramNames.length) * 100, 100);
                })() : 0;

                const phase = suggestions?.phase ?? campaign.phases[campaign.phases.length - 1]?.name ?? "Unknown";
                const contextItems = [
                  { icon: <Trophy size={14} />, label: "Best Found", value: bestVal.toFixed(4), sub: objKey },
                  { icon: <Target size={14} />, label: "Recent Trend", value: recentImproving ? "Improving" : "Plateaued", sub: `Last ${recentN} trials` },
                  { icon: <Compass size={14} />, label: "Param Diversity", value: `${paramDiversity.toFixed(0)}%`, sub: "Recent spread" },
                  { icon: <Rocket size={14} />, label: "Phase", value: String(phase).charAt(0).toUpperCase() + String(phase).slice(1), sub: `${trials.length} trials total` },
                ];

                return (
                  <div style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))",
                    gap: "10px",
                    marginBottom: "16px",
                  }}>
                    {contextItems.map((item, i) => (
                      <div key={i} style={{
                        background: "var(--color-card-bg)",
                        border: "1px solid var(--color-border)",
                        borderRadius: "8px",
                        padding: "10px 12px",
                        display: "flex",
                        flexDirection: "column",
                        gap: "4px",
                      }}>
                        <div style={{ display: "flex", alignItems: "center", gap: "6px", color: "var(--color-text-muted)" }}>
                          {item.icon}
                          <span style={{ fontSize: "0.72rem", fontWeight: 500, textTransform: "uppercase", letterSpacing: "0.04em" }}>
                            {item.label}
                          </span>
                        </div>
                        <div style={{ fontSize: "1.1rem", fontWeight: 600, fontFamily: "var(--font-mono)", color: "var(--color-text-primary)" }}>
                          {item.value}
                        </div>
                        <div style={{ fontSize: "0.7rem", color: "var(--color-text-muted)" }}>
                          {item.sub}
                        </div>
                      </div>
                    ))}
                  </div>
                );
              })()}

              {/* EI Decomposition View */}
              {suggestions && suggestions.suggestions.length > 0 && trials.length >= 5 && (() => {
                const objKey = Object.keys(trials[0].kpis)[0];
                const objVals = trials.map(t => Number(t.kpis[objKey]) || 0);
                const bestSoFar = Math.min(...objVals);
                const meanObj = objVals.reduce((a, b) => a + b, 0) / objVals.length;
                const stdObj = Math.sqrt(objVals.reduce((a, v) => a + (v - meanObj) ** 2, 0) / objVals.length) || 1;

                // For each suggestion, estimate exploitation (predicted improvement) and exploration (novelty/uncertainty)
                const eiData = suggestions.suggestions.map((sug, idx) => {
                  const params = Object.values(sug).map(Number);
                  // Exploitation: predicted improvement based on nearest neighbor interpolation
                  let minDist = Infinity;
                  let nearestVal = meanObj;
                  for (const t of trials) {
                    const tParams = Object.values(t.parameters).map(Number);
                    const dist = Math.sqrt(tParams.reduce((sum, v, i) => sum + (v - (params[i] || 0)) ** 2, 0));
                    if (dist < minDist) { minDist = dist; nearestVal = Number(t.kpis[objKey]) || 0; }
                  }
                  const exploitation = Math.max(0, bestSoFar - nearestVal + stdObj * 0.3) / (stdObj * 2);
                  // Exploration: average distance to all trials (normalized)
                  const avgDist = trials.reduce((sum, t) => {
                    const tParams = Object.values(t.parameters).map(Number);
                    return sum + Math.sqrt(tParams.reduce((s, v, i) => s + (v - (params[i] || 0)) ** 2, 0));
                  }, 0) / trials.length;
                  const maxPossibleDist = stdObj * Math.sqrt(params.length) * 3;
                  const exploration = Math.min(avgDist / (maxPossibleDist || 1), 1);
                  return { idx: idx + 1, exploitation: Math.min(exploitation, 1), exploration: Math.min(exploration, 1), total: exploitation + exploration };
                });

                const maxTotal = Math.max(...eiData.map(d => d.total), 0.01);
                const barH = 18, gap = 6;
                const W = 380, padL = 42, padR = 16;
                const barW = W - padL - padR;
                const H = eiData.length * (barH + gap) + 40;

                return (
                  <div className="card" style={{ marginBottom: "16px" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "4px" }}>
                      <BarChart2 size={16} style={{ color: "var(--color-primary)" }} />
                      <h2 style={{ margin: 0 }}>Acquisition Decomposition</h2>
                    </div>
                    <p style={{ fontSize: "0.78rem", color: "var(--color-text-muted)", margin: "0 0 8px" }}>
                      Each suggestion's acquisition value split into exploitation (predicted improvement) and exploration (uncertainty/novelty).
                    </p>
                    <svg width={W} height={H} style={{ display: "block", width: "100%", maxWidth: `${W}px` }}>
                      {eiData.map((d, i) => {
                        const y = 24 + i * (barH + gap);
                        const exploitW = (d.exploitation / maxTotal) * barW;
                        const exploreW = (d.exploration / maxTotal) * barW;
                        return (
                          <Fragment key={i}>
                            <text x={padL - 6} y={y + barH / 2 + 3} textAnchor="end" fontSize="10" fontFamily="var(--font-mono)" fill="var(--color-text-muted)">
                              #{d.idx}
                            </text>
                            {/* Exploitation bar */}
                            <rect x={padL} y={y} width={Math.max(exploitW, 1)} height={barH} rx="3" fill="rgba(34,197,94,0.55)" />
                            {/* Exploration bar */}
                            <rect x={padL + exploitW} y={y} width={Math.max(exploreW, 1)} height={barH} rx="3" fill="rgba(59,130,246,0.5)" />
                            {/* Value label */}
                            <text x={padL + exploitW + exploreW + 4} y={y + barH / 2 + 3} fontSize="9" fontFamily="var(--font-mono)" fill="var(--color-text-muted)">
                              {d.total.toFixed(2)}
                            </text>
                          </Fragment>
                        );
                      })}
                    </svg>
                    <div style={{ display: "flex", gap: "16px", marginTop: "4px" }}>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: 10, height: 10, borderRadius: 2, background: "rgba(34,197,94,0.55)", marginRight: 4, verticalAlign: "middle" }} />Exploitation</span>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: 10, height: 10, borderRadius: 2, background: "rgba(59,130,246,0.5)", marginRight: 4, verticalAlign: "middle" }} />Exploration</span>
                    </div>
                  </div>
                );
              })()}

              {/* Surrogate Calibration Scatter */}
              {suggestions && suggestions.predicted_values && trials.length >= 8 && (() => {
                // Compare predicted vs actual for past trials via leave-one-out style
                // Use last N trials as "test" points: compare actual to nearest-neighbor predicted
                const scObjKey = Object.keys(trials[0].kpis)[0];
                const scN = Math.min(trials.length, 40);
                const scTrials = trials.slice(-scN);
                const scActual = scTrials.map(t => Number(t.kpis[scObjKey]) || 0);
                // Approximate predictions using LOO nearest-neighbor from remaining trials
                const scPredicted = scTrials.map((t, idx) => {
                  const others = trials.filter((_, j) => j !== trials.length - scN + idx);
                  if (others.length === 0) return scActual[idx];
                  const pKeys = Object.keys(t.parameters);
                  let bestDist = Infinity, bestVal = scActual[idx];
                  for (const o of others) {
                    let dist = 0;
                    for (const k of pKeys) {
                      const diff = (Number(t.parameters[k]) || 0) - (Number(o.parameters[k]) || 0);
                      dist += diff * diff;
                    }
                    if (dist < bestDist) {
                      bestDist = dist;
                      bestVal = Number(o.kpis[scObjKey]) || 0;
                    }
                  }
                  return bestVal;
                });

                const allVals = [...scActual, ...scPredicted];
                const scMin = Math.min(...allVals);
                const scMax = Math.max(...allVals);
                const scRange = scMax - scMin || 1;
                const scW = 280, scH = 260, scPadL = 48, scPadR = 16, scPadT = 16, scPadB = 36;
                const scPlotW = scW - scPadL - scPadR;
                const scPlotH = scH - scPadT - scPadB;
                const toSX = (v: number) => scPadL + ((v - scMin) / scRange) * scPlotW;
                const toSY = (v: number) => scPadT + scPlotH - ((v - scMin) / scRange) * scPlotH;

                // R² calculation
                const meanActual = scActual.reduce((a, b) => a + b, 0) / scActual.length;
                const ssTot = scActual.reduce((s, v) => s + (v - meanActual) ** 2, 0);
                const ssRes = scActual.reduce((s, v, i) => s + (v - scPredicted[i]) ** 2, 0);
                const r2 = ssTot > 0 ? 1 - ssRes / ssTot : 0;

                // RMSE
                const rmse = Math.sqrt(ssRes / scActual.length);

                return (
                  <div className="card" style={{ marginBottom: 18 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
                      <BarChart2 size={18} style={{ color: "var(--color-text-muted)" }} />
                      <div>
                        <h2 style={{ margin: 0 }}>Surrogate Calibration</h2>
                        <p className="stat-label" style={{ margin: 0, textTransform: "none" }}>
                          Predicted vs actual comparison for last {scN} trials (LOO nearest-neighbor).
                        </p>
                      </div>
                      <span className="findings-badge" style={{ marginLeft: "auto" }}>
                        R² = {r2.toFixed(3)}
                      </span>
                    </div>
                    <div style={{ overflowX: "auto" }}>
                      <svg width={scW} height={scH} viewBox={`0 0 ${scW} ${scH}`} style={{ display: "block", margin: "0 auto" }}>
                        {/* Grid lines */}
                        {[0, 0.25, 0.5, 0.75, 1].map(f => {
                          const v = scMin + f * scRange;
                          return (
                            <g key={f}>
                              <line x1={toSX(v)} y1={scPadT} x2={toSX(v)} y2={scPadT + scPlotH} stroke="var(--color-border)" strokeWidth={0.5} />
                              <line x1={scPadL} y1={toSY(v)} x2={scPadL + scPlotW} y2={toSY(v)} stroke="var(--color-border)" strokeWidth={0.5} />
                            </g>
                          );
                        })}
                        {/* Perfect calibration line */}
                        <line
                          x1={toSX(scMin)} y1={toSY(scMin)}
                          x2={toSX(scMax)} y2={toSY(scMax)}
                          stroke="var(--color-text-muted)"
                          strokeWidth={1.5}
                          strokeDasharray="6,3"
                          opacity={0.5}
                        />
                        {/* Data points */}
                        {scActual.map((actual, i) => {
                          const pred = scPredicted[i];
                          const error = Math.abs(actual - pred) / scRange;
                          const pointColor = error < 0.1 ? "rgba(34,197,94,0.7)" : error < 0.25 ? "rgba(234,179,8,0.7)" : "rgba(239,68,68,0.7)";
                          return (
                            <circle
                              key={i}
                              cx={toSX(pred)}
                              cy={toSY(actual)}
                              r={4}
                              fill={pointColor}
                              stroke="rgba(255,255,255,0.4)"
                              strokeWidth={0.5}
                            >
                              <title>Trial {scN - scActual.length + i + 1}: pred={pred.toPrecision(4)}, actual={actual.toPrecision(4)}</title>
                            </circle>
                          );
                        })}
                        {/* Axes */}
                        <text x={scPadL + scPlotW / 2} y={scH - 2} textAnchor="middle" fontSize="10" fill="var(--color-text-muted)" fontFamily="var(--font-mono)">
                          Predicted
                        </text>
                        <text x={10} y={scPadT + scPlotH / 2} textAnchor="middle" fontSize="10" fill="var(--color-text-muted)" fontFamily="var(--font-mono)" transform={`rotate(-90,10,${scPadT + scPlotH / 2})`}>
                          Actual
                        </text>
                        {/* Axis tick labels */}
                        {[0, 0.5, 1].map(f => {
                          const v = scMin + f * scRange;
                          return (
                            <g key={f}>
                              <text x={toSX(v)} y={scPadT + scPlotH + 14} textAnchor="middle" fontSize="8" fill="var(--color-text-muted)" fontFamily="var(--font-mono)">
                                {v.toPrecision(3)}
                              </text>
                              <text x={scPadL - 4} y={toSY(v) + 3} textAnchor="end" fontSize="8" fill="var(--color-text-muted)" fontFamily="var(--font-mono)">
                                {v.toPrecision(3)}
                              </text>
                            </g>
                          );
                        })}
                      </svg>
                    </div>
                    <div style={{ display: "flex", gap: "16px", marginTop: "6px", flexWrap: "wrap", alignItems: "center" }}>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: 10, height: 10, borderRadius: "50%", background: "rgba(34,197,94,0.7)", marginRight: 4, verticalAlign: "middle" }} />&lt;10% error</span>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: 10, height: 10, borderRadius: "50%", background: "rgba(234,179,8,0.7)", marginRight: 4, verticalAlign: "middle" }} />10-25% error</span>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: 10, height: 10, borderRadius: "50%", background: "rgba(239,68,68,0.7)", marginRight: 4, verticalAlign: "middle" }} />&gt;25% error</span>
                      <span style={{ marginLeft: "auto", fontSize: "0.78rem", color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}>
                        RMSE: {rmse.toPrecision(3)}
                      </span>
                    </div>
                  </div>
                );
              })()}

              {/* Suggestion Stability Scores */}
              {suggestions && suggestions.suggestions.length > 0 && trials.length >= 8 && (() => {
                // Bootstrap stability: resample trials N times, re-rank by nearest-neighbor predicted obj
                const ssObjKey = Object.keys(trials[0].kpis)[0];
                const ssPKeys = Object.keys(trials[0].parameters);
                const nBoot = 20;

                const ssScores = suggestions.suggestions.map((sug) => {
                  const sugParams = ssPKeys.map(k => Number(sug[k]) || 0);
                  // For each bootstrap, find NN prediction and rank this suggestion
                  const predictions: number[] = [];
                  for (let b = 0; b < nBoot; b++) {
                    // Resample trials with replacement
                    const bootTrials: typeof trials = [];
                    for (let i = 0; i < trials.length; i++) {
                      bootTrials.push(trials[Math.floor(Math.random() * trials.length)]);
                    }
                    // NN prediction
                    let bestDist = Infinity, pred = 0;
                    for (const bt of bootTrials) {
                      let dist = 0;
                      for (let k = 0; k < ssPKeys.length; k++) {
                        const diff = sugParams[k] - (Number(bt.parameters[ssPKeys[k]]) || 0);
                        dist += diff * diff;
                      }
                      if (dist < bestDist) {
                        bestDist = dist;
                        pred = Number(bt.kpis[ssObjKey]) || 0;
                      }
                    }
                    predictions.push(pred);
                  }
                  // Stability = 1 - (std / range)
                  const meanPred = predictions.reduce((a, b) => a + b, 0) / nBoot;
                  const stdPred = Math.sqrt(predictions.reduce((s, p) => s + (p - meanPred) ** 2, 0) / nBoot);
                  const allObjVals = trials.map(t => Number(t.kpis[ssObjKey]) || 0);
                  const objRange = Math.max(...allObjVals) - Math.min(...allObjVals) || 1;
                  const stability = Math.max(0, Math.min(1, 1 - stdPred / (objRange * 0.5)));
                  return { stability: Math.round(stability * 100), meanPred, stdPred };
                });

                const avgStability = ssScores.reduce((s, sc) => s + sc.stability, 0) / ssScores.length;

                return (
                  <div className="card" style={{ marginBottom: 18 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
                      <Flag size={18} style={{ color: "var(--color-text-muted)" }} />
                      <div>
                        <h2 style={{ margin: 0 }}>Suggestion Stability</h2>
                        <p className="stat-label" style={{ margin: 0, textTransform: "none" }}>
                          Bootstrap reliability of each suggestion ({nBoot} resamples). Higher = more robust.
                        </p>
                      </div>
                      <span className="findings-badge" style={{ marginLeft: "auto" }}>
                        avg {avgStability.toFixed(0)}%
                      </span>
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                      {ssScores.map((sc, i) => {
                        const color = sc.stability >= 80 ? "rgba(34,197,94,0.7)" : sc.stability >= 50 ? "rgba(234,179,8,0.7)" : "rgba(239,68,68,0.7)";
                        const label = sc.stability >= 80 ? "Stable" : sc.stability >= 50 ? "Moderate" : "Unstable";
                        return (
                          <div key={i} style={{ display: "flex", alignItems: "center", gap: 10, padding: "4px 0" }}>
                            <span style={{ width: 32, fontSize: "0.82rem", fontWeight: 600, color: "var(--color-text-muted)", textAlign: "right", fontFamily: "var(--font-mono)", flexShrink: 0 }}>
                              #{i + 1}
                            </span>
                            <div style={{ flex: 1, height: 14, background: "var(--color-border)", borderRadius: 4, overflow: "hidden", position: "relative" }}>
                              <div style={{ width: `${sc.stability}%`, height: "100%", background: color, borderRadius: 4, transition: "width 0.3s ease", minWidth: 2 }} />
                            </div>
                            <span style={{ width: 42, fontSize: "0.78rem", fontFamily: "var(--font-mono)", fontWeight: 600, color, textAlign: "right", flexShrink: 0 }}>
                              {sc.stability}%
                            </span>
                            <span style={{ width: 60, fontSize: "0.72rem", color: "var(--color-text-muted)", flexShrink: 0 }}>
                              {label}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                    <div style={{ display: "flex", gap: "16px", marginTop: "8px", flexWrap: "wrap" }}>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: 10, height: 10, borderRadius: 2, background: "rgba(34,197,94,0.7)", marginRight: 4, verticalAlign: "middle" }} />Stable (&ge;80%)</span>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: 10, height: 10, borderRadius: 2, background: "rgba(234,179,8,0.7)", marginRight: 4, verticalAlign: "middle" }} />Moderate (50-80%)</span>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: 10, height: 10, borderRadius: 2, background: "rgba(239,68,68,0.7)", marginRight: 4, verticalAlign: "middle" }} />Unstable (&lt;50%)</span>
                    </div>
                  </div>
                );
              })()}

              {/* Suggestion Novelty Scores */}
              {suggestions && suggestions.suggestions.length > 0 && trials.length >= 3 && (() => {
                const snPKeys = Object.keys(trials[0].parameters);
                const snSpecs = campaign.spec?.parameters || [];
                const snRanges = snPKeys.map(k => {
                  const sp = snSpecs.find((s: { name: string }) => s.name === k);
                  const lo = sp && (sp as { lower?: number }).lower != null ? (sp as { lower: number }).lower : Math.min(...trials.map(t => Number(t.parameters[k]) || 0));
                  const hi = sp && (sp as { upper?: number }).upper != null ? (sp as { upper: number }).upper : Math.max(...trials.map(t => Number(t.parameters[k]) || 0));
                  return hi - lo || 1;
                });
                // Normalize trial coords
                const snTrialCoords = trials.map(t => snPKeys.map((k, d) => (Number(t.parameters[k]) || 0) / snRanges[d]));
                const snScores = suggestions.suggestions.map((sug) => {
                  const sugCoords = snPKeys.map((k, d) => (Number(sug[k]) || 0) / snRanges[d]);
                  let minDist = Infinity;
                  for (const tc of snTrialCoords) {
                    const dist = Math.sqrt(tc.reduce((s, v, d) => s + (v - sugCoords[d]) ** 2, 0));
                    if (dist < minDist) minDist = dist;
                  }
                  return minDist;
                });
                // Normalize to 0-1 (max possible dist in unit cube = sqrt(dims))
                const snMaxDist = Math.sqrt(snPKeys.length);
                const snNorm = snScores.map(d => Math.min(d / (snMaxDist * 0.5), 1));
                const snAvg = snNorm.reduce((a, b) => a + b, 0) / snNorm.length;
                const snW = 320, snH = 80, snPad = 16;
                const snCircleR = 18;
                const snSpacing = Math.min((snW - 2 * snPad) / snNorm.length, snCircleR * 2.8);
                const snStartX = snPad + (snW - 2 * snPad - (snNorm.length - 1) * snSpacing) / 2;
                return (
                  <div className="card">
                    <div style={{ display: "flex", alignItems: "flex-start", gap: "10px", marginBottom: "8px" }}>
                      <Circle size={16} style={{ color: "var(--color-primary)", marginTop: 2 }} />
                      <div style={{ flex: 1 }}>
                        <h2 style={{ margin: 0 }}>Suggestion Novelty</h2>
                        <p style={{ margin: "2px 0 0", fontSize: "0.78rem", color: "var(--color-text-muted)" }}>
                          Min distance from each suggestion to historical trials. Higher = more novel.
                        </p>
                      </div>
                      <span className="findings-badge" style={{ background: snAvg > 0.5 ? "rgba(59,130,246,0.12)" : snAvg > 0.25 ? "rgba(234,179,8,0.12)" : "rgba(239,68,68,0.12)", color: snAvg > 0.5 ? "#3b82f6" : snAvg > 0.25 ? "#eab308" : "#ef4444", border: `1px solid ${snAvg > 0.5 ? "#3b82f644" : snAvg > 0.25 ? "#eab30844" : "#ef444444"}` }}>
                        avg {(snAvg * 100).toFixed(0)}% novel
                      </span>
                    </div>
                    <svg width={snW} height={snH} viewBox={`0 0 ${snW} ${snH}`} role="img" aria-label="Suggestion novelty scores" style={{ display: "block", margin: "0 auto" }}>
                      {snNorm.map((nov, i) => {
                        const cx = snStartX + i * snSpacing;
                        const cy = snH / 2;
                        const fillColor = nov > 0.5 ? "#3b82f6" : nov > 0.25 ? "#eab308" : "#ef4444";
                        const fillH = snCircleR * 2 * nov;
                        const clipId = `sn-clip-${i}`;
                        return (
                          <g key={i}>
                            <defs>
                              <clipPath id={clipId}>
                                <circle cx={cx} cy={cy} r={snCircleR - 1} />
                              </clipPath>
                            </defs>
                            {/* Background circle */}
                            <circle cx={cx} cy={cy} r={snCircleR} fill="none" stroke="var(--color-border)" strokeWidth={1.5} />
                            {/* Fill from bottom */}
                            <rect x={cx - snCircleR} y={cy + snCircleR - fillH} width={snCircleR * 2} height={fillH} fill={fillColor} opacity={0.6} clipPath={`url(#${clipId})`} />
                            {/* Label */}
                            <text x={cx} y={cy + 4} textAnchor="middle" fontSize="9" fontWeight="600" fill="var(--color-text-primary)" fontFamily="var(--font-mono)">
                              {(nov * 100).toFixed(0)}%
                            </text>
                            {/* Rank below */}
                            <text x={cx} y={cy + snCircleR + 12} textAnchor="middle" fontSize="8" fill="var(--color-text-muted)">
                              #{i + 1}
                            </text>
                            <title>Suggestion #{i + 1}: {(nov * 100).toFixed(1)}% novelty (min distance: {snScores[i].toPrecision(3)})</title>
                          </g>
                        );
                      })}
                    </svg>
                    <div style={{ display: "flex", gap: "16px", marginTop: "4px", flexWrap: "wrap", alignItems: "center" }}>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: 10, height: 10, borderRadius: "50%", background: "#3b82f6", opacity: 0.6, marginRight: 4, verticalAlign: "middle" }} />High (&gt;50%)</span>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: 10, height: 10, borderRadius: "50%", background: "#eab308", opacity: 0.6, marginRight: 4, verticalAlign: "middle" }} />Moderate (25-50%)</span>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: 10, height: 10, borderRadius: "50%", background: "#ef4444", opacity: 0.6, marginRight: 4, verticalAlign: "middle" }} />Low (&lt;25%)</span>
                    </div>
                  </div>
                );
              })()}

              {/* Trust Region Membership */}
              {suggestions && suggestions.suggestions.length > 0 && trials.length >= 3 && (() => {
                const trSpecs = campaign.spec?.parameters?.filter((s: { name: string; type: string; lower?: number; upper?: number }) => s.type === "continuous" && s.lower != null && s.upper != null) || [];
                if (trSpecs.length === 0) return null;
                // Normalize observations
                const trNorm = (val: number, sp: { lower?: number; upper?: number }) => {
                  const lo = sp.lower ?? 0, hi = sp.upper ?? 1;
                  return hi > lo ? (val - lo) / (hi - lo) : 0.5;
                };
                const trObsNormed = trials.map(t => trSpecs.map((s: { name: string; lower?: number; upper?: number }) => trNorm(t.parameters[s.name] ?? 0, s)));
                const trDist = (a: number[], b: number[]) => Math.sqrt(a.reduce((s, v, i) => s + (v - (b[i] ?? 0)) ** 2, 0));
                // Get objective values
                const trObjKey = campaign.objective_names?.[0] || Object.keys(trials[0].kpis)[0];
                const trObjDir = (campaign.objective_directions?.[trObjKey] || "minimize") === "minimize" ? -1 : 1;
                const trScored = trObsNormed.map((n, i) => ({ normed: n, score: (trials[i].kpis[trObjKey] ?? 0) * trObjDir }));
                trScored.sort((a, b) => b.score - a.score);
                // Greedy select local optima
                const trOptima: { normed: number[]; radius: number }[] = [];
                const trMaxK = 3;
                for (const obs of trScored) {
                  if (trOptima.length >= trMaxK) break;
                  const tooClose = trOptima.some(o => trDist(o.normed, obs.normed) < 0.25);
                  if (!tooClose) {
                    trOptima.push({ normed: obs.normed, radius: 0.15 + 0.1 * (1 - trOptima.length / trMaxK) });
                  }
                }
                if (trOptima.length === 0) return null;
                // PCA-like 1D projection: use first continuous parameter dimension
                const trProject = (normed: number[]) => normed[0] ?? 0.5;
                // Classify suggestions
                const trSugs = suggestions.suggestions.map((sug: Record<string, number>, idx: number) => {
                  const normed = trSpecs.map((s: { name: string; lower?: number; upper?: number }) => trNorm(sug[s.name] ?? 0, s));
                  const minDist = Math.min(...trOptima.map(o => trDist(o.normed, normed) - o.radius));
                  return { idx: idx + 1, normed, proj: trProject(normed), exploit: minDist < 0 };
                });
                const trExploitCount = trSugs.filter((s: { exploit: boolean }) => s.exploit).length;
                const trExploreCount = trSugs.length - trExploitCount;
                const trStripW = 200, trStripH = 44;
                const trPadX = 10;
                const trMapX = (v: number) => trPadX + v * (trStripW - 2 * trPadX);
                return (
                  <div className="card" style={{ padding: "14px 16px" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                      <Scan size={16} style={{ color: "var(--color-text-muted)" }} />
                      <h3 style={{ margin: 0, fontSize: "0.92rem" }}>Trust Region Membership</h3>
                      <span className="findings-badge" style={{ background: trExploitCount > trExploreCount ? "#3b82f618" : "#f9731618", color: trExploitCount > trExploreCount ? "#3b82f6" : "#f97316", marginLeft: "auto" }}>
                        {trExploitCount} exploit / {trExploreCount} explore
                      </span>
                    </div>
                    <svg width="100%" viewBox={`0 0 ${trStripW} ${trStripH}`} style={{ display: "block" }}>
                      {/* Strip background */}
                      <rect x={trPadX} y={12} width={trStripW - 2 * trPadX} height={16} rx={8} fill="var(--color-border)" opacity={0.5} />
                      {/* Trust regions */}
                      {trOptima.map((o, i) => {
                        const cx = trMapX(trProject(o.normed));
                        const rw = o.radius * (trStripW - 2 * trPadX);
                        return (
                          <Fragment key={`trr${i}`}>
                            <rect x={Math.max(trPadX, cx - rw)} y={12} width={Math.min(rw * 2, trStripW - 2 * trPadX)} height={16} rx={8} fill="#3b82f6" opacity={0.12} />
                            <polygon points={`${cx - 3},12 ${cx + 3},12 ${cx},8`} fill="#eab308" />
                          </Fragment>
                        );
                      })}
                      {/* Suggestion dots */}
                      {trSugs.map((s: { idx: number; proj: number; exploit: boolean }) => (
                        <circle key={`trs${s.idx}`} cx={trMapX(s.proj)} cy={20} r={4.5} fill={s.exploit ? "#3b82f6" : "#f97316"} stroke="white" strokeWidth="1" />
                      ))}
                      {/* Labels */}
                      <text x={trPadX} y={trStripH - 1} fontSize="6" fill="var(--color-text-muted)">0</text>
                      <text x={trStripW - trPadX} y={trStripH - 1} fontSize="6" fill="var(--color-text-muted)" textAnchor="end">1</text>
                      <text x={trStripW / 2} y={trStripH - 1} fontSize="5.5" fill="var(--color-text-muted)" textAnchor="middle">{trSpecs[0]?.name || "dim 1"} (normalized)</text>
                    </svg>
                    <div style={{ display: "flex", gap: 12, fontSize: "0.72rem", color: "var(--color-text-muted)", marginTop: 4 }}>
                      <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
                        <span style={{ display: "inline-block", width: 8, height: 8, borderRadius: "50%", background: "#3b82f6" }} /> Exploit
                      </span>
                      <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
                        <span style={{ display: "inline-block", width: 8, height: 8, borderRadius: "50%", background: "#f97316" }} /> Explore
                      </span>
                      <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
                        <span style={{ display: "inline-block", width: 0, height: 0, borderLeft: "4px solid transparent", borderRight: "4px solid transparent", borderBottom: "6px solid #eab308" }} /> Local optima
                      </span>
                      <span style={{ marginLeft: "auto" }}>
                        <span style={{ display: "inline-block", width: 16, height: 8, background: "#3b82f6", opacity: 0.12, borderRadius: 3, marginRight: 3, verticalAlign: "middle" }} /> Trust regions
                      </span>
                    </div>
                  </div>
                );
              })()}

              {/* D-Optimality Scores */}
              {suggestions && suggestions.suggestions.length > 0 && trials.length >= 5 && (() => {
                const doSpecs = campaign.spec?.parameters?.filter((s: { name: string; type: string; lower?: number; upper?: number }) => s.type === "continuous" && s.lower != null && s.upper != null) || [];
                if (doSpecs.length === 0) return null;
                const doD = doSpecs.length;
                const doNorm = (val: number, sp: { lower?: number; upper?: number }) => {
                  const lo = sp.lower ?? 0, hi = sp.upper ?? 1;
                  return hi > lo ? (val - lo) / (hi - lo) : 0.5;
                };
                // Existing design matrix (normalized)
                const doX = trials.map(t => doSpecs.map((s: { name: string; lower?: number; upper?: number }) => doNorm(t.parameters[s.name] ?? 0, s)));
                // Compute min distance to existing points for each suggestion (space-filling score)
                const doScores = suggestions.suggestions.map((sug: Record<string, number>, idx: number) => {
                  const sugNorm = doSpecs.map((s: { name: string; lower?: number; upper?: number }) => doNorm(sug[s.name] ?? 0, s));
                  // D-optimality proxy: min distance to existing points (space-filling)
                  const minDist = Math.min(...doX.map((obs: number[]) => Math.sqrt(obs.reduce((a: number, v: number, i: number) => a + (v - sugNorm[i]) ** 2, 0))));
                  // A-optimality proxy: average distance to k nearest
                  const allDists = doX.map((obs: number[]) => Math.sqrt(obs.reduce((a: number, v: number, i: number) => a + (v - sugNorm[i]) ** 2, 0))).sort((a: number, b: number) => a - b);
                  const kNearest = allDists.slice(0, 5);
                  const avgKDist = kNearest.reduce((a: number, b: number) => a + b, 0) / kNearest.length;
                  // E-optimality proxy: distance from centroid
                  const centroid = doSpecs.map((_: unknown, di: number) => doX.reduce((a: number, obs: number[]) => a + obs[di], 0) / doX.length);
                  const centDist = Math.sqrt(centroid.reduce((a: number, c: number, i: number) => a + (c - sugNorm[i]) ** 2, 0));
                  return { idx: idx + 1, dScore: minDist, aScore: avgKDist, eScore: centDist };
                });
                // Normalize scores to 0-100
                const doMaxD = Math.max(...doScores.map((s: { dScore: number }) => s.dScore), 0.01);
                const doMaxA = Math.max(...doScores.map((s: { aScore: number }) => s.aScore), 0.01);
                const doMaxE = Math.max(...doScores.map((s: { eScore: number }) => s.eScore), 0.01);
                const doNormed = doScores.map((s: { idx: number; dScore: number; aScore: number; eScore: number }) => ({
                  idx: s.idx,
                  d: (s.dScore / doMaxD) * 100,
                  a: (s.aScore / doMaxA) * 100,
                  e: (s.eScore / doMaxE) * 100,
                }));
                const doBarW = 14, doGap = 6, doGroupW = doBarW * 3 + doGap;
                const doSvgW = doNormed.length * (doGroupW + 12) + 30;
                const doSvgH = 60;
                const doPadY = 10, doPadX = 20;
                const doChartH = doSvgH - 2 * doPadY;
                return (
                  <div className="card" style={{ padding: "14px 16px" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                      <Diamond size={16} style={{ color: "var(--color-text-muted)" }} />
                      <h3 style={{ margin: 0, fontSize: "0.92rem" }}>Design Optimality</h3>
                      <span className="findings-badge" style={{ background: "#3b82f618", color: "#3b82f6", marginLeft: "auto" }}>
                        {doD}D space-filling
                      </span>
                    </div>
                    <svg width="100%" viewBox={`0 0 ${doSvgW} ${doSvgH}`} style={{ display: "block" }}>
                      {/* 50% reference line */}
                      <line x1={doPadX} y1={doPadY + doChartH * 0.5} x2={doSvgW - 5} y2={doPadY + doChartH * 0.5} stroke="var(--color-border)" strokeWidth="0.5" strokeDasharray="3 2" />
                      <text x={doPadX - 2} y={doPadY + doChartH * 0.5 + 2} fontSize="5" fill="var(--color-text-muted)" textAnchor="end">50</text>
                      {/* Bars for each suggestion */}
                      {doNormed.map((s: { idx: number; d: number; a: number; e: number }, i: number) => {
                        const gx = doPadX + i * (doGroupW + 12);
                        const barH = (v: number) => (v / 100) * doChartH;
                        return (
                          <Fragment key={`do${i}`}>
                            <rect x={gx} y={doPadY + doChartH - barH(s.d)} width={doBarW} height={barH(s.d)} rx={2} fill="#3b82f6" opacity={0.8} />
                            <rect x={gx + doBarW} y={doPadY + doChartH - barH(s.a)} width={doBarW} height={barH(s.a)} rx={2} fill="#22c55e" opacity={0.8} />
                            <rect x={gx + doBarW * 2} y={doPadY + doChartH - barH(s.e)} width={doBarW} height={barH(s.e)} rx={2} fill="#8b5cf6" opacity={0.8} />
                            <text x={gx + doGroupW / 2} y={doSvgH - 1} fontSize="6" fill="var(--color-text-muted)" textAnchor="middle">#{s.idx}</text>
                          </Fragment>
                        );
                      })}
                    </svg>
                    <div style={{ display: "flex", gap: 10, fontSize: "0.72rem", color: "var(--color-text-muted)", marginTop: 4 }}>
                      <span style={{ display: "flex", alignItems: "center", gap: 3 }}>
                        <span style={{ display: "inline-block", width: 8, height: 8, borderRadius: 2, background: "#3b82f6" }} /> D-opt (min dist)
                      </span>
                      <span style={{ display: "flex", alignItems: "center", gap: 3 }}>
                        <span style={{ display: "inline-block", width: 8, height: 8, borderRadius: 2, background: "#22c55e" }} /> A-opt (k-NN avg)
                      </span>
                      <span style={{ display: "flex", alignItems: "center", gap: 3 }}>
                        <span style={{ display: "inline-block", width: 8, height: 8, borderRadius: 2, background: "#8b5cf6" }} /> E-opt (centroid)
                      </span>
                    </div>
                  </div>
                );
              })()}

              {/* Boundary Proximity Radar */}
              {suggestions && suggestions.suggestions.length > 0 && (() => {
                const bpSpecs = campaign.spec?.parameters?.filter((s: { name: string; type: string; lower?: number; upper?: number }) => s.type === "continuous" && s.lower != null && s.upper != null) || [];
                if (bpSpecs.length < 3) return null;
                const bpParams = bpSpecs.slice(0, 8);
                const bpN = bpParams.length;
                const bpSugs = suggestions.suggestions.slice(0, 5);

                // For each suggestion, compute normalized boundary proximity per param
                // proximity = how close to nearest bound (0=center, 1=at boundary)
                const bpData = bpSugs.map((sug: Record<string, number>) => {
                  return bpParams.map((s: { name: string; lower?: number; upper?: number }) => {
                    const lo = s.lower ?? 0, hi = s.upper ?? 1;
                    const norm = hi > lo ? (Number(sug[s.name] ?? 0) - lo) / (hi - lo) : 0.5;
                    return Math.abs(norm - 0.5) * 2; // 0=center, 1=boundary
                  });
                });

                const bpW = 200, bpH = 200, bpCx = bpW / 2, bpCy = bpH / 2, bpR = 72;
                const bpColors = ["rgba(59,130,246,0.6)", "rgba(34,197,94,0.6)", "rgba(234,179,8,0.6)", "rgba(239,68,68,0.5)", "rgba(168,85,247,0.5)"];

                // Polygon points helper
                const bpPoly = (values: number[]) => {
                  return values.map((v, i) => {
                    const angle = (2 * Math.PI * i) / bpN - Math.PI / 2;
                    const r = v * bpR;
                    return `${(bpCx + r * Math.cos(angle)).toFixed(1)},${(bpCy + r * Math.sin(angle)).toFixed(1)}`;
                  }).join(" ");
                };

                const bpAvgProx = bpData.flat().reduce((a: number, b: number) => a + b, 0) / (bpData.flat().length || 1);
                const bpLabel = bpAvgProx > 0.6 ? "Boundary-seeking" : bpAvgProx > 0.3 ? "Balanced" : "Interior-focused";

                return (
                  <div className="card" style={{ padding: "14px 16px" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                      <Orbit size={16} style={{ color: "var(--color-text-muted)" }} />
                      <h3 style={{ margin: 0, fontSize: "0.92rem" }}>Boundary Proximity</h3>
                      <span className="findings-badge" style={{ marginLeft: "auto", color: bpAvgProx > 0.6 ? "#f59e0b" : "#22c55e", borderColor: bpAvgProx > 0.6 ? "rgba(245,158,11,0.3)" : "rgba(34,197,94,0.3)" }}>
                        {bpLabel}
                      </span>
                    </div>
                    <svg width="100%" viewBox={`0 0 ${bpW} ${bpH}`} style={{ display: "block" }}>
                      {/* Grid circles */}
                      {[0.25, 0.5, 0.75, 1].map(f => (
                        <circle key={f} cx={bpCx} cy={bpCy} r={f * bpR} fill="none" stroke="var(--color-border)" strokeWidth="0.5" opacity={0.5} />
                      ))}
                      {/* Axis lines and labels */}
                      {bpParams.map((s: { name: string }, i: number) => {
                        const angle = (2 * Math.PI * i) / bpN - Math.PI / 2;
                        const lx = bpCx + (bpR + 16) * Math.cos(angle);
                        const ly = bpCy + (bpR + 16) * Math.sin(angle);
                        return (
                          <g key={i}>
                            <line x1={bpCx} y1={bpCy} x2={bpCx + bpR * Math.cos(angle)} y2={bpCy + bpR * Math.sin(angle)} stroke="var(--color-border)" strokeWidth="0.5" opacity={0.4} />
                            <text x={lx} y={ly + 3} textAnchor="middle" fontSize="6.5" fill="var(--color-text-muted)" fontFamily="var(--font-mono)">
                              {s.name.length > 6 ? s.name.slice(0, 5) + "…" : s.name}
                            </text>
                          </g>
                        );
                      })}
                      {/* Suggestion polygons */}
                      {bpData.map((vals: number[], si: number) => (
                        <polygon key={si} points={bpPoly(vals)} fill={bpColors[si % bpColors.length]} fillOpacity={0.12} stroke={bpColors[si % bpColors.length]} strokeWidth="1.5" />
                      ))}
                      {/* Center dot */}
                      <circle cx={bpCx} cy={bpCy} r="2" fill="var(--color-text-muted)" opacity={0.4} />
                      {/* Scale labels */}
                      <text x={bpCx + 3} y={bpCy - bpR + 3} fontSize="5.5" fill="var(--color-text-muted)">edge</text>
                      <text x={bpCx + 3} y={bpCy - 3} fontSize="5.5" fill="var(--color-text-muted)">center</text>
                    </svg>
                    <div style={{ display: "flex", gap: "10px", marginTop: "2px", flexWrap: "wrap", alignItems: "center", justifyContent: "center" }}>
                      {bpSugs.map((_: Record<string, number>, i: number) => (
                        <span key={i} className="efficiency-legend-item">
                          <span style={{ display: "inline-block", width: 10, height: 3, background: bpColors[i % bpColors.length], marginRight: 3, verticalAlign: "middle", borderRadius: 1 }} />
                          #{i + 1}
                        </span>
                      ))}
                    </div>
                  </div>
                );
              })()}

              {/* Batch Diversity Matrix */}
              {suggestions && suggestions.suggestions.length >= 2 && (() => {
                const bdSpecs = campaign.spec?.parameters?.filter((s: { name: string; type: string; lower?: number; upper?: number }) => s.type === "continuous" && s.lower != null && s.upper != null) || [];
                if (bdSpecs.length === 0) return null;
                const bdSugs = suggestions.suggestions;
                const bdN = bdSugs.length;

                // Normalize each suggestion parameter to [0,1]
                const bdNormalized = bdSugs.map(sug => {
                  const norm: number[] = [];
                  for (const sp of bdSpecs) {
                    const lo = (sp as { lower: number }).lower;
                    const hi = (sp as { upper: number }).upper;
                    const range = hi - lo || 1;
                    norm.push(((Number(sug[(sp as { name: string }).name]) || 0) - lo) / range);
                  }
                  return norm;
                });

                // Compute pairwise Euclidean distance matrix
                const bdDistMatrix: number[][] = [];
                let bdMaxDist = 0;
                for (let i = 0; i < bdN; i++) {
                  bdDistMatrix[i] = [];
                  for (let j = 0; j < bdN; j++) {
                    if (i === j) { bdDistMatrix[i][j] = 0; continue; }
                    let dist = 0;
                    for (let k = 0; k < bdNormalized[i].length; k++) {
                      dist += (bdNormalized[i][k] - bdNormalized[j][k]) ** 2;
                    }
                    dist = Math.sqrt(dist);
                    bdDistMatrix[i][j] = dist;
                    if (dist > bdMaxDist) bdMaxDist = dist;
                  }
                }

                // Average pairwise distance
                let bdSumDist = 0, bdCount = 0;
                for (let i = 0; i < bdN; i++) {
                  for (let j = i + 1; j < bdN; j++) {
                    bdSumDist += bdDistMatrix[i][j];
                    bdCount++;
                  }
                }
                const bdAvgDist = bdCount > 0 ? bdSumDist / bdCount : 0;
                const bdMaxPossible = Math.sqrt(bdSpecs.length); // max dist in normalized space
                const bdDiversityPct = bdMaxPossible > 0 ? (bdAvgDist / bdMaxPossible) * 100 : 0;
                const bdLabel = bdDiversityPct > 60 ? "Diverse" : bdDiversityPct > 30 ? "Moderate" : "Clustered";
                const bdLabelColor = bdDiversityPct > 60 ? "#22c55e" : bdDiversityPct > 30 ? "#f59e0b" : "#ef4444";

                // SVG heatmap
                const bdCellSize = Math.min(32, Math.floor(200 / bdN));
                const bdPadL = 28, bdPadT = 28;
                const bdW = bdPadL + bdN * bdCellSize + 10;
                const bdH = bdPadT + bdN * bdCellSize + 10;

                const bdColor = (d: number) => {
                  const t = bdMaxDist > 0 ? d / bdMaxDist : 0;
                  // Interpolate from blue (close) to green (far)
                  const r = Math.round(34 * (1 - t) + 34 * t);
                  const g = Math.round(130 * (1 - t) + 197 * t);
                  const b2 = Math.round(246 * (1 - t) + 94 * t);
                  return `rgb(${r},${g},${b2})`;
                };

                return (
                  <div className="card" style={{ marginBottom: "16px" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                      <LayoutGrid size={16} style={{ color: "var(--color-primary)" }} />
                      <h3 style={{ margin: 0, fontSize: "0.92rem" }}>Batch Diversity Matrix</h3>
                      <span className="findings-badge" style={{ background: bdLabelColor + "18", color: bdLabelColor, marginLeft: "auto" }}>
                        {bdLabel}
                      </span>
                    </div>
                    <div style={{ fontSize: "0.78rem", color: "var(--color-text-muted)", marginBottom: 8 }}>
                      Pairwise Euclidean distance (normalized) — avg {bdAvgDist.toFixed(3)}
                    </div>
                    <svg width={bdW} height={bdH} viewBox={`0 0 ${bdW} ${bdH}`} style={{ width: "100%", maxWidth: bdW, height: "auto" }}>
                      {/* Column headers */}
                      {bdSugs.map((_, i) => (
                        <text key={`bdch${i}`} x={bdPadL + i * bdCellSize + bdCellSize / 2} y={bdPadT - 6} fontSize="7" fill="var(--color-text-muted)" textAnchor="middle">#{i + 1}</text>
                      ))}
                      {/* Row headers */}
                      {bdSugs.map((_, i) => (
                        <text key={`bdrh${i}`} x={bdPadL - 4} y={bdPadT + i * bdCellSize + bdCellSize / 2 + 3} fontSize="7" fill="var(--color-text-muted)" textAnchor="end">#{i + 1}</text>
                      ))}
                      {/* Cells */}
                      {bdDistMatrix.map((row, i) =>
                        row.map((d, j) => (
                          <g key={`bdc${i}-${j}`}>
                            <rect
                              x={bdPadL + j * bdCellSize}
                              y={bdPadT + i * bdCellSize}
                              width={bdCellSize - 1}
                              height={bdCellSize - 1}
                              rx={2}
                              fill={i === j ? "var(--color-bg-secondary)" : bdColor(d)}
                              opacity={i === j ? 0.3 : 0.7 + 0.3 * (d / (bdMaxDist || 1))}
                            />
                            {bdCellSize >= 20 && (
                              <text
                                x={bdPadL + j * bdCellSize + (bdCellSize - 1) / 2}
                                y={bdPadT + i * bdCellSize + (bdCellSize - 1) / 2 + 3}
                                fontSize="6"
                                fill={i === j ? "var(--color-text-muted)" : "white"}
                                textAnchor="middle"
                                fontWeight="500"
                              >
                                {i === j ? "—" : d.toFixed(2)}
                              </text>
                            )}
                          </g>
                        ))
                      )}
                      {/* Color legend */}
                      <defs>
                        <linearGradient id="bdGrad" x1="0" y1="0" x2="1" y2="0">
                          <stop offset="0%" stopColor="rgb(34,130,246)" />
                          <stop offset="100%" stopColor="rgb(34,197,94)" />
                        </linearGradient>
                      </defs>
                      <rect x={bdPadL} y={bdPadT + bdN * bdCellSize + 2} width={bdN * bdCellSize - 1} height={4} rx={2} fill="url(#bdGrad)" opacity="0.6" />
                      <text x={bdPadL} y={bdPadT + bdN * bdCellSize + 12} fontSize="5" fill="var(--color-text-muted)">Close</text>
                      <text x={bdPadL + bdN * bdCellSize - 1} y={bdPadT + bdN * bdCellSize + 12} fontSize="5" fill="var(--color-text-muted)" textAnchor="end">Far</text>
                    </svg>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.72rem", color: "var(--color-text-muted)", marginTop: 4 }}>
                      <span>{bdSpecs.length}D space, {bdN} suggestions</span>
                      <span>diversity: {bdDiversityPct.toFixed(1)}%</span>
                    </div>
                  </div>
                );
              })()}

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
                  {suggestions.suggestions.map((sug, i) => {
                    const diversity = computeDiversityScore(sug);
                    return (
                      <div key={i} className={`suggestion-card-wrapper ${selectedSuggestions.has(i) ? "suggestion-selected" : ""}`}>
                        <div className="suggestion-select-check" onClick={(e) => { e.stopPropagation(); toggleSuggestionSelect(i); }}>
                          <CheckSquare
                            size={14}
                            className={selectedSuggestions.has(i) ? "sug-check-active" : "sug-check-idle"}
                            fill={selectedSuggestions.has(i) ? "currentColor" : "none"}
                          />
                        </div>
                        {diversity !== null && (
                          <div
                            className={`diversity-badge ${diversity > 0.5 ? "diversity-high" : diversity > 0.25 ? "diversity-mid" : "diversity-low"}`}
                            title={`Diversity: ${(diversity * 100).toFixed(0)}% — ${diversity > 0.5 ? "Highly exploratory" : diversity > 0.25 ? "Moderately novel" : "Close to recent trials"}`}
                          >
                            <Activity size={11} />
                            {diversity > 0.5 ? "Explore" : diversity > 0.25 ? "Balanced" : "Refine"}
                          </div>
                        )}
                        <RealSuggestionCard
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
                          onReject={() => handleRejectSuggestion(sug, i + 1)}
                        />
                      </div>
                    );
                  })}
                </div>
              )}

              {/* Rejected Suggestions Stack */}
              {rejectedSuggestions.length > 0 && (
                <div className="rejected-stack">
                  <button
                    className="rejected-stack-toggle"
                    onClick={() => setShowRejectedStack(s => !s)}
                  >
                    <Undo2 size={14} />
                    <span>Recently Rejected ({rejectedSuggestions.length})</span>
                    {showRejectedStack ? <ArrowUp size={14} /> : <ArrowDown size={14} />}
                  </button>
                  {showRejectedStack && (
                    <div className="rejected-stack-list">
                      {rejectedSuggestions.map((r, idx) => {
                        const paramEntries = Object.entries(r.suggestion).slice(0, 4);
                        return (
                          <div key={r.timestamp} className="rejected-item">
                            <div className="rejected-item-header">
                              <span className="rejected-item-index">#{r.index}</span>
                              <span className="rejected-item-time">
                                {new Date(r.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                              </span>
                              <button
                                className="btn btn-sm btn-secondary rejected-reconsider-btn"
                                onClick={() => handleReconsider(idx)}
                              >
                                <Undo2 size={12} /> Reconsider
                              </button>
                            </div>
                            <div className="rejected-item-params">
                              {paramEntries.map(([k, v]) => (
                                <span key={k} className="rejected-param">
                                  <span className="rejected-param-name">{k}</span>
                                  <span className="mono">{typeof v === "number" ? v.toFixed(3) : String(v)}</span>
                                </span>
                              ))}
                              {Object.keys(r.suggestion).length > 4 && (
                                <span className="rejected-param-more">+{Object.keys(r.suggestion).length - 4} more</span>
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              )}

              {/* Batch Planner Bar */}
              {suggestions && selectedSuggestions.size > 0 && (
                <div className="batch-planner-bar">
                  <div className="batch-planner-left">
                    <Package size={15} />
                    <span className="batch-planner-count">{selectedSuggestions.size} selected</span>
                    {(() => {
                      if (!suggestions || selectedSuggestions.size < 2) return null;
                      const selected = suggestions.suggestions.filter((_, i) => selectedSuggestions.has(i));
                      const specs = campaign.spec?.parameters?.filter(s => s.type === "continuous" && s.lower != null && s.upper != null) ?? [];
                      if (specs.length === 0 || selected.length < 2) return null;
                      let totalDist = 0;
                      let pairs = 0;
                      for (let a = 0; a < selected.length; a++) {
                        for (let b = a + 1; b < selected.length; b++) {
                          let sumSq = 0;
                          for (const spec of specs) {
                            const r = (spec.upper! - spec.lower!) || 1;
                            sumSq += (((selected[a][spec.name] ?? 0) - (selected[b][spec.name] ?? 0)) / r) ** 2;
                          }
                          totalDist += Math.sqrt(sumSq / specs.length);
                          pairs++;
                        }
                      }
                      const batchDiversity = pairs > 0 ? totalDist / pairs : 0;
                      return (
                        <span className="batch-diversity-pill">
                          <Activity size={11} />
                          Batch diversity: {(batchDiversity * 100).toFixed(0)}%
                        </span>
                      );
                    })()}
                  </div>
                  <div className="batch-planner-right">
                    <button className="btn btn-sm btn-secondary" onClick={handleExportSelectedCSV}>
                      <FileDown size={13} /> Export CSV
                    </button>
                    <button className="btn btn-sm btn-secondary" onClick={() => setSelectedSuggestions(new Set())}>
                      Clear
                    </button>
                  </div>
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
              {/* Trial Outcome Distribution */}
              {trials.length >= 5 && (() => {
                const objKey = Object.keys(trials[0].kpis)[0];
                const vals = trials.map(t => Number(t.kpis[objKey]) || 0).sort((a, b) => a - b);
                const minV = vals[0];
                const maxV = vals[vals.length - 1];
                const range = maxV - minV;
                if (range === 0) return null;
                const nBins = Math.min(20, Math.max(8, Math.ceil(Math.sqrt(vals.length))));
                const binWidth = range / nBins;
                const bins = Array.from({ length: nBins }, (_, i) => ({
                  low: minV + i * binWidth,
                  high: minV + (i + 1) * binWidth,
                  count: 0,
                }));
                vals.forEach(v => {
                  const bi = Math.min(Math.floor((v - minV) / binWidth), nBins - 1);
                  bins[bi].count++;
                });
                const maxCount = Math.max(...bins.map(b => b.count));
                const q1 = vals[Math.floor(vals.length * 0.25)];
                const median = vals[Math.floor(vals.length * 0.5)];
                const q3 = vals[Math.floor(vals.length * 0.75)];
                const bestV = vals[0]; // sorted ascending, best is min for minimize
                const W = 460, H = 120, padL = 4, padR = 4, padT = 8, padB = 24;
                const plotW = W - padL - padR;
                const plotH = H - padT - padB;
                const barW = plotW / nBins - 1;
                const qX = (v: number) => padL + ((v - minV) / range) * plotW;

                return (
                  <div className="card histogram-card">
                    <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "6px" }}>
                      <BarChart3 size={16} style={{ color: "var(--color-primary)" }} />
                      <h2 style={{ margin: 0 }}>Outcome Distribution</h2>
                      <span style={{ fontSize: "0.75rem", color: "var(--color-text-muted)", marginLeft: "auto" }}>
                        {objKey} &middot; {trials.length} trials
                      </span>
                    </div>
                    <svg width={W} height={H} style={{ display: "block", width: "100%", maxWidth: `${W}px` }}>
                      {/* Bins */}
                      {bins.map((b, i) => {
                        const x = padL + (i / nBins) * plotW;
                        const barH = maxCount > 0 ? (b.count / maxCount) * plotH : 0;
                        const pctile = (b.low + b.high) / 2;
                        const color = pctile <= q1 ? "rgba(34,197,94,0.6)" : pctile >= q3 ? "rgba(239,68,68,0.5)" : "rgba(59,130,246,0.4)";
                        return (
                          <g key={i}>
                            <rect x={x} y={padT + plotH - barH} width={barW} height={barH} rx="2" fill={color}>
                              <title>{`${b.low.toFixed(4)} – ${b.high.toFixed(4)}: ${b.count} trials`}</title>
                            </rect>
                          </g>
                        );
                      })}
                      {/* Quartile markers */}
                      {[
                        { v: q1, label: "Q1", color: "var(--color-text-muted)" },
                        { v: median, label: "Med", color: "var(--color-primary)" },
                        { v: q3, label: "Q3", color: "var(--color-text-muted)" },
                        { v: bestV, label: "Best", color: "#22c55e" },
                      ].map(({ v, label, color }) => (
                        <g key={label}>
                          <line x1={qX(v)} y1={padT} x2={qX(v)} y2={padT + plotH + 4} stroke={color} strokeWidth="1" strokeDasharray="3,2" />
                          <text x={qX(v)} y={H - 4} textAnchor="middle" fontSize="9" fontFamily="var(--font-mono)" fill={color} fontWeight={label === "Best" ? 600 : 400}>
                            {label}
                          </text>
                        </g>
                      ))}
                      {/* X-axis labels */}
                      <text x={padL} y={H - 4} textAnchor="start" fontSize="9" fontFamily="var(--font-mono)" fill="var(--color-text-muted)">
                        {minV.toFixed(3)}
                      </text>
                      <text x={padL + plotW} y={H - 4} textAnchor="end" fontSize="9" fontFamily="var(--font-mono)" fill="var(--color-text-muted)">
                        {maxV.toFixed(3)}
                      </text>
                    </svg>
                  </div>
                );
              })()}

              {/* Exploration vs Exploitation Timeline */}
              {trials.length >= 8 && (() => {
                const chrono = [...trials].sort((a, b) => a.iteration - b.iteration);
                const paramNames = Object.keys(chrono[0].parameters);
                const objKey = Object.keys(chrono[0].kpis)[0];

                // Classify each trial as explore or exploit
                const classifications: Array<{ iteration: number; score: number; label: "explore" | "exploit" }> = [];
                let runningBestIdx = 0;
                let runningBestVal = Number(chrono[0].kpis[objKey]) || 0;

                for (let i = 0; i < chrono.length; i++) {
                  const curVal = Number(chrono[i].kpis[objKey]) || 0;
                  if (curVal < runningBestVal) {
                    runningBestVal = curVal;
                    runningBestIdx = i;
                  }
                  if (i === 0) {
                    classifications.push({ iteration: chrono[i].iteration, score: 1, label: "explore" });
                    continue;
                  }

                  // Novelty: min normalized distance to all previous trials
                  let minDist = Infinity;
                  for (let j = 0; j < i; j++) {
                    let dist = 0;
                    for (const p of paramNames) {
                      const d = (Number(chrono[i].parameters[p]) || 0) - (Number(chrono[j].parameters[p]) || 0);
                      dist += d * d;
                    }
                    minDist = Math.min(minDist, Math.sqrt(dist));
                  }
                  // Distance to best-so-far
                  let bestDist = 0;
                  for (const p of paramNames) {
                    const d = (Number(chrono[i].parameters[p]) || 0) - (Number(chrono[runningBestIdx].parameters[p]) || 0);
                    bestDist += d * d;
                  }
                  bestDist = Math.sqrt(bestDist);

                  // Exploration score: high novelty & far from best = explore; low novelty & near best = exploit
                  const noveltyNorm = Math.min(minDist / (Math.sqrt(paramNames.length) * 0.3 + 0.01), 1);
                  const bestDistNorm = Math.min(bestDist / (Math.sqrt(paramNames.length) * 0.3 + 0.01), 1);
                  const score = 0.5 * noveltyNorm + 0.5 * bestDistNorm;
                  classifications.push({
                    iteration: chrono[i].iteration,
                    score,
                    label: score > 0.45 ? "explore" : "exploit",
                  });
                }

                const exploreCount = classifications.filter(c => c.label === "explore").length;
                const exploitCount = classifications.length - exploreCount;
                const explorePct = ((exploreCount / classifications.length) * 100).toFixed(0);

                const W = 460, H = 80, padL = 4, padR = 4, padT = 8, padB = 20;
                const plotW = W - padL - padR;
                const plotH = H - padT - padB;
                const barW = Math.max(1, plotW / classifications.length - 0.5);

                return (
                  <div className="card">
                    <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}>
                      <Shuffle size={16} style={{ color: "var(--color-primary)" }} />
                      <h2 style={{ margin: 0 }}>Explore vs Exploit</h2>
                      <span style={{ fontSize: "0.72rem", color: "var(--color-text-muted)", marginLeft: "auto", fontFamily: "var(--font-mono)" }}>
                        {explorePct}% explore · {100 - Number(explorePct)}% exploit
                      </span>
                    </div>
                    <p style={{ fontSize: "0.78rem", color: "var(--color-text-muted)", margin: "0 0 6px" }}>
                      Each bar represents a trial. Blue = exploring new regions, orange = refining near known good results.
                    </p>
                    <svg width={W} height={H} style={{ display: "block", width: "100%", maxWidth: `${W}px` }}>
                      {classifications.map((c, i) => {
                        const x = padL + (i / classifications.length) * plotW;
                        const barH = padT + (1 - c.score) * plotH;
                        const color = c.label === "explore" ? "rgba(59,130,246,0.7)" : "rgba(249,115,22,0.65)";
                        return (
                          <rect key={i} x={x} y={barH} width={barW} height={padT + plotH - barH} fill={color} rx="1">
                            <title>Trial #{c.iteration}: {c.label} (score {c.score.toFixed(2)})</title>
                          </rect>
                        );
                      })}
                      {/* Threshold line */}
                      <line x1={padL} y1={padT + 0.55 * plotH} x2={padL + plotW} y2={padT + 0.55 * plotH}
                        stroke="var(--color-text-muted)" strokeWidth="0.5" strokeDasharray="4,3" opacity="0.5" />
                      {/* X labels */}
                      <text x={padL} y={H - 4} fontSize="9" fontFamily="var(--font-mono)" fill="var(--color-text-muted)">0</text>
                      <text x={padL + plotW} y={H - 4} textAnchor="end" fontSize="9" fontFamily="var(--font-mono)" fill="var(--color-text-muted)">{classifications.length}</text>
                      <text x={padL + plotW / 2} y={H - 4} textAnchor="middle" fontSize="10" fill="var(--color-text-muted)">Trial</text>
                    </svg>
                    <div className="efficiency-legend" style={{ maxWidth: `${W}px` }}>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: 10, height: 10, borderRadius: 2, background: "rgba(59,130,246,0.7)", marginRight: 4, verticalAlign: "middle" }} />Explore ({exploreCount})</span>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: 10, height: 10, borderRadius: 2, background: "rgba(249,115,22,0.65)", marginRight: 4, verticalAlign: "middle" }} />Exploit ({exploitCount})</span>
                    </div>
                  </div>
                );
              })()}

              {/* Trial Clustering Visualization */}
              {trials.length >= 12 && (() => {
                const paramNames = Object.keys(trials[0].parameters);
                if (paramNames.length < 2) return null;
                const objKey = Object.keys(trials[0].kpis)[0];

                // Normalize parameter values to [0,1]
                const mins = paramNames.map(p => Math.min(...trials.map(t => Number(t.parameters[p]) || 0)));
                const maxs = paramNames.map(p => Math.max(...trials.map(t => Number(t.parameters[p]) || 0)));
                const ranges = paramNames.map((_, i) => maxs[i] - mins[i] || 1);
                const normalized = trials.map(t =>
                  paramNames.map((p, i) => ((Number(t.parameters[p]) || 0) - mins[i]) / ranges[i])
                );

                // Simple k-means clustering (k=3)
                const k = Math.min(3, Math.floor(trials.length / 4));
                // Initialize centroids via spread
                const centroids: number[][] = [];
                for (let c = 0; c < k; c++) {
                  const idx = Math.floor((c / k) * normalized.length);
                  centroids.push([...normalized[idx]]);
                }

                // Run 15 iterations of k-means
                const assignments = new Array(trials.length).fill(0);
                for (let iter = 0; iter < 15; iter++) {
                  // Assign
                  for (let i = 0; i < normalized.length; i++) {
                    let bestDist = Infinity;
                    for (let c = 0; c < k; c++) {
                      let dist = 0;
                      for (let d = 0; d < paramNames.length; d++) {
                        dist += (normalized[i][d] - centroids[c][d]) ** 2;
                      }
                      if (dist < bestDist) { bestDist = dist; assignments[i] = c; }
                    }
                  }
                  // Update centroids
                  for (let c = 0; c < k; c++) {
                    const members = normalized.filter((_, i) => assignments[i] === c);
                    if (members.length === 0) continue;
                    for (let d = 0; d < paramNames.length; d++) {
                      centroids[c][d] = members.reduce((a, m) => a + m[d], 0) / members.length;
                    }
                  }
                }

                // Cluster stats
                const clusterColors = ["rgba(59,130,246,0.7)", "rgba(239,68,68,0.65)", "rgba(34,197,94,0.7)"];
                const clusterLabels = ["Cluster A", "Cluster B", "Cluster C"];
                const clusterStats = Array.from({ length: k }, (_, c) => {
                  const members = trials.filter((_, i) => assignments[i] === c);
                  const objVals = members.map(t => Number(t.kpis[objKey]) || 0);
                  return {
                    count: members.length,
                    bestObj: objVals.length > 0 ? Math.min(...objVals) : Infinity,
                    meanObj: objVals.length > 0 ? objVals.reduce((a, b) => a + b, 0) / objVals.length : 0,
                  };
                });

                // Plot using first two parameters
                const xParam = paramNames[0];
                const yParam = paramNames[1];
                const W = 460, H = 200, padL = 50, padR = 12, padT = 10, padB = 32;
                const plotW = W - padL - padR;
                const plotH = H - padT - padB;
                const xVals = trials.map(t => Number(t.parameters[xParam]) || 0);
                const yVals = trials.map(t => Number(t.parameters[yParam]) || 0);
                const xMin = Math.min(...xVals), xMax = Math.max(...xVals);
                const yMin = Math.min(...yVals), yMax = Math.max(...yVals);
                const xRange = xMax - xMin || 1;
                const yRange = yMax - yMin || 1;
                const sx = (v: number) => padL + ((v - xMin) / xRange) * plotW;
                const sy = (v: number) => padT + (1 - (v - yMin) / yRange) * plotH;

                return (
                  <div className="card">
                    <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}>
                      <GitBranch size={16} style={{ color: "var(--color-primary)" }} />
                      <h2 style={{ margin: 0 }}>Trial Clusters</h2>
                      <span style={{ fontSize: "0.72rem", color: "var(--color-text-muted)", marginLeft: "auto", fontFamily: "var(--font-mono)" }}>
                        k={k} · {paramNames.length}D
                      </span>
                    </div>
                    <p style={{ fontSize: "0.78rem", color: "var(--color-text-muted)", margin: "0 0 8px" }}>
                      K-means clustering of trials in parameter space. Colors indicate cluster membership, diamonds mark centroids.
                    </p>
                    <svg width={W} height={H} style={{ display: "block", width: "100%", maxWidth: `${W}px` }}>
                      {/* Grid */}
                      {[0, 0.5, 1].map(f => (
                        <Fragment key={f}>
                          <line x1={padL} y1={padT + f * plotH} x2={padL + plotW} y2={padT + f * plotH} stroke="var(--color-border)" strokeWidth="0.5" strokeDasharray="3,3" />
                          <line x1={padL + f * plotW} y1={padT} x2={padL + f * plotW} y2={padT + plotH} stroke="var(--color-border)" strokeWidth="0.5" strokeDasharray="3,3" />
                        </Fragment>
                      ))}
                      {/* Data points */}
                      {trials.map((t, i) => (
                        <circle
                          key={i}
                          cx={sx(xVals[i])}
                          cy={sy(yVals[i])}
                          r="3.5"
                          fill={clusterColors[assignments[i] % k]}
                          stroke="white"
                          strokeWidth="0.5"
                          opacity="0.8"
                        >
                          <title>Trial #{t.iteration}: {clusterLabels[assignments[i]]} · {objKey}={Number(t.kpis[objKey]).toFixed(4)}</title>
                        </circle>
                      ))}
                      {/* Centroids as diamonds */}
                      {centroids.map((c, ci) => {
                        const cx = padL + c[0] * plotW;
                        const cy = padT + (1 - c[1]) * plotH;
                        const s = 6;
                        return (
                          <polygon
                            key={ci}
                            points={`${cx},${cy - s} ${cx + s},${cy} ${cx},${cy + s} ${cx - s},${cy}`}
                            fill={clusterColors[ci % k]}
                            stroke="white"
                            strokeWidth="1.5"
                          >
                            <title>{clusterLabels[ci]}: {clusterStats[ci].count} trials, best={clusterStats[ci].bestObj.toFixed(4)}</title>
                          </polygon>
                        );
                      })}
                      {/* Axes */}
                      <line x1={padL} y1={padT} x2={padL} y2={padT + plotH} stroke="var(--color-border)" strokeWidth="1" />
                      <line x1={padL} y1={padT + plotH} x2={padL + plotW} y2={padT + plotH} stroke="var(--color-border)" strokeWidth="1" />
                      {/* Axis labels */}
                      {[0, 0.5, 1].map(f => (
                        <Fragment key={`a${f}`}>
                          <text x={padL + f * plotW} y={H - 8} textAnchor="middle" fontSize="9" fontFamily="var(--font-mono)" fill="var(--color-text-muted)">
                            {(xMin + f * xRange).toFixed(2)}
                          </text>
                          <text x={padL - 4} y={padT + (1 - f) * plotH + 3} textAnchor="end" fontSize="9" fontFamily="var(--font-mono)" fill="var(--color-text-muted)">
                            {(yMin + f * yRange).toFixed(2)}
                          </text>
                        </Fragment>
                      ))}
                      <text x={padL + plotW / 2} y={H - 0} textAnchor="middle" fontSize="10" fill="var(--color-text-muted)">{xParam}</text>
                      <text x={10} y={padT + plotH / 2} textAnchor="middle" fontSize="10" fill="var(--color-text-muted)" transform={`rotate(-90, 10, ${padT + plotH / 2})`}>{yParam}</text>
                    </svg>
                    {/* Cluster summary */}
                    <div style={{ display: "flex", gap: "16px", flexWrap: "wrap", marginTop: "8px" }}>
                      {clusterStats.map((cs, ci) => (
                        <div key={ci} style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "0.78rem" }}>
                          <span style={{ display: "inline-block", width: 10, height: 10, borderRadius: 2, background: clusterColors[ci % k] }} />
                          <span style={{ fontWeight: 500 }}>{clusterLabels[ci]}</span>
                          <span style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}>
                            n={cs.count} · best={cs.bestObj.toFixed(4)} · avg={cs.meanObj.toFixed(4)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                );
              })()}

              {/* Fidelity-Aware Trial Timeline */}
              {trials.length >= 8 && (() => {
                const ftObjKey = Object.keys(trials[0].kpis)[0];
                const ftObjVals = trials.map(t => Number(t.kpis[ftObjKey]) || 0);
                const ftMinObj = Math.min(...ftObjVals), ftMaxObj = Math.max(...ftObjVals);
                const ftObjRange = ftMaxObj - ftMinObj || 1;
                // Infer fidelity from trial position and objective variance
                // Heuristic: early trials = low fidelity (exploration), mid = medium, late = high (exploitation)
                const ftTiers = trials.map((_t, i) => {
                  if (i < trials.length * 0.3) return "low";
                  if (i < trials.length * 0.7) return "medium";
                  return "high";
                });

                const tierConfig: Record<string, { color: string; label: string; order: number }> = {
                  low: { color: "rgba(147,197,253,0.7)", label: "Low Fidelity", order: 0 },
                  medium: { color: "rgba(59,130,246,0.7)", label: "Medium Fidelity", order: 1 },
                  high: { color: "rgba(30,64,175,0.85)", label: "High Fidelity", order: 2 },
                };

                const W = 420, H = 200, padL = 52, padR = 16, padT = 28, padB = 32;
                const plotW = W - padL - padR, plotH = H - padT - padB;
                const n = trials.length;

                const ftToX = (i: number) => padL + (i / Math.max(n - 1, 1)) * plotW;
                const ftToY = (v: number) => padT + (1 - (v - ftMinObj) / ftObjRange) * plotH;

                // Running best line
                let ftRunBest = ftObjVals[0];
                const ftBestLine = ftObjVals.map(v => { ftRunBest = Math.min(ftRunBest, v); return ftRunBest; });
                const ftBestPath = ftBestLine.map((v, i) => `${i === 0 ? "M" : "L"}${ftToX(i).toFixed(1)},${ftToY(v).toFixed(1)}`).join(" ");

                const tierCounts = Object.entries(tierConfig).map(([k, cfg]) => ({
                  ...cfg, key: k, count: ftTiers.filter(t => t === k).length,
                })).filter(tc => tc.count > 0).sort((a, b) => a.order - b.order);

                return (
                  <div className="card">
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "4px" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                        <Layers size={16} style={{ color: "var(--color-primary)" }} />
                        <h2 style={{ margin: 0 }}>Fidelity Timeline</h2>
                      </div>
                      <span style={{ fontSize: "0.75rem", fontFamily: "var(--font-mono)", color: "var(--color-text-muted)" }}>
                        {tierCounts.map(tc => `${tc.count} ${tc.key}`).join(" · ")}
                      </span>
                    </div>
                    <p style={{ fontSize: "0.78rem", color: "var(--color-text-muted)", margin: "0 0 8px" }}>
                      Trial performance colored by inferred fidelity level. Bold line tracks the running best.
                    </p>
                    <svg width={W} height={H} style={{ display: "block", width: "100%", maxWidth: `${W}px` }}>
                      {/* Grid */}
                      {[0, 0.25, 0.5, 0.75, 1].map(f => (
                        <Fragment key={f}>
                          <line x1={padL} y1={padT + (1 - f) * plotH} x2={padL + plotW} y2={padT + (1 - f) * plotH} stroke="var(--color-border)" strokeWidth="0.5" strokeDasharray="2,3" />
                          <text x={padL - 6} y={padT + (1 - f) * plotH + 3} textAnchor="end" fontSize="8" fill="var(--color-text-muted)" fontFamily="var(--font-mono)">
                            {(ftMinObj + f * ftObjRange).toFixed(2)}
                          </text>
                        </Fragment>
                      ))}
                      {/* X-axis */}
                      <text x={padL + plotW / 2} y={H - 4} textAnchor="middle" fontSize="9" fill="var(--color-text-muted)">Trial Index</text>
                      {/* Running best line */}
                      <path d={ftBestPath} fill="none" stroke="var(--color-primary)" strokeWidth="1.5" strokeDasharray="4,2" opacity="0.5" />
                      {/* Data points */}
                      {trials.map((t, i) => {
                        const tier = ftTiers[i];
                        const cfg = tierConfig[tier] || tierConfig.medium;
                        return (
                          <circle
                            key={i}
                            cx={ftToX(i)}
                            cy={ftToY(ftObjVals[i])}
                            r={tier === "high" ? 4 : tier === "medium" ? 3.5 : 3}
                            fill={cfg.color}
                            stroke="white"
                            strokeWidth="0.5"
                            opacity="0.85"
                          >
                            <title>Trial #{t.iteration}: {ftObjKey}={ftObjVals[i].toFixed(4)} ({tier} fidelity)</title>
                          </circle>
                        );
                      })}
                    </svg>
                    {/* Legend */}
                    <div style={{ display: "flex", gap: "14px", marginTop: "4px" }}>
                      {tierCounts.map(tc => (
                        <span key={tc.key} className="efficiency-legend-item">
                          <span style={{ display: "inline-block", width: 10, height: 10, borderRadius: "50%", background: tc.color, marginRight: 4, verticalAlign: "middle" }} />
                          {tc.label}
                        </span>
                      ))}
                    </div>
                  </div>
                );
              })()}

              {/* Trial Diversity Timeline */}
              {trials.length >= 10 && (() => {
                // Measure pairwise diversity over sliding windows
                const tdObjKey = Object.keys(trials[0].kpis)[0];
                const pKeys = Object.keys(trials[0].parameters);
                const windowSize = Math.max(3, Math.floor(trials.length / 8));
                const tdWindows: { start: number; end: number; diversity: number; meanObj: number }[] = [];

                // Normalize parameter values for distance calculation
                const tdParamRanges = pKeys.map(k => {
                  const vals = trials.map(t => Number(t.parameters[k]) || 0);
                  const min = Math.min(...vals);
                  const max = Math.max(...vals);
                  return { k, min, range: max - min || 1 };
                });

                for (let i = 0; i <= trials.length - windowSize; i += Math.max(1, Math.floor(windowSize / 2))) {
                  const windowTrials = trials.slice(i, i + windowSize);
                  // Average pairwise distance (normalized)
                  let totalDist = 0, pairs = 0;
                  for (let a = 0; a < windowTrials.length; a++) {
                    for (let b = a + 1; b < windowTrials.length; b++) {
                      let dist = 0;
                      for (const pr of tdParamRanges) {
                        const diff = ((Number(windowTrials[a].parameters[pr.k]) || 0) - (Number(windowTrials[b].parameters[pr.k]) || 0)) / pr.range;
                        dist += diff * diff;
                      }
                      totalDist += Math.sqrt(dist / pKeys.length);
                      pairs++;
                    }
                  }
                  const diversity = pairs > 0 ? totalDist / pairs : 0;
                  const meanObj = windowTrials.reduce((s, t) => s + (Number(t.kpis[tdObjKey]) || 0), 0) / windowTrials.length;
                  tdWindows.push({ start: i, end: i + windowSize - 1, diversity, meanObj });
                }

                if (tdWindows.length < 3) return null;

                const tdW = 440, tdH = 140, tdPadL = 44, tdPadR = 16, tdPadT = 16, tdPadB = 28;
                const tdPlotW = tdW - tdPadL - tdPadR;
                const tdPlotH = tdH - tdPadT - tdPadB;
                const tdDivs = tdWindows.map(w => w.diversity);
                const tdMaxDiv = Math.max(...tdDivs);
                const tdMinDiv = Math.min(...tdDivs);
                const tdDivRange = tdMaxDiv - tdMinDiv || 1;

                const toTdX = (i: number) => tdPadL + (i / (tdWindows.length - 1)) * tdPlotW;
                const toTdY = (d: number) => tdPadT + tdPlotH - ((d - tdMinDiv) / tdDivRange) * tdPlotH;

                const tdLinePath = tdWindows.map((w, i) => `${i === 0 ? "M" : "L"}${toTdX(i).toFixed(1)},${toTdY(w.diversity).toFixed(1)}`).join(" ");
                const tdAreaPath = `${tdLinePath} L${toTdX(tdWindows.length - 1)},${tdPadT + tdPlotH} L${tdPadL},${tdPadT + tdPlotH} Z`;

                // Detect exploration vs exploitation phases
                const tdMidDiv = (tdMaxDiv + tdMinDiv) / 2;
                const explorationPct = (tdWindows.filter(w => w.diversity > tdMidDiv).length / tdWindows.length * 100);

                // Trend: compare first vs last third
                const thirdN = Math.max(1, Math.floor(tdWindows.length / 3));
                const earlyDiv = tdWindows.slice(0, thirdN).reduce((s, w) => s + w.diversity, 0) / thirdN;
                const lateDiv = tdWindows.slice(-thirdN).reduce((s, w) => s + w.diversity, 0) / thirdN;
                const divTrend = lateDiv > earlyDiv * 1.05 ? "expanding" : lateDiv < earlyDiv * 0.95 ? "contracting" : "stable";
                const divTrendColor = divTrend === "expanding" ? "var(--color-blue)" : divTrend === "contracting" ? "rgba(234,179,8,0.8)" : "var(--color-text-muted)";

                return (
                  <div className="card" style={{ marginBottom: 18 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
                      <Layers size={18} style={{ color: "var(--color-text-muted)" }} />
                      <div>
                        <h2 style={{ margin: 0 }}>Trial Diversity Timeline</h2>
                        <p className="stat-label" style={{ margin: 0, textTransform: "none" }}>
                          Pairwise parameter diversity across sliding windows of {windowSize} trials.
                        </p>
                      </div>
                      <span className="findings-badge" style={{ marginLeft: "auto", color: divTrendColor, borderColor: divTrendColor }}>
                        {divTrend === "expanding" ? "↑" : divTrend === "contracting" ? "↓" : "→"} {divTrend}
                      </span>
                    </div>
                    <div style={{ overflowX: "auto" }}>
                      <svg width={tdW} height={tdH} viewBox={`0 0 ${tdW} ${tdH}`} style={{ display: "block", margin: "0 auto" }}>
                        {/* Grid */}
                        {[0, 0.25, 0.5, 0.75, 1].map(f => (
                          <line key={f} x1={tdPadL} y1={tdPadT + (1 - f) * tdPlotH} x2={tdPadL + tdPlotW} y2={tdPadT + (1 - f) * tdPlotH} stroke="var(--color-border)" strokeWidth={0.5} />
                        ))}
                        {/* Area fill */}
                        <path d={tdAreaPath} fill="rgba(99,102,241,0.12)" />
                        {/* Line */}
                        <path d={tdLinePath} fill="none" stroke="rgba(99,102,241,0.7)" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />
                        {/* Midline (exploration threshold) */}
                        <line x1={tdPadL} y1={toTdY(tdMidDiv)} x2={tdPadL + tdPlotW} y2={toTdY(tdMidDiv)} stroke="var(--color-text-muted)" strokeWidth={0.8} strokeDasharray="4,3" opacity={0.4} />
                        <text x={tdPadL + tdPlotW + 2} y={toTdY(tdMidDiv) + 3} fontSize="7" fill="var(--color-text-muted)" fontFamily="var(--font-mono)">mid</text>
                        {/* Window dots */}
                        {tdWindows.map((w, i) => (
                          <circle key={i} cx={toTdX(i)} cy={toTdY(w.diversity)} r={2.5} fill={w.diversity > tdMidDiv ? "rgba(99,102,241,0.8)" : "rgba(234,179,8,0.7)"} />
                        ))}
                        {/* Axis labels */}
                        <text x={tdPadL + tdPlotW / 2} y={tdH - 2} textAnchor="middle" fontSize="10" fill="var(--color-text-muted)" fontFamily="var(--font-mono)">
                          Window
                        </text>
                        <text x={8} y={tdPadT + tdPlotH / 2} textAnchor="middle" fontSize="10" fill="var(--color-text-muted)" fontFamily="var(--font-mono)" transform={`rotate(-90,8,${tdPadT + tdPlotH / 2})`}>
                          Diversity
                        </text>
                        {/* Y-axis ticks */}
                        {[0, 0.5, 1].map(f => (
                          <text key={f} x={tdPadL - 4} y={tdPadT + (1 - f) * tdPlotH + 3} textAnchor="end" fontSize="8" fill="var(--color-text-muted)" fontFamily="var(--font-mono)">
                            {(tdMinDiv + f * tdDivRange).toFixed(2)}
                          </text>
                        ))}
                      </svg>
                    </div>
                    <div style={{ display: "flex", gap: "16px", marginTop: "6px", flexWrap: "wrap", alignItems: "center" }}>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: 10, height: 10, borderRadius: "50%", background: "rgba(99,102,241,0.8)", marginRight: 4, verticalAlign: "middle" }} />Exploring</span>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: 10, height: 10, borderRadius: "50%", background: "rgba(234,179,8,0.7)", marginRight: 4, verticalAlign: "middle" }} />Exploiting</span>
                      <span style={{ marginLeft: "auto", fontSize: "0.78rem", color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}>
                        {explorationPct.toFixed(0)}% exploring windows
                      </span>
                    </div>
                  </div>
                );
              })()}

              {/* Parameter Lock-In Detector */}
              {trials.length >= 15 && (() => {
                const liSpecs = campaign.spec?.parameters || [];
                const liContSpecs = liSpecs.filter((s: { name: string; type: string; lower?: number; upper?: number }) => s.type === "continuous" && s.lower != null && s.upper != null);
                if (liContSpecs.length === 0) return null;

                const liWinSize = Math.max(5, Math.floor(trials.length / 8));
                const liWindows: { start: number; end: number }[] = [];
                for (let i = 0; i <= trials.length - liWinSize; i += Math.max(1, Math.floor(liWinSize / 2))) {
                  liWindows.push({ start: i, end: i + liWinSize - 1 });
                }
                if (liWindows.length < 3) return null;

                // For each parameter, compute range spread in each window (normalized to [0,1])
                const liData = liContSpecs.map((s: { name: string; type: string; lower?: number; upper?: number }) => {
                  const fullRange = (s.upper ?? 1) - (s.lower ?? 0) || 1;
                  const windowSpreads = liWindows.map(w => {
                    const vals = trials.slice(w.start, w.end + 1).map(t => Number(t.parameters[s.name]) || 0);
                    const min = Math.min(...vals);
                    const max = Math.max(...vals);
                    return (max - min) / fullRange;
                  });
                  // Lock-in detected if late spread < 30% of early spread
                  const earlyAvg = windowSpreads.slice(0, Math.ceil(windowSpreads.length / 3)).reduce((a, b) => a + b, 0) / Math.ceil(windowSpreads.length / 3);
                  const lateAvg = windowSpreads.slice(-Math.ceil(windowSpreads.length / 3)).reduce((a, b) => a + b, 0) / Math.ceil(windowSpreads.length / 3);
                  const locked = earlyAvg > 0.05 && lateAvg < earlyAvg * 0.3;
                  return { name: s.name, windowSpreads, earlyAvg, lateAvg, locked };
                });

                const lockedCount = liData.filter(d => d.locked).length;
                const liW = 440, liH = 130, liPadL = 80, liPadR = 16, liPadT = 8, liPadB = 24;
                const liPlotW = liW - liPadL - liPadR;
                const liPlotH = liH - liPadT - liPadB;
                const rowH = liPlotH / liData.length;

                // Color palette for parameters
                const liColors = ["rgba(59,130,246,0.6)", "rgba(34,197,94,0.6)", "rgba(234,179,8,0.6)", "rgba(239,68,68,0.5)", "rgba(168,85,247,0.5)", "rgba(236,72,153,0.5)", "rgba(20,184,166,0.5)", "rgba(249,115,22,0.5)"];

                return (
                  <div className="card" style={{ marginBottom: 18 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
                      <TrendingDown size={18} style={{ color: "var(--color-text-muted)" }} />
                      <div>
                        <h2 style={{ margin: 0 }}>Parameter Lock-In Detector</h2>
                        <p className="stat-label" style={{ margin: 0, textTransform: "none" }}>
                          Sampling range evolution per parameter. Narrowing = potential premature convergence.
                        </p>
                      </div>
                      {lockedCount > 0 && (
                        <span className="findings-badge" style={{ marginLeft: "auto", color: "rgba(239,68,68,0.8)", borderColor: "rgba(239,68,68,0.3)" }}>
                          {lockedCount} locked
                        </span>
                      )}
                    </div>
                    <div style={{ overflowX: "auto" }}>
                      <svg width={liW} height={liH} viewBox={`0 0 ${liW} ${liH}`} style={{ display: "block", margin: "0 auto" }}>
                        {/* Row backgrounds and parameter names */}
                        {liData.map((d, pi) => (
                          <g key={pi}>
                            {pi % 2 === 0 && (
                              <rect x={liPadL} y={liPadT + pi * rowH} width={liPlotW} height={rowH} fill="var(--color-border)" opacity={0.2} />
                            )}
                            <text x={liPadL - 6} y={liPadT + pi * rowH + rowH / 2 + 3} textAnchor="end" fontSize="9" fontFamily="var(--font-mono)" fontWeight={d.locked ? 700 : 400} fill={d.locked ? "rgba(239,68,68,0.8)" : "var(--color-text-muted)"}>
                              {d.name.length > 10 ? d.name.slice(0, 9) + "…" : d.name}
                              {d.locked ? " ⚠" : ""}
                            </text>
                          </g>
                        ))}
                        {/* Stream bands - spread for each parameter across windows */}
                        {liData.map((d, pi) => {
                          const centerY = liPadT + pi * rowH + rowH / 2;
                          const maxSpread = Math.max(...d.windowSpreads, 0.01);
                          const halfH = (rowH * 0.4);
                          const topPath = d.windowSpreads.map((sp, wi) => {
                            const x = liPadL + (wi / (liWindows.length - 1)) * liPlotW;
                            const y = centerY - (sp / maxSpread) * halfH;
                            return `${wi === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
                          }).join(" ");
                          const bottomPath = d.windowSpreads.map((_sp, wi) => {
                            const x = liPadL + ((liWindows.length - 1 - wi) / (liWindows.length - 1)) * liPlotW;
                            const y = centerY + (d.windowSpreads[liWindows.length - 1 - wi] / maxSpread) * halfH;
                            return `L${x.toFixed(1)},${y.toFixed(1)}`;
                          }).join(" ");
                          return (
                            <path key={pi} d={`${topPath} ${bottomPath} Z`} fill={liColors[pi % liColors.length]} opacity={0.7} />
                          );
                        })}
                        {/* X axis */}
                        <text x={liPadL + liPlotW / 2} y={liH - 2} textAnchor="middle" fontSize="9" fill="var(--color-text-muted)" fontFamily="var(--font-mono)">
                          Window →
                        </text>
                      </svg>
                    </div>
                    <div style={{ display: "flex", gap: "16px", marginTop: "6px", flexWrap: "wrap", alignItems: "center" }}>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: 20, height: 8, background: "rgba(59,130,246,0.4)", marginRight: 4, verticalAlign: "middle", borderRadius: 2 }} />Wide spread</span>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: 6, height: 8, background: "rgba(239,68,68,0.5)", marginRight: 4, verticalAlign: "middle", borderRadius: 2 }} />Narrowing</span>
                      <span style={{ marginLeft: "auto", fontSize: "0.72rem", color: "var(--color-text-muted)" }}>
                        ⚠ = locked in (late spread &lt; 30% of early)
                      </span>
                    </div>
                  </div>
                );
              })()}

              {/* Posterior Contraction Timeline */}
              {trials.length >= 15 && (() => {
                const pcObjKey = Object.keys(trials[0].kpis)[0];
                const pcPKeys = Object.keys(trials[0].parameters);
                const pcSpecs = campaign.spec?.parameters || [];
                const pcRanges = pcPKeys.map(k => {
                  const sp = pcSpecs.find((s: { name: string }) => s.name === k);
                  const range = sp && (sp as { upper?: number }).upper != null && (sp as { lower?: number }).lower != null ? ((sp as { upper: number }).upper - (sp as { lower: number }).lower) : 1;
                  return range || 1;
                });
                // At each trial, compute IQR-volume of top-k trials so far
                const pcK = Math.max(5, Math.floor(trials.length / 5));
                const pcWinSize = Math.max(3, Math.floor(trials.length / 20));
                const pcPoints: Array<{ idx: number; volume: number }> = [];
                for (let i = pcK - 1; i < trials.length; i += pcWinSize) {
                  // Get top-k by objective (minimize = lowest values)
                  const subset = trials.slice(0, i + 1).map((t, j) => ({
                    j,
                    val: Number(t.kpis[pcObjKey]) || 0,
                    params: pcPKeys.map((k, d) => (Number(t.parameters[k]) || 0) / pcRanges[d]),
                  })).sort((a, b) => a.val - b.val).slice(0, pcK);
                  // Compute normalized IQR volume: product of IQRs per dimension
                  let logVol = 0;
                  for (let d = 0; d < pcPKeys.length; d++) {
                    const dimVals = subset.map(s => s.params[d]).sort((a, b) => a - b);
                    const q1 = dimVals[Math.floor(dimVals.length * 0.25)];
                    const q3 = dimVals[Math.floor(dimVals.length * 0.75)];
                    const iqr = Math.max(q3 - q1, 0.001);
                    logVol += Math.log(iqr);
                  }
                  pcPoints.push({ idx: i, volume: Math.exp(logVol / pcPKeys.length) }); // Geometric mean of IQRs
                }
                if (pcPoints.length < 2) return null;
                const pcMaxVol = Math.max(...pcPoints.map(p => p.volume));
                const pcMinVol = Math.min(...pcPoints.map(p => p.volume));
                const pcVolRange = pcMaxVol - pcMinVol || 1;
                const pcW = 320, pcH = 120, pcPadL = 40, pcPadR = 20, pcPadT = 12, pcPadB = 24;
                const pcPlotW = pcW - pcPadL - pcPadR;
                const pcPlotH = pcH - pcPadT - pcPadB;
                const pcMaxIdx = trials.length - 1;
                const pcPathPts = pcPoints.map(p => ({
                  x: pcPadL + (p.idx / pcMaxIdx) * pcPlotW,
                  y: pcPadT + (1 - (p.volume - pcMinVol) / pcVolRange) * pcPlotH,
                }));
                const pcAreaPath = `M${pcPathPts.map(p => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" L")} L${pcPathPts[pcPathPts.length - 1].x.toFixed(1)},${pcPadT + pcPlotH} L${pcPathPts[0].x.toFixed(1)},${pcPadT + pcPlotH} Z`;
                const pcLinePath = pcPathPts.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
                // Trend: compare first third vs last third
                const pcThird = Math.max(1, Math.floor(pcPoints.length / 3));
                const pcEarlyAvg = pcPoints.slice(0, pcThird).reduce((s, p) => s + p.volume, 0) / pcThird;
                const pcLateAvg = pcPoints.slice(-pcThird).reduce((s, p) => s + p.volume, 0) / pcThird;
                const pcContraction = pcEarlyAvg > 0 ? ((pcEarlyAvg - pcLateAvg) / pcEarlyAvg) * 100 : 0;
                const pcTrend = pcContraction > 20 ? "Converging" : pcContraction > 5 ? "Narrowing" : pcContraction > -5 ? "Stable" : "Diverging";
                const pcTrendColor = pcContraction > 20 ? "#22c55e" : pcContraction > 5 ? "#3b82f6" : pcContraction > -5 ? "#eab308" : "#ef4444";
                return (
                  <div className="card">
                    <div style={{ display: "flex", alignItems: "flex-start", gap: "10px", marginBottom: "8px" }}>
                      <Minimize2 size={16} style={{ color: "var(--color-primary)", marginTop: 2 }} />
                      <div style={{ flex: 1 }}>
                        <h2 style={{ margin: 0 }}>Posterior Contraction</h2>
                        <p style={{ margin: "2px 0 0", fontSize: "0.78rem", color: "var(--color-text-muted)" }}>
                          Search volume of top-{pcK} trials over time. Shrinking = spatial convergence.
                        </p>
                      </div>
                      <span className="findings-badge" style={{ background: pcTrendColor + "22", color: pcTrendColor, border: `1px solid ${pcTrendColor}44` }}>
                        {pcTrend} ({pcContraction > 0 ? "-" : "+"}{Math.abs(pcContraction).toFixed(0)}%)
                      </span>
                    </div>
                    <svg width={pcW} height={pcH} viewBox={`0 0 ${pcW} ${pcH}`} role="img" aria-label="Posterior contraction timeline" style={{ display: "block" }}>
                      {/* Grid */}
                      {[0, 0.5, 1].map(f => (
                        <g key={f}>
                          <line x1={pcPadL} y1={pcPadT + f * pcPlotH} x2={pcPadL + pcPlotW} y2={pcPadT + f * pcPlotH} stroke="var(--color-border)" strokeWidth={0.5} />
                          <text x={pcPadL - 4} y={pcPadT + f * pcPlotH + 3} textAnchor="end" fontSize="8" fill="var(--color-text-muted)" fontFamily="var(--font-mono)">
                            {(pcMaxVol - f * pcVolRange).toPrecision(2)}
                          </text>
                        </g>
                      ))}
                      {/* Area fill */}
                      <path d={pcAreaPath} fill={pcTrendColor} opacity={0.15} />
                      {/* Line */}
                      <path d={pcLinePath} fill="none" stroke={pcTrendColor} strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />
                      {/* Dots */}
                      {pcPathPts.map((p, i) => (
                        <circle key={i} cx={p.x} cy={p.y} r={2.5} fill={pcTrendColor}>
                          <title>Trial {pcPoints[i].idx}: volume {pcPoints[i].volume.toPrecision(3)}</title>
                        </circle>
                      ))}
                      {/* X axis */}
                      <text x={pcPadL + pcPlotW / 2} y={pcH - 2} textAnchor="middle" fontSize="9" fill="var(--color-text-muted)" fontFamily="var(--font-mono)">Trial</text>
                      <text x={pcPadL} y={pcH - 2} textAnchor="middle" fontSize="8" fill="var(--color-text-muted)" fontFamily="var(--font-mono)">0</text>
                      <text x={pcPadL + pcPlotW} y={pcH - 2} textAnchor="middle" fontSize="8" fill="var(--color-text-muted)" fontFamily="var(--font-mono)">{pcMaxIdx}</text>
                    </svg>
                    <div style={{ display: "flex", gap: "16px", marginTop: "4px", flexWrap: "wrap", alignItems: "center" }}>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: 10, height: 10, borderRadius: "50%", background: pcTrendColor, marginRight: 4, verticalAlign: "middle" }} />IQR Volume (top-{pcK})</span>
                      <span style={{ marginLeft: "auto", fontSize: "0.72rem", color: "var(--color-text-muted)" }}>Shrinking area = convergence in parameter space</span>
                    </div>
                  </div>
                );
              })()}

              {/* Search Coverage Progression */}
              {trials.length >= 5 && (() => {
                const scSpecs = campaign.spec?.parameters?.filter((s: { name: string; type: string; lower?: number; upper?: number }) => s.type === "continuous" && s.lower != null && s.upper != null) || [];
                if (scSpecs.length === 0) return null;
                const scSorted = [...trials].sort((a, b) => a.iteration - b.iteration);
                const scG = 20; // grid cells per dimension
                const scN = scSorted.length;
                const scStep = Math.max(1, Math.floor(scN / 20));
                const scCheckpoints: { iter: number; coverage: number }[] = [];
                // Track which cells are covered per dimension
                const scCovered: Set<number>[] = scSpecs.map(() => new Set<number>());
                for (let i = 0; i < scN; i++) {
                  const t = scSorted[i];
                  for (let d = 0; d < scSpecs.length; d++) {
                    const sp = scSpecs[d] as { name: string; lower?: number; upper?: number };
                    const lo = sp.lower ?? 0, hi = sp.upper ?? 1;
                    const norm = hi > lo ? (t.parameters[sp.name] - lo) / (hi - lo) : 0.5;
                    const cell = Math.min(scG - 1, Math.max(0, Math.floor(norm * scG)));
                    scCovered[d].add(cell);
                  }
                  if ((i + 1) % scStep === 0 || i === scN - 1) {
                    const avgCov = scCovered.reduce((a, s) => a + s.size / scG, 0) / scSpecs.length;
                    scCheckpoints.push({ iter: t.iteration, coverage: avgCov });
                  }
                }
                if (scCheckpoints.length < 2) return null;
                const scCurrent = scCheckpoints[scCheckpoints.length - 1].coverage;
                // Stall detection: last 5 checkpoints delta < 0.01
                const scLastN = scCheckpoints.slice(-5);
                const scStalled = scLastN.length >= 5 && Math.abs(scLastN[scLastN.length - 1].coverage - scLastN[0].coverage) < 0.01;
                const scW = 200, scH = 70, scPadX = 20, scPadY = 12;
                const scChartW = scW - 2 * scPadX, scChartH = scH - 2 * scPadY;
                const scPts = scCheckpoints.map((c, i) => ({
                  x: scPadX + (i / (scCheckpoints.length - 1)) * scChartW,
                  y: scH - scPadY - c.coverage * scChartH,
                }));
                const scAreaPath = `M${scPts[0].x},${scH - scPadY} ` + scPts.map(p => `L${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ") + ` L${scPts[scPts.length - 1].x},${scH - scPadY} Z`;
                const scLinePath = scPts.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
                const scTargetY = scH - scPadY - 0.8 * scChartH;
                const scCovColor = scCurrent >= 0.7 ? "#22c55e" : scCurrent >= 0.4 ? "#f59e0b" : "#ef4444";
                return (
                  <div className="card" style={{ padding: "14px 16px" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                      <MapPin size={16} style={{ color: "var(--color-text-muted)" }} />
                      <h3 style={{ margin: 0, fontSize: "0.92rem" }}>Search Coverage</h3>
                      <span className="findings-badge" style={{ background: scCovColor + "18", color: scCovColor, marginLeft: "auto" }}>
                        {(scCurrent * 100).toFixed(0)}% covered{scStalled ? " (stalled)" : ""}
                      </span>
                    </div>
                    <svg width="100%" viewBox={`0 0 ${scW} ${scH}`} style={{ display: "block" }}>
                      <defs>
                        <linearGradient id="sc-grad" x1="0" y1="1" x2="0" y2="0">
                          <stop offset="0%" stopColor={scCovColor} stopOpacity="0.02" />
                          <stop offset="100%" stopColor={scCovColor} stopOpacity="0.25" />
                        </linearGradient>
                      </defs>
                      {/* Grid */}
                      {[0.25, 0.5, 0.75].map(t => (
                        <line key={`scg${t}`} x1={scPadX} y1={scH - scPadY - t * scChartH} x2={scW - scPadX} y2={scH - scPadY - t * scChartH} stroke="var(--color-border)" strokeWidth="0.5" />
                      ))}
                      {/* Target line at 80% */}
                      <line x1={scPadX} y1={scTargetY} x2={scW - scPadX} y2={scTargetY} stroke="#22c55e" strokeWidth="0.8" strokeDasharray="3 2" opacity={0.5} />
                      <text x={scW - scPadX + 2} y={scTargetY + 3} fontSize="5.5" fill="#22c55e" opacity={0.7}>80%</text>
                      {/* Area fill */}
                      <path d={scAreaPath} fill="url(#sc-grad)" />
                      {/* Line */}
                      <path d={scLinePath} fill="none" stroke={scCovColor} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
                      {/* Endpoint */}
                      <circle cx={scPts[scPts.length - 1].x} cy={scPts[scPts.length - 1].y} r="3" fill={scCovColor} />
                      {/* Current value label */}
                      <text x={scPts[scPts.length - 1].x + 2} y={scPts[scPts.length - 1].y - 4} fontSize="7" fill={scCovColor} fontWeight="600" fontFamily="var(--font-mono)">{(scCurrent * 100).toFixed(0)}%</text>
                      {/* Axes labels */}
                      <text x={scPadX} y={scH - 1} fontSize="5.5" fill="var(--color-text-muted)">iter {scCheckpoints[0].iter}</text>
                      <text x={scW - scPadX} y={scH - 1} fontSize="5.5" fill="var(--color-text-muted)" textAnchor="end">iter {scCheckpoints[scCheckpoints.length - 1].iter}</text>
                      <text x={scPadX - 3} y={scH - scPadY + 2} fontSize="5" fill="var(--color-text-muted)" textAnchor="end">0</text>
                      <text x={scPadX - 3} y={scPadY + 3} fontSize="5" fill="var(--color-text-muted)" textAnchor="end">1</text>
                    </svg>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.72rem", color: "var(--color-text-muted)", marginTop: 2 }}>
                      <span>{scSpecs.length} dims × {scG} grid cells</span>
                      {scStalled && <span style={{ color: "#f59e0b", fontWeight: 500 }}>Coverage stalled — consider forcing exploration</span>}
                    </div>
                  </div>
                );
              })()}

              {/* Learning Velocity Timeline */}
              {trials.length >= 12 && (() => {
                const lvSorted = [...trials].sort((a, b) => a.iteration - b.iteration);
                const lvKpi = Object.keys(lvSorted[0]?.kpis || {})[0];
                if (!lvKpi) return null;
                const lvN = lvSorted.length;
                const lvWin = Math.max(3, Math.floor(lvN / 12));

                // Compute rolling best and rolling variance for windows
                const lvWindows: { iter: number; improvement: number; varReduction: number }[] = [];
                let lvRunBest = Number(lvSorted[0].kpis[lvKpi]) || 0;
                // Determine if minimizing (assume minimize if first values are more negative)
                const lvFirstAvg = lvSorted.slice(0, 5).reduce((s, t) => s + (Number(t.kpis[lvKpi]) || 0), 0) / 5;
                const lvLastAvg = lvSorted.slice(-5).reduce((s, t) => s + (Number(t.kpis[lvKpi]) || 0), 0) / 5;
                const lvMinimize = lvLastAvg <= lvFirstAvg;

                for (let i = lvWin; i <= lvN; i += Math.max(1, Math.floor(lvWin / 2))) {
                  const winSlice = lvSorted.slice(i - lvWin, i);
                  const winKpis = winSlice.map(t => Number(t.kpis[lvKpi]) || 0);

                  // Improvement: how much best improved in this window
                  const winBest = lvMinimize ? Math.min(...winKpis) : Math.max(...winKpis);
                  const prevBest = lvRunBest;
                  if (lvMinimize ? winBest < lvRunBest : winBest > lvRunBest) {
                    lvRunBest = winBest;
                  }
                  const improvement = Math.abs(lvRunBest - prevBest);

                  // Variance reduction: compare window variance to previous window
                  const winMean = winKpis.reduce((a, b) => a + b, 0) / winKpis.length;
                  const winVar = winKpis.reduce((s, v) => s + (v - winMean) ** 2, 0) / winKpis.length;
                  const prevSlice = lvSorted.slice(Math.max(0, i - 2 * lvWin), i - lvWin);
                  let varReduction = 0;
                  if (prevSlice.length > 2) {
                    const prevKpis = prevSlice.map(t => Number(t.kpis[lvKpi]) || 0);
                    const prevMean = prevKpis.reduce((a, b) => a + b, 0) / prevKpis.length;
                    const prevVar = prevKpis.reduce((s, v) => s + (v - prevMean) ** 2, 0) / prevKpis.length;
                    varReduction = prevVar > 0 ? Math.max(0, (prevVar - winVar) / prevVar) : 0;
                  }

                  lvWindows.push({ iter: winSlice[winSlice.length - 1].iteration, improvement, varReduction });
                }

                if (lvWindows.length < 3) return null;

                const lvMaxImp = Math.max(...lvWindows.map(w => w.improvement), 0.0001);
                const lvW = 440, lvH = 110, lvPadL = 46, lvPadR = 46, lvPadT = 10, lvPadB = 24;
                const lvPlotW = lvW - lvPadL - lvPadR;
                const lvPlotH = lvH - lvPadT - lvPadB;

                // Phase detection
                const lvRecentImps = lvWindows.slice(-3).map(w => w.improvement);
                const lvRecentAvg = lvRecentImps.reduce((a, b) => a + b, 0) / lvRecentImps.length;
                const lvOverallAvg = lvWindows.map(w => w.improvement).reduce((a, b) => a + b, 0) / lvWindows.length;
                const lvPhase = lvRecentAvg > lvOverallAvg * 1.5 ? "Accelerating" : lvRecentAvg > lvOverallAvg * 0.3 ? "Steady" : "Plateau";
                const lvPhaseColor = lvPhase === "Accelerating" ? "#22c55e" : lvPhase === "Steady" ? "#3b82f6" : "#f59e0b";

                // Plot points
                const lvImpPts = lvWindows.map((w, i) => ({
                  x: lvPadL + (i / (lvWindows.length - 1)) * lvPlotW,
                  y: lvPadT + (1 - w.improvement / lvMaxImp) * lvPlotH,
                }));
                const lvVarPts = lvWindows.map((w, i) => ({
                  x: lvPadL + (i / (lvWindows.length - 1)) * lvPlotW,
                  y: lvPadT + (1 - w.varReduction) * lvPlotH,
                }));

                const lvImpPath = lvImpPts.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
                const lvVarPath = lvVarPts.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");

                // Background phase regions
                const lvThird = Math.floor(lvWindows.length / 3);

                return (
                  <div className="card" style={{ marginBottom: 18 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
                      <Thermometer size={18} style={{ color: "var(--color-text-muted)" }} />
                      <div>
                        <h2 style={{ margin: 0 }}>Learning Velocity</h2>
                        <p className="stat-label" style={{ margin: 0, textTransform: "none" }}>
                          Rate of improvement and variance reduction per trial window.
                        </p>
                      </div>
                      <span className="findings-badge" style={{ marginLeft: "auto", color: lvPhaseColor, borderColor: lvPhaseColor + "40" }}>
                        {lvPhase}
                      </span>
                    </div>
                    <div style={{ overflowX: "auto" }}>
                      <svg width={lvW} height={lvH} viewBox={`0 0 ${lvW} ${lvH}`} style={{ display: "block", margin: "0 auto" }}>
                        {/* Phase background regions */}
                        {lvThird > 0 && (
                          <>
                            <rect x={lvPadL} y={lvPadT} width={(lvThird / (lvWindows.length - 1)) * lvPlotW} height={lvPlotH} fill="#22c55e" opacity={0.04} />
                            <rect x={lvPadL + (lvThird / (lvWindows.length - 1)) * lvPlotW} y={lvPadT} width={((lvThird) / (lvWindows.length - 1)) * lvPlotW} height={lvPlotH} fill="#3b82f6" opacity={0.04} />
                            <rect x={lvPadL + (2 * lvThird / (lvWindows.length - 1)) * lvPlotW} y={lvPadT} width={((lvWindows.length - 1 - 2 * lvThird) / (lvWindows.length - 1)) * lvPlotW} height={lvPlotH} fill="#f59e0b" opacity={0.04} />
                          </>
                        )}
                        {/* Grid lines */}
                        {[0.25, 0.5, 0.75].map(f => (
                          <line key={f} x1={lvPadL} y1={lvPadT + (1 - f) * lvPlotH} x2={lvW - lvPadR} y2={lvPadT + (1 - f) * lvPlotH} stroke="var(--color-border)" strokeWidth="0.5" />
                        ))}
                        {/* Zero line */}
                        <line x1={lvPadL} y1={lvPadT + lvPlotH} x2={lvW - lvPadR} y2={lvPadT + lvPlotH} stroke="var(--color-border)" strokeWidth="0.8" />
                        {/* Improvement line */}
                        <path d={lvImpPath} fill="none" stroke="#3b82f6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                        {lvImpPts.map((p, i) => (
                          <circle key={`imp${i}`} cx={p.x} cy={p.y} r="2.5" fill="#3b82f6" />
                        ))}
                        {/* Variance reduction line (dashed) */}
                        <path d={lvVarPath} fill="none" stroke="#8b5cf6" strokeWidth="1.5" strokeDasharray="4 2" strokeLinecap="round" />
                        {/* Y axis labels */}
                        <text x={lvPadL - 4} y={lvPadT + 3} textAnchor="end" fontSize="6.5" fill="var(--color-text-muted)" fontFamily="var(--font-mono)">max</text>
                        <text x={lvPadL - 4} y={lvPadT + lvPlotH + 3} textAnchor="end" fontSize="6.5" fill="var(--color-text-muted)" fontFamily="var(--font-mono)">0</text>
                        {/* Right Y axis labels */}
                        <text x={lvW - lvPadR + 4} y={lvPadT + 3} fontSize="6.5" fill="#8b5cf6" fontFamily="var(--font-mono)">100%</text>
                        <text x={lvW - lvPadR + 4} y={lvPadT + lvPlotH + 3} fontSize="6.5" fill="#8b5cf6" fontFamily="var(--font-mono)">0%</text>
                        {/* X axis */}
                        <text x={lvPadL} y={lvH - 2} fontSize="7" fill="var(--color-text-muted)" fontFamily="var(--font-mono)">iter {lvWindows[0].iter}</text>
                        <text x={lvW - lvPadR} y={lvH - 2} fontSize="7" fill="var(--color-text-muted)" fontFamily="var(--font-mono)" textAnchor="end">iter {lvWindows[lvWindows.length - 1].iter}</text>
                        <text x={lvPadL + lvPlotW / 2} y={lvH - 2} fontSize="7" fill="var(--color-text-muted)" textAnchor="middle">Window →</text>
                        {/* Peak annotation */}
                        {(() => {
                          const peakIdx = lvWindows.reduce((best, w, i) => w.improvement > lvWindows[best].improvement ? i : best, 0);
                          if (lvWindows[peakIdx].improvement > 0) {
                            const px = lvImpPts[peakIdx].x;
                            const py = lvImpPts[peakIdx].y;
                            return (
                              <text x={px} y={py - 6} textAnchor="middle" fontSize="6.5" fill="#3b82f6" fontWeight="600">
                                Peak t={lvWindows[peakIdx].iter}
                              </text>
                            );
                          }
                          return null;
                        })()}
                      </svg>
                    </div>
                    <div style={{ display: "flex", gap: "16px", marginTop: "4px", flexWrap: "wrap", alignItems: "center" }}>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: 14, height: 3, background: "#3b82f6", marginRight: 4, verticalAlign: "middle", borderRadius: 1 }} />Improvement rate</span>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: 14, height: 0, borderTop: "2px dashed #8b5cf6", marginRight: 4, verticalAlign: "middle" }} />Var reduction</span>
                      <span style={{ marginLeft: "auto", fontSize: "0.72rem", color: "var(--color-text-muted)" }}>
                        {lvWin}-trial windows
                      </span>
                    </div>
                  </div>
                );
              })()}

              {/* Objective Distribution Evolution */}
              {trials.length >= 12 && (() => {
                const deSorted = [...trials].sort((a, b) => a.iteration - b.iteration);
                const deKpiKey = Object.keys(deSorted[0]?.kpis || {})[0];
                if (!deKpiKey) return null;
                const deN = deSorted.length;
                const deNWindows = Math.min(8, Math.floor(deN / 3));
                if (deNWindows < 3) return null;
                const deWinSize = Math.floor(deN / deNWindows);

                // Compute quartile stats for each window
                const deWindows: { iter: number; q0: number; q25: number; q50: number; q75: number; q100: number }[] = [];
                for (let w = 0; w < deNWindows; w++) {
                  const start = w * deWinSize;
                  const end = w === deNWindows - 1 ? deN : (w + 1) * deWinSize;
                  const vals = deSorted.slice(start, end).map(t => t.kpis[deKpiKey] ?? 0).sort((a, b) => a - b);
                  const qAt = (p: number) => {
                    const idx = p * (vals.length - 1);
                    const lo = Math.floor(idx), hi = Math.ceil(idx);
                    return lo === hi ? vals[lo] : vals[lo] + (vals[hi] - vals[lo]) * (idx - lo);
                  };
                  deWindows.push({
                    iter: Math.round((start + end) / 2),
                    q0: vals[0],
                    q25: qAt(0.25),
                    q50: qAt(0.5),
                    q75: qAt(0.75),
                    q100: vals[vals.length - 1],
                  });
                }

                // Global y range
                const deYMin = Math.min(...deWindows.map(w => w.q0));
                const deYMax = Math.max(...deWindows.map(w => w.q100));
                const deYRange = deYMax - deYMin || 1;

                // Detect shift
                const deFirst = deWindows[0];
                const deLast = deWindows[deWindows.length - 1];
                const deMedianShift = deLast.q50 - deFirst.q50;
                const deIQRFirst = deFirst.q75 - deFirst.q25 || 0.001;
                const deIQRLast = deLast.q75 - deLast.q25 || 0.001;
                const deNarrowing = deIQRLast < deIQRFirst * 0.7;
                const deShiftLabel = deNarrowing ? "Converging" : Math.abs(deMedianShift) / deIQRFirst > 0.5 ? "Shifting" : "Stable";
                const deShiftColor = deNarrowing ? "#22c55e" : Math.abs(deMedianShift) / deIQRFirst > 0.5 ? "#3b82f6" : "var(--color-text-muted)";

                // SVG layout
                const deW = 320, deH = 140, dePadL = 40, dePadR = 10, dePadT = 10, dePadB = 24;
                const dePlotW = deW - dePadL - dePadR;
                const dePlotH = deH - dePadT - dePadB;
                const deXScale = (i: number) => dePadL + (i / (deNWindows - 1)) * dePlotW;
                const deYScale = (v: number) => dePadT + dePlotH - ((v - deYMin) / deYRange) * dePlotH;
                const deBandW = Math.max(6, dePlotW / deNWindows * 0.6);

                return (
                  <div className="card" style={{ marginBottom: 18 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
                      <BarChart2 size={18} style={{ color: "var(--color-text-muted)" }} />
                      <div>
                        <h2 style={{ margin: 0 }}>Objective Distribution Evolution</h2>
                        <div style={{ fontSize: "0.78rem", color: "var(--color-text-muted)" }}>Quartile shift across iteration windows</div>
                      </div>
                      <span className="findings-badge" style={{ marginLeft: "auto", background: deShiftColor + "18", color: deShiftColor }}>
                        {deShiftLabel}
                      </span>
                    </div>
                    <svg width={deW} height={deH} viewBox={`0 0 ${deW} ${deH}`} style={{ width: "100%", height: "auto" }}>
                      {/* Y axis labels */}
                      {[0, 0.25, 0.5, 0.75, 1].map(f => {
                        const v = deYMin + f * deYRange;
                        return (
                          <Fragment key={`dey${f}`}>
                            <line x1={dePadL} y1={deYScale(v)} x2={dePadL + dePlotW} y2={deYScale(v)} stroke="var(--color-border)" strokeWidth="0.5" strokeDasharray="3,3" />
                            <text x={dePadL - 4} y={deYScale(v) + 3} fontSize="6" fill="var(--color-text-muted)" textAnchor="end">{v.toPrecision(3)}</text>
                          </Fragment>
                        );
                      })}
                      {/* Box plots for each window */}
                      {deWindows.map((w, i) => {
                        const x = deXScale(i);
                        const half = deBandW / 2;
                        return (
                          <g key={`dew${i}`}>
                            {/* Whisker: q0 to q25 */}
                            <line x1={x} y1={deYScale(w.q0)} x2={x} y2={deYScale(w.q25)} stroke="var(--color-text-muted)" strokeWidth="0.8" />
                            {/* Whisker: q75 to q100 */}
                            <line x1={x} y1={deYScale(w.q75)} x2={x} y2={deYScale(w.q100)} stroke="var(--color-text-muted)" strokeWidth="0.8" />
                            {/* Whisker caps */}
                            <line x1={x - half * 0.5} y1={deYScale(w.q0)} x2={x + half * 0.5} y2={deYScale(w.q0)} stroke="var(--color-text-muted)" strokeWidth="0.8" />
                            <line x1={x - half * 0.5} y1={deYScale(w.q100)} x2={x + half * 0.5} y2={deYScale(w.q100)} stroke="var(--color-text-muted)" strokeWidth="0.8" />
                            {/* IQR box */}
                            <rect
                              x={x - half}
                              y={deYScale(w.q75)}
                              width={deBandW}
                              height={Math.max(1, deYScale(w.q25) - deYScale(w.q75))}
                              rx={2}
                              fill={i < deNWindows / 3 ? "rgba(59,130,246,0.15)" : i < 2 * deNWindows / 3 ? "rgba(168,85,247,0.15)" : "rgba(34,197,94,0.15)"}
                              stroke={i < deNWindows / 3 ? "#3b82f6" : i < 2 * deNWindows / 3 ? "#a855f7" : "#22c55e"}
                              strokeWidth="1"
                            />
                            {/* Median line */}
                            <line
                              x1={x - half}
                              y1={deYScale(w.q50)}
                              x2={x + half}
                              y2={deYScale(w.q50)}
                              stroke={i < deNWindows / 3 ? "#3b82f6" : i < 2 * deNWindows / 3 ? "#a855f7" : "#22c55e"}
                              strokeWidth="2"
                            />
                            {/* Window label */}
                            <text x={x} y={deH - dePadB + 12} fontSize="6" fill="var(--color-text-muted)" textAnchor="middle">t{w.iter}</text>
                          </g>
                        );
                      })}
                      {/* Median trend line */}
                      <polyline
                        points={deWindows.map((w, i) => `${deXScale(i)},${deYScale(w.q50)}`).join(" ")}
                        fill="none"
                        stroke="var(--color-primary)"
                        strokeWidth="1"
                        strokeDasharray="4,3"
                        opacity="0.5"
                      />
                    </svg>
                    <div style={{ display: "flex", gap: "16px", marginTop: 4, flexWrap: "wrap" }}>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: 14, height: 8, background: "rgba(59,130,246,0.3)", border: "1px solid #3b82f6", marginRight: 4, verticalAlign: "middle", borderRadius: 2 }} />Early</span>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: 14, height: 8, background: "rgba(168,85,247,0.3)", border: "1px solid #a855f7", marginRight: 4, verticalAlign: "middle", borderRadius: 2 }} />Mid</span>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: 14, height: 8, background: "rgba(34,197,94,0.3)", border: "1px solid #22c55e", marginRight: 4, verticalAlign: "middle", borderRadius: 2 }} />Late</span>
                      <span className="efficiency-legend-item" style={{ marginLeft: "auto" }}>IQR: {deIQRFirst.toPrecision(3)} → {deIQRLast.toPrecision(3)}</span>
                    </div>
                  </div>
                );
              })()}

              {/* Hypothesis Rejection Timeline */}
              {trials.length >= 10 && (() => {
                const hrSpecs = campaign.spec?.parameters?.filter((s: { name: string; type: string; lower?: number; upper?: number }) => s.type === "continuous" && s.lower != null && s.upper != null) || [];
                if (hrSpecs.length === 0) return null;
                const hrSorted = [...trials].sort((a, b) => a.iteration - b.iteration);
                const hrN = hrSorted.length;

                // --- Hypothesis 1: Monotonicity per parameter (running Spearman rank correlation) ---
                const hrMonoHyps = hrSpecs.slice(0, 4).map((s: { name: string; type: string; lower?: number; upper?: number }) => {
                  const kpiKey = Object.keys(hrSorted[0]?.kpis || {})[0];
                  if (!kpiKey) return null;
                  // Compute Spearman correlation in expanding windows
                  const hrSnapshots: { iter: number; rho: number; pReject: boolean }[] = [];
                  const step = Math.max(1, Math.floor(hrN / 16));
                  for (let w = 8; w <= hrN; w += step) {
                    const slice = hrSorted.slice(0, w);
                    const xs = slice.map(t => Number(t.parameters[s.name]) || 0);
                    const ys = slice.map(t => Number(t.kpis[kpiKey]) || 0);
                    // Rank arrays
                    const rank = (arr: number[]) => {
                      const sorted = [...arr].map((v, i) => ({ v, i })).sort((a, b) => a.v - b.v);
                      const ranks = new Array(arr.length);
                      sorted.forEach((el, r) => { ranks[el.i] = r + 1; });
                      return ranks;
                    };
                    const rx = rank(xs);
                    const ry = rank(ys);
                    const n = rx.length;
                    const d2 = rx.reduce((sum, rxi, i) => sum + (rxi - ry[i]) ** 2, 0);
                    const rho = 1 - (6 * d2) / (n * (n * n - 1));
                    // |rho| < 0.3 with enough data => reject monotonicity
                    const pReject = n >= 10 && Math.abs(rho) < 0.3;
                    hrSnapshots.push({ iter: slice[slice.length - 1].iteration, rho, pReject });
                  }
                  return {
                    label: `Mono(${s.name.length > 6 ? s.name.slice(0, 5) + "…" : s.name})`,
                    snapshots: hrSnapshots,
                    type: "monotonicity" as const,
                  };
                }).filter(Boolean) as Array<{ label: string; snapshots: Array<{ iter: number; rho: number; pReject: boolean }>; type: string }>;

                // --- Hypothesis 2: Optimum in center vs edge ---
                const hrOptHyp = (() => {
                  const kpiKey = Object.keys(hrSorted[0]?.kpis || {})[0];
                  if (!kpiKey || hrSpecs.length < 1) return null;
                  const hrSnapCenter: { iter: number; rho: number; pReject: boolean }[] = [];
                  const step = Math.max(1, Math.floor(hrN / 16));
                  for (let w = 8; w <= hrN; w += step) {
                    const slice = hrSorted.slice(0, w);
                    // Check if best trials are in center 40% of parameter space
                    const kpiVals = slice.map(t => Number(t.kpis[kpiKey]) || 0);
                    const kpiMin = Math.min(...kpiVals);
                    const kpiMax = Math.max(...kpiVals);
                    const kpiThresh = kpiMin + (kpiMax - kpiMin) * 0.7;
                    const topTrials = slice.filter(t => (Number(t.kpis[kpiKey]) || 0) >= kpiThresh);
                    const centerCount = topTrials.filter(t => {
                      return hrSpecs.every((sp: { name: string; lower?: number; upper?: number }) => {
                        const lo = sp.lower ?? 0, hi = sp.upper ?? 1;
                        const norm = hi > lo ? (Number(t.parameters[sp.name]) - lo) / (hi - lo) : 0.5;
                        return norm >= 0.3 && norm <= 0.7;
                      });
                    }).length;
                    const centerFrac = topTrials.length > 0 ? centerCount / topTrials.length : 0.5;
                    // If most top trials are NOT centered => reject center hypothesis
                    const pReject = topTrials.length >= 3 && centerFrac < 0.3;
                    hrSnapCenter.push({ iter: slice[slice.length - 1].iteration, rho: centerFrac, pReject });
                  }
                  return {
                    label: "Opt@Center",
                    snapshots: hrSnapCenter,
                    type: "center" as const,
                  };
                })();

                // --- Hypothesis 3: No interaction effects (independence test) ---
                const hrInterHyp = (() => {
                  const kpiKey = Object.keys(hrSorted[0]?.kpis || {})[0];
                  if (!kpiKey || hrSpecs.length < 2) return null;
                  const s0 = hrSpecs[0] as { name: string; lower?: number; upper?: number };
                  const s1 = hrSpecs[1] as { name: string; lower?: number; upper?: number };
                  const hrSnapInter: { iter: number; rho: number; pReject: boolean }[] = [];
                  const step = Math.max(1, Math.floor(hrN / 16));
                  for (let w = 10; w <= hrN; w += step) {
                    const slice = hrSorted.slice(0, w);
                    // Compute residual-product correlation as interaction proxy
                    const xs = slice.map(t => Number(t.parameters[s0.name]) || 0);
                    const ys = slice.map(t => Number(t.parameters[s1.name]) || 0);
                    const zs = slice.map(t => Number(t.kpis[kpiKey]) || 0);
                    const mean = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;
                    const mx = mean(xs), my = mean(ys), mz = mean(zs);
                    // Product term correlation with kpi
                    const prods = xs.map((x, i) => (x - mx) * (ys[i] - my));
                    const mp = mean(prods);
                    const covPZ = prods.reduce((s, p, i) => s + (p - mp) * (zs[i] - mz), 0) / slice.length;
                    const stdP = Math.sqrt(prods.reduce((s, p) => s + (p - mp) ** 2, 0) / slice.length) || 1;
                    const stdZ = Math.sqrt(zs.reduce((s, z) => s + (z - mz) ** 2, 0) / slice.length) || 1;
                    const interCorr = covPZ / (stdP * stdZ);
                    // Reject independence (meaning interaction detected) if |corr| > 0.4
                    const pReject = slice.length >= 10 && Math.abs(interCorr) > 0.4;
                    hrSnapInter.push({ iter: slice[slice.length - 1].iteration, rho: interCorr, pReject });
                  }
                  return {
                    label: `Indep(${s0.name.slice(0, 3)}×${s1.name.slice(0, 3)})`,
                    snapshots: hrSnapInter,
                    type: "interaction" as const,
                  };
                })();

                const hrAllHyps = [...hrMonoHyps, ...(hrOptHyp ? [hrOptHyp] : []), ...(hrInterHyp ? [hrInterHyp] : [])];
                if (hrAllHyps.length === 0) return null;

                const hrRejected = hrAllHyps.filter(h => h.snapshots.length > 0 && h.snapshots[h.snapshots.length - 1].pReject).length;
                const hrW = 440, hrH = 24 * hrAllHyps.length + 32, hrPadL = 100, hrPadR = 16, hrPadT = 6, hrPadB = 22;
                const hrPlotW = hrW - hrPadL - hrPadR;
                const hrRowH = (hrH - hrPadT - hrPadB) / hrAllHyps.length;

                // Find global iteration range across all hypotheses
                const hrAllIters = hrAllHyps.flatMap(h => h.snapshots.map(s => s.iter));
                const hrMinIter = Math.min(...hrAllIters);
                const hrMaxIter = Math.max(...hrAllIters);
                const hrIterRange = hrMaxIter - hrMinIter || 1;

                return (
                  <div className="card" style={{ marginBottom: 18 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
                      <FlaskRound size={18} style={{ color: "var(--color-text-muted)" }} />
                      <div>
                        <h2 style={{ margin: 0 }}>Hypothesis Rejection Timeline</h2>
                        <p className="stat-label" style={{ margin: 0, textTransform: "none" }}>
                          Structural hypotheses tested via rolling statistics. Red = rejected.
                        </p>
                      </div>
                      <span className="findings-badge" style={{ marginLeft: "auto", color: hrRejected > 0 ? "rgba(239,68,68,0.8)" : "#22c55e", borderColor: hrRejected > 0 ? "rgba(239,68,68,0.3)" : "rgba(34,197,94,0.3)" }}>
                        {hrRejected} rejected / {hrAllHyps.length} tested
                      </span>
                    </div>
                    <div style={{ overflowX: "auto" }}>
                      <svg width={hrW} height={hrH} viewBox={`0 0 ${hrW} ${hrH}`} style={{ display: "block", margin: "0 auto" }}>
                        {/* Swim lane backgrounds */}
                        {hrAllHyps.map((h, hi) => (
                          <g key={hi}>
                            {hi % 2 === 0 && (
                              <rect x={hrPadL} y={hrPadT + hi * hrRowH} width={hrPlotW} height={hrRowH} fill="var(--color-border)" opacity={0.15} />
                            )}
                            {/* Label */}
                            <text x={hrPadL - 6} y={hrPadT + hi * hrRowH + hrRowH / 2 + 3} textAnchor="end" fontSize="8" fontFamily="var(--font-mono)" fontWeight={500} fill={h.snapshots.length > 0 && h.snapshots[h.snapshots.length - 1].pReject ? "rgba(239,68,68,0.8)" : "var(--color-text-secondary)"}>
                              {h.label}
                            </text>
                            {/* Status segments: green=active, red=rejected */}
                            {h.snapshots.map((snap, si) => {
                              const x1 = hrPadL + ((snap.iter - hrMinIter) / hrIterRange) * hrPlotW;
                              const nextIter = si < h.snapshots.length - 1 ? h.snapshots[si + 1].iter : hrMaxIter;
                              const x2 = hrPadL + ((nextIter - hrMinIter) / hrIterRange) * hrPlotW;
                              const y = hrPadT + hi * hrRowH + hrRowH * 0.3;
                              const barH = hrRowH * 0.4;
                              return (
                                <rect key={si} x={x1} y={y} width={Math.max(2, x2 - x1)} height={barH} rx={2} fill={snap.pReject ? "rgba(239,68,68,0.45)" : "rgba(34,197,94,0.35)"} />
                              );
                            })}
                            {/* Rejection marker */}
                            {h.snapshots.map((snap, si) => {
                              if (!snap.pReject) return null;
                              // Show rejection marker at first rejection point
                              const prevNotRejected = si === 0 || !h.snapshots[si - 1].pReject;
                              if (!prevNotRejected) return null;
                              const x = hrPadL + ((snap.iter - hrMinIter) / hrIterRange) * hrPlotW;
                              const y = hrPadT + hi * hrRowH + hrRowH / 2;
                              return (
                                <g key={`rej${si}`}>
                                  <line x1={x} y1={y - hrRowH * 0.3} x2={x} y2={y + hrRowH * 0.3} stroke="rgba(239,68,68,0.7)" strokeWidth="1.5" />
                                  <text x={x} y={y - hrRowH * 0.35} textAnchor="middle" fontSize="7" fill="rgba(239,68,68,0.8)" fontWeight="600">✗</text>
                                </g>
                              );
                            })}
                          </g>
                        ))}
                        {/* X axis */}
                        <line x1={hrPadL} y1={hrH - hrPadB} x2={hrPadL + hrPlotW} y2={hrH - hrPadB} stroke="var(--color-border)" strokeWidth="0.5" />
                        <text x={hrPadL} y={hrH - 4} fontSize="7" fill="var(--color-text-muted)" fontFamily="var(--font-mono)">iter {hrMinIter}</text>
                        <text x={hrPadL + hrPlotW} y={hrH - 4} fontSize="7" fill="var(--color-text-muted)" fontFamily="var(--font-mono)" textAnchor="end">iter {hrMaxIter}</text>
                        <text x={hrPadL + hrPlotW / 2} y={hrH - 4} fontSize="7" fill="var(--color-text-muted)" textAnchor="middle">Iteration →</text>
                      </svg>
                    </div>
                    <div style={{ display: "flex", gap: "16px", marginTop: "4px", flexWrap: "wrap", alignItems: "center" }}>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: 14, height: 8, background: "rgba(34,197,94,0.35)", marginRight: 4, verticalAlign: "middle", borderRadius: 2 }} />Active</span>
                      <span className="efficiency-legend-item"><span style={{ display: "inline-block", width: 14, height: 8, background: "rgba(239,68,68,0.45)", marginRight: 4, verticalAlign: "middle", borderRadius: 2 }} />Rejected</span>
                      <span className="efficiency-legend-item"><span style={{ fontWeight: 600, color: "rgba(239,68,68,0.8)", marginRight: 4 }}>✗</span>Rejection point</span>
                      <span style={{ marginLeft: "auto", fontSize: "0.72rem", color: "var(--color-text-muted)" }}>
                        Spearman ρ, center fraction, product correlation
                      </span>
                    </div>
                  </div>
                );
              })()}

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
                    {bestResult && (
                      <button
                        className="btn btn-sm btn-accent"
                        onClick={() => {
                          // Find best trial's position in sorted list
                          const filterLower2 = historyFilter.toLowerCase();
                          let filtered2 = trials;
                          if (filterLower2) {
                            filtered2 = trials.filter((t) => {
                              const iterStr = String(t.iteration);
                              const paramStr = Object.entries(t.parameters).map(([k, v]) => `${k}=${typeof v === "number" ? v.toFixed(4) : v}`).join(" ");
                              const kpiStr = Object.entries(t.kpis).map(([k, v]) => `${k}=${typeof v === "number" ? v.toFixed(4) : v}`).join(" ");
                              return `${iterStr} ${paramStr} ${kpiStr}`.toLowerCase().includes(filterLower2);
                            });
                          }
                          let sorted2 = [...filtered2];
                          if (historySortCol) {
                            sorted2.sort((a, b) => {
                              let va: number, vb: number;
                              if (historySortCol === "__iter__") { va = a.iteration; vb = b.iteration; }
                              else if (historySortCol.startsWith("p:")) { const key = historySortCol.slice(2); va = Number(a.parameters[key]) || 0; vb = Number(b.parameters[key]) || 0; }
                              else { const key = historySortCol.slice(2); va = Number(a.kpis[key]) || 0; vb = Number(b.kpis[key]) || 0; }
                              return historySortDir === "asc" ? va - vb : vb - va;
                            });
                          } else {
                            sorted2.reverse();
                          }
                          const bestIdx = sorted2.findIndex((t) => t.iteration === bestResult.iteration);
                          if (bestIdx >= 0) {
                            const targetPage = Math.floor(bestIdx / HISTORY_PAGE_SIZE);
                            setHistoryPage(targetPage);
                            // Expand the best trial and scroll to it
                            setExpandedTrialId(bestResult.id);
                            setTimeout(() => {
                              document.querySelector(".history-row-best")?.scrollIntoView({ behavior: "smooth", block: "center" });
                            }, 100);
                          }
                        }}
                        title="Jump to best trial"
                      >
                        <Trophy size={13} /> Jump to Best
                      </button>
                    )}
                    {bookmarks.size > 0 && (
                      <button
                        className={`btn btn-sm ${showBookmarkedOnly ? "btn-bookmark-active" : "btn-secondary"}`}
                        onClick={() => { setShowBookmarkedOnly(p => !p); setHistoryPage(0); }}
                        title={showBookmarkedOnly ? "Show all trials" : "Show bookmarked only"}
                      >
                        <Star size={13} fill={showBookmarkedOnly ? "currentColor" : "none"} /> {bookmarks.size}
                      </button>
                    )}
                    <button
                      className="btn btn-sm btn-secondary"
                      onClick={handleExportHistoryCSV}
                      title="Download all trials as CSV"
                    >
                      <Download size={13} /> CSV
                    </button>
                    <button
                      className="btn btn-sm btn-secondary"
                      onClick={handleExportHistoryJSON}
                      title="Download all trials as JSON"
                    >
                      <FileJson size={13} /> JSON
                    </button>
                    {compareSet.size >= 2 && (
                      <button
                        className="btn btn-sm btn-primary"
                        onClick={() => setShowCompareModal(true)}
                        title="Compare selected trials"
                      >
                        <GitCompare size={13} /> Compare ({compareSet.size})
                      </button>
                    )}
                    <span className="history-sort-hint">Click headers to sort</span>
                  </div>
                </div>
                {trials.length > 0 ? (() => {
                  // Apply filter
                  const filterLower = historyFilter.toLowerCase();
                  let filtered = trials;
                  if (showBookmarkedOnly) {
                    filtered = filtered.filter((t) => bookmarks.has(t.id));
                  }
                  if (filterLower) {
                    filtered = filtered.filter((t) => {
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

                  // Build parameter spec lookup for range bars
                  const specMap = new Map<string, { lower: number; upper: number }>();
                  if (campaign.spec?.parameters) {
                    for (const s of campaign.spec.parameters) {
                      if (s.lower != null && s.upper != null) specMap.set(s.name, { lower: s.lower, upper: s.upper });
                    }
                  }

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
                              <th style={{ width: "32px", textAlign: "center", padding: "8px 2px" }} title="Select for comparison">
                                <GitCompare size={12} />
                              </th>
                              <th style={{ width: "36px", textAlign: "center", padding: "8px 4px" }}><Star size={13} /></th>
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
                                    className={`history-row-clickable ${isBest ? "history-row-best" : ""} ${isExpanded ? "history-row-expanded" : ""} ${bookmarks.has(trial.id) ? "history-row-bookmarked" : ""}`}
                                    onClick={() => setExpandedTrialId(isExpanded ? null : trial.id)}
                                  >
                                    <td className="history-compare-cell" onClick={(e) => { e.stopPropagation(); toggleCompare(trial.id); }}>
                                      <CheckSquare
                                        size={13}
                                        className={`compare-check ${compareSet.has(trial.id) ? "compare-active" : ""}`}
                                        fill={compareSet.has(trial.id) ? "currentColor" : "none"}
                                      />
                                    </td>
                                    <td className="history-bookmark-cell" onClick={(e) => { e.stopPropagation(); toggleBookmark(trial.id); }}>
                                      <Star
                                        size={14}
                                        className={`bookmark-star ${bookmarks.has(trial.id) ? "bookmarked" : ""}`}
                                        fill={bookmarks.has(trial.id) ? "currentColor" : "none"}
                                      />
                                    </td>
                                    <td className="history-iter">
                                      {trial.iteration}
                                      {(trialTags[trial.id]?.length > 0) && (
                                        <span className="trial-tag-dot" title={trialTags[trial.id].join(", ")} />
                                      )}
                                    </td>
                                    {paramKeys.map((p) => {
                                      const spec = specMap.get(p);
                                      const val = trial.parameters[p];
                                      return (
                                        <td key={p} className="mono history-param-cell">
                                          <span>{typeof val === "number" ? val.toFixed(3) : String(val)}</span>
                                          {spec && typeof val === "number" && (
                                            <MiniRangeBar value={val} lower={spec.lower} upper={spec.upper} />
                                          )}
                                        </td>
                                      );
                                    })}
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
                                      <td colSpan={3 + paramKeys.length + kpiKeys.length}>
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
                                          <div className="history-detail-note">
                                            <span className="history-detail-label">Note</span>
                                            <input
                                              type="text"
                                              className="history-note-input"
                                              placeholder="Add a note about this trial..."
                                              value={trialNotes[trial.id] || ""}
                                              onChange={(e) => setTrialNote(trial.id, e.target.value)}
                                              onClick={(e) => e.stopPropagation()}
                                            />
                                          </div>
                                          <div className="history-detail-tags" onClick={(e) => e.stopPropagation()}>
                                            <span className="history-detail-label"><Tag size={12} /> Tags</span>
                                            <div className="trial-tag-row">
                                              {TRIAL_TAG_OPTIONS.map(tag => {
                                                const active = (trialTags[trial.id] || []).includes(tag);
                                                return (
                                                  <button
                                                    key={tag}
                                                    className={`trial-tag-btn ${active ? "trial-tag-active" : ""} trial-tag-${tag}`}
                                                    onClick={() => toggleTrialTag(trial.id, tag)}
                                                  >
                                                    {tag}
                                                  </button>
                                                );
                                              })}
                                            </div>
                                          </div>
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

              {/* Statistical Quick-Compare */}
              {trials.length >= 8 && (() => {
                const objKey = Object.keys(trials[0].kpis)[0];
                const allVals = trials.map(t => Number(t.kpis[objKey]) || 0).sort((a, b) => a - b);
                const q25idx = Math.floor(allVals.length * 0.25);
                const q75idx = Math.ceil(allVals.length * 0.75);
                const topVals = allVals.slice(0, q25idx); // best 25% (lowest for minimize)
                const bottomVals = allVals.slice(q75idx); // worst 25%
                const stats = (vals: number[]) => {
                  if (vals.length === 0) return { mean: 0, std: 0, min: 0, max: 0, n: 0 };
                  const n = vals.length;
                  const mean = vals.reduce((a, b) => a + b, 0) / n;
                  const variance = vals.reduce((a, v) => a + (v - mean) ** 2, 0) / n;
                  return { mean, std: Math.sqrt(variance), min: Math.min(...vals), max: Math.max(...vals), n };
                };
                const topS = stats(topVals);
                const botS = stats(bottomVals);
                // Cohen's d effect size
                const pooledStd = Math.sqrt(((topS.std ** 2) * topS.n + (botS.std ** 2) * botS.n) / (topS.n + botS.n));
                const cohenD = pooledStd > 0 ? Math.abs(topS.mean - botS.mean) / pooledStd : 0;
                const effectLabel = cohenD > 0.8 ? "Large" : cohenD > 0.5 ? "Medium" : cohenD > 0.2 ? "Small" : "Negligible";
                const effectColor = cohenD > 0.8 ? "#22c55e" : cohenD > 0.5 ? "#eab308" : "#94a3b8";
                // Improvement %
                const improvement = botS.mean !== 0 ? ((botS.mean - topS.mean) / Math.abs(botS.mean) * 100) : 0;

                return (
                  <div className="card stat-compare-card">
                    <div className="stat-compare-header" onClick={() => setShowStatCompare(p => !p)} style={{ cursor: "pointer" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                        <BarChart2 size={16} style={{ color: "var(--color-primary)" }} />
                        <h2 style={{ margin: 0 }}>Statistical Quick-Compare</h2>
                        <span className="stat-compare-badge" style={{ background: effectColor }}>{effectLabel} effect</span>
                      </div>
                      <ChevronRight size={16} style={{ transform: showStatCompare ? "rotate(90deg)" : "none", transition: "transform 0.2s", color: "var(--color-text-muted)" }} />
                    </div>
                    {showStatCompare && (
                      <div className="stat-compare-body">
                        <p className="range-desc" style={{ marginBottom: "12px" }}>Comparing best 25% of trials vs worst 25% by {objKey}.</p>
                        <div className="stat-compare-grid">
                          <div className="stat-compare-group stat-compare-top">
                            <div className="stat-compare-group-title">Best 25% ({topS.n} trials)</div>
                            <div className="stat-compare-row">
                              <span>Mean</span><span className="mono">{topS.mean.toFixed(4)}</span>
                            </div>
                            <div className="stat-compare-row">
                              <span>Std Dev</span><span className="mono">{topS.std.toFixed(4)}</span>
                            </div>
                            <div className="stat-compare-row">
                              <span>Range</span><span className="mono">[{topS.min.toFixed(4)}, {topS.max.toFixed(4)}]</span>
                            </div>
                          </div>
                          <div className="stat-compare-group stat-compare-bottom">
                            <div className="stat-compare-group-title">Worst 25% ({botS.n} trials)</div>
                            <div className="stat-compare-row">
                              <span>Mean</span><span className="mono">{botS.mean.toFixed(4)}</span>
                            </div>
                            <div className="stat-compare-row">
                              <span>Std Dev</span><span className="mono">{botS.std.toFixed(4)}</span>
                            </div>
                            <div className="stat-compare-row">
                              <span>Range</span><span className="mono">[{botS.min.toFixed(4)}, {botS.max.toFixed(4)}]</span>
                            </div>
                          </div>
                        </div>
                        <div className="stat-compare-summary">
                          <div className="stat-compare-row">
                            <span>Effect Size (Cohen's d)</span>
                            <span className="mono" style={{ color: effectColor, fontWeight: 600 }}>{cohenD.toFixed(3)} ({effectLabel})</span>
                          </div>
                          <div className="stat-compare-row">
                            <span>Improvement</span>
                            <span className="mono" style={{ fontWeight: 600 }}>{improvement > 0 ? "+" : ""}{improvement.toFixed(1)}%</span>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })()}

              {/* Trial Comparison Modal */}
              {showCompareModal && compareSet.size >= 2 && (() => {
                const compareTrials = trials.filter(t => compareSet.has(t.id));
                if (compareTrials.length < 2) return null;
                const paramKeys2 = Object.keys(compareTrials[0].parameters);
                const kpiKeys2 = Object.keys(compareTrials[0].kpis);
                const bestCompareKpi = Math.min(...compareTrials.map(t => Number(Object.values(t.kpis)[0]) || 0));
                return (
                  <div className="shortcut-overlay" onClick={() => setShowCompareModal(false)}>
                    <div className="compare-modal" onClick={(e) => e.stopPropagation()}>
                      <div className="shortcut-modal-header">
                        <h3><GitCompare size={16} /> Trial Comparison</h3>
                        <button className="shortcut-close" onClick={() => setShowCompareModal(false)}><span>&times;</span></button>
                      </div>
                      <div className="compare-table-wrap">
                        <table className="compare-table">
                          <thead>
                            <tr>
                              <th className="compare-label-col">Metric</th>
                              {compareTrials.map(t => (
                                <th key={t.id} className={`compare-trial-col ${Number(Object.values(t.kpis)[0]) === bestCompareKpi ? "compare-best-col" : ""}`}>
                                  Trial #{t.iteration}
                                  {Number(Object.values(t.kpis)[0]) === bestCompareKpi && <span className="compare-best-tag">Best</span>}
                                </th>
                              ))}
                              {compareTrials.length === 2 && <th className="compare-delta-col">Delta</th>}
                            </tr>
                          </thead>
                          <tbody>
                            {kpiKeys2.map(k => {
                              const vals = compareTrials.map(t => Number(t.kpis[k]) || 0);
                              const best = Math.min(...vals);
                              return (
                                <tr key={`k-${k}`} className="compare-kpi-row">
                                  <td className="compare-label">{k}</td>
                                  {vals.map((v, i) => (
                                    <td key={i} className={`mono compare-val ${v === best ? "compare-val-best" : ""}`}>
                                      {v.toFixed(4)}
                                    </td>
                                  ))}
                                  {vals.length === 2 && (
                                    <td className={`mono compare-delta ${vals[1] - vals[0] < 0 ? "compare-delta-better" : vals[1] - vals[0] > 0 ? "compare-delta-worse" : ""}`}>
                                      {(vals[1] - vals[0]).toFixed(4)}
                                    </td>
                                  )}
                                </tr>
                              );
                            })}
                            {paramKeys2.map(p => {
                              const vals = compareTrials.map(t => Number(t.parameters[p]) || 0);
                              return (
                                <tr key={`p-${p}`}>
                                  <td className="compare-label">{p}</td>
                                  {vals.map((v, i) => (
                                    <td key={i} className="mono compare-val">{v.toFixed(4)}</td>
                                  ))}
                                  {vals.length === 2 && (
                                    <td className="mono compare-delta">{(vals[1] - vals[0]).toFixed(4)}</td>
                                  )}
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                      <div className="compare-footer">
                        <button className="btn btn-sm btn-secondary" onClick={() => { setCompareSet(new Set()); setShowCompareModal(false); }}>
                          Clear Selection
                        </button>
                      </div>
                    </div>
                  </div>
                );
              })()}
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
              <div className="shortcut-item"><kbd>b</kbd> Jump to best trial</div>
              <div className="shortcut-item"><kbd>g</kbd> Generate suggestions</div>
              <div className="shortcut-item"><kbd>f</kbd> Focus filter input</div>
              <div className="shortcut-item"><kbd>?</kbd> Toggle this help</div>
              <div className="shortcut-item"><kbd>Esc</kbd> Close modal / collapse</div>
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
          animation: fadeSlideUp 0.25s ease;
        }

        @keyframes fadeSlideUp {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: translateY(0); }
        }

        /* Staggered card reveals */
        .suggestions-grid > * {
          animation: fadeSlideUp 0.3s ease both;
        }
        .suggestions-grid > *:nth-child(2) { animation-delay: 50ms; }
        .suggestions-grid > *:nth-child(3) { animation-delay: 100ms; }
        .suggestions-grid > *:nth-child(4) { animation-delay: 150ms; }
        .suggestions-grid > *:nth-child(5) { animation-delay: 200ms; }
        .suggestions-grid > *:nth-child(n+6) { animation-delay: 250ms; }

        .stats-row > * {
          animation: fadeSlideUp 0.25s ease both;
        }
        .stats-row > *:nth-child(2) { animation-delay: 40ms; }
        .stats-row > *:nth-child(3) { animation-delay: 80ms; }
        .stats-row > *:nth-child(4) { animation-delay: 120ms; }

        @media (prefers-reduced-motion: reduce) {
          .tab-panel, .suggestions-grid > *, .stats-row > * {
            animation: none;
          }
        }

        /* Bookmark styles */
        .history-bookmark-cell {
          text-align: center;
          padding: 8px 4px !important;
          cursor: pointer;
          width: 36px;
        }
        .bookmark-star {
          color: var(--color-text-muted);
          opacity: 0.25;
          transition: all 0.15s;
        }
        .bookmark-star:hover {
          opacity: 0.7;
          color: #f59e0b;
        }
        .bookmark-star.bookmarked {
          color: #f59e0b;
          opacity: 1;
        }
        .history-row-bookmarked {
          border-left: 3px solid #f59e0b !important;
        }
        .btn-bookmark-active {
          background: linear-gradient(135deg, #f59e0b, #d97706) !important;
          color: white !important;
          border-color: #d97706 !important;
          font-weight: 600;
        }
        .history-note-input {
          flex: 1;
          padding: 4px 8px;
          border: 1px solid var(--color-border);
          border-radius: 6px;
          font-size: 0.82rem;
          font-family: inherit;
          background: var(--color-surface);
          color: var(--color-text);
          width: 100%;
          max-width: 400px;
          transition: border-color 0.15s;
        }
        .history-note-input:focus {
          outline: none;
          border-color: var(--color-primary);
          box-shadow: 0 0 0 2px rgba(79, 110, 247, 0.1);
        }
        .history-detail-note {
          display: flex;
          align-items: center;
          gap: 8px;
          grid-column: 1 / -1;
          padding-top: 8px;
          border-top: 1px solid var(--color-border-subtle);
          margin-top: 4px;
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

        /* ── Diversity badges ── */
        .suggestion-card-wrapper {
          position: relative;
        }
        .diversity-badge {
          position: absolute;
          top: -6px;
          right: 12px;
          z-index: 2;
          display: inline-flex;
          align-items: center;
          gap: 4px;
          padding: 2px 10px;
          font-size: 0.68rem;
          font-weight: 700;
          letter-spacing: 0.03em;
          border-radius: 10px;
          text-transform: uppercase;
        }
        .diversity-high {
          background: #dbeafe;
          color: #1d4ed8;
          border: 1px solid #93c5fd;
        }
        .diversity-mid {
          background: #fef3c7;
          color: #92400e;
          border: 1px solid #fcd34d;
        }
        .diversity-low {
          background: #f0fdf4;
          color: #166534;
          border: 1px solid #86efac;
        }
        [data-theme="dark"] .diversity-high {
          background: #1e3a5f;
          color: #93c5fd;
          border-color: #1d4ed8;
        }
        [data-theme="dark"] .diversity-mid {
          background: #451a03;
          color: #fcd34d;
          border-color: #92400e;
        }
        [data-theme="dark"] .diversity-low {
          background: #052e16;
          color: #86efac;
          border-color: #166534;
        }

        /* ── Rejected suggestion stack ── */
        .rejected-stack {
          margin-top: 16px;
          border: 1px solid var(--color-border);
          border-radius: var(--radius-lg, 12px);
          background: var(--color-surface);
          overflow: hidden;
        }
        .rejected-stack-toggle {
          display: flex;
          align-items: center;
          gap: 8px;
          width: 100%;
          padding: 12px 16px;
          background: none;
          border: none;
          font-size: 0.85rem;
          font-weight: 500;
          color: var(--color-text-muted);
          cursor: pointer;
          font-family: inherit;
          transition: color 0.15s, background 0.15s;
        }
        .rejected-stack-toggle:hover {
          color: var(--color-text);
          background: var(--color-bg);
        }
        .rejected-stack-list {
          border-top: 1px solid var(--color-border);
          animation: fadeSlideUp 0.2s ease;
        }
        .rejected-item {
          padding: 12px 16px;
          border-bottom: 1px solid var(--color-border-subtle);
        }
        .rejected-item:last-child {
          border-bottom: none;
        }
        .rejected-item-header {
          display: flex;
          align-items: center;
          gap: 10px;
          margin-bottom: 6px;
        }
        .rejected-item-index {
          font-weight: 700;
          font-size: 0.88rem;
          color: var(--color-text-muted);
        }
        .rejected-item-time {
          font-size: 0.72rem;
          color: var(--color-text-muted);
          opacity: 0.6;
        }
        .rejected-reconsider-btn {
          margin-left: auto;
          font-size: 0.72rem !important;
          padding: 3px 10px !important;
        }
        .rejected-item-params {
          display: flex;
          flex-wrap: wrap;
          gap: 8px;
          font-size: 0.78rem;
        }
        .rejected-param {
          display: flex;
          gap: 4px;
          padding: 2px 8px;
          background: var(--color-bg);
          border-radius: 6px;
        }
        .rejected-param-name {
          color: var(--color-text-muted);
          font-weight: 500;
        }
        .rejected-param-more {
          color: var(--color-text-muted);
          font-size: 0.72rem;
          padding: 2px 6px;
          opacity: 0.6;
        }

        /* ── Checkpoint styles ── */
        .checkpoint-controls {
          display: flex;
          align-items: center;
          gap: 12px;
          margin-bottom: 16px;
        }
        .checkpoint-count {
          font-size: 0.78rem;
          color: var(--color-text-muted);
        }
        .checkpoint-modal {
          min-width: 380px;
        }
        .checkpoint-modal-desc {
          font-size: 0.85rem;
          color: var(--color-text-muted);
          margin: 0 0 16px;
          line-height: 1.5;
        }
        .checkpoint-modal-snapshot {
          background: var(--color-bg);
          border-radius: 8px;
          padding: 12px 14px;
          margin-bottom: 16px;
        }
        .checkpoint-snapshot-row {
          display: flex;
          justify-content: space-between;
          font-size: 0.82rem;
          padding: 4px 0;
        }
        .checkpoint-snapshot-row + .checkpoint-snapshot-row {
          border-top: 1px solid var(--color-border-subtle);
        }
        .checkpoint-title-input {
          width: 100%;
          max-width: 100%;
          margin-bottom: 16px;
          padding: 8px 12px;
        }
        .checkpoint-modal-actions {
          display: flex;
          justify-content: flex-end;
          gap: 8px;
        }
        .checkpoint-list {
          display: flex;
          flex-direction: column;
          gap: 4px;
          margin-bottom: 16px;
          background: var(--color-surface);
          border: 1px solid var(--color-border);
          border-radius: var(--radius-lg, 12px);
          padding: 8px;
        }
        .checkpoint-item {
          display: flex;
          align-items: center;
          gap: 10px;
          padding: 8px 10px;
          border-radius: 8px;
          transition: background 0.15s;
        }
        .checkpoint-item:hover {
          background: var(--color-bg);
        }
        .checkpoint-dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          background: var(--color-primary);
          flex-shrink: 0;
        }
        .checkpoint-info {
          flex: 1;
          min-width: 0;
        }
        .checkpoint-title {
          display: block;
          font-size: 0.85rem;
          font-weight: 600;
          color: var(--color-text);
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }
        .checkpoint-meta {
          display: block;
          font-size: 0.72rem;
          color: var(--color-text-muted);
          margin-top: 2px;
        }
        .checkpoint-remove {
          background: none;
          border: none;
          color: var(--color-text-muted);
          opacity: 0;
          cursor: pointer;
          font-size: 1rem;
          padding: 2px 6px;
          transition: opacity 0.15s, color 0.15s;
        }
        .checkpoint-item:hover .checkpoint-remove {
          opacity: 0.5;
        }
        .checkpoint-remove:hover {
          opacity: 1 !important;
          color: #ef4444;
        }

        /* ── Parameter Variance Sentinel ── */
        .sentinel-card {
          transition: border-color 0.3s;
        }
        .sentinel-alert {
          border-color: #fca5a5 !important;
        }
        .sentinel-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 10px;
          margin-bottom: 12px;
        }
        .sentinel-header-left {
          display: flex;
          align-items: center;
          gap: 10px;
        }
        .sentinel-header-left h2 {
          font-size: 1rem;
        }
        .sentinel-alert-badge {
          display: inline-flex;
          padding: 2px 8px;
          font-size: 0.68rem;
          font-weight: 700;
          border-radius: 8px;
          background: #fef2f2;
          color: #dc2626;
          border: 1px solid #fca5a5;
          text-transform: uppercase;
          letter-spacing: 0.03em;
        }
        [data-theme="dark"] .sentinel-alert-badge {
          background: #450a0a;
          color: #fca5a5;
          border-color: #991b1b;
        }
        .sentinel-desc {
          font-size: 0.82rem;
          color: var(--color-text-muted);
          margin: 0 0 14px;
        }
        .sentinel-body {
          animation: fadeSlideUp 0.2s ease;
        }
        .sentinel-row {
          display: flex;
          align-items: center;
          gap: 10px;
          padding: 6px 0;
          border-bottom: 1px solid var(--color-border-subtle);
        }
        .sentinel-row:last-child {
          border-bottom: none;
        }
        .sentinel-row-alert {
          background: rgba(239, 68, 68, 0.04);
          margin: 0 -12px;
          padding: 6px 12px;
          border-radius: 6px;
        }
        .sentinel-param-name {
          width: 100px;
          font-size: 0.78rem;
          font-weight: 500;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
          flex-shrink: 0;
        }
        .sentinel-variance-bar {
          flex: 1;
          height: 5px;
          background: var(--color-border);
          border-radius: 3px;
          overflow: hidden;
        }
        .sentinel-variance-fill {
          height: 100%;
          border-radius: 3px;
          transition: width 0.4s ease;
          min-width: 2px;
        }
        .sentinel-std {
          width: 45px;
          font-size: 0.72rem;
          font-weight: 600;
          text-align: right;
          flex-shrink: 0;
        }
        .sentinel-warn-tag {
          font-size: 0.62rem;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 0.04em;
          color: #ef4444;
          flex-shrink: 0;
        }

        /* ── Goal Tracker ── */
        .goal-tracker-card { }
        .goal-tracker-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-bottom: 14px;
        }
        .goal-tracker-header-left {
          display: flex;
          align-items: center;
          gap: 10px;
        }
        .goal-tracker-header-left h2 { font-size: 1rem; }
        .goal-empty {
          font-size: 0.85rem;
          color: var(--color-text-muted);
          font-style: italic;
          margin: 0;
        }
        .goal-list { display: flex; flex-direction: column; gap: 10px; }
        .goal-item {
          padding: 12px 14px;
          border: 1px solid var(--color-border);
          border-radius: 10px;
          background: var(--color-surface);
          transition: border-color 0.2s;
        }
        .goal-item-reached { border-color: #22c55e; background: rgba(34, 197, 94, 0.04); }
        .goal-item-on-track { border-color: #3b82f6; }
        .goal-item-at-risk { border-color: #eab308; }
        .goal-item-behind { border-color: #ef4444; }
        .goal-item-top {
          display: flex;
          align-items: center;
          gap: 8px;
          margin-bottom: 8px;
        }
        .goal-item-name {
          font-weight: 600;
          font-size: 0.9rem;
          flex: 1;
        }
        .goal-status-badge {
          font-size: 0.68rem;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 0.03em;
          padding: 2px 8px;
          border-radius: 8px;
        }
        .goal-status-reached { background: #f0fdf4; color: #166534; }
        .goal-status-on-track { background: #eff6ff; color: #1d4ed8; }
        .goal-status-at-risk { background: #fffbeb; color: #92400e; }
        .goal-status-behind { background: #fef2f2; color: #991b1b; }
        [data-theme="dark"] .goal-status-reached { background: #052e16; color: #86efac; }
        [data-theme="dark"] .goal-status-on-track { background: #172554; color: #93c5fd; }
        [data-theme="dark"] .goal-status-at-risk { background: #451a03; color: #fcd34d; }
        [data-theme="dark"] .goal-status-behind { background: #450a0a; color: #fca5a5; }
        .goal-remove {
          background: none;
          border: none;
          color: var(--color-text-muted);
          opacity: 0;
          cursor: pointer;
          font-size: 1rem;
          padding: 0 4px;
          transition: opacity 0.15s;
        }
        .goal-item:hover .goal-remove { opacity: 0.5; }
        .goal-remove:hover { opacity: 1 !important; color: #ef4444; }
        .goal-progress-row {
          display: flex;
          align-items: center;
          gap: 10px;
          margin-bottom: 4px;
        }
        .goal-current, .goal-target-val {
          font-size: 0.75rem;
          font-weight: 600;
          color: var(--color-text-muted);
          width: 60px;
          flex-shrink: 0;
        }
        .goal-target-val { text-align: right; }
        .goal-progress-bar {
          flex: 1;
          height: 6px;
          background: var(--color-border);
          border-radius: 3px;
          overflow: hidden;
        }
        .goal-progress-fill {
          height: 100%;
          background: var(--color-primary);
          border-radius: 3px;
          transition: width 0.4s ease;
          min-width: 2px;
        }
        .goal-item-reached .goal-progress-fill { background: #22c55e; }
        .goal-item-at-risk .goal-progress-fill { background: #eab308; }
        .goal-item-behind .goal-progress-fill { background: #ef4444; }
        .goal-meta-row {
          display: flex;
          justify-content: space-between;
          font-size: 0.72rem;
          color: var(--color-text-muted);
        }
        .goal-eta { font-weight: 500; }
        .goal-modal-fields { display: flex; flex-direction: column; gap: 12px; margin-bottom: 16px; }
        .goal-modal-label {
          display: flex;
          flex-direction: column;
          gap: 4px;
          font-size: 0.82rem;
          font-weight: 500;
          color: var(--color-text-muted);
        }
        .goal-modal-row { display: flex; gap: 12px; }

        /* ── What-If Analysis ── */
        .whatif-card { }
        .whatif-header {
          display: flex;
          align-items: center;
          gap: 10px;
          margin-bottom: 8px;
        }
        .whatif-header h2 { font-size: 1rem; }
        .whatif-controls {
          display: flex;
          flex-direction: column;
          gap: 12px;
          margin-bottom: 16px;
        }
        .whatif-label {
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 0.82rem;
          font-weight: 500;
          color: var(--color-text-muted);
        }
        .whatif-slider-group { display: flex; flex-direction: column; gap: 4px; }
        .whatif-slider-labels {
          display: flex;
          justify-content: space-between;
          font-size: 0.72rem;
          color: var(--color-text-muted);
        }
        .whatif-current-val {
          font-weight: 700;
          color: var(--color-primary);
          font-size: 0.82rem;
        }
        .whatif-slider {
          width: 100%;
          accent-color: var(--color-primary);
          cursor: pointer;
        }
        .whatif-results {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
          gap: 10px;
          margin-bottom: 16px;
        }
        .whatif-result-box {
          padding: 10px 12px;
          background: var(--color-bg);
          border-radius: 8px;
          text-align: center;
        }
        .whatif-result-label {
          font-size: 0.72rem;
          font-weight: 500;
          color: var(--color-text-muted);
          margin-bottom: 2px;
        }
        .whatif-result-value {
          font-size: 1.1rem;
          font-weight: 700;
        }
        .whatif-result-ci {
          font-size: 0.68rem;
          color: var(--color-text-muted);
        }
        .whatif-better { color: #22c55e; }
        .whatif-worse { color: #ef4444; }
        .whatif-scatter {
          display: block;
          width: 100%;
          max-width: 300px;
          background: var(--color-bg);
          border-radius: 8px;
          padding: 2px;
        }

        /* ── Trial Comparison ── */
        .history-compare-cell {
          text-align: center;
          padding: 8px 2px !important;
          cursor: pointer;
          width: 32px;
        }
        .compare-check {
          color: var(--color-text-muted);
          opacity: 0.2;
          transition: all 0.15s;
        }
        .compare-check:hover { opacity: 0.6; }
        .compare-active {
          color: var(--color-primary) !important;
          opacity: 1 !important;
        }
        .compare-modal {
          background: var(--color-surface);
          border: 1px solid var(--color-border);
          border-radius: 16px;
          padding: 24px 28px;
          min-width: 500px;
          max-width: 700px;
          max-height: 80vh;
          overflow-y: auto;
          box-shadow: var(--shadow-lg);
        }
        .compare-table-wrap { overflow-x: auto; margin-bottom: 16px; }
        .compare-table {
          width: 100%;
          border-collapse: collapse;
          font-size: 0.82rem;
        }
        .compare-table th {
          padding: 8px 12px;
          font-weight: 600;
          text-align: center;
          border-bottom: 2px solid var(--color-border);
          font-size: 0.78rem;
        }
        .compare-table td {
          padding: 6px 12px;
          border-bottom: 1px solid var(--color-border-subtle);
        }
        .compare-label-col {
          text-align: left !important;
          width: 100px;
        }
        .compare-label {
          font-weight: 500;
          color: var(--color-text-muted);
          font-family: var(--font-mono);
          font-size: 0.78rem;
        }
        .compare-val { text-align: center; }
        .compare-val-best {
          font-weight: 700;
          color: var(--color-green);
        }
        .compare-best-col {
          background: rgba(34, 197, 94, 0.06);
        }
        .compare-best-tag {
          display: inline-block;
          margin-left: 6px;
          font-size: 0.6rem;
          font-weight: 700;
          text-transform: uppercase;
          color: #22c55e;
          background: rgba(34, 197, 94, 0.12);
          padding: 1px 5px;
          border-radius: 4px;
        }
        .compare-delta-col {
          width: 80px;
          text-align: center;
        }
        .compare-delta {
          text-align: center;
          font-size: 0.78rem;
          color: var(--color-text-muted);
        }
        .compare-delta-better { color: #22c55e !important; font-weight: 600; }
        .compare-delta-worse { color: #ef4444 !important; font-weight: 600; }
        .compare-kpi-row {
          background: rgba(79, 110, 247, 0.03);
        }
        .compare-footer {
          display: flex;
          justify-content: flex-end;
        }

        /* ── Batch Planner ── */
        .suggestion-select-check {
          position: absolute;
          top: 8px;
          left: 8px;
          z-index: 3;
          cursor: pointer;
          padding: 4px;
          border-radius: 4px;
          transition: background 0.15s;
        }
        .suggestion-select-check:hover {
          background: var(--color-bg);
        }
        .sug-check-idle {
          color: var(--color-text-muted);
          opacity: 0.25;
          transition: opacity 0.15s;
        }
        .suggestion-card-wrapper:hover .sug-check-idle {
          opacity: 0.6;
        }
        .sug-check-active {
          color: var(--color-primary);
          opacity: 1;
        }
        .suggestion-selected {
          outline: 2px solid var(--color-primary);
          outline-offset: -1px;
          border-radius: var(--radius-lg, 12px);
        }
        .batch-planner-bar {
          position: sticky;
          bottom: 0;
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 12px 18px;
          background: var(--color-surface);
          border: 1px solid var(--color-primary);
          border-radius: 12px;
          box-shadow: 0 -4px 20px rgba(0, 0, 0, 0.08);
          margin-top: 16px;
          animation: fadeSlideUp 0.2s ease;
        }
        .batch-planner-left {
          display: flex;
          align-items: center;
          gap: 10px;
          color: var(--color-primary);
          font-weight: 600;
          font-size: 0.88rem;
        }
        .batch-planner-count { color: var(--color-text); }
        .batch-diversity-pill {
          display: inline-flex;
          align-items: center;
          gap: 4px;
          padding: 2px 10px;
          background: var(--color-primary-subtle, rgba(79, 110, 247, 0.08));
          border-radius: 10px;
          font-size: 0.72rem;
          font-weight: 600;
          color: var(--color-primary);
        }
        .batch-planner-right {
          display: flex;
          gap: 8px;
        }

        /* ── Decision Journal ── */
        .decision-journal-card { padding: 16px 20px; }
        .decision-journal-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 12px;
        }
        .journal-count {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          min-width: 20px;
          height: 20px;
          padding: 0 6px;
          background: var(--color-primary-subtle, rgba(79, 110, 247, 0.1));
          color: var(--color-primary);
          border-radius: 10px;
          font-size: 0.72rem;
          font-weight: 600;
        }
        .journal-input-row {
          display: flex;
          gap: 8px;
          margin-bottom: 12px;
        }
        .journal-input {
          flex: 1;
          padding: 8px 12px;
          border: 1px solid var(--color-border);
          border-radius: 8px;
          background: var(--color-bg);
          color: var(--color-text);
          font-family: var(--font-mono);
          font-size: 0.82rem;
        }
        .journal-input:focus { outline: none; border-color: var(--color-primary); box-shadow: 0 0 0 3px rgba(79, 110, 247, 0.1); }
        .journal-empty {
          font-size: 0.82rem;
          color: var(--color-text-muted);
          font-style: italic;
          padding: 8px 0;
          margin: 0;
        }
        .journal-entries {
          display: flex;
          flex-direction: column;
          gap: 8px;
          max-height: 280px;
          overflow-y: auto;
        }
        .journal-entry {
          padding: 10px 12px;
          background: var(--color-bg);
          border: 1px solid var(--color-border-subtle, var(--color-border));
          border-radius: 8px;
          border-left: 3px solid var(--color-primary);
        }
        .journal-entry-meta {
          display: flex;
          align-items: center;
          gap: 8px;
          margin-bottom: 4px;
          font-size: 0.72rem;
          color: var(--color-text-muted);
        }
        .journal-iter {
          font-weight: 600;
          color: var(--color-primary);
          background: var(--color-primary-subtle, rgba(79, 110, 247, 0.08));
          padding: 1px 6px;
          border-radius: 4px;
        }
        .journal-delete {
          margin-left: auto;
          background: none;
          border: none;
          color: var(--color-text-muted);
          cursor: pointer;
          padding: 2px;
          border-radius: 4px;
          opacity: 0;
          transition: opacity 0.15s;
        }
        .journal-entry:hover .journal-delete { opacity: 1; }
        .journal-delete:hover { color: var(--color-red, #ef4444); background: rgba(239, 68, 68, 0.08); }
        .journal-entry-text {
          font-size: 0.85rem;
          color: var(--color-text);
          line-height: 1.5;
        }

        /* ── Replay Controls ── */
        .convergence-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 8px;
        }
        .convergence-header h2 { margin: 0; }
        .replay-controls {
          display: flex;
          align-items: center;
          gap: 6px;
        }
        .replay-counter {
          font-family: var(--font-mono);
          font-size: 0.75rem;
          color: var(--color-text-muted);
          padding: 2px 8px;
          background: var(--color-bg);
          border-radius: 6px;
          border: 1px solid var(--color-border);
        }
        .replay-info {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 6px 12px;
          margin-top: 8px;
          background: var(--color-primary-subtle, rgba(79, 110, 247, 0.06));
          border-radius: 8px;
          font-size: 0.8rem;
          font-family: var(--font-mono);
          color: var(--color-text);
        }
        .replay-dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          background: var(--color-primary);
          animation: pulse 1s ease-in-out infinite;
        }
        .replay-best { font-weight: 600; color: var(--color-primary); }

        /* ── Trial Tags ── */
        .history-detail-tags {
          margin-top: 8px;
          display: flex;
          align-items: flex-start;
          gap: 8px;
        }
        .history-detail-tags .history-detail-label {
          display: flex;
          align-items: center;
          gap: 4px;
          padding-top: 3px;
        }
        .trial-tag-row {
          display: flex;
          flex-wrap: wrap;
          gap: 4px;
        }
        .trial-tag-btn {
          padding: 2px 10px;
          border-radius: 12px;
          border: 1px solid var(--color-border);
          background: var(--color-bg);
          color: var(--color-text-muted);
          font-size: 0.72rem;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.15s ease;
        }
        .trial-tag-btn:hover { border-color: var(--color-primary); color: var(--color-primary); }
        .trial-tag-active { border-color: transparent !important; color: white !important; }
        .trial-tag-active.trial-tag-promising { background: #22c55e; }
        .trial-tag-active.trial-tag-anomaly { background: #f59e0b; }
        .trial-tag-active.trial-tag-investigate { background: #3b82f6; }
        .trial-tag-active.trial-tag-baseline { background: #8b5cf6; }
        .trial-tag-active.trial-tag-outlier { background: #ef4444; }
        .trial-tag-active.trial-tag-equipment-issue { background: #64748b; }
        .trial-tag-dot {
          display: inline-block;
          width: 6px;
          height: 6px;
          border-radius: 50%;
          background: var(--color-primary);
          margin-left: 4px;
          vertical-align: middle;
        }

        /* ── Overview Actions Row ── */
        .overview-actions-row {
          display: flex;
          justify-content: flex-end;
          margin-bottom: 8px;
        }

        /* ── Histogram ── */
        .histogram-card {
          margin-bottom: 0;
        }

        /* ── Sample Efficiency ── */
        .efficiency-badge {
          font-size: 0.72rem;
          font-weight: 600;
          padding: 2px 8px;
          border-radius: 10px;
          background: var(--color-primary);
          color: white;
        }
        .efficiency-legend {
          display: flex;
          justify-content: center;
          gap: 16px;
          margin-top: 6px;
          font-size: 0.75rem;
          color: var(--color-text-muted);
        }
        .efficiency-legend-item {
          display: flex;
          align-items: center;
        }

        /* ── Parallel Coordinates Plot ── */
        .pcoord-line {
          transition: opacity 0.15s;
        }
        .pcoord-line:hover {
          stroke-width: 2.5 !important;
          opacity: 1 !important;
        }
        svg:has(.pcoord-line:hover) .pcoord-line:not(:hover) {
          opacity: 0.15 !important;
        }
        .pcoord-legend {
          display: flex;
          justify-content: center;
          gap: 16px;
          margin-top: 8px;
          font-size: 0.75rem;
          color: var(--color-text-muted);
        }
        .pcoord-legend-item {
          display: flex;
          align-items: center;
          gap: 5px;
        }
        .pcoord-dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          display: inline-block;
        }

        /* ── Statistical Quick-Compare ── */
        .stat-compare-card {}
        .stat-compare-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
        }
        .stat-compare-badge {
          font-size: 0.7rem;
          font-weight: 600;
          padding: 2px 8px;
          border-radius: 10px;
          color: white;
          text-transform: uppercase;
          letter-spacing: 0.03em;
        }
        .stat-compare-body {
          margin-top: 12px;
        }
        .stat-compare-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 12px;
          margin-bottom: 12px;
        }
        .stat-compare-group {
          padding: 10px 14px;
          border-radius: 8px;
          border: 1px solid var(--color-border);
        }
        .stat-compare-top {
          background: rgba(34, 197, 94, 0.04);
          border-color: rgba(34, 197, 94, 0.2);
        }
        .stat-compare-bottom {
          background: rgba(239, 68, 68, 0.04);
          border-color: rgba(239, 68, 68, 0.2);
        }
        .stat-compare-group-title {
          font-weight: 600;
          font-size: 0.82rem;
          margin-bottom: 6px;
        }
        .stat-compare-top .stat-compare-group-title { color: #22c55e; }
        .stat-compare-bottom .stat-compare-group-title { color: #ef4444; }
        .stat-compare-row {
          display: flex;
          justify-content: space-between;
          font-size: 0.82rem;
          padding: 3px 0;
        }
        .stat-compare-summary {
          padding: 10px 14px;
          background: var(--color-bg);
          border-radius: 8px;
          border: 1px solid var(--color-border);
        }

        /* ── Experiment Cost / Time Tracker ── */
        .cost-tracker-card {}
        .cost-tracker-grid {
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: 10px;
          margin-bottom: 14px;
        }
        .cost-tracker-item {
          text-align: center;
          padding: 8px;
          background: var(--color-bg);
          border-radius: 8px;
          border: 1px solid var(--color-border);
        }
        .cost-tracker-label {
          display: block;
          font-size: 0.72rem;
          text-transform: uppercase;
          letter-spacing: 0.04em;
          color: var(--color-text-muted);
          margin-bottom: 4px;
        }
        .cost-tracker-value {
          display: block;
          font-size: 1.1rem;
          font-weight: 700;
          font-family: var(--font-mono);
        }
        .cost-tracker-bar-section { margin-top: 4px; }
        .cost-tracker-bar-header {
          display: flex;
          justify-content: space-between;
          font-size: 0.78rem;
          margin-bottom: 4px;
          color: var(--color-text-muted);
        }
        .cost-tracker-bar {
          height: 8px;
          background: var(--color-border);
          border-radius: 4px;
          overflow: hidden;
        }
        .cost-tracker-bar-fill {
          height: 100%;
          border-radius: 4px;
          transition: width 0.4s ease;
        }
        .cost-tracker-warn {
          display: flex;
          align-items: center;
          gap: 6px;
          margin-top: 8px;
          font-size: 0.78rem;
          color: var(--color-yellow, #eab308);
          font-weight: 500;
        }

        /* ── Key Findings Summary ── */
        .findings-card {}
        .findings-badge {
          font-size: 0.72rem;
          font-weight: 700;
          padding: 2px 7px;
          border-radius: 10px;
          background: var(--color-primary);
          color: white;
        }
        .findings-list {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }
        .findings-item {
          display: flex;
          align-items: flex-start;
          gap: 10px;
          padding: 10px 14px;
          border-radius: 8px;
          font-size: 0.84rem;
          line-height: 1.5;
          border-left: 3px solid transparent;
        }
        .findings-success {
          background: rgba(34, 197, 94, 0.06);
          border-left-color: #22c55e;
        }
        .findings-info {
          background: rgba(59, 130, 246, 0.06);
          border-left-color: #3b82f6;
        }
        .findings-warning {
          background: rgba(234, 179, 8, 0.06);
          border-left-color: #eab308;
        }
        .findings-icon {
          flex-shrink: 0;
          margin-top: 2px;
          color: var(--color-text-muted);
        }
        .findings-success .findings-icon { color: #22c55e; }
        .findings-info .findings-icon { color: #3b82f6; }
        .findings-warning .findings-icon { color: #eab308; }
        .findings-text {
          color: var(--color-text);
        }

        /* ── Radar Chart ── */
        .radar-legend {
          display: flex;
          justify-content: center;
          gap: 16px;
          margin-top: 8px;
          font-size: 0.75rem;
          color: var(--color-text-muted);
        }
        .radar-legend-item {
          display: flex;
          align-items: center;
          gap: 5px;
        }
        .radar-dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          display: inline-block;
        }
      `}</style>
    </div>
    </ErrorBoundary>
  );
}
