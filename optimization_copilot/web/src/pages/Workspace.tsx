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
