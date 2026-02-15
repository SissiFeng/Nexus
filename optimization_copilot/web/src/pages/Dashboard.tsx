import { useEffect, useState, useCallback, useMemo, useRef } from "react";
import { Link, useNavigate } from "react-router-dom";
import type { LucideIcon } from "lucide-react";
import {
  Search,
  Archive,
  ArchiveRestore,
  Trash2,
  ExternalLink,
  Activity,
  Trophy,
  FlaskConical,
  LayoutGrid,
  Upload,
  BookOpen,
  Sparkles,
  ArrowRight,
  Clock,
  GitCompareArrows,
  X,
  CheckSquare,
  Square,
} from "lucide-react";
import {
  fetchCampaigns,
  deleteCampaign,
  updateCampaignStatus,
  type CampaignSummary,
} from "../api";
import ErrorBoundary from "../components/ErrorBoundary";
import EmptyState from "../components/EmptyState";

/* ── Helpers ── */

function formatRelativeTime(epochSeconds: number): string {
  const now = Date.now() / 1000;
  const diff = now - epochSeconds;

  if (diff < 60) return "just now";
  if (diff < 3600) {
    const mins = Math.floor(diff / 60);
    return `${mins} minute${mins !== 1 ? "s" : ""} ago`;
  }
  if (diff < 86400) {
    const hrs = Math.floor(diff / 3600);
    return `${hrs} hour${hrs !== 1 ? "s" : ""} ago`;
  }
  if (diff < 2592000) {
    const days = Math.floor(diff / 86400);
    return `${days} day${days !== 1 ? "s" : ""} ago`;
  }
  const months = Math.floor(diff / 2592000);
  return `${months} month${months !== 1 ? "s" : ""} ago`;
}

type StatusFilter = "all" | "running" | "completed" | "archived";
type SortOption = "latest" | "oldest" | "best_kpi" | "most_trials";

/* ── Sub-components ── */

function StatusBadge({ status }: { status: string }) {
  return <span className={`badge badge-${status}`}>{status}</span>;
}

function StatCard({
  label,
  value,
  icon: Icon,
  color,
}: {
  label: string;
  value: string | number;
  icon: LucideIcon;
  color?: string;
}) {
  return (
    <div className="stat-card">
      <div className="stat-card-header">
        <span className="stat-label">{label}</span>
        <Icon size={18} color={color || "var(--color-text-muted)"} />
      </div>
      <div className="stat-value" style={{ color: color || undefined }}>
        {value}
      </div>
    </div>
  );
}

function CampaignCard({
  campaign,
  onArchive,
  onUnarchive,
  onDelete,
  selected,
  onToggleSelect,
}: {
  campaign: CampaignSummary;
  onArchive: (id: string) => void;
  onUnarchive: (id: string) => void;
  onDelete: (id: string) => void;
  selected: boolean;
  onToggleSelect: (id: string) => void;
}) {
  const [confirmDelete, setConfirmDelete] = useState(false);
  const c = campaign;

  return (
    <div className={`dashboard-card ${c.status === "archived" ? "dashboard-card-archived" : ""} ${selected ? "dashboard-card-selected" : ""}`}>
      <div className="dashboard-card-top">
        <button
          className="dashboard-card-checkbox"
          onClick={(e) => { e.stopPropagation(); onToggleSelect(c.campaign_id); }}
          title={selected ? "Deselect" : "Select for comparison"}
        >
          {selected ? <CheckSquare size={16} /> : <Square size={16} />}
        </button>
        <Link to={`/workspace/${c.campaign_id}`} className="dashboard-card-name">
          {c.name}
        </Link>
        <StatusBadge status={c.status} />
      </div>

      {c.best_kpi !== null && (
        <div className="dashboard-card-kpi">
          <span className="dashboard-card-kpi-label">Best KPI</span>
          <span className="dashboard-card-kpi-value">{c.best_kpi.toFixed(4)}</span>
        </div>
      )}

      <div className="dashboard-card-progress-section">
        <div className="dashboard-card-progress-header">
          <span className="dashboard-card-progress-label">
            {c.total_trials > 0
              ? `Trial ${c.iteration} / ${c.total_trials}`
              : `Iteration ${c.iteration}`}
          </span>
          {c.total_trials > 0 && (
            <span className="dashboard-card-progress-pct">
              {Math.round((c.iteration / c.total_trials) * 100)}%
            </span>
          )}
        </div>
        {c.total_trials > 0 && (
          <div className="dashboard-card-progress-track">
            <div
              className={`dashboard-card-progress-fill ${c.status === "completed" ? "progress-complete" : ""}`}
              style={{ width: `${Math.min(100, (c.iteration / c.total_trials) * 100)}%` }}
            />
          </div>
        )}
      </div>

      <div className="dashboard-card-meta">
        <span className="dashboard-card-date">{formatRelativeTime(c.created_at)}</span>
        {c.updated_at && c.updated_at !== c.created_at && (
          <span className="dashboard-card-updated">
            <Clock size={11} />
            {formatRelativeTime(c.updated_at)}
          </span>
        )}
      </div>

      {c.objective_names.length > 0 && (
        <div className="dashboard-card-tags">
          {c.objective_names.map((name) => (
            <span key={name} className="dashboard-card-tag">
              {name}
            </span>
          ))}
        </div>
      )}

      <div className="dashboard-card-actions">
        <Link to={`/workspace/${c.campaign_id}`} className="btn btn-sm btn-primary">
          <ExternalLink size={14} />
          <span>Open</span>
        </Link>

        {c.status !== "archived" ? (
          <button
            className="btn btn-sm btn-secondary"
            onClick={() => onArchive(c.campaign_id)}
            title="Archive campaign"
          >
            <Archive size={14} />
            <span>Archive</span>
          </button>
        ) : (
          <button
            className="btn btn-sm btn-secondary"
            onClick={() => onUnarchive(c.campaign_id)}
            title="Unarchive campaign"
          >
            <ArchiveRestore size={14} />
            <span>Unarchive</span>
          </button>
        )}

        {!confirmDelete ? (
          <button
            className="btn btn-sm btn-danger-outline"
            onClick={() => setConfirmDelete(true)}
            title="Delete campaign"
          >
            <Trash2 size={14} />
          </button>
        ) : (
          <div className="dashboard-card-confirm">
            <span className="dashboard-card-confirm-text">Delete?</span>
            <button
              className="btn btn-sm btn-danger"
              onClick={() => {
                onDelete(c.campaign_id);
                setConfirmDelete(false);
              }}
            >
              Yes
            </button>
            <button
              className="btn btn-sm btn-secondary"
              onClick={() => setConfirmDelete(false)}
            >
              No
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

/* ── Dashboard ── */

export default function Dashboard() {
  const navigate = useNavigate();
  const [allCampaigns, setAllCampaigns] = useState<CampaignSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Search
  const [searchQuery, setSearchQuery] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Selection for batch compare
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());

  const toggleSelect = useCallback((id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  // Filters
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("all");
  const [sortBy, setSortBy] = useState<SortOption>("latest");

  // Load campaigns
  const loadCampaigns = useCallback(async () => {
    try {
      const data = await fetchCampaigns();
      setAllCampaigns(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load campaigns");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    setLoading(true);
    loadCampaigns();
  }, [loadCampaigns]);

  // Debounce search input
  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      setDebouncedSearch(searchQuery);
    }, 300);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [searchQuery]);

  // Filtered and sorted campaigns
  const filteredCampaigns = useMemo(() => {
    let result = [...allCampaigns];

    // Search filter (case-insensitive substring)
    if (debouncedSearch.trim()) {
      const q = debouncedSearch.trim().toLowerCase();
      result = result.filter((c) => c.name.toLowerCase().includes(q));
    }

    // Status filter
    if (statusFilter === "running") {
      result = result.filter((c) => c.status === "running" || c.status === "paused");
    } else if (statusFilter === "completed") {
      result = result.filter((c) => c.status === "completed");
    } else if (statusFilter === "archived") {
      result = result.filter((c) => c.status === "archived");
    } else {
      // "all" hides archived by default
      result = result.filter((c) => c.status !== "archived");
    }

    // Sort
    switch (sortBy) {
      case "latest":
        result.sort((a, b) => b.created_at - a.created_at);
        break;
      case "oldest":
        result.sort((a, b) => a.created_at - b.created_at);
        break;
      case "best_kpi":
        result.sort((a, b) => {
          if (a.best_kpi === null && b.best_kpi === null) return 0;
          if (a.best_kpi === null) return 1;
          if (b.best_kpi === null) return -1;
          return a.best_kpi - b.best_kpi;
        });
        break;
      case "most_trials":
        result.sort((a, b) => b.total_trials - a.total_trials);
        break;
    }

    return result;
  }, [allCampaigns, debouncedSearch, statusFilter, sortBy]);

  // Summary stats (computed from all non-archived campaigns)
  const stats = useMemo(() => {
    const nonArchived = allCampaigns.filter((c) => c.status !== "archived");
    const activeCampaigns = nonArchived.filter(
      (c) => c.status === "running" || c.status === "paused"
    );
    const allKpis = nonArchived
      .map((c) => c.best_kpi)
      .filter((v): v is number => v !== null);
    const bestOverallKpi =
      allKpis.length > 0 ? Math.min(...allKpis) : null;
    const totalTrials = nonArchived.reduce((sum, c) => sum + c.total_trials, 0);

    return {
      total: nonArchived.length,
      active: activeCampaigns.length,
      bestKpi: bestOverallKpi,
      totalTrials,
    };
  }, [allCampaigns]);

  // Has any campaigns at all (not filtered)
  const hasCampaigns = allCampaigns.length > 0;
  // Whether current filter/search produced no results
  const isFilterEmpty = filteredCampaigns.length === 0 && hasCampaigns;

  // Actions
  const handleArchive = async (id: string) => {
    try {
      await updateCampaignStatus(id, "archived");
      await loadCampaigns();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to archive campaign");
    }
  };

  const handleUnarchive = async (id: string) => {
    try {
      await updateCampaignStatus(id, "draft");
      await loadCampaigns();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to unarchive campaign");
    }
  };

  const handleDelete = async (id: string) => {
    try {
      await deleteCampaign(id);
      await loadCampaigns();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete campaign");
    }
  };

  return (
    <ErrorBoundary>
      <div className="page">
        {/* Header */}
        <div className="page-header">
          <h1>Campaigns</h1>
          <button
            className="btn btn-primary"
            onClick={() => navigate("/new-campaign")}
          >
            + New Campaign
          </button>
        </div>

        {error && <div className="error-banner">{error}</div>}

        {loading ? (
          <div className="dashboard-skeleton">
            <div className="dashboard-stats">
              {[1, 2, 3, 4].map((i) => (
                <div key={i} className="stat-card skel-pulse">
                  <div className="skel-line" style={{ width: "60%", height: "12px" }} />
                  <div className="skel-line" style={{ width: "40%", height: "24px", marginTop: "8px" }} />
                </div>
              ))}
            </div>
            <div className="skel-line" style={{ width: "280px", height: "36px", borderRadius: "8px", margin: "16px 0" }} />
            <div className="dashboard-grid">
              {[1, 2, 3, 4, 5, 6].map((i) => (
                <div key={i} className="dashboard-card skel-pulse">
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <div className="skel-line" style={{ width: "65%", height: "16px" }} />
                    <div className="skel-line" style={{ width: "60px", height: "20px", borderRadius: "10px" }} />
                  </div>
                  <div className="skel-line" style={{ width: "100%", height: "48px", borderRadius: "8px" }} />
                  <div className="skel-line" style={{ width: "80%", height: "4px", borderRadius: "2px" }} />
                  <div className="skel-line" style={{ width: "45%", height: "12px" }} />
                </div>
              ))}
            </div>
          </div>
        ) : !hasCampaigns ? (
          <>
            {/* Getting Started */}
            <div className="getting-started">
              <div className="gs-hero">
                <h2 className="gs-hero-title">Welcome to Nexus</h2>
                <p className="gs-hero-subtitle">
                  An intelligent platform that helps scientists discover optimal experimental
                  conditions using Bayesian optimization. Upload your data, and let the AI
                  suggest your next experiments.
                </p>
              </div>

              <div className="gs-steps">
                <div className="gs-step">
                  <div className="gs-step-number">1</div>
                  <div className="gs-step-content">
                    <h3>Upload Data</h3>
                    <p>Upload a CSV with your experimental results — parameters and measured outcomes.</p>
                  </div>
                </div>
                <div className="gs-step-arrow"><ArrowRight size={16} /></div>
                <div className="gs-step">
                  <div className="gs-step-number">2</div>
                  <div className="gs-step-content">
                    <h3>Map Columns</h3>
                    <p>Tell us which columns are parameters (inputs) and which are objectives (outputs).</p>
                  </div>
                </div>
                <div className="gs-step-arrow"><ArrowRight size={16} /></div>
                <div className="gs-step">
                  <div className="gs-step-number">3</div>
                  <div className="gs-step-content">
                    <h3>Get Suggestions</h3>
                    <p>The AI analyzes patterns and suggests the most promising experiments to try next.</p>
                  </div>
                </div>
              </div>

              <div className="gs-actions">
                <button className="btn btn-primary gs-action-btn" onClick={() => navigate("/new-campaign")}>
                  <Upload size={16} />
                  Upload Your Data
                </button>
                <button className="btn btn-secondary gs-action-btn" onClick={() => navigate("/demos")}>
                  <BookOpen size={16} />
                  Explore Demo Datasets
                </button>
              </div>

              <div className="gs-features">
                <div className="gs-feature">
                  <Sparkles size={18} className="gs-feature-icon" />
                  <div>
                    <strong>Explainable Suggestions</strong>
                    <span>Every suggestion comes with a "why" — understand the reasoning behind each experiment.</span>
                  </div>
                </div>
                <div className="gs-feature">
                  <FlaskConical size={18} className="gs-feature-icon" />
                  <div>
                    <strong>Works with Small Data</strong>
                    <span>Start optimizing with as few as 5-10 experiments. No large datasets required.</span>
                  </div>
                </div>
                <div className="gs-feature">
                  <Activity size={18} className="gs-feature-icon" />
                  <div>
                    <strong>Real-time Diagnostics</strong>
                    <span>Monitor convergence, exploration coverage, and model health as your campaign progresses.</span>
                  </div>
                </div>
              </div>
            </div>

            <style>{`
              .getting-started {
                max-width: 720px;
                margin: 0 auto;
                padding: 20px 0 40px;
              }
              .gs-hero {
                text-align: center;
                margin-bottom: 36px;
              }
              .gs-hero-title {
                font-size: 1.5rem;
                font-weight: 700;
                margin: 0 0 10px;
                color: var(--color-text);
              }
              .gs-hero-subtitle {
                font-size: 0.92rem;
                color: var(--color-text-muted);
                line-height: 1.6;
                margin: 0;
                max-width: 540px;
                margin-left: auto;
                margin-right: auto;
              }
              .gs-steps {
                display: flex;
                align-items: flex-start;
                gap: 8px;
                margin-bottom: 32px;
              }
              .gs-step {
                flex: 1;
                display: flex;
                flex-direction: column;
                align-items: center;
                text-align: center;
                gap: 10px;
              }
              .gs-step-number {
                width: 36px;
                height: 36px;
                border-radius: 50%;
                background: var(--color-primary);
                color: white;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 0.88rem;
                font-weight: 700;
              }
              .gs-step-content h3 {
                font-size: 0.9rem;
                font-weight: 600;
                margin: 0 0 4px;
                color: var(--color-text);
              }
              .gs-step-content p {
                font-size: 0.8rem;
                color: var(--color-text-muted);
                line-height: 1.5;
                margin: 0;
              }
              .gs-step-arrow {
                color: var(--color-border);
                margin-top: 10px;
                flex-shrink: 0;
              }
              .gs-actions {
                display: flex;
                gap: 12px;
                justify-content: center;
                margin-bottom: 36px;
              }
              .gs-action-btn {
                display: flex;
                align-items: center;
                gap: 8px;
                padding: 12px 24px;
                font-weight: 600;
              }
              .gs-features {
                display: flex;
                flex-direction: column;
                gap: 14px;
                padding: 20px;
                background: var(--color-surface);
                border: 1px solid var(--color-border);
                border-radius: 12px;
              }
              .gs-feature {
                display: flex;
                align-items: flex-start;
                gap: 12px;
                font-size: 0.85rem;
                line-height: 1.5;
              }
              .gs-feature-icon {
                flex-shrink: 0;
                color: var(--color-primary);
                margin-top: 1px;
              }
              .gs-feature strong {
                display: block;
                font-weight: 600;
                color: var(--color-text);
                margin-bottom: 2px;
              }
              .gs-feature span {
                color: var(--color-text-muted);
              }
              @media (max-width: 640px) {
                .gs-steps {
                  flex-direction: column;
                  align-items: stretch;
                }
                .gs-step {
                  flex-direction: row;
                  text-align: left;
                }
                .gs-step-arrow {
                  display: none;
                }
                .gs-actions {
                  flex-direction: column;
                }
              }
            `}</style>
          </>
        ) : (
          <>
            {/* Stats Summary */}
            <div className="dashboard-stats">
              <StatCard
                label="Total Campaigns"
                value={stats.total}
                icon={LayoutGrid}
                color="var(--color-primary)"
              />
              <StatCard
                label="Active"
                value={stats.active}
                icon={Activity}
                color="var(--color-green)"
              />
              <StatCard
                label="Best KPI"
                value={stats.bestKpi !== null ? stats.bestKpi.toFixed(4) : "--"}
                icon={Trophy}
                color="var(--color-purple)"
              />
              <StatCard
                label="Total Experiments"
                value={stats.totalTrials}
                icon={FlaskConical}
                color="var(--color-blue)"
              />
            </div>

            {/* Search Bar */}
            <div className="dashboard-search">
              <div className="dashboard-search-input-wrapper">
                <Search size={18} className="dashboard-search-icon" />
                <input
                  type="text"
                  className="dashboard-search-input"
                  placeholder="Search campaigns by name..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
              </div>
            </div>

            {/* Filter Controls */}
            <div className="dashboard-filters">
              <div className="dashboard-filter-group">
                {(
                  [
                    ["all", "All"],
                    ["running", "Running"],
                    ["completed", "Completed"],
                    ["archived", "Archived"],
                  ] as [StatusFilter, string][]
                ).map(([value, label]) => (
                  <button
                    key={value}
                    className={`dashboard-filter-chip ${statusFilter === value ? "active" : ""}`}
                    onClick={() => setStatusFilter(value)}
                  >
                    {label}
                  </button>
                ))}
              </div>

              <div className="dashboard-sort">
                <label htmlFor="sort-select" className="dashboard-sort-label">
                  Sort by
                </label>
                <select
                  id="sort-select"
                  className="dashboard-sort-select"
                  value={sortBy}
                  onChange={(e) => setSortBy(e.target.value as SortOption)}
                >
                  <option value="latest">Latest</option>
                  <option value="oldest">Oldest</option>
                  <option value="best_kpi">Best KPI</option>
                  <option value="most_trials">Most Trials</option>
                </select>
              </div>
            </div>

            {/* Campaign Grid or No Results */}
            {isFilterEmpty ? (
              <EmptyState
                icon={Search}
                title="No matching campaigns"
                description={
                  debouncedSearch.trim()
                    ? `No campaigns match "${debouncedSearch.trim()}". Try a different search term or adjust your filters.`
                    : "No campaigns match the selected filters. Try changing the status filter."
                }
              />
            ) : (
              <div className="dashboard-grid">
                {filteredCampaigns.map((c) => (
                  <CampaignCard
                    key={c.campaign_id}
                    campaign={c}
                    onArchive={handleArchive}
                    onUnarchive={handleUnarchive}
                    onDelete={handleDelete}
                    selected={selectedIds.has(c.campaign_id)}
                    onToggleSelect={toggleSelect}
                  />
                ))}
              </div>
            )}
          </>
        )}

        {/* Floating comparison toolbar */}
        {selectedIds.size >= 2 && (
          <div className="compare-toolbar">
            <span className="compare-toolbar-count">
              {selectedIds.size} campaigns selected
            </span>
            <button
              className="btn btn-primary btn-sm"
              onClick={() => {
                const ids = Array.from(selectedIds);
                navigate(`/compare?ids=${ids.join(",")}`);
              }}
            >
              <GitCompareArrows size={14} />
              Compare Selected
            </button>
            <button
              className="compare-toolbar-clear"
              onClick={() => setSelectedIds(new Set())}
              title="Clear selection"
            >
              <X size={14} />
            </button>
          </div>
        )}
      </div>
    </ErrorBoundary>
  );
}
