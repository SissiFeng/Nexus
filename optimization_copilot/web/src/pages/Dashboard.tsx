import { useEffect, useState, useCallback, useMemo, useRef } from "react";
import { Link, useNavigate } from "react-router-dom";
import type { LucideIcon } from "lucide-react";
import {
  Beaker,
  Search,
  Archive,
  ArchiveRestore,
  Trash2,
  ExternalLink,
  Activity,
  Trophy,
  FlaskConical,
  LayoutGrid,
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
}: {
  campaign: CampaignSummary;
  onArchive: (id: string) => void;
  onUnarchive: (id: string) => void;
  onDelete: (id: string) => void;
}) {
  const [confirmDelete, setConfirmDelete] = useState(false);
  const c = campaign;

  return (
    <div className={`dashboard-card ${c.status === "archived" ? "dashboard-card-archived" : ""}`}>
      <div className="dashboard-card-top">
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

      <div className="dashboard-card-progress">
        {c.total_trials > 0
          ? `Trial ${c.iteration}/${c.total_trials}`
          : `Iteration ${c.iteration}`}
      </div>

      <div className="dashboard-card-meta">
        <span className="dashboard-card-date">{formatRelativeTime(c.created_at)}</span>
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
          <div className="loading">Loading campaigns...</div>
        ) : !hasCampaigns ? (
          <EmptyState
            icon={Beaker}
            title="No campaigns yet"
            description="Create your first optimization campaign by uploading experimental data. The AI agent will help you discover patterns and suggest better experiments."
            actionLabel="+ Create First Campaign"
            onAction={() => navigate("/new-campaign")}
          />
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
                  />
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </ErrorBoundary>
  );
}
