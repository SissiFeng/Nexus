import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import {
  Beaker,
  FlaskConical,
  Atom,
  Zap,
  Droplets,
  Microscope,
  TestTube,
  Loader2,
  ArrowRight,
  Database,
  SlidersHorizontal,
  Target,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import {
  fetchDemoDatasets,
  fetchDemoDataset,
  createCampaignFromUpload,
} from "../api";
import type { DemoDatasetSummary } from "../api";

/* ── Icon map keyed by dataset id ── */
const ICON_MAP: Record<string, LucideIcon> = {
  oer_catalyst: Beaker,
  suzuki_miyaura: FlaskConical,
  hplc_separation: Droplets,
  additives: TestTube,
  c2_yield: Zap,
  bh_reaction: Atom,
  vapor_diffusion: Microscope,
};

/* ── Tag color palette ── */
const TAG_COLORS: Record<string, { bg: string; fg: string }> = {
  Chemistry: { bg: "#dbeafe", fg: "#1e40af" },
  Catalysis: { bg: "#fef9c3", fg: "#854d0e" },
  Materials: { bg: "#dcfce7", fg: "#166534" },
  Organic: { bg: "#fce7f3", fg: "#9d174d" },
  "Cross-coupling": { bg: "#ede9fe", fg: "#5b21b6" },
  Analytical: { bg: "#e0f2fe", fg: "#075985" },
  Chromatography: { bg: "#ccfbf1", fg: "#115e59" },
  Screening: { bg: "#fff7ed", fg: "#9a3412" },
  "Gas-phase": { bg: "#f3e8ff", fg: "#6b21a8" },
  Crystallography: { bg: "#fef3c7", fg: "#92400e" },
};

const DEFAULT_TAG_COLOR = { bg: "#f1f5f9", fg: "#475569" };

export default function DemoGallery() {
  const navigate = useNavigate();
  const [datasets, setDatasets] = useState<DemoDatasetSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [loadingId, setLoadingId] = useState<string | null>(null);

  useEffect(() => {
    fetchDemoDatasets()
      .then(setDatasets)
      .catch((err) => setError(err.message || "Failed to load demo datasets"))
      .finally(() => setLoading(false));
  }, []);

  const handleTryDataset = async (datasetId: string, datasetName: string) => {
    setLoadingId(datasetId);
    setError(null);

    try {
      // Fetch the dataset with up to 50 rows
      const detail = await fetchDemoDataset(datasetId, 50);

      // Create a campaign directly from the demo data
      const result = await createCampaignFromUpload({
        name: `Demo: ${datasetName}`,
        description: detail.description,
        data: detail.data,
        mapping: detail.mapping,
        batch_size: 5,
        exploration_weight: 0.5,
      });

      const campaignId = result.campaign_id;
      navigate(`/workspace/${campaignId}`);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to create campaign from demo dataset"
      );
    } finally {
      setLoadingId(null);
    }
  };

  if (loading) {
    return (
      <div className="page">
        <div className="loading">Loading demo datasets...</div>
      </div>
    );
  }

  return (
    <div className="page">
      <div className="demo-gallery-header">
        <div>
          <h1>Demo Dataset Gallery</h1>
          <p className="demo-gallery-subtitle">
            Explore real scientific optimization datasets. Click "Try This Dataset" to
            instantly create a campaign and start optimizing.
          </p>
        </div>
      </div>

      {error && <div className="error-banner">{error}</div>}

      <div className="demo-grid">
        {datasets.map((ds) => {
          const Icon = ICON_MAP[ds.id] || Beaker;
          const isLoading = loadingId === ds.id;

          return (
            <div key={ds.id} className="demo-card">
              <div className="demo-card-icon-row">
                <div className="demo-card-icon">
                  <Icon size={28} />
                </div>
                <div className="demo-card-tags">
                  {ds.tags.map((tag) => {
                    const color = TAG_COLORS[tag] || DEFAULT_TAG_COLOR;
                    return (
                      <span
                        key={tag}
                        className="demo-tag"
                        style={{ background: color.bg, color: color.fg }}
                      >
                        {tag}
                      </span>
                    );
                  })}
                </div>
              </div>

              <h3 className="demo-card-name">{ds.name}</h3>
              <p className="demo-card-desc">{ds.description}</p>

              <div className="demo-card-stats">
                <div className="demo-stat">
                  <Database size={14} />
                  <span className="demo-stat-value">{ds.row_count.toLocaleString()}</span>
                  <span className="demo-stat-label">rows</span>
                </div>
                <div className="demo-stat">
                  <SlidersHorizontal size={14} />
                  <span className="demo-stat-value">{ds.n_parameters}</span>
                  <span className="demo-stat-label">params</span>
                </div>
                <div className="demo-stat">
                  <Target size={14} />
                  <span className="demo-stat-value">{ds.n_objectives}</span>
                  <span className="demo-stat-label">{ds.n_objectives === 1 ? "objective" : "objectives"}</span>
                </div>
              </div>

              <button
                className="btn btn-primary demo-card-btn"
                onClick={() => handleTryDataset(ds.id, ds.name)}
                disabled={loadingId !== null}
              >
                {isLoading ? (
                  <>
                    <Loader2 size={16} className="demo-spinner" />
                    Creating...
                  </>
                ) : (
                  <>
                    Try This Dataset
                    <ArrowRight size={16} />
                  </>
                )}
              </button>
            </div>
          );
        })}
      </div>

      <style>{`
        .demo-gallery-header {
          margin-bottom: 28px;
        }

        .demo-gallery-header h1 {
          font-size: 1.5rem;
          font-weight: 700;
          margin-bottom: 6px;
        }

        .demo-gallery-subtitle {
          color: var(--color-text-muted);
          font-size: 0.95rem;
          line-height: 1.5;
        }

        .demo-grid {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 20px;
        }

        .demo-card {
          background: var(--color-surface);
          border: 1px solid var(--color-border);
          border-radius: var(--radius);
          padding: 24px;
          display: flex;
          flex-direction: column;
          box-shadow: var(--shadow-sm);
          transition: box-shadow 0.2s ease, transform 0.2s ease;
        }

        .demo-card:hover {
          box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
          transform: translateY(-2px);
        }

        .demo-card-icon-row {
          display: flex;
          align-items: flex-start;
          justify-content: space-between;
          margin-bottom: 16px;
          gap: 12px;
        }

        .demo-card-icon {
          width: 48px;
          height: 48px;
          border-radius: 12px;
          background: linear-gradient(135deg, #eff6ff, #dbeafe);
          display: flex;
          align-items: center;
          justify-content: center;
          color: var(--color-primary);
          flex-shrink: 0;
        }

        .demo-card-tags {
          display: flex;
          flex-wrap: wrap;
          gap: 4px;
          justify-content: flex-end;
        }

        .demo-tag {
          display: inline-block;
          padding: 2px 8px;
          border-radius: 10px;
          font-size: 0.72rem;
          font-weight: 600;
          white-space: nowrap;
          letter-spacing: 0.02em;
        }

        .demo-card-name {
          font-size: 1.05rem;
          font-weight: 700;
          margin-bottom: 8px;
          color: var(--color-text);
        }

        .demo-card-desc {
          font-size: 0.85rem;
          color: var(--color-text-muted);
          line-height: 1.5;
          margin-bottom: 16px;
          flex: 1;
        }

        .demo-card-stats {
          display: flex;
          gap: 16px;
          padding: 12px 0;
          border-top: 1px solid var(--color-border);
          border-bottom: 1px solid var(--color-border);
          margin-bottom: 16px;
        }

        .demo-stat {
          display: flex;
          align-items: center;
          gap: 4px;
          color: var(--color-text-muted);
        }

        .demo-stat-value {
          font-weight: 700;
          font-size: 0.88rem;
          color: var(--color-text);
          font-family: var(--font-mono);
        }

        .demo-stat-label {
          font-size: 0.78rem;
        }

        .demo-card-btn {
          width: 100%;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 8px;
          padding: 10px 16px;
          font-weight: 600;
        }

        .demo-spinner {
          animation: spin 1s linear infinite;
        }

        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }

        /* Responsive */
        @media (max-width: 1024px) {
          .demo-grid {
            grid-template-columns: repeat(2, 1fr);
          }
        }

        @media (max-width: 640px) {
          .demo-grid {
            grid-template-columns: 1fr;
          }

          .demo-card-stats {
            flex-wrap: wrap;
            gap: 10px;
          }
        }
      `}</style>
    </div>
  );
}
