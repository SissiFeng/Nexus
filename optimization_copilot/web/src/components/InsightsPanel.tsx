import React, { useEffect, useState, useCallback } from "react";
import { fetchInsights, type InsightsData, type InsightSummary, type CorrelationInsight, type OptimalRegion, type TopCondition } from "../api";

interface InsightsPanelProps {
  campaignId: string;
}

const CATEGORY_STYLES: Record<InsightSummary["category"], { bg: string; border: string; text: string; label: string }> = {
  discovery: { bg: "#e3f2fd", border: "#1565c0", text: "#1565c0", label: "Discovery" },
  warning: { bg: "#fff3e0", border: "#e65100", text: "#e65100", label: "Warning" },
  recommendation: { bg: "#e8f5e9", border: "#2e7d32", text: "#2e7d32", label: "Recommendation" },
  trend: { bg: "#f3e5f5", border: "#6a1b9a", text: "#6a1b9a", label: "Trend" },
};

const STRENGTH_COLORS: Record<string, string> = { strong: "#1565c0", moderate: "#f59e0b", weak: "#94a3b8" };

const S: Record<string, React.CSSProperties> = {
  container: { padding: "1rem" },
  header: { display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1.25rem" },
  heading: { fontSize: "1.25rem", fontWeight: 600, margin: 0 },
  meta: { fontSize: "0.8rem", color: "#718096" },
  btn: { padding: "6px 14px", fontSize: "0.85rem", fontWeight: 500, background: "#f1f5f9", border: "1px solid #e2e8f0", borderRadius: "6px", cursor: "pointer", color: "#334155" },
  section: { marginBottom: "1.5rem" },
  secTitle: { fontSize: "1rem", fontWeight: 600, marginBottom: "0.75rem", color: "#1e293b", borderBottom: "1px solid #e2e8f0", paddingBottom: "0.4rem" },
  grid: { display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: "12px" },
  empty: { fontSize: "0.85rem", color: "#94a3b8", fontStyle: "italic", padding: "0.5rem 0" },
  tblWrap: { overflowX: "auto" as const },
  tbl: { width: "100%", borderCollapse: "collapse" as const, fontSize: "0.85rem" },
  th: { textAlign: "left" as const, padding: "8px 12px", fontWeight: 600, borderBottom: "2px solid #e2e8f0", color: "#475569", fontSize: "0.8rem", textTransform: "uppercase" as const, letterSpacing: "0.03em" },
  td: { padding: "8px 12px", borderBottom: "1px solid #f1f5f9" },
  mono: { fontFamily: "monospace", fontSize: "0.85rem" },
  row: { display: "flex", justifyContent: "space-between", fontSize: "0.85rem", marginBottom: "4px" },
  bar: { height: "6px", background: "#f1f5f9", borderRadius: "3px", overflow: "hidden" },
};

function SummaryCard({ summary }: { summary: InsightSummary }) {
  const c = CATEGORY_STYLES[summary.category];
  return (
    <div style={{ background: c.bg, borderLeft: `4px solid ${c.border}`, borderRadius: "6px", padding: "12px 14px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "6px" }}>
        <span style={{ fontWeight: 600, fontSize: "0.9rem", color: c.text }}>{summary.title}</span>
        <span style={{ fontSize: "0.7rem", fontWeight: 600, textTransform: "uppercase", color: c.text, opacity: 0.8, flexShrink: 0, marginLeft: "8px" }}>{c.label}</span>
      </div>
      <div style={{ fontSize: "0.85rem", color: "#334155", lineHeight: 1.5 }}>{summary.body}</div>
    </div>
  );
}

function TopConditionsTable({ conditions }: { conditions: TopCondition[] }) {
  if (conditions.length === 0) return <div style={S.empty}>No top conditions available yet.</div>;
  const paramKeys = Array.from(new Set(conditions.flatMap((c) => Object.keys(c.parameters))));
  return (
    <div style={S.tblWrap}>
      <table style={S.tbl}>
        <thead>
          <tr>
            <th style={S.th}>Rank</th>
            {paramKeys.map((k) => <th key={k} style={S.th}>{k}</th>)}
            <th style={S.th}>Objective</th>
            <th style={S.th}>Value</th>
          </tr>
        </thead>
        <tbody>
          {conditions.map((c) => (
            <tr key={c.rank}>
              <td style={S.td}>#{c.rank}</td>
              {paramKeys.map((k) => (
                <td key={k} style={{ ...S.td, ...S.mono }}>
                  {c.parameters[k] != null ? (typeof c.parameters[k] === "number" ? (c.parameters[k] as number).toPrecision(4) : String(c.parameters[k])) : "-"}
                </td>
              ))}
              <td style={S.td}>{c.objective_name}</td>
              <td style={{ ...S.td, ...S.mono, fontWeight: 600 }}>{c.objective_value.toPrecision(4)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function CorrelationItem({ c }: { c: CorrelationInsight }) {
  const color = STRENGTH_COLORS[c.strength] || "#94a3b8";
  const arrow = c.direction === "positive" ? "\u2191" : "\u2193";
  return (
    <div style={{ marginBottom: "10px" }}>
      <div style={S.row}>
        <span><span style={{ fontWeight: 500 }}>{c.parameter}</span><span style={{ color: "#718096" }}> vs </span><span style={{ fontWeight: 500 }}>{c.objective}</span></span>
        <span style={{ color, fontWeight: 600 }}>{arrow} {c.correlation.toFixed(3)} ({c.strength})</span>
      </div>
      <div style={S.bar}>
        <div style={{ width: `${Math.abs(c.correlation) * 100}%`, height: "100%", background: color, borderRadius: "3px", transition: "width 0.3s ease" }} />
      </div>
    </div>
  );
}

function OptimalRegionBar({ region }: { region: OptimalRegion }) {
  const [oLow, oHigh] = region.overall_range;
  const [bLow, bHigh] = region.best_range;
  const span = oHigh - oLow;
  if (span <= 0) return null;
  const leftPct = ((bLow - oLow) / span) * 100;
  const widthPct = ((bHigh - bLow) / span) * 100;
  return (
    <div style={{ marginBottom: "14px" }}>
      <div style={S.row}>
        <span style={{ fontWeight: 500 }}>{region.parameter}</span>
        <span style={{ color: "#2e7d32", fontWeight: 600, fontSize: "0.8rem" }}>+{region.improvement_pct.toFixed(1)}% improvement</span>
      </div>
      <div style={{ position: "relative", height: "16px", background: "#f1f5f9", borderRadius: "4px", overflow: "hidden" }}>
        <div style={{ position: "absolute", left: `${leftPct}%`, width: `${Math.max(widthPct, 2)}%`, height: "100%", background: "linear-gradient(90deg, #3b82f6, #2563eb)", borderRadius: "3px", transition: "all 0.3s ease" }} />
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.75rem", color: "#94a3b8", marginTop: "2px" }}>
        <span>{oLow.toPrecision(3)}</span>
        <span style={{ color: "#3b82f6", fontWeight: 500 }}>Best: [{bLow.toPrecision(3)}, {bHigh.toPrecision(3)}]</span>
        <span>{oHigh.toPrecision(3)}</span>
      </div>
    </div>
  );
}

/* ── Main component ── */

export default function InsightsPanel({ campaignId }: InsightsPanelProps) {
  const [data, setData] = useState<InsightsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    fetchInsights(campaignId)
      .then((d) => setData(d))
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to fetch insights"))
      .finally(() => setLoading(false));
  }, [campaignId]);

  useEffect(() => { load(); }, [load]);

  if (loading) return <div className="loading">Loading insights...</div>;
  if (error) {
    return (
      <div style={S.container}>
        <div className="error-banner">{error}</div>
        <button style={{ ...S.btn, marginTop: "8px" }} onClick={load}>Retry</button>
      </div>
    );
  }
  if (!data) return null;

  const { summaries, top_conditions, correlations, optimal_regions, interactions, trends, failure_patterns } = data;

  return (
    <div style={S.container}>
      {/* Header */}
      <div style={S.header}>
        <div>
          <h2 style={S.heading}>Insights</h2>
          <div style={S.meta}>
            {data.n_observations} observations &middot; {data.n_parameters} parameters &middot; {data.n_objectives} objective{data.n_objectives !== 1 ? "s" : ""}
          </div>
        </div>
        <button style={S.btn} onClick={load}>Refresh</button>
      </div>

      {/* Summary cards */}
      {summaries.length > 0 && (
        <div style={S.section}>
          <div style={S.secTitle}>Summary</div>
          <div style={S.grid}>
            {[...summaries].sort((a, b) => b.importance - a.importance).map((s, i) => <SummaryCard key={i} summary={s} />)}
          </div>
        </div>
      )}

      {/* Top conditions */}
      <div style={S.section}>
        <div style={S.secTitle}>Top Conditions</div>
        <TopConditionsTable conditions={top_conditions} />
      </div>

      {/* Correlations */}
      <div style={S.section}>
        <div style={S.secTitle}>Correlations</div>
        {correlations.length === 0
          ? <div style={S.empty}>Not enough data for correlation analysis.</div>
          : correlations.map((c, i) => <CorrelationItem key={i} c={c} />)}
      </div>

      {/* Optimal regions */}
      <div style={S.section}>
        <div style={S.secTitle}>Optimal Regions</div>
        {optimal_regions.length === 0
          ? <div style={S.empty}>Not enough data to identify optimal regions.</div>
          : optimal_regions.map((r, i) => <OptimalRegionBar key={i} region={r} />)}
      </div>

      {/* Interactions */}
      {interactions.length > 0 && (
        <div style={S.section}>
          <div style={S.secTitle}>Parameter Interactions</div>
          <div style={S.tblWrap}>
            <table style={S.tbl}>
              <thead>
                <tr>
                  <th style={S.th}>Parameter A</th>
                  <th style={S.th}>Parameter B</th>
                  <th style={S.th}>Strength</th>
                  <th style={S.th}>Description</th>
                </tr>
              </thead>
              <tbody>
                {interactions.map((ix, i) => (
                  <tr key={i}>
                    <td style={{ ...S.td, fontWeight: 500 }}>{ix.param_a}</td>
                    <td style={{ ...S.td, fontWeight: 500 }}>{ix.param_b}</td>
                    <td style={{ ...S.td, ...S.mono }}>{ix.interaction_strength.toFixed(3)}</td>
                    <td style={{ ...S.td, color: "#475569" }}>{ix.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Failure patterns */}
      {failure_patterns.length > 0 && (
        <div style={S.section}>
          <div style={S.secTitle}>Failure Patterns</div>
          {failure_patterns.map((fp, i) => (
            <div key={i} style={{ background: "#fff3e0", borderLeft: "4px solid #e65100", borderRadius: "6px", padding: "10px 14px", marginBottom: "8px", fontSize: "0.85rem" }}>
              <div style={{ fontWeight: 500, marginBottom: "4px" }}>{fp.description}</div>
              <div style={{ color: "#718096" }}>
                <span style={S.mono}>{fp.parameter}</span> risky range: [{fp.risky_range[0].toPrecision(3)}, {fp.risky_range[1].toPrecision(3)}]
                &mdash; failure rate {(fp.failure_rate_in_range * 100).toFixed(1)}% vs overall {(fp.overall_failure_rate * 100).toFixed(1)}%
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Trends */}
      {trends.length > 0 && (
        <div style={S.section}>
          <div style={S.secTitle}>Trends</div>
          {trends.map((t, i) => (
            <div key={i} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "8px 0", borderBottom: i < trends.length - 1 ? "1px solid #f1f5f9" : "none", fontSize: "0.85rem" }}>
              <span>{t.description}</span>
              <span style={{ ...S.mono, color: "#6a1b9a", fontWeight: 600 }}>{t.metric}: {typeof t.value === "number" ? t.value.toPrecision(4) : t.value}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
