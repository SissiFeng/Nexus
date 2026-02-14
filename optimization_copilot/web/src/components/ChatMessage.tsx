import { User, Bot, Beaker, AlertCircle, Lightbulb } from "lucide-react";

export interface InsightMeta {
  title: string;
  body: string;
  category: "discovery" | "warning" | "recommendation" | "trend";
  importance: number;
}

export interface CorrelationMeta {
  parameter: string;
  objective: string;
  correlation: number;
  strength: "strong" | "moderate" | "weak";
  direction: "positive" | "negative";
}

export interface TopConditionMeta {
  rank: number;
  parameters: Record<string, unknown>;
  objective_value: number;
  objective_name: string;
}

export interface OptimalRegionMeta {
  parameter: string;
  best_range: [number, number];
  overall_range: [number, number];
  improvement_pct: number;
}

export interface ChatMsg {
  id: string;
  role: "user" | "system" | "agent" | "suggestion";
  content: string;
  timestamp: number;
  metadata?: {
    confidence?: number;
    hypothesis?: string;
    recommendations?: string[];
    suggestions?: Array<Record<string, number>>;
    diagnostics?: Record<string, number>;
    insights?: InsightMeta[];
    correlations?: CorrelationMeta[];
    top_conditions?: TopConditionMeta[];
    optimal_regions?: OptimalRegionMeta[];
    interactions?: Array<{
      param_a: string;
      param_b: string;
      interaction_strength: number;
      description: string;
    }>;
    trends?: Array<{ description: string; metric: string; value: number }>;
    failure_patterns?: Array<{
      description: string;
      parameter: string;
      risky_range: [number, number];
      failure_rate_in_range: number;
    }>;
  };
}

interface ChatMessageProps {
  message: ChatMsg;
}

const CATEGORY_COLORS: Record<
  string,
  { bg: string; border: string; text: string; badge: string }
> = {
  discovery: {
    bg: "#f0f7ff",
    border: "#3b82f6",
    text: "#1e40af",
    badge: "#dbeafe",
  },
  warning: {
    bg: "#fff8f0",
    border: "#f59e0b",
    text: "#92400e",
    badge: "#fef3c7",
  },
  recommendation: {
    bg: "#f0fdf4",
    border: "#22c55e",
    text: "#166534",
    badge: "#dcfce7",
  },
  trend: {
    bg: "#faf5ff",
    border: "#a855f7",
    text: "#6b21a8",
    badge: "#f3e8ff",
  },
};

function InsightCards({ insights }: { insights: InsightMeta[] }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "8px", marginTop: "10px" }}>
      {insights.map((ins, i) => {
        const colors = CATEGORY_COLORS[ins.category] || CATEGORY_COLORS.discovery;
        return (
          <div
            key={i}
            style={{
              background: colors.bg,
              borderLeft: `3px solid ${colors.border}`,
              borderRadius: "6px",
              padding: "8px 10px",
              fontSize: "0.82rem",
            }}
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                marginBottom: "4px",
              }}
            >
              <span style={{ fontWeight: 600, color: colors.text, fontSize: "0.83rem" }}>
                {ins.title}
              </span>
              <span
                style={{
                  background: colors.badge,
                  color: colors.text,
                  padding: "1px 6px",
                  borderRadius: "8px",
                  fontSize: "0.7rem",
                  fontWeight: 500,
                  textTransform: "uppercase",
                }}
              >
                {ins.category}
              </span>
            </div>
            <div style={{ color: "#4b5563", lineHeight: 1.4 }}>{ins.body}</div>
          </div>
        );
      })}
    </div>
  );
}

function CorrelationBars({ correlations }: { correlations: CorrelationMeta[] }) {
  if (!correlations.length) return null;
  return (
    <div style={{ marginTop: "10px" }}>
      <div
        style={{
          fontSize: "0.78rem",
          fontWeight: 600,
          color: "#374151",
          marginBottom: "6px",
        }}
      >
        Parameter-Objective Correlations
      </div>
      {correlations.map((c, i) => {
        const absR = Math.abs(c.correlation);
        const barColor =
          c.strength === "strong"
            ? c.direction === "positive"
              ? "#22c55e"
              : "#ef4444"
            : c.direction === "positive"
              ? "#86efac"
              : "#fca5a5";
        return (
          <div
            key={i}
            style={{
              display: "flex",
              alignItems: "center",
              gap: "6px",
              marginBottom: "4px",
              fontSize: "0.78rem",
            }}
          >
            <span
              style={{
                width: "90px",
                textAlign: "right",
                color: "#6b7280",
                flexShrink: 0,
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
              }}
            >
              {c.parameter}
            </span>
            <div
              style={{
                flex: 1,
                height: "12px",
                background: "#f3f4f6",
                borderRadius: "6px",
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  width: `${absR * 100}%`,
                  height: "100%",
                  background: barColor,
                  borderRadius: "6px",
                  transition: "width 0.3s",
                }}
              />
            </div>
            <span style={{ width: "55px", color: "#374151", fontFamily: "monospace", fontSize: "0.75rem" }}>
              {c.direction === "positive" ? "+" : ""}
              {c.correlation.toFixed(3)}
            </span>
          </div>
        );
      })}
    </div>
  );
}

function TopConditionsTable({ conditions }: { conditions: TopConditionMeta[] }) {
  if (!conditions.length) return null;
  const paramKeys = Object.keys(conditions[0].parameters).slice(0, 4);
  return (
    <div style={{ marginTop: "10px" }}>
      <div
        style={{
          fontSize: "0.78rem",
          fontWeight: 600,
          color: "#374151",
          marginBottom: "6px",
        }}
      >
        Top Performing Conditions
      </div>
      <div style={{ overflowX: "auto" }}>
        <table
          style={{
            width: "100%",
            fontSize: "0.75rem",
            borderCollapse: "collapse",
            fontFamily: "monospace",
          }}
        >
          <thead>
            <tr style={{ borderBottom: "1px solid #e5e7eb" }}>
              <th style={{ padding: "3px 6px", textAlign: "left", color: "#6b7280", fontWeight: 500 }}>#</th>
              {paramKeys.map((k) => (
                <th key={k} style={{ padding: "3px 6px", textAlign: "right", color: "#6b7280", fontWeight: 500 }}>
                  {k}
                </th>
              ))}
              <th style={{ padding: "3px 6px", textAlign: "right", color: "#6b7280", fontWeight: 500 }}>
                {conditions[0].objective_name}
              </th>
            </tr>
          </thead>
          <tbody>
            {conditions.map((tc) => (
              <tr
                key={tc.rank}
                style={{
                  borderBottom: "1px solid #f3f4f6",
                  background: tc.rank === 1 ? "#f0fdf4" : "transparent",
                }}
              >
                <td style={{ padding: "3px 6px", fontWeight: tc.rank === 1 ? 600 : 400 }}>
                  {tc.rank}
                </td>
                {paramKeys.map((k) => {
                  const v = tc.parameters[k];
                  return (
                    <td key={k} style={{ padding: "3px 6px", textAlign: "right" }}>
                      {typeof v === "number" ? v.toPrecision(3) : String(v).slice(0, 12)}
                    </td>
                  );
                })}
                <td
                  style={{
                    padding: "3px 6px",
                    textAlign: "right",
                    fontWeight: 600,
                    color: "#1e40af",
                  }}
                >
                  {tc.objective_value.toPrecision(4)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function OptimalRegions({ regions }: { regions: OptimalRegionMeta[] }) {
  if (!regions.length) return null;
  return (
    <div style={{ marginTop: "10px" }}>
      <div
        style={{
          fontSize: "0.78rem",
          fontWeight: 600,
          color: "#374151",
          marginBottom: "6px",
        }}
      >
        Optimal Parameter Ranges
      </div>
      {regions.map((r, i) => {
        const overall = r.overall_range[1] - r.overall_range[0];
        const leftPct = overall > 0 ? ((r.best_range[0] - r.overall_range[0]) / overall) * 100 : 0;
        const widthPct = overall > 0 ? ((r.best_range[1] - r.best_range[0]) / overall) * 100 : 100;
        const impColor = r.improvement_pct > 0 ? "#166534" : "#92400e";
        return (
          <div key={i} style={{ marginBottom: "8px" }}>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                fontSize: "0.78rem",
                marginBottom: "2px",
              }}
            >
              <span style={{ color: "#374151", fontWeight: 500 }}>{r.parameter}</span>
              <span style={{ color: impColor, fontWeight: 600, fontSize: "0.75rem" }}>
                {r.improvement_pct > 0 ? "+" : ""}
                {r.improvement_pct.toFixed(1)}%
              </span>
            </div>
            <div
              style={{
                height: "10px",
                background: "#f3f4f6",
                borderRadius: "5px",
                position: "relative",
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  position: "absolute",
                  left: `${leftPct}%`,
                  width: `${Math.max(widthPct, 3)}%`,
                  height: "100%",
                  background: "linear-gradient(90deg, #3b82f6, #60a5fa)",
                  borderRadius: "5px",
                }}
              />
            </div>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                fontSize: "0.68rem",
                color: "#9ca3af",
                marginTop: "1px",
              }}
            >
              <span>{r.overall_range[0].toPrecision(3)}</span>
              <span style={{ color: "#3b82f6", fontWeight: 500 }}>
                [{r.best_range[0].toPrecision(3)}, {r.best_range[1].toPrecision(3)}]
              </span>
              <span>{r.overall_range[1].toPrecision(3)}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

export function ChatMessage({ message }: ChatMessageProps) {
  const formatTime = (timestamp: number) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  if (message.role === "system") {
    return (
      <div className="chat-message-system">
        <span className="chat-system-text">{message.content}</span>
        <span className="chat-timestamp">{formatTime(message.timestamp)}</span>
      </div>
    );
  }

  if (message.role === "user") {
    return (
      <div className="chat-message-wrapper chat-message-user-wrapper">
        <div className="chat-message-bubble chat-message-user">
          <div className="chat-message-header">
            <User size={14} className="chat-message-icon" />
            <span className="chat-timestamp">{formatTime(message.timestamp)}</span>
          </div>
          <div className="chat-message-content">{message.content}</div>
        </div>
      </div>
    );
  }

  if (message.role === "agent") {
    const hasInsights = message.metadata?.insights && message.metadata.insights.length > 0;

    return (
      <div className="chat-message-wrapper chat-message-agent-wrapper">
        <div className="chat-message-bubble chat-message-agent">
          <div className="chat-message-header">
            {hasInsights ? (
              <Lightbulb size={14} className="chat-message-icon" style={{ color: "#f59e0b" }} />
            ) : (
              <Bot size={14} className="chat-message-icon" />
            )}
            <span className="chat-timestamp">{formatTime(message.timestamp)}</span>
          </div>
          <div className="chat-message-content">{message.content}</div>

          {/* Rich insight rendering */}
          {hasInsights && (
            <InsightCards insights={message.metadata!.insights!} />
          )}

          {message.metadata?.correlations && message.metadata.correlations.length > 0 && (
            <CorrelationBars correlations={message.metadata.correlations} />
          )}

          {message.metadata?.top_conditions && message.metadata.top_conditions.length > 0 && (
            <TopConditionsTable conditions={message.metadata.top_conditions} />
          )}

          {message.metadata?.optimal_regions && message.metadata.optimal_regions.length > 0 && (
            <OptimalRegions regions={message.metadata.optimal_regions} />
          )}

          {message.metadata?.interactions && message.metadata.interactions.length > 0 && (
            <div style={{ marginTop: "10px" }}>
              <div style={{ fontSize: "0.78rem", fontWeight: 600, color: "#374151", marginBottom: "4px" }}>
                Parameter Interactions
              </div>
              {message.metadata.interactions.map((ix, i) => (
                <div
                  key={i}
                  style={{
                    fontSize: "0.78rem",
                    color: "#4b5563",
                    padding: "3px 0",
                    borderBottom: "1px solid #f3f4f6",
                  }}
                >
                  <span style={{ fontWeight: 500 }}>
                    {ix.param_a} × {ix.param_b}
                  </span>{" "}
                  <span style={{ fontFamily: "monospace", color: "#6b7280" }}>
                    (r={ix.interaction_strength.toFixed(3)})
                  </span>
                </div>
              ))}
            </div>
          )}

          {message.metadata?.failure_patterns && message.metadata.failure_patterns.length > 0 && (
            <div style={{ marginTop: "10px" }}>
              <div style={{ fontSize: "0.78rem", fontWeight: 600, color: "#92400e", marginBottom: "4px" }}>
                Risk Zones
              </div>
              {message.metadata.failure_patterns.map((fp, i) => (
                <div
                  key={i}
                  style={{
                    fontSize: "0.78rem",
                    color: "#92400e",
                    background: "#fff8f0",
                    padding: "4px 8px",
                    borderRadius: "4px",
                    marginBottom: "4px",
                  }}
                >
                  {fp.description}
                </div>
              ))}
            </div>
          )}

          {/* Non-insight metadata */}
          {message.metadata?.hypothesis && (
            <div className="chat-hypothesis-callout">
              <div className="chat-callout-label">
                <Beaker size={12} />
                Hypothesis
              </div>
              <div className="chat-callout-text">
                {message.metadata.hypothesis}
              </div>
            </div>
          )}

          {message.metadata?.recommendations &&
            message.metadata.recommendations.length > 0 && (
              <div className="chat-recommendations">
                <div className="chat-recommendations-label">
                  Recommendations:
                </div>
                <ol className="chat-recommendations-list">
                  {message.metadata.recommendations.map((rec, idx) => (
                    <li key={idx}>{rec}</li>
                  ))}
                </ol>
              </div>
            )}

          {message.metadata?.confidence !== undefined && (
            <div className="chat-confidence">
              <div className="chat-confidence-label">
                Confidence: {Math.round(message.metadata.confidence * 100)}%
              </div>
              <div className="chat-confidence-bar-bg">
                <div
                  className="chat-confidence-bar-fill"
                  style={{
                    width: `${message.metadata.confidence * 100}%`,
                  }}
                />
              </div>
            </div>
          )}
        </div>
      </div>
    );
  }

  if (message.role === "suggestion") {
    return (
      <div className="chat-message-wrapper chat-message-agent-wrapper">
        <div className="chat-message-bubble chat-message-suggestion">
          <div className="chat-message-header">
            <AlertCircle size={14} className="chat-message-icon" />
            <span className="chat-timestamp">{formatTime(message.timestamp)}</span>
          </div>
          <div className="chat-message-content">{message.content}</div>

          {message.metadata?.suggestions &&
            message.metadata.suggestions.length > 0 && (
              <div className="chat-suggestions-table-wrapper">
                <table className="chat-suggestions-table">
                  <thead>
                    <tr>
                      <th>#</th>
                      {Object.keys(message.metadata.suggestions[0]).map(
                        (key) => (
                          <th key={key}>{key}</th>
                        )
                      )}
                    </tr>
                  </thead>
                  <tbody>
                    {message.metadata.suggestions.map((suggestion, idx) => (
                      <tr key={idx}>
                        <td className="chat-suggestion-rank">{idx + 1}</td>
                        {Object.entries(suggestion).map(([key, value]) => {
                          const isHighlighted =
                            idx > 0 &&
                            message.metadata?.suggestions?.[0][key] !== value;
                          return (
                            <td
                              key={key}
                              className={
                                isHighlighted
                                  ? "chat-suggestion-highlighted"
                                  : ""
                              }
                            >
                              {typeof value === "number"
                                ? value.toFixed(3)
                                : value}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

          {message.metadata?.diagnostics && (
            <div className="chat-diagnostics">
              <div className="chat-diagnostics-label">Diagnostics:</div>
              <div className="chat-diagnostics-grid">
                {Object.entries(message.metadata.diagnostics).map(
                  ([key, value]) => (
                    <div key={key} className="chat-diagnostic-item">
                      <span className="chat-diagnostic-key">{key}:</span>
                      <span className="chat-diagnostic-value">
                        {typeof value === "number" ? value.toFixed(3) : value}
                      </span>
                    </div>
                  )
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    );
  }

  return null;
}
