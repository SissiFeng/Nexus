import { useMemo } from "react";

interface ParamDef {
  name: string;
  type: "continuous" | "categorical";
  lower?: number;
  upper?: number;
}

interface ObjDef {
  name: string;
  direction: "minimize" | "maximize";
}

interface DataQualityReportProps {
  data: Array<Record<string, string>>;
  parameters: ParamDef[];
  objectives: ObjDef[];
}

interface ColumnQuality {
  name: string;
  role: "parameter" | "objective";
  missingCount: number;
  missingPct: number;
  cv: number | null;
  outlierCount: number;
  outlierPct: number;
  isNumeric: boolean;
}

interface CorrelationWarning {
  paramA: string;
  paramB: string;
  r: number;
}

interface QualityReport {
  overallScore: number;
  sampleSize: number;
  sampleSizeRating: "red" | "yellow" | "green";
  sampleSizeMessage: string;
  columns: ColumnQuality[];
  correlationWarnings: CorrelationWarning[];
  completenessScore: number;
  varianceScore: number;
  outlierScore: number;
  sampleSizeScore: number;
}

function parseNumericColumn(
  data: Array<Record<string, string>>,
  col: string
): number[] {
  return data
    .map((row) => {
      const val = row[col];
      if (val === undefined || val === null || val === "") return NaN;
      return parseFloat(val);
    })
    .filter((v) => !isNaN(v));
}

function mean(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function std(values: number[]): number {
  if (values.length < 2) return 0;
  const m = mean(values);
  const variance = values.reduce((sum, v) => sum + (v - m) ** 2, 0) / (values.length - 1);
  return Math.sqrt(variance);
}

function pearsonCorrelation(xs: number[], ys: number[]): number | null {
  if (xs.length !== ys.length || xs.length < 3) return null;
  const n = xs.length;
  const mx = mean(xs);
  const my = mean(ys);
  let num = 0;
  let denX = 0;
  let denY = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx;
    const dy = ys[i] - my;
    num += dx * dy;
    denX += dx * dx;
    denY += dy * dy;
  }
  const den = Math.sqrt(denX * denY);
  if (den === 0) return null;
  return num / den;
}

function computeReport(
  data: Array<Record<string, string>>,
  parameters: ParamDef[],
  objectives: ObjDef[]
): QualityReport {
  const sampleSize = data.length;

  // Sample size rating
  let sampleSizeRating: "red" | "yellow" | "green";
  let sampleSizeMessage: string;
  if (sampleSize < 10) {
    sampleSizeRating = "red";
    sampleSizeMessage = "Very small -- suggestions will be exploratory";
  } else if (sampleSize <= 50) {
    sampleSizeRating = "yellow";
    sampleSizeMessage = "Moderate -- model learning";
  } else {
    sampleSizeRating = "green";
    sampleSizeMessage = "Good -- reliable analysis";
  }

  // Analyze each column
  const allColumns: { name: string; role: "parameter" | "objective"; isNumeric: boolean }[] = [
    ...parameters.map((p) => ({
      name: p.name,
      role: "parameter" as const,
      isNumeric: p.type === "continuous",
    })),
    ...objectives.map((o) => ({
      name: o.name,
      role: "objective" as const,
      isNumeric: true,
    })),
  ];

  const columns: ColumnQuality[] = allColumns.map((col) => {
    const missingCount = data.filter((row) => {
      const val = row[col.name];
      return val === undefined || val === null || val === "";
    }).length;
    const missingPct = sampleSize > 0 ? (missingCount / sampleSize) * 100 : 0;

    let cv: number | null = null;
    let outlierCount = 0;
    let outlierPct = 0;

    if (col.isNumeric) {
      const numericValues = parseNumericColumn(data, col.name);
      if (numericValues.length > 1) {
        const m = mean(numericValues);
        const s = std(numericValues);
        cv = m !== 0 ? s / Math.abs(m) : null;

        // Outlier detection: beyond mean +/- 3*std
        if (s > 0) {
          outlierCount = numericValues.filter(
            (v) => Math.abs(v - m) > 3 * s
          ).length;
          outlierPct =
            numericValues.length > 0
              ? (outlierCount / numericValues.length) * 100
              : 0;
        }
      }
    }

    return {
      name: col.name,
      role: col.role,
      missingCount,
      missingPct,
      cv,
      outlierCount,
      outlierPct,
      isNumeric: col.isNumeric,
    };
  });

  // Correlation check for continuous parameters
  const continuousParams = parameters.filter((p) => p.type === "continuous");
  const correlationWarnings: CorrelationWarning[] = [];

  for (let i = 0; i < continuousParams.length; i++) {
    for (let j = i + 1; j < continuousParams.length; j++) {
      const nameA = continuousParams[i].name;
      const nameB = continuousParams[j].name;

      // Build paired values (only rows where both have valid numbers)
      const paired: { a: number; b: number }[] = [];
      for (const row of data) {
        const va = parseFloat(row[nameA]);
        const vb = parseFloat(row[nameB]);
        if (!isNaN(va) && !isNaN(vb)) {
          paired.push({ a: va, b: vb });
        }
      }

      if (paired.length >= 3) {
        const r = pearsonCorrelation(
          paired.map((p) => p.a),
          paired.map((p) => p.b)
        );
        if (r !== null && Math.abs(r) > 0.95) {
          correlationWarnings.push({ paramA: nameA, paramB: nameB, r });
        }
      }
    }
  }

  // Compute sub-scores (0-100 each)

  // Completeness: average (100 - missingPct) across columns
  const completenessScore =
    columns.length > 0
      ? columns.reduce((sum, c) => sum + (100 - c.missingPct), 0) / columns.length
      : 100;

  // Variance: percentage of numeric columns with sufficient variation (CV >= 0.01)
  const numericCols = columns.filter((c) => c.isNumeric && c.cv !== null);
  const varianceScore =
    numericCols.length > 0
      ? (numericCols.filter((c) => c.cv !== null && c.cv >= 0.01).length /
          numericCols.length) *
        100
      : 100;

  // Outlier: percentage of columns without excessive outliers (<= 5%)
  const outlierScore =
    numericCols.length > 0
      ? (numericCols.filter((c) => c.outlierPct <= 5).length / numericCols.length) *
        100
      : 100;

  // Sample size score
  let sampleSizeScore: number;
  if (sampleSize < 10) {
    sampleSizeScore = Math.max(0, sampleSize * 5); // 0-45
  } else if (sampleSize <= 50) {
    sampleSizeScore = 50 + ((sampleSize - 10) / 40) * 30; // 50-80
  } else {
    sampleSizeScore = Math.min(100, 80 + ((sampleSize - 50) / 50) * 20); // 80-100
  }

  // Overall score: weighted average
  const overallScore = Math.round(
    completenessScore * 0.3 +
      varianceScore * 0.25 +
      outlierScore * 0.25 +
      sampleSizeScore * 0.2
  );

  return {
    overallScore,
    sampleSize,
    sampleSizeRating,
    sampleSizeMessage,
    columns,
    correlationWarnings,
    completenessScore: Math.round(completenessScore),
    varianceScore: Math.round(varianceScore),
    outlierScore: Math.round(outlierScore),
    sampleSizeScore: Math.round(sampleSizeScore),
  };
}

function scoreColor(score: number): string {
  if (score >= 80) return "#16a34a";
  if (score >= 50) return "#ca8a04";
  return "#dc2626";
}

function scoreBg(score: number): string {
  if (score >= 80) return "#dcfce7";
  if (score >= 50) return "#fef9c3";
  return "#fee2e2";
}

function ratingColor(rating: "red" | "yellow" | "green"): string {
  if (rating === "green") return "#16a34a";
  if (rating === "yellow") return "#ca8a04";
  return "#dc2626";
}

function ratingBg(rating: "red" | "yellow" | "green"): string {
  if (rating === "green") return "#dcfce7";
  if (rating === "yellow") return "#fef9c3";
  return "#fee2e2";
}

function missingColor(pct: number): string {
  if (pct >= 20) return "#dc2626";
  if (pct >= 5) return "#ca8a04";
  return "#16a34a";
}

function missingBg(pct: number): string {
  if (pct >= 20) return "#fee2e2";
  if (pct >= 5) return "#fef9c3";
  return "#dcfce7";
}

export default function DataQualityReport({
  data,
  parameters,
  objectives,
}: DataQualityReportProps) {
  const report = useMemo(
    () => computeReport(data, parameters, objectives),
    [data, parameters, objectives]
  );

  const columnsWithMissing = report.columns.filter((c) => c.missingPct > 0);
  const columnsWithLowVariance = report.columns.filter(
    (c) => c.isNumeric && c.cv !== null && c.cv < 0.01
  );
  const columnsWithOutliers = report.columns.filter(
    (c) => c.isNumeric && c.outlierPct > 5
  );

  return (
    <div style={styles.container}>
      <h3 style={styles.title}>Data Quality Report</h3>

      {/* Overall Score */}
      <div style={styles.scoreSection}>
        <div
          style={{
            ...styles.scoreBadge,
            background: scoreBg(report.overallScore),
            color: scoreColor(report.overallScore),
          }}
        >
          {report.overallScore}
        </div>
        <div style={styles.scoreDetails}>
          <div style={styles.scoreLabel}>Overall Quality Score</div>
          <div style={styles.subScores}>
            <span style={styles.subScore}>
              Completeness: {report.completenessScore}
            </span>
            <span style={styles.subScore}>
              Variance: {report.varianceScore}
            </span>
            <span style={styles.subScore}>
              Outliers: {report.outlierScore}
            </span>
            <span style={styles.subScore}>
              Sample Size: {report.sampleSizeScore}
            </span>
          </div>
        </div>
      </div>

      {/* Sample Size */}
      <div style={styles.card}>
        <div style={styles.cardHeader}>
          <span style={styles.cardIcon}>&#x1F4CA;</span>
          <span style={styles.cardTitle}>Sample Size</span>
        </div>
        <div style={styles.cardBody}>
          <span
            style={{
              ...styles.inlineBadge,
              background: ratingBg(report.sampleSizeRating),
              color: ratingColor(report.sampleSizeRating),
            }}
          >
            {report.sampleSize} rows
          </span>
          <span style={{ color: ratingColor(report.sampleSizeRating), marginLeft: 12 }}>
            {report.sampleSizeMessage}
          </span>
        </div>
      </div>

      {/* Missing Values */}
      {columnsWithMissing.length > 0 && (
        <div style={styles.card}>
          <div style={styles.cardHeader}>
            <span style={styles.cardIcon}>&#x26A0;</span>
            <span style={styles.cardTitle}>Missing Values</span>
          </div>
          <div style={styles.cardBody}>
            {columnsWithMissing.map((col) => (
              <div key={col.name} style={styles.issueRow}>
                <strong>{col.name}</strong>
                <span
                  style={{
                    ...styles.inlineBadge,
                    background: missingBg(col.missingPct),
                    color: missingColor(col.missingPct),
                  }}
                >
                  {col.missingPct.toFixed(1)}% missing ({col.missingCount}/{report.sampleSize})
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Variance Check */}
      {columnsWithLowVariance.length > 0 && (
        <div style={styles.card}>
          <div style={styles.cardHeader}>
            <span style={styles.cardIcon}>&#x26A0;</span>
            <span style={styles.cardTitle}>Insufficient Variance</span>
          </div>
          <div style={styles.cardBody}>
            {columnsWithLowVariance.map((col) => (
              <div key={col.name} style={styles.issueRow}>
                <strong>{col.name}</strong>
                <span style={{ ...styles.inlineBadge, background: "#fef9c3", color: "#854d0e" }}>
                  CV = {col.cv !== null ? col.cv.toFixed(4) : "N/A"}
                </span>
                <span style={styles.issueMessage}>
                  Insufficient variation -- optimization unlikely to find patterns
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Outlier Detection */}
      {columnsWithOutliers.length > 0 && (
        <div style={styles.card}>
          <div style={styles.cardHeader}>
            <span style={styles.cardIcon}>&#x26A0;</span>
            <span style={styles.cardTitle}>Outlier Detection</span>
          </div>
          <div style={styles.cardBody}>
            {columnsWithOutliers.map((col) => (
              <div key={col.name} style={styles.issueRow}>
                <strong>{col.name}</strong>
                <span style={{ ...styles.inlineBadge, background: "#fee2e2", color: "#dc2626" }}>
                  {col.outlierCount} outliers ({col.outlierPct.toFixed(1)}%)
                </span>
                <span style={styles.issueMessage}>
                  Values beyond mean +/- 3 std
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Correlation Check */}
      {report.correlationWarnings.length > 0 && (
        <div style={styles.card}>
          <div style={styles.cardHeader}>
            <span style={styles.cardIcon}>&#x26A0;</span>
            <span style={styles.cardTitle}>Parameter Correlation</span>
          </div>
          <div style={styles.cardBody}>
            {report.correlationWarnings.map((w) => (
              <div key={`${w.paramA}-${w.paramB}`} style={styles.issueRow}>
                <strong>
                  {w.paramA} / {w.paramB}
                </strong>
                <span style={{ ...styles.inlineBadge, background: "#fee2e2", color: "#dc2626" }}>
                  r = {w.r.toFixed(3)}
                </span>
                <span style={styles.issueMessage}>
                  Highly correlated -- consider removing one
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* All clear message */}
      {columnsWithMissing.length === 0 &&
        columnsWithLowVariance.length === 0 &&
        columnsWithOutliers.length === 0 &&
        report.correlationWarnings.length === 0 && (
          <div style={{ ...styles.card, borderColor: "#bbf7d0" }}>
            <div style={styles.cardBody}>
              <span style={{ color: "#16a34a", fontWeight: 600 }}>
                No data quality issues detected.
              </span>
            </div>
          </div>
        )}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    marginBottom: 24,
    padding: 20,
    background: "var(--color-bg, #f8fafc)",
    border: "1px solid var(--color-border, #e2e8f0)",
    borderRadius: 8,
  },
  title: {
    fontSize: "1rem",
    fontWeight: 600,
    marginBottom: 16,
    marginTop: 0,
  },
  scoreSection: {
    display: "flex",
    alignItems: "center",
    gap: 16,
    marginBottom: 20,
    padding: 16,
    background: "var(--color-surface, #fff)",
    borderRadius: 8,
    border: "1px solid var(--color-border, #e2e8f0)",
  },
  scoreBadge: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    width: 64,
    height: 64,
    borderRadius: "50%",
    fontSize: "1.5rem",
    fontWeight: 700,
    flexShrink: 0,
  },
  scoreDetails: {
    display: "flex",
    flexDirection: "column",
    gap: 6,
  },
  scoreLabel: {
    fontSize: "0.95rem",
    fontWeight: 600,
  },
  subScores: {
    display: "flex",
    gap: 16,
    flexWrap: "wrap",
  },
  subScore: {
    fontSize: "0.8rem",
    color: "#64748b",
  },
  card: {
    marginBottom: 12,
    border: "1px solid var(--color-border, #e2e8f0)",
    borderRadius: 8,
    background: "var(--color-surface, #fff)",
    overflow: "hidden",
  },
  cardHeader: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    padding: "10px 16px",
    borderBottom: "1px solid var(--color-border, #e2e8f0)",
    background: "var(--color-bg, #f8fafc)",
  },
  cardIcon: {
    fontSize: "1rem",
  },
  cardTitle: {
    fontSize: "0.9rem",
    fontWeight: 600,
  },
  cardBody: {
    padding: "12px 16px",
  },
  issueRow: {
    display: "flex",
    alignItems: "center",
    gap: 12,
    padding: "6px 0",
    flexWrap: "wrap" as const,
    fontSize: "0.88rem",
  },
  inlineBadge: {
    display: "inline-block",
    padding: "2px 10px",
    borderRadius: 12,
    fontSize: "0.8rem",
    fontWeight: 600,
  },
  issueMessage: {
    color: "#64748b",
    fontSize: "0.82rem",
    fontStyle: "italic",
  },
};
