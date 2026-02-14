import { useState, useMemo } from 'react';
import {
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ZAxis,
  Line,
  ComposedChart,
} from 'recharts';

/* ── Types ── */

interface Observation {
  iteration: number;
  parameters: Record<string, number>;
  kpi_values: Record<string, number>;
}

interface ParetoPlotProps {
  observations: Observation[];
  objectiveNames: string[];
  objectiveDirections: string[]; // "minimize" or "maximize"
}

/* ── Pareto Dominance Algorithm ── */

/**
 * Determines if point A dominates point B, respecting objective directions.
 * A dominates B if: for all objectives A is at least as good as B,
 * and for at least one objective A is strictly better.
 */
function dominates(
  a: Record<string, number>,
  b: Record<string, number>,
  objectiveNames: string[],
  objectiveDirections: string[]
): boolean {
  let strictlyBetterInAtLeastOne = false;

  for (let i = 0; i < objectiveNames.length; i++) {
    const name = objectiveNames[i];
    const dir = objectiveDirections[i];
    const aVal = a[name] ?? 0;
    const bVal = b[name] ?? 0;

    if (dir === 'maximize') {
      if (aVal < bVal) return false; // A is worse in this objective
      if (aVal > bVal) strictlyBetterInAtLeastOne = true;
    } else {
      // minimize
      if (aVal > bVal) return false; // A is worse in this objective
      if (aVal < bVal) strictlyBetterInAtLeastOne = true;
    }
  }

  return strictlyBetterInAtLeastOne;
}

/**
 * Compute the set of non-dominated (Pareto optimal) indices.
 */
function computeParetoFront(
  observations: Observation[],
  objectiveNames: string[],
  objectiveDirections: string[]
): Set<number> {
  const paretoIndices = new Set<number>();

  for (let i = 0; i < observations.length; i++) {
    let isDominated = false;
    for (let j = 0; j < observations.length; j++) {
      if (i === j) continue;
      if (
        dominates(
          observations[j].kpi_values,
          observations[i].kpi_values,
          objectiveNames,
          objectiveDirections
        )
      ) {
        isDominated = true;
        break;
      }
    }
    if (!isDominated) {
      paretoIndices.add(i);
    }
  }

  return paretoIndices;
}

/**
 * Find the "knee point" -- the Pareto-optimal point closest to the ideal
 * point in normalized objective space.
 */
function findKneePoint(
  paretoObs: Observation[],
  objectiveNames: string[],
  objectiveDirections: string[]
): number {
  if (paretoObs.length === 0) return -1;
  if (paretoObs.length === 1) return 0;

  // Find min/max for normalization
  const mins: number[] = [];
  const maxs: number[] = [];
  for (let i = 0; i < objectiveNames.length; i++) {
    const name = objectiveNames[i];
    const vals = paretoObs.map((o) => o.kpi_values[name] ?? 0);
    mins.push(Math.min(...vals));
    maxs.push(Math.max(...vals));
  }

  // The ideal point is (0, 0, ...) in normalized space where 0 = best
  let bestIdx = 0;
  let bestDist = Infinity;

  for (let p = 0; p < paretoObs.length; p++) {
    let distSq = 0;
    for (let i = 0; i < objectiveNames.length; i++) {
      const name = objectiveNames[i];
      const val = paretoObs[p].kpi_values[name] ?? 0;
      const range = maxs[i] - mins[i] || 1;
      // Normalize: 0 = best, 1 = worst
      const norm =
        objectiveDirections[i] === 'maximize'
          ? (maxs[i] - val) / range
          : (val - mins[i]) / range;
      distSq += norm * norm;
    }
    if (distSq < bestDist) {
      bestDist = distSq;
      bestIdx = p;
    }
  }

  return bestIdx;
}

/* ── Formatters ── */

function formatAxisLabel(name: string, direction: string): string {
  const arrow = direction === 'maximize' ? '\u2191' : '\u2193';
  return `${name} (${direction} ${arrow})`;
}

function formatValue(v: number): string {
  if (Math.abs(v) >= 1000) return v.toFixed(1);
  if (Math.abs(v) >= 1) return v.toFixed(3);
  return v.toFixed(4);
}

/* ── Component ── */

export default function ParetoPlot({
  observations,
  objectiveNames,
  objectiveDirections,
}: ParetoPlotProps) {
  // For 3+ objectives, allow selecting which 2 to plot
  const [xObjective, setXObjective] = useState(0);
  const [yObjective, setYObjective] = useState(
    objectiveNames.length > 1 ? 1 : 0
  );
  const [sortCol, setSortCol] = useState<string | null>(null);
  const [sortAsc, setSortAsc] = useState(true);

  // Compute Pareto front
  const paretoIndices = useMemo(
    () => computeParetoFront(observations, objectiveNames, objectiveDirections),
    [observations, objectiveNames, objectiveDirections]
  );

  // Pareto-optimal observations
  const paretoObs = useMemo(
    () => observations.filter((_, i) => paretoIndices.has(i)),
    [observations, paretoIndices]
  );

  // Knee point
  const kneeIdx = useMemo(
    () => findKneePoint(paretoObs, objectiveNames, objectiveDirections),
    [paretoObs, objectiveNames, objectiveDirections]
  );

  // Single objective guard
  if (objectiveNames.length < 2) {
    return (
      <div
        style={{
          padding: '32px 24px',
          textAlign: 'center',
          color: 'var(--color-text-muted)',
          background: 'var(--color-bg)',
          borderRadius: 'var(--radius)',
          border: '1px solid var(--color-border)',
        }}
      >
        <p style={{ fontSize: '0.95rem', marginBottom: '8px' }}>
          Pareto analysis requires 2+ objectives.
        </p>
        <p style={{ fontSize: '0.85rem' }}>
          Switch to the <strong>Convergence plot</strong> for single-objective
          optimization.
        </p>
      </div>
    );
  }

  if (observations.length === 0) {
    return <p className="empty-state">No observation data for Pareto analysis.</p>;
  }

  const xName = objectiveNames[xObjective];
  const yName = objectiveNames[yObjective];
  const xDir = objectiveDirections[xObjective];
  const yDir = objectiveDirections[yObjective];

  // Build chart data
  const paretoData = observations
    .filter((_, i) => paretoIndices.has(i))
    .map((obs) => ({
      x: obs.kpi_values[xName] ?? 0,
      y: obs.kpi_values[yName] ?? 0,
      iteration: obs.iteration,
      parameters: obs.parameters,
      kpi_values: obs.kpi_values,
    }));

  const dominatedData = observations
    .filter((_, i) => !paretoIndices.has(i))
    .map((obs) => ({
      x: obs.kpi_values[xName] ?? 0,
      y: obs.kpi_values[yName] ?? 0,
      iteration: obs.iteration,
      parameters: obs.parameters,
      kpi_values: obs.kpi_values,
    }));

  // Sort Pareto front points by X for the connecting line (step-wise)
  const sortedParetoLine = [...paretoData].sort((a, b) => a.x - b.x);

  // Build step-wise line data for Pareto front
  const stepLineData: Array<{ x: number; y: number }> = [];
  for (let i = 0; i < sortedParetoLine.length; i++) {
    const pt = sortedParetoLine[i];
    stepLineData.push({ x: pt.x, y: pt.y });
    if (i < sortedParetoLine.length - 1) {
      const next = sortedParetoLine[i + 1];
      // Horizontal step then vertical
      stepLineData.push({ x: next.x, y: pt.y });
    }
  }

  // Custom tooltip
  const CustomTooltip = ({ active, payload }: { active?: boolean; payload?: Array<{ payload: typeof paretoData[0] }> }) => {
    if (!active || !payload || payload.length === 0) return null;
    const point = payload[0].payload;
    if (!point || !point.kpi_values) return null;

    return (
      <div
        style={{
          background: 'var(--color-surface)',
          border: '1px solid var(--color-border)',
          borderRadius: '6px',
          padding: '10px 14px',
          fontSize: '0.85rem',
          boxShadow: 'var(--shadow-md)',
          maxWidth: '300px',
        }}
      >
        <div
          style={{
            fontWeight: 600,
            marginBottom: '6px',
            color: 'var(--color-primary)',
            fontSize: '0.82rem',
          }}
        >
          Iteration {point.iteration}
        </div>
        <div style={{ marginBottom: '6px' }}>
          {objectiveNames.map((name) => (
            <div key={name} style={{ display: 'flex', justifyContent: 'space-between', gap: '16px' }}>
              <span style={{ color: 'var(--color-text-muted)' }}>{name}:</span>
              <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 500 }}>
                {formatValue(point.kpi_values[name] ?? 0)}
              </span>
            </div>
          ))}
        </div>
        {Object.keys(point.parameters).length > 0 && (
          <div
            style={{
              borderTop: '1px solid var(--color-border)',
              paddingTop: '6px',
              fontSize: '0.8rem',
            }}
          >
            {Object.entries(point.parameters)
              .slice(0, 5)
              .map(([k, v]) => (
                <div key={k} style={{ display: 'flex', justifyContent: 'space-between', gap: '16px' }}>
                  <span style={{ color: 'var(--color-text-muted)' }}>{k}:</span>
                  <span style={{ fontFamily: 'var(--font-mono)' }}>
                    {typeof v === 'number' ? formatValue(v) : String(v)}
                  </span>
                </div>
              ))}
          </div>
        )}
      </div>
    );
  };

  // Sorted Pareto table data
  const paretoTableData = paretoObs.map((obs, idx) => ({
    obs,
    isKnee: idx === kneeIdx,
    originalIdx: idx,
  }));

  if (sortCol) {
    paretoTableData.sort((a, b) => {
      const aVal = a.obs.kpi_values[sortCol] ?? 0;
      const bVal = b.obs.kpi_values[sortCol] ?? 0;
      return sortAsc ? aVal - bVal : bVal - aVal;
    });
  }

  const handleSort = (col: string) => {
    if (sortCol === col) {
      setSortAsc(!sortAsc);
    } else {
      setSortCol(col);
      setSortAsc(true);
    }
  };

  const paretoCount = paretoIndices.size;
  const dominatedCount = observations.length - paretoCount;

  return (
    <div>
      {/* Objective selectors for 3+ objectives */}
      {objectiveNames.length > 2 && (
        <div
          style={{
            display: 'flex',
            gap: '16px',
            marginBottom: '16px',
            alignItems: 'center',
            flexWrap: 'wrap',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <label
              htmlFor="pareto-x-obj"
              style={{ fontWeight: 500, fontSize: '0.9rem' }}
            >
              X-axis:
            </label>
            <select
              id="pareto-x-obj"
              value={xObjective}
              onChange={(e) => setXObjective(Number(e.target.value))}
              className="input"
              style={{ width: '200px' }}
            >
              {objectiveNames.map((name, i) => (
                <option key={name} value={i}>
                  {formatAxisLabel(name, objectiveDirections[i])}
                </option>
              ))}
            </select>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <label
              htmlFor="pareto-y-obj"
              style={{ fontWeight: 500, fontSize: '0.9rem' }}
            >
              Y-axis:
            </label>
            <select
              id="pareto-y-obj"
              value={yObjective}
              onChange={(e) => setYObjective(Number(e.target.value))}
              className="input"
              style={{ width: '200px' }}
            >
              {objectiveNames.map((name, i) => (
                <option key={name} value={i}>
                  {formatAxisLabel(name, objectiveDirections[i])}
                </option>
              ))}
            </select>
          </div>
        </div>
      )}

      {/* Scatter plot */}
      <ResponsiveContainer width="100%" height={420}>
        <ComposedChart margin={{ top: 20, right: 30, left: 20, bottom: 30 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />

          <XAxis
            type="number"
            dataKey="x"
            name={xName}
            label={{
              value: formatAxisLabel(xName, xDir),
              position: 'insideBottom',
              offset: -15,
              fontSize: 12,
              fill: 'var(--color-text-muted)',
            }}
            tick={{ fontSize: 11 }}
          />

          <YAxis
            type="number"
            dataKey="y"
            name={yName}
            label={{
              value: formatAxisLabel(yName, yDir),
              angle: -90,
              position: 'insideLeft',
              fontSize: 12,
              fill: 'var(--color-text-muted)',
            }}
            tick={{ fontSize: 11 }}
          />

          <ZAxis range={[40, 40]} />

          <Tooltip content={<CustomTooltip />} />

          {/* Pareto front step-wise connecting line */}
          {stepLineData.length > 1 && (
            <Line
              data={stepLineData}
              type="linear"
              dataKey="y"
              stroke="#6366f1"
              strokeWidth={2}
              strokeDasharray="6 3"
              dot={false}
              isAnimationActive={false}
              name="Pareto Front"
              style={{
                filter: 'drop-shadow(0 0 3px rgba(99, 102, 241, 0.4))',
              }}
            />
          )}

          {/* Dominated points -- gray, small, translucent */}
          {dominatedData.length > 0 && (
            <Scatter
              name="Dominated"
              data={dominatedData}
              fill="#94a3b8"
              opacity={0.5}
            >
              {dominatedData.map((_, index) => (
                <circle
                  key={`dom-${index}`}
                  r={3.5}
                  fill="#94a3b8"
                  fillOpacity={0.5}
                  stroke="#94a3b8"
                  strokeOpacity={0.3}
                  strokeWidth={1}
                />
              ))}
            </Scatter>
          )}

          {/* Pareto optimal points -- indigo, larger, filled */}
          <Scatter
            name="Pareto Optimal"
            data={paretoData}
            fill="#4f46e5"
          >
            {paretoData.map((_, index) => (
              <circle
                key={`par-${index}`}
                r={5.5}
                fill="#4f46e5"
                stroke="#312e81"
                strokeWidth={1.5}
              />
            ))}
          </Scatter>
        </ComposedChart>
      </ResponsiveContainer>

      {/* Legend */}
      <div
        style={{
          display: 'flex',
          gap: '24px',
          marginTop: '12px',
          justifyContent: 'center',
          fontSize: '0.85rem',
          flexWrap: 'wrap',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span
            style={{
              display: 'inline-block',
              width: '12px',
              height: '12px',
              borderRadius: '50%',
              background: '#4f46e5',
              border: '1.5px solid #312e81',
            }}
          />
          <span style={{ color: 'var(--color-text)' }}>
            Pareto Optimal ({paretoCount} point{paretoCount !== 1 ? 's' : ''})
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span
            style={{
              display: 'inline-block',
              width: '10px',
              height: '10px',
              borderRadius: '50%',
              background: '#94a3b8',
              opacity: 0.5,
              border: '1px solid #94a3b8',
            }}
          />
          <span style={{ color: 'var(--color-text-muted)' }}>
            Dominated ({dominatedCount} point{dominatedCount !== 1 ? 's' : ''})
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span
            style={{
              display: 'inline-block',
              width: '20px',
              height: '0',
              borderTop: '2px dashed #6366f1',
              filter: 'drop-shadow(0 0 2px rgba(99, 102, 241, 0.5))',
            }}
          />
          <span style={{ color: 'var(--color-text-muted)' }}>Pareto Front</span>
        </div>
      </div>

      {/* ── Pareto Table ── */}
      {paretoObs.length > 0 && (
        <div style={{ marginTop: '24px' }}>
          <h3
            style={{
              fontSize: '1rem',
              fontWeight: 600,
              marginBottom: '12px',
            }}
          >
            Pareto-Optimal Points
          </h3>

          <div className="table-wrapper">
            <table className="data-table">
              <thead>
                <tr>
                  <th style={{ width: '60px' }}>#</th>
                  <th style={{ width: '80px' }}>Iter.</th>
                  {objectiveNames.map((name) => (
                    <th
                      key={name}
                      className="sortable"
                      onClick={() => handleSort(name)}
                      style={{ cursor: 'pointer' }}
                    >
                      {name}{' '}
                      {sortCol === name ? (sortAsc ? '\u25B2' : '\u25BC') : '\u25B4'}
                    </th>
                  ))}
                  {Object.keys(paretoObs[0]?.parameters ?? {})
                    .slice(0, 4)
                    .map((pName) => (
                      <th key={pName} style={{ color: 'var(--color-text-muted)' }}>
                        {pName}
                      </th>
                    ))}
                  <th style={{ width: '70px' }}>Note</th>
                </tr>
              </thead>
              <tbody>
                {paretoTableData.map((row, idx) => (
                  <tr
                    key={`pareto-row-${row.originalIdx}`}
                    style={
                      row.isKnee
                        ? {
                            background: '#f0fdf4',
                            borderLeft: '3px solid var(--color-green)',
                          }
                        : undefined
                    }
                  >
                    <td style={{ fontWeight: 600, color: 'var(--color-primary)' }}>
                      {idx + 1}
                    </td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>
                      {row.obs.iteration}
                    </td>
                    {objectiveNames.map((name) => (
                      <td
                        key={name}
                        style={{ fontFamily: 'var(--font-mono)', fontWeight: 500 }}
                      >
                        {formatValue(row.obs.kpi_values[name] ?? 0)}
                      </td>
                    ))}
                    {Object.keys(paretoObs[0]?.parameters ?? {})
                      .slice(0, 4)
                      .map((pName) => (
                        <td
                          key={pName}
                          style={{
                            fontFamily: 'var(--font-mono)',
                            fontSize: '0.85rem',
                            color: 'var(--color-text-muted)',
                          }}
                        >
                          {typeof row.obs.parameters[pName] === 'number'
                            ? formatValue(row.obs.parameters[pName])
                            : String(row.obs.parameters[pName] ?? '')}
                        </td>
                      ))}
                    <td>
                      {row.isKnee && (
                        <span
                          style={{
                            background: 'var(--color-green)',
                            color: 'white',
                            padding: '2px 8px',
                            borderRadius: '10px',
                            fontSize: '0.75rem',
                            fontWeight: 600,
                            whiteSpace: 'nowrap',
                          }}
                        >
                          Knee
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <style>{`
        .pareto-plot-wrapper {
          position: relative;
        }
      `}</style>
    </div>
  );
}
