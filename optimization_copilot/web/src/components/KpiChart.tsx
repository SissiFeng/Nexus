interface KpiChartProps {
  data: {
    iterations: number[];
    values: number[];
  };
}

const CHART_WIDTH = 600;
const CHART_HEIGHT = 300;
const PADDING = { top: 20, right: 20, bottom: 40, left: 60 };

export default function KpiChart({ data }: KpiChartProps) {
  if (data.iterations.length === 0) {
    return <p className="empty-state">No data to display.</p>;
  }

  const plotW = CHART_WIDTH - PADDING.left - PADDING.right;
  const plotH = CHART_HEIGHT - PADDING.top - PADDING.bottom;

  const xMin = Math.min(...data.iterations);
  const xMax = Math.max(...data.iterations);
  const yMin = Math.min(...data.values);
  const yMax = Math.max(...data.values);

  const yRange = yMax - yMin || 1;
  const xRange = xMax - xMin || 1;

  const yPad = yRange * 0.1;
  const yLo = yMin - yPad;
  const yHi = yMax + yPad;
  const ySpan = yHi - yLo;

  const scaleX = (v: number) =>
    PADDING.left + ((v - xMin) / xRange) * plotW;
  const scaleY = (v: number) =>
    PADDING.top + plotH - ((v - yLo) / ySpan) * plotH;

  // Build polyline points
  const points = data.iterations
    .map((iter, i) => `${scaleX(iter)},${scaleY(data.values[i])}`)
    .join(" ");

  // Y-axis ticks (5 ticks)
  const yTicks = Array.from({ length: 5 }, (_, i) => {
    const val = yLo + (ySpan * i) / 4;
    return { val, y: scaleY(val) };
  });

  // X-axis ticks (up to 6 ticks)
  const tickCount = Math.min(data.iterations.length, 6);
  const xTicks = Array.from({ length: tickCount }, (_, i) => {
    const idx = Math.round((i / (tickCount - 1)) * (data.iterations.length - 1));
    const val = data.iterations[idx];
    return { val, x: scaleX(val) };
  });

  return (
    <svg
      viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`}
      className="kpi-chart"
      preserveAspectRatio="xMidYMid meet"
    >
      {/* Grid lines */}
      {yTicks.map((t, i) => (
        <line
          key={`grid-${i}`}
          x1={PADDING.left}
          y1={t.y}
          x2={CHART_WIDTH - PADDING.right}
          y2={t.y}
          stroke="#e0e0e0"
          strokeWidth={1}
        />
      ))}

      {/* Axes */}
      <line
        x1={PADDING.left}
        y1={PADDING.top}
        x2={PADDING.left}
        y2={PADDING.top + plotH}
        stroke="#333"
        strokeWidth={1.5}
      />
      <line
        x1={PADDING.left}
        y1={PADDING.top + plotH}
        x2={CHART_WIDTH - PADDING.right}
        y2={PADDING.top + plotH}
        stroke="#333"
        strokeWidth={1.5}
      />

      {/* Y-axis labels */}
      {yTicks.map((t, i) => (
        <text
          key={`ylabel-${i}`}
          x={PADDING.left - 8}
          y={t.y + 4}
          textAnchor="end"
          fontSize={11}
          fill="#666"
        >
          {t.val.toPrecision(3)}
        </text>
      ))}

      {/* X-axis labels */}
      {xTicks.map((t, i) => (
        <text
          key={`xlabel-${i}`}
          x={t.x}
          y={PADDING.top + plotH + 20}
          textAnchor="middle"
          fontSize={11}
          fill="#666"
        >
          {t.val}
        </text>
      ))}

      {/* Axis titles */}
      <text
        x={CHART_WIDTH / 2}
        y={CHART_HEIGHT - 4}
        textAnchor="middle"
        fontSize={12}
        fill="#333"
      >
        Iteration
      </text>
      <text
        x={14}
        y={CHART_HEIGHT / 2}
        textAnchor="middle"
        fontSize={12}
        fill="#333"
        transform={`rotate(-90, 14, ${CHART_HEIGHT / 2})`}
      >
        KPI
      </text>

      {/* Data line */}
      <polyline
        points={points}
        fill="none"
        stroke="#3b82f6"
        strokeWidth={2}
        strokeLinejoin="round"
      />

      {/* Data points */}
      {data.iterations.map((iter, i) => (
        <circle
          key={`dot-${i}`}
          cx={scaleX(iter)}
          cy={scaleY(data.values[i])}
          r={3.5}
          fill="#3b82f6"
          stroke="#fff"
          strokeWidth={1.5}
        >
          <title>
            Iteration {iter}: {data.values[i].toPrecision(4)}
          </title>
        </circle>
      ))}
    </svg>
  );
}
