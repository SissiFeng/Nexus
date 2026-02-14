import { useState } from 'react';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ZAxis,
} from 'recharts';

interface ScatterMatrixProps {
  data: Array<Record<string, number>>;
  parameters: string[];
  objectiveName: string;
  objectiveDirection: 'minimize' | 'maximize';
  xParam?: string;
  yParam?: string;
  suggestions?: Array<Record<string, number>>;
  onParamChange?: (axis: 'x' | 'y', param: string) => void;
}

export default function ScatterMatrix({
  data,
  parameters,
  objectiveName,
  objectiveDirection,
  xParam,
  yParam,
  suggestions = [],
  onParamChange,
}: ScatterMatrixProps) {
  const [selectedX, setSelectedX] = useState(xParam || parameters[0] || '');
  const [selectedY, setSelectedY] = useState(yParam || parameters[1] || parameters[0] || '');

  if (data.length === 0 || parameters.length === 0) {
    return <p className="empty-state">No data to display.</p>;
  }

  const handleXChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newX = e.target.value;
    setSelectedX(newX);
    onParamChange?.('x', newX);
  };

  const handleYChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newY = e.target.value;
    setSelectedY(newY);
    onParamChange?.('y', newY);
  };

  // Get objective values for color mapping
  const objectiveValues = data.map((d) => d[objectiveName] || 0);
  const minObj = Math.min(...objectiveValues);
  const maxObj = Math.max(...objectiveValues);

  // Color mapping: viridis-like scale
  const getColor = (objValue: number) => {
    const normalized = (objValue - minObj) / (maxObj - minObj || 1);

    // For minimize: blue (good) -> yellow (mid) -> red (bad)
    // For maximize: red (bad) -> yellow (mid) -> blue (good)
    const ratio = objectiveDirection === 'minimize' ? normalized : 1 - normalized;

    if (ratio < 0.5) {
      // Blue to yellow
      const t = ratio * 2;
      return `rgb(${Math.round(68 + t * 187)}, ${Math.round(138 + t * 68)}, ${Math.round(255 - t * 50)})`;
    } else {
      // Yellow to red
      const t = (ratio - 0.5) * 2;
      return `rgb(${Math.round(255)}, ${Math.round(206 - t * 140)}, ${Math.round(205 - t * 205)})`;
    }
  };

  // Prepare chart data
  const chartData = data.map((d) => ({
    x: d[selectedX] || 0,
    y: d[selectedY] || 0,
    objective: d[objectiveName] || 0,
    fill: getColor(d[objectiveName] || 0),
  }));

  const suggestionData = suggestions.map((s) => ({
    x: s[selectedX] || 0,
    y: s[selectedY] || 0,
    fill: '#22c55e',
  }));

  // Custom tooltip
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length > 0) {
      const point = payload[0].payload;
      const dataPoint = data.find(
        (d) => d[selectedX] === point.x && d[selectedY] === point.y
      );

      if (!dataPoint) return null;

      return (
        <div
          style={{
            background: '#ffffff',
            border: '1px solid #e2e8f0',
            borderRadius: '6px',
            padding: '8px 12px',
            fontSize: '0.85rem',
          }}
        >
          <div><strong>{selectedX}:</strong> {point.x.toFixed(3)}</div>
          <div><strong>{selectedY}:</strong> {point.y.toFixed(3)}</div>
          <div><strong>{objectiveName}:</strong> {point.objective.toFixed(4)}</div>
          <div style={{ marginTop: '4px', fontSize: '0.78rem', color: '#718096' }}>
            {Object.entries(dataPoint)
              .filter(([k]) => k !== selectedX && k !== selectedY && k !== objectiveName)
              .slice(0, 3)
              .map(([k, v]) => (
                <div key={k}>
                  {k}: {typeof v === 'number' ? v.toFixed(3) : v}
                </div>
              ))}
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div>
      {/* Parameter selectors */}
      <div style={{ display: 'flex', gap: '16px', marginBottom: '16px', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <label htmlFor="x-param" style={{ fontWeight: 500, fontSize: '0.9rem' }}>
            X-axis:
          </label>
          <select
            id="x-param"
            value={selectedX}
            onChange={handleXChange}
            className="input"
            style={{ width: '180px' }}
          >
            {parameters.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <label htmlFor="y-param" style={{ fontWeight: 500, fontSize: '0.9rem' }}>
            Y-axis:
          </label>
          <select
            id="y-param"
            value={selectedY}
            onChange={handleYChange}
            className="input"
            style={{ width: '180px' }}
          >
            {parameters.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Scatter chart */}
      <ResponsiveContainer width="100%" height={400}>
        <ScatterChart margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />

          <XAxis
            type="number"
            dataKey="x"
            name={selectedX}
            label={{ value: selectedX, position: 'insideBottom', offset: -10, fontSize: 12 }}
            tick={{ fontSize: 11 }}
          />

          <YAxis
            type="number"
            dataKey="y"
            name={selectedY}
            label={{ value: selectedY, angle: -90, position: 'insideLeft', fontSize: 12 }}
            tick={{ fontSize: 11 }}
          />

          <ZAxis range={[60, 60]} />

          <Tooltip content={<CustomTooltip />} />

          {/* Data points */}
          <Scatter
            name="Trials"
            data={chartData}
            fill="#3b82f6"
          >
            {chartData.map((entry, index) => (
              <circle key={`cell-${index}`} r={4} fill={entry.fill} />
            ))}
          </Scatter>

          {/* Suggestion points */}
          {suggestionData.length > 0 && (
            <Scatter
              name="Suggestions"
              data={suggestionData}
              fill="#22c55e"
              shape="star"
            />
          )}
        </ScatterChart>
      </ResponsiveContainer>

      {/* Color scale legend */}
      <div style={{ marginTop: '16px', display: 'flex', alignItems: 'center', gap: '12px' }}>
        <span style={{ fontSize: '0.85rem', fontWeight: 500, color: '#718096' }}>
          {objectiveName} ({objectiveDirection}):
        </span>
        <div style={{ flex: 1, maxWidth: '300px', height: '20px', position: 'relative' }}>
          <div
            style={{
              width: '100%',
              height: '100%',
              background:
                objectiveDirection === 'minimize'
                  ? 'linear-gradient(to right, #448aff, #ffce42, #ff3333)'
                  : 'linear-gradient(to right, #ff3333, #ffce42, #448aff)',
              borderRadius: '4px',
            }}
          />
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              fontSize: '0.78rem',
              color: '#718096',
              marginTop: '4px',
            }}
          >
            <span>
              {objectiveDirection === 'minimize' ? 'Best' : 'Worst'}: {minObj.toFixed(3)}
            </span>
            <span>
              {objectiveDirection === 'minimize' ? 'Worst' : 'Best'}: {maxObj.toFixed(3)}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
