import {
  ComposedChart,
  Line,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceArea,
} from 'recharts';

interface ConvergencePlotProps {
  data: Array<{ iteration: number; value: number; best: number }>;
  objectiveName: string;
  direction: 'minimize' | 'maximize';
  phases?: Array<{ name: string; start: number; end: number; color: string }>;
}

export default function ConvergencePlot({
  data,
  objectiveName,
  direction,
  phases = [],
}: ConvergencePlotProps) {
  if (data.length === 0) {
    return <p className="empty-state">No convergence data to display.</p>;
  }

  return (
    <ResponsiveContainer width="100%" height={400}>
      <ComposedChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />

        {/* Phase regions */}
        {phases.map((phase, idx) => (
          <ReferenceArea
            key={`phase-${idx}`}
            x1={phase.start}
            x2={phase.end}
            fill={phase.color}
            fillOpacity={0.1}
            label={{
              value: phase.name,
              position: 'top',
              fontSize: 11,
              fill: '#718096',
            }}
          />
        ))}

        <XAxis
          dataKey="iteration"
          label={{ value: 'Iteration', position: 'insideBottom', offset: -10, fontSize: 12 }}
          tick={{ fontSize: 11 }}
        />
        <YAxis
          label={{
            value: `${objectiveName} (${direction})`,
            angle: -90,
            position: 'insideLeft',
            fontSize: 12,
          }}
          tick={{ fontSize: 11 }}
        />

        <Tooltip
          contentStyle={{
            background: '#ffffff',
            border: '1px solid #e2e8f0',
            borderRadius: '6px',
            fontSize: '0.85rem',
          }}
          formatter={(value: unknown, name: unknown) => [
            typeof value === 'number' ? value.toFixed(4) : '-',
            name === 'value' ? 'Trial Value' : 'Best So Far',
          ]}
          labelFormatter={(label) => `Iteration ${label}`}
        />

        <Legend
          wrapperStyle={{ fontSize: '0.85rem', paddingTop: '10px' }}
          iconType="line"
        />

        {/* Individual trial values */}
        <Scatter
          dataKey="value"
          fill="#94a3b8"
          opacity={0.6}
          name="Trial Values"
        />

        {/* Best-so-far line */}
        <Line
          type="stepAfter"
          dataKey="best"
          stroke="#3b82f6"
          strokeWidth={2.5}
          dot={false}
          name="Best So Far"
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
