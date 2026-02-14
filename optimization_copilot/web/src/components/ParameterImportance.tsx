import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';

interface ParameterImportanceProps {
  data: Array<{ name: string; importance: number }>;
  onParameterClick?: (name: string) => void;
}

export default function ParameterImportance({
  data,
  onParameterClick,
}: ParameterImportanceProps) {
  if (data.length === 0) {
    return <p className="empty-state">No parameter importance data available.</p>;
  }

  // Sort by importance (highest first)
  const sortedData = [...data].sort((a, b) => b.importance - a.importance);

  // Generate color based on importance (dark blue for high, light blue for low)
  const getColor = (importance: number, maxImportance: number) => {
    const ratio = importance / maxImportance;
    const hue = 210; // Blue hue
    const saturation = 70 + ratio * 30; // 70-100%
    const lightness = 60 - ratio * 25; // 60-35%
    return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
  };

  const maxImportance = Math.max(...sortedData.map((d) => d.importance));

  const handleClick = (data: any) => {
    if (onParameterClick && data.name) {
      onParameterClick(data.name);
    }
  };

  return (
    <ResponsiveContainer width="100%" height={Math.max(300, sortedData.length * 35)}>
      <BarChart
        data={sortedData}
        layout="vertical"
        margin={{ top: 20, right: 30, left: 150, bottom: 20 }}
      >
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" horizontal={false} />

        <XAxis
          type="number"
          domain={[0, 1]}
          label={{
            value: 'Importance Score',
            position: 'insideBottom',
            offset: -10,
            fontSize: 12,
          }}
          tick={{ fontSize: 11 }}
        />

        <YAxis
          type="category"
          dataKey="name"
          tick={{ fontSize: 11 }}
          width={140}
        />

        <Tooltip
          contentStyle={{
            background: '#ffffff',
            border: '1px solid #e2e8f0',
            borderRadius: '6px',
            fontSize: '0.85rem',
          }}
          formatter={(value: unknown) => [typeof value === 'number' ? value.toFixed(4) : '-', 'Importance']}
        />

        <Bar
          dataKey="importance"
          onClick={handleClick}
          cursor={onParameterClick ? 'pointer' : 'default'}
        >
          {sortedData.map((entry, index) => (
            <Cell
              key={`cell-${index}`}
              fill={getColor(entry.importance, maxImportance)}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
