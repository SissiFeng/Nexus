interface Phase {
  name: string;
  start: number;
  end: number;
}

interface PhaseTimelineProps {
  phases: Phase[];
}

const PHASE_COLORS = [
  "#3b82f6", // blue
  "#10b981", // green
  "#f59e0b", // amber
  "#8b5cf6", // purple
  "#ef4444", // red
  "#06b6d4", // cyan
  "#f97316", // orange
];

export default function PhaseTimeline({ phases }: PhaseTimelineProps) {
  if (phases.length === 0) {
    return <p className="empty-state">No phases recorded.</p>;
  }

  const globalStart = Math.min(...phases.map((p) => p.start));
  const globalEnd = Math.max(...phases.map((p) => p.end));
  const totalSpan = globalEnd - globalStart || 1;

  return (
    <div className="phase-timeline">
      <div className="timeline-bar">
        {phases.map((phase, i) => {
          const left = ((phase.start - globalStart) / totalSpan) * 100;
          const width = ((phase.end - phase.start) / totalSpan) * 100;
          const color = PHASE_COLORS[i % PHASE_COLORS.length];

          return (
            <div
              key={i}
              className="timeline-segment"
              style={{
                left: `${left}%`,
                width: `${Math.max(width, 1)}%`,
                backgroundColor: color,
              }}
              title={`${phase.name}: ${phase.start} - ${phase.end}`}
            >
              <span className="timeline-label">{phase.name}</span>
            </div>
          );
        })}
      </div>
      <div className="timeline-legend">
        {phases.map((phase, i) => (
          <div key={i} className="legend-item">
            <span
              className="legend-dot"
              style={{
                backgroundColor: PHASE_COLORS[i % PHASE_COLORS.length],
              }}
            />
            <span className="legend-text">
              {phase.name} ({phase.start}&ndash;{phase.end})
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
