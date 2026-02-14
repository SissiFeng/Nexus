import {
  TrendingDown,
  Search,
  AlertTriangle,
  Volume2,
  Pause,
  Signal,
  Trophy,
  Zap,
} from 'lucide-react';

interface DiagnosticCardsProps {
  diagnostics: {
    convergence_trend: number;
    exploration_coverage: number;
    failure_rate: number;
    noise_estimate: number;
    plateau_length: number;
    signal_to_noise: number;
    best_kpi_value: number;
    improvement_velocity: number;
  };
  tooltips?: Record<string, string>;
}

interface MetricConfig {
  label: string;
  icon: React.ElementType;
  format: (value: number) => string;
  description: string;
  getStatus: (value: number) => 'green' | 'yellow' | 'red' | 'neutral';
}

const METRICS: Record<string, MetricConfig> = {
  convergence_trend: {
    label: 'Convergence Trend',
    icon: TrendingDown,
    format: (v) => v.toFixed(3),
    description: 'Rate of improvement per iteration',
    getStatus: (v) => (v < -0.1 ? 'green' : v < 0.05 ? 'yellow' : 'red'),
  },
  exploration_coverage: {
    label: 'Exploration Coverage',
    icon: Search,
    format: (v) => `${(v * 100).toFixed(1)}%`,
    description: 'Search space coverage',
    getStatus: (v) => (v > 0.6 ? 'green' : v > 0.3 ? 'yellow' : 'red'),
  },
  failure_rate: {
    label: 'Failure Rate',
    icon: AlertTriangle,
    format: (v) => `${(v * 100).toFixed(1)}%`,
    description: 'Proportion of failed trials',
    getStatus: (v) => (v < 0.1 ? 'green' : v < 0.3 ? 'yellow' : 'red'),
  },
  noise_estimate: {
    label: 'Noise Estimate',
    icon: Volume2,
    format: (v) => v.toFixed(3),
    description: 'Estimated measurement noise',
    getStatus: (v) => (v < 0.1 ? 'green' : v < 0.3 ? 'yellow' : 'red'),
  },
  plateau_length: {
    label: 'Plateau Length',
    icon: Pause,
    format: (v) => `${Math.round(v)} iter`,
    description: 'Iterations without improvement',
    getStatus: (v) => (v < 5 ? 'green' : v < 15 ? 'yellow' : 'red'),
  },
  signal_to_noise: {
    label: 'Signal to Noise',
    icon: Signal,
    format: (v) => v.toFixed(2),
    description: 'Ratio of signal to noise',
    getStatus: (v) => (v > 10 ? 'green' : v > 3 ? 'yellow' : 'red'),
  },
  best_kpi_value: {
    label: 'Best KPI Value',
    icon: Trophy,
    format: (v) => v.toFixed(4),
    description: 'Best objective value achieved',
    getStatus: () => 'neutral',
  },
  improvement_velocity: {
    label: 'Improvement Velocity',
    icon: Zap,
    format: (v) => v.toFixed(4),
    description: 'Recent rate of improvement',
    getStatus: (v) => (v < -0.05 ? 'green' : v < 0 ? 'yellow' : 'red'),
  },
};

export default function DiagnosticCards({ diagnostics, tooltips }: DiagnosticCardsProps) {
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'green':
        return '#22c55e';
      case 'yellow':
        return '#eab308';
      case 'red':
        return '#ef4444';
      default:
        return '#94a3b8';
    }
  };

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
        gap: '16px',
      }}
    >
      {Object.entries(diagnostics).map(([key, value]) => {
        const config = METRICS[key];
        if (!config) return null;

        const Icon = config.icon;
        const status = config.getStatus(value);
        const statusColor = getStatusColor(status);

        return (
          <div key={key} className="stat-card" title={tooltips?.[key] ?? config.description}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
              <Icon size={18} style={{ color: "var(--color-text-muted)" }} />
              <span className="stat-label" style={{ textTransform: 'none', margin: 0 }}>
                {config.label}
              </span>
              <div
                style={{
                  marginLeft: 'auto',
                  width: '10px',
                  height: '10px',
                  borderRadius: '50%',
                  backgroundColor: statusColor,
                }}
              />
            </div>
            <div
              className="stat-value"
              style={{ marginBottom: '4px', fontSize: '1.3rem' }}
            >
              {config.format(value)}
            </div>
            <div style={{ fontSize: '0.78rem', color: 'var(--color-text-muted)' }}>
              {config.description}
            </div>
          </div>
        );
      })}
    </div>
  );
}
