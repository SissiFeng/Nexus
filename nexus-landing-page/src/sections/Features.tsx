function Features() {
  const features = [
    { number: '148+', label: 'Inline Visualizations', description: 'SVG micro-viz, no charting libs' },
    { number: '17', label: 'Diagnostic Signals', description: 'Real-time health monitoring' },
    { number: '10+', label: 'Optimization Backends', description: 'GP-BO, TPE, CMA-ES, NSGA-II...' },
    { number: '520+', label: 'Python Files', description: 'Pure implementations, zero ML deps' }
  ]

  return (
    <section id="features" className="section">
      <div className="container">
        <div className="section-header">
          <span className="section-tag">Features</span>
          <h2 className="section-title">
            Everything You Need
          </h2>
          <p className="section-subtitle">
            Built from the ground up for scientific rigor and transparency.
          </p>
        </div>

        <div className="features-grid">
          {features.map((feature, index) => (
            <div key={index} className="feature-card">
              <div className="feature-number">{feature.number}</div>
              <div className="feature-label">{feature.label}</div>
              <p style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)', marginTop: '8px' }}>
                {feature.description}
              </p>
            </div>
          ))}
        </div>

        <div style={{ marginTop: '64px', display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '24px' }}>
          <div style={{ 
            padding: '24px', 
            background: 'var(--color-surface)', 
            borderRadius: 'var(--radius-lg)',
            border: '1px solid var(--color-border)'
          }}>
            <h4 style={{ marginBottom: '12px', color: 'var(--color-primary)' }}>🎯 Multi-Objective</h4>
            <p style={{ fontSize: '0.875rem' }}>
              Pareto frontier analysis, NSGA-II, MOBO, and scalarization methods for complex trade-offs.
            </p>
          </div>
          
          <div style={{ 
            padding: '24px', 
            background: 'var(--color-surface)', 
            borderRadius: 'var(--radius-lg)',
            border: '1px solid var(--color-border)'
          }}>
            <h4 style={{ marginBottom: '12px', color: 'var(--color-green)' }}>🧪 Causal Discovery</h4>
            <p style={{ fontSize: '0.875rem' }}>
              PC algorithm to find causal relationships, not just correlations. Understand mechanism, not just pattern.
            </p>
          </div>
          
          <div style={{ 
            padding: '24px', 
            background: 'var(--color-surface)', 
            borderRadius: 'var(--radius-lg)',
            border: '1px solid var(--color-border)'
          }}>
            <h4 style={{ marginBottom: '12px', color: 'var(--color-purple)' }}>🔒 Safety First</h4>
            <p style={{ fontSize: '0.875rem' }}>
              Hazard classification, emergency protocols, and execution guards for high-stakes experiments.
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}

export default Features
