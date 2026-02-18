function Features() {
  const features = [
    { number: '148+', label: 'Inline Visualizations', description: 'SVG micro-viz, no charting libs' },
    { number: '17', label: 'Diagnostic Signals', description: 'Real-time health monitoring' },
    { number: '10+', label: 'Optimization Backends', description: 'GP-BO, TPE, CMA-ES, NSGA-II...' },
    { number: '37+', label: 'New Tests Added', description: 'For intelligent features' }
  ]

  const intelligentFeatures = [
    {
      icon: '🧠',
      title: 'Smart Algorithm Selection',
      color: '#3d5af1',
      description: 'Automatically selects TPE, CMA-ES, GP-BO, or Random based on noise level, dimensionality, and data characteristics. Explains why.',
      badges: ['Noise-Aware', 'Dimension-Adaptive', 'Explainable']
    },
    {
      icon: '🔗',
      title: 'Causal Discovery',
      color: '#0fa968',
      description: 'PC algorithm learns causal graphs to answer "which variables truly drive the objective?" Distinguishes correlation from causation.',
      badges: ['Root Cause Analysis', 'Confounder Detection', 'Intervention Guidance']
    },
    {
      icon: '📊',
      title: 'Multi-Fidelity Optimization',
      color: '#d4940a',
      description: 'Cheap simulation → expensive experiment pipeline. Adaptive promotion with uncertainty quantification. Cost-aware budget allocation.',
      badges: ['Successive Halving', 'Transfer Learning', 'Budget Optimization']
    },
    {
      icon: '⚡',
      title: 'Async SDL Integration',
      color: '#dc3545',
      description: 'Priority-based experiment scheduling for Self-Driving Labs. Resource-aware dispatch with out-of-order result handling.',
      badges: ['Priority Queue', 'Resource Management', 'Dependency Tracking']
    }
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

        {/* Intelligent Optimization Features */}
        <div style={{ marginTop: '80px' }}>
          <div className="section-header" style={{ marginBottom: '48px' }}>
            <span className="section-tag">New in v0.2.0</span>
            <h3 style={{ fontSize: '2rem', marginTop: '16px' }}>
              Intelligent Optimization
            </h3>
            <p className="section-subtitle">
              Four powerful capabilities that transform how you run experiments
            </p>
          </div>

          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: 'repeat(2, 1fr)', 
            gap: '24px'
          }}>
            {intelligentFeatures.map((feature, index) => (
              <div key={index} style={{ 
                padding: '28px', 
                background: 'var(--color-surface)', 
                borderRadius: 'var(--radius-lg)',
                border: '1px solid var(--color-border)',
                transition: 'all 0.2s',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = feature.color
                e.currentTarget.style.boxShadow = `0 4px 20px ${feature.color}15`
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = 'var(--color-border)'
                e.currentTarget.style.boxShadow = 'none'
              }}
              >
                <div style={{ 
                  fontSize: '2rem', 
                  marginBottom: '16px',
                  width: '56px',
                  height: '56px',
                  background: `${feature.color}15`,
                  borderRadius: 'var(--radius)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}>
                  {feature.icon}
                </div>
                
                <h4 style={{ 
                  marginBottom: '12px', 
                  color: feature.color,
                  fontSize: '1.25rem'
                }}>
                  {feature.title}
                </h4>
                
                <p style={{ 
                  fontSize: '0.875rem', 
                  color: 'var(--color-text-muted)',
                  lineHeight: 1.6,
                  marginBottom: '16px'
                }}>
                  {feature.description}
                </p>
                
                <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                  {feature.badges.map((badge, i) => (
                    <span key={i} style={{
                      padding: '4px 10px',
                      background: `${feature.color}10`,
                      color: feature.color,
                      borderRadius: '100px',
                      fontSize: '0.7rem',
                      fontWeight: 600,
                      textTransform: 'uppercase',
                      letterSpacing: '0.5px'
                    }}>
                      {badge}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Original Feature Cards */}
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
            <h4 style={{ marginBottom: '12px', color: 'var(--color-green)' }}>🧪 Wet Lab Safety</h4>
            <p style={{ fontSize: '0.875rem' }}>
              Hazard classification, emergency protocols, and execution guards for high-stakes experiments.
            </p>
          </div>
          
          <div style={{ 
            padding: '24px', 
            background: 'var(--color-surface)', 
            borderRadius: 'var(--radius-lg)',
            border: '1px solid var(--color-border)'
          }}>
            <h4 style={{ marginBottom: '12px', color: 'var(--color-purple)' }}>🔌 MCP Server</h4>
            <p style={{ fontSize: '0.875rem' }}>
              Model Context Protocol integration lets Claude drive campaigns through tool calls.
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}

export default Features
