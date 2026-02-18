function Hero() {
  const scrollToPlayground = () => {
    const element = document.getElementById('playground')
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' })
    }
  }

  return (
    <section className="hero">
      <div className="container hero-content">
        <div className="hero-text">
          <div className="hero-badge">
            <span>🚀</span>
            Intelligent Optimization Platform
          </div>
          
          <h1 className="hero-title">
            Transform Experiments from<br />
            Black Box to Research Partner
          </h1>
          
          <p className="hero-subtitle">
            Nexus wraps Bayesian optimization with a diagnostic intelligence layer. 
            Zero-code setup, 148+ real-time visualizations, and AI-powered insights 
            for scientific experiments.
          </p>
          
          <div className="hero-buttons">
            <a 
              href="https://codespaces.new/SissiFeng/Nexus?quickstart=1"
              target="_blank"
              rel="noopener noreferrer"
              className="btn btn-primary btn-large"
            >
              <svg viewBox="0 0 16 16" width="18" height="18" fill="currentColor" style={{ marginRight: '8px' }}>
                <path d="M4.25 2A2.25 2.25 0 0 0 2 4.25v7.5A2.25 2.25 0 0 0 4.25 14h7.5A2.25 2.25 0 0 0 14 11.75v-7.5A2.25 2.25 0 0 0 11.75 2h-7.5Zm0 1.5h7.5a.75.75 0 0 1 .75.75v7.5a.75.75 0 0 1-.75.75h-7.5a.75.75 0 0 1-.75-.75v-7.5a.75.75 0 0 1 .75-.75ZM5 5.75A.75.75 0 0 1 5.75 5h.5a.75.75 0 0 1 0 1.5h-.5A.75.75 0 0 1 5 5.75Zm3 0A.75.75 0 0 1 8.75 5h.5a.75.75 0 0 1 0 1.5h-.5A.75.75 0 0 1 8 5.75Zm-3 3A.75.75 0 0 1 5.75 8h.5a.75.75 0 0 1 0 1.5h-.5A.75.75 0 0 1 5 8.75Zm3 0A.75.75 0 0 1 8.75 8h.5a.75.75 0 0 1 0 1.5h-.5A.75.75 0 0 1 8 8.75Zm3 0A.75.75 0 0 1 11.75 8h.5a.75.75 0 0 1 0 1.5h-.5A.75.75 0 0 1 11 8.75ZM5 11.75a.75.75 0 0 1 .75-.75h4.5a.75.75 0 0 1 0 1.5h-4.5a.75.75 0 0 1-.75-.75Z"/>
              </svg>
              Try in Browser — Free
            </a>
            <button 
              onClick={scrollToPlayground}
              className="btn btn-secondary btn-large"
            >
              <span>▶</span>
              Watch Demo
            </button>
          </div>
          
          {/* GitHub Badge Row */}
          <div style={{ 
            display: 'flex', 
            gap: '16px', 
            alignItems: 'center',
            flexWrap: 'wrap',
            marginTop: '8px'
          }}>
            <a 
              href="https://github.com/SissiFeng/Nexus"
              target="_blank"
              rel="noopener noreferrer"
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '6px',
                fontSize: '0.875rem',
                color: 'var(--color-text-muted)',
                textDecoration: 'none',
                transition: 'color 0.2s'
              }}
              onMouseEnter={(e) => e.currentTarget.style.color = 'var(--color-primary)'}
              onMouseLeave={(e) => e.currentTarget.style.color = 'var(--color-text-muted)'}
            >
              <svg viewBox="0 0 16 16" width="16" height="16" fill="currentColor">
                <path d="M8 0c4.42 0 8 3.58 8 8a8.013 8.013 0 0 1-5.45 7.59c-.4.08-.55-.17-.55-.38 0-.27.01-1.13.01-2.2 0-.75-.25-1.23-.54-1.48 1.78-.2 3.65-.88 3.65-3.95 0-.88-.31-1.59-.82-2.15.08-.2.36-1.02-.08-2.12 0 0-.67-.22-2.2.82-.64-.18-1.32-.27-2-.27-.68 0-1.36.09-2 .27-1.53-1.03-2.2-.82-2.2-.82-.44 1.1-.16 1.92-.08 2.12-.51.56-.82 1.28-.82 2.15 0 3.06 1.86 3.75 3.64 3.95-.23.2-.44.55-.51 1.07-.46.21-1.61.55-2.33-.66-.15-.24-.6-.83-1.23-.82-.67.01-.27.38.01.53.34.19.73.9.82 1.13.16.45.68 1.31 2.69.94 0 .67.01 1.3.01 1.49 0 .21-.15.45-.55.38A7.995 7.995 0 0 1 0 8c0-4.42 3.58-8 8-8Z"/>
              </svg>
              Star on GitHub
            </a>
            <a 
              href="https://github.com/SissiFeng/Nexus/releases/tag/v0.1.0"
              target="_blank"
              rel="noopener noreferrer"
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '6px',
                fontSize: '0.875rem',
                color: 'var(--color-text-muted)',
                textDecoration: 'none',
                transition: 'color 0.2s'
              }}
              onMouseEnter={(e) => e.currentTarget.style.color = 'var(--color-primary)'}
              onMouseLeave={(e) => e.currentTarget.style.color = 'var(--color-text-muted)'}
            >
              <span style={{ 
                padding: '2px 8px', 
                background: 'var(--color-primary-subtle)', 
                color: 'var(--color-primary)',
                borderRadius: '100px',
                fontSize: '0.75rem',
                fontWeight: 600
              }}>v0.1.0</span>
              Latest Release
            </a>
          </div>
          
          <div className="hero-stats">
            <div className="hero-stat">
              <span className="hero-stat-value">148+</span>
              <span className="hero-stat-label">Visualizations</span>
            </div>
            <div className="hero-stat">
              <span className="hero-stat-value">10+</span>
              <span className="hero-stat-label">Backends</span>
            </div>
            <div className="hero-stat">
              <span className="hero-stat-value">17</span>
              <span className="hero-stat-label">Diagnostics</span>
            </div>
          </div>
        </div>

        <div className="hero-visual">
          <div className="hero-dashboard">
            <div className="dashboard-header">
              <div className="dot dot-red"></div>
              <div className="dot dot-yellow"></div>
              <div className="dot dot-green"></div>
              <span style={{ marginLeft: 'auto', fontSize: '0.75rem', color: 'var(--color-text-muted)' }}>
                Campaign: OER_Catalyst_v2
              </span>
            </div>
            
            <div className="dashboard-content">
              <div className="metric-card">
                <div className="metric-label">Best KPI</div>
                <div className="metric-value">0.847</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Iterations</div>
                <div className="metric-value">42</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Phase</div>
                <div className="metric-value" style={{ fontSize: '1rem' }}>Exploitation</div>
              </div>
            </div>
            
            <div style={{ 
              marginTop: '16px', 
              padding: '12px', 
              background: 'var(--color-bg)', 
              borderRadius: 'var(--radius)',
              fontSize: '0.75rem',
              color: 'var(--color-text-muted)',
              fontFamily: 'var(--font-mono)'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                <span>convergence_trend</span>
                <span style={{ color: 'var(--color-green)' }}>+0.23 ▲</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                <span>exploration_coverage</span>
                <span style={{ color: 'var(--color-yellow)' }}>0.67 ⚠</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>noise_estimate</span>
                <span style={{ color: 'var(--color-green)' }}>0.04 ✓</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

export default Hero
