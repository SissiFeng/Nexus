function Stats() {
  const stats = [
    { value: '520+', label: 'Python Files' },
    { value: '181K+', label: 'Lines of Code' },
    { value: '155', label: 'Test Files' },
    { value: '6,300+', label: 'Tests' },
    { value: '148+', label: 'Visualizations' }
  ]

  return (
    <section className="section stats-section">
      <div className="container">
        <div className="stats-grid">
          {stats.map((stat, index) => (
            <div key={index} className="stat-item">
              <div className="stat-value">{stat.value}</div>
              <div className="stat-label">{stat.label}</div>
            </div>
          ))}
        </div>
        
        <div style={{ 
          textAlign: 'center', 
          marginTop: '64px',
          padding: '32px',
          background: 'rgba(255, 255, 255, 0.05)',
          borderRadius: 'var(--radius-lg)',
          border: '1px solid rgba(255, 255, 255, 0.1)'
        }}>
          <p style={{ 
            fontSize: '1.125rem', 
            color: 'rgba(255, 255, 255, 0.8)',
            fontStyle: 'italic'
          }}>
            "Built entirely with Claude Code — 37+ feature batches, 99+ commits, 
            18,000+ lines in the main Workspace component"
          </p>
        </div>
      </div>
    </section>
  )
}

export default Stats
