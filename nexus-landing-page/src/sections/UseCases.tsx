function UseCases() {
  const useCases = [
    {
      icon: '⚗️',
      title: 'Chemical Synthesis',
      description: 'Optimize reaction conditions, catalyst selection, and yield maximization'
    },
    {
      icon: '🔋',
      title: 'Materials Discovery',
      description: 'Discover battery electrolytes, superconductors, and novel compounds'
    },
    {
      icon: '💊',
      title: 'Drug Formulation',
      description: 'Formulate stable, bioavailable pharmaceutical products'
    },
    {
      icon: '🧬',
      title: 'Bioprocess Engineering',
      description: 'Optimize fermentation, cell culture, and protein production'
    }
  ]

  return (
    <section id="usecases" className="section section-alt">
      <div className="container">
        <div className="section-header">
          <span className="section-tag">Use Cases</span>
          <h2 className="section-title">
            Trusted Across Disciplines
          </h2>
          <p className="section-subtitle">
            Nexus adapts to any experimental optimization problem 
            where parameters affect measurable outcomes.
          </p>
        </div>

        <div className="usecases-grid">
          {useCases.map((useCase, index) => (
            <div key={index} className="usecase-card">
              <div className="usecase-icon">{useCase.icon}</div>
              <h3 className="usecase-title">{useCase.title}</h3>
              <p className="usecase-desc">{useCase.description}</p>
            </div>
          ))}
        </div>

        <div style={{ 
          marginTop: '64px', 
          textAlign: 'center',
          padding: '48px',
          background: 'var(--color-surface)',
          borderRadius: 'var(--radius-lg)',
          border: '1px solid var(--color-border)'
        }}>
          <h3 style={{ marginBottom: '16px' }}>🎓 Demo Gallery Included</h3>
          <p style={{ maxWidth: '600px', margin: '0 auto 24px' }}>
            Start with pre-configured datasets: OER Catalyst Optimization, 
            Suzuki Coupling Yield, Battery Electrolyte Formulation, and more.
          </p>
          <a 
            href="https://github.com/SissiFeng/Nexus/tree/main/demo-datasets"
            target="_blank"
            rel="noopener noreferrer"
            className="btn btn-primary"
          >
            Browse Demo Gallery
          </a>
        </div>
      </div>
    </section>
  )
}

export default UseCases
