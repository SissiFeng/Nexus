function Solution() {
  const solutions = [
    {
      icon: '⚡',
      title: 'Zero-Code Setup',
      description: 'Upload a CSV, map columns visually, and start optimizing. No Python, no notebooks, no configuration files.'
    },
    {
      icon: '📊',
      title: '148+ Inline Visualizations',
      description: 'Real-time diagnostics that surface problems before they waste trials. Every decision is explained with data.'
    },
    {
      icon: '🔍',
      title: 'Transparent Suggestions',
      description: 'Every recommendation includes novelty scores, risk profiles, and provenance. Know why, not just what.'
    },
    {
      icon: '🤖',
      title: 'AI Chat Interface',
      description: 'Ask "Why did you switch strategies?" and get answers backed by computed signals, not black-box magic.'
    },
    {
      icon: '🧠',
      title: 'Campaign Memory',
      description: 'Decision journals, learning curves, and hypothesis tracking. Your experiments learn from each other.'
    },
    {
      icon: '🔬',
      title: 'Scientific Rigor',
      description: 'FAIR metadata, audit trails, and reproducibility built-in. Publish with confidence.'
    }
  ]

  return (
    <section id="solution" className="section section-alt">
      <div className="container">
        <div className="section-header">
          <span className="section-tag">The Solution</span>
          <h2 className="section-title">
            Diagnostic Intelligence Layer
          </h2>
          <p className="section-subtitle">
            Nexus wraps Bayesian optimization with transparency and explainability, 
            turning a black-box tool into a trusted research partner.
          </p>
        </div>

        <div className="solution-features">
          {solutions.map((solution, index) => (
            <div key={index} className="solution-feature">
              <div className="solution-icon">{solution.icon}</div>
              <div className="solution-content">
                <h3>{solution.title}</h3>
                <p>{solution.description}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

export default Solution
