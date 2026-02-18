function HowItWorks() {
  const steps = [
    {
      title: 'Upload Your Data',
      description: 'Drop a CSV file with your experimental results. Nexus automatically detects data types, suggests mappings, and validates format.'
    },
    {
      title: 'Configure Visually',
      description: 'Use the Column Mapper to assign parameters (inputs), objectives (outputs), and metadata. Set bounds, directions, and constraints with point-and-click simplicity.'
    },
    {
      title: 'Monitor in Real-Time',
      description: 'The Workspace shows 17 diagnostic signals, 148+ visualizations, and AI-generated insights. Watch convergence, detect drift, and understand behavior as it happens.'
    },
    {
      title: 'Get Intelligent Suggestions',
      description: 'Generate next experiments with predicted outcomes, uncertainty estimates, and novelty scores. Every suggestion includes risk-return profiles and provenance.'
    },
    {
      title: 'Run & Feed Back',
      description: 'Execute suggested experiments in your lab, then upload results. The optimizer adapts automatically, updating diagnostics and refining suggestions.'
    },
    {
      title: 'Learn & Improve',
      description: 'Campaign memory tracks hypotheses, decisions, and outcomes. Future campaigns benefit from accumulated knowledge through meta-learning.'
    }
  ]

  return (
    <section id="how-it-works" className="section section-alt">
      <div className="container">
        <div className="section-header">
          <span className="section-tag">How It Works</span>
          <h2 className="section-title">
            Six Steps to Smarter Experiments
          </h2>
          <p className="section-subtitle">
            From data upload to optimized results, Nexus guides you through 
            the entire experimental workflow.
          </p>
        </div>

        <div className="steps">
          {steps.map((step, index) => (
            <div key={index} className="step">
              <div className="step-number">{index + 1}</div>
              <div className="step-content">
                <h3>{step.title}</h3>
                <p>{step.description}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

export default HowItWorks
