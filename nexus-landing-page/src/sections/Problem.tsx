function Problem() {
  return (
    <section id="problem" className="section">
      <div className="container">
        <div className="section-header">
          <span className="section-tag">The Problem</span>
          <h2 className="section-title">
            Scientists Face Two Major Barriers
          </h2>
          <p className="section-subtitle">
            Running experimental optimization shouldn't require a PhD in machine learning.
            Yet existing tools create friction that slows down discovery.
          </p>
        </div>

        <div className="problem-grid">
          <div className="problem-card">
            <div className="problem-icon">🕶️</div>
            <h3 className="problem-title">Opacity</h3>
            <p className="problem-desc">
              Existing tools don't explain <strong>why</strong> a suggestion was made. 
              Researchers can't trust recommendations, diagnose stalled optimizations, 
              or understand the optimizer's reasoning.
            </p>
          </div>

          <div className="problem-card">
            <div className="problem-icon">💻</div>
            <h3 className="problem-title">Programming Required</h3>
            <p className="problem-desc">
              Most tools demand ML expertise to configure. Scientists spend more time 
              wrestling with code than running experiments, creating a barrier to adoption.
            </p>
          </div>

          <div className="problem-card">
            <div className="problem-icon">😵</div>
            <h3 className="problem-title">The Result?</h3>
            <p className="problem-desc">
              Researchers don't trust suggestions, can't diagnose problems, 
              or abandon the tool entirely—wasting time, money, and scientific opportunity.
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}

export default Problem
