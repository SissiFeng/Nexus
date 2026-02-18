function CTA() {
  const scrollToPlayground = () => {
    const element = document.getElementById('playground')
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' })
    }
  }

  return (
    <section className="section cta-section">
      <div className="container">
        <h2 className="cta-title">
          Ready to Transform Your Experiments?
        </h2>
        <p className="cta-subtitle">
          Try Nexus instantly in your browser — no installation, no coding required. 
          Or explore the open-source code on GitHub.
        </p>
        
        <div className="cta-buttons">
          <button 
            onClick={scrollToPlayground}
            className="btn btn-primary btn-large"
          >
            <span>▶</span>
            Try It Now — Free
          </button>
          <a 
            href="https://github.com/SissiFeng/Nexus"
            target="_blank"
            rel="noopener noreferrer"
            className="btn btn-secondary btn-large"
          >
            <span>💻</span>
            View on GitHub
          </a>
        </div>
        
        <div style={{ 
          marginTop: '48px',
          padding: '24px',
          background: 'var(--color-bg)',
          borderRadius: 'var(--radius-lg)',
          display: 'inline-flex',
          alignItems: 'center',
          gap: '16px'
        }}>
          <div style={{ display: 'flex', gap: '-8px' }}>
            {['⭐', '🧪', '🔬', '💡'].map((emoji, i) => (
              <div key={i} style={{
                width: '40px',
                height: '40px',
                background: 'var(--color-surface)',
                borderRadius: '50%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '1.25rem',
                border: '2px solid var(--color-border)',
                marginLeft: i > 0 ? '-8px' : 0,
                zIndex: 4 - i
              }}>
                {emoji}
              </div>
            ))}
          </div>
          <div style={{ textAlign: 'left' }}>
            <div style={{ fontSize: '0.875rem', fontWeight: 600 }}>
              Open Source & Free
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)' }}>
              MIT License — Contributions Welcome
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

export default CTA
