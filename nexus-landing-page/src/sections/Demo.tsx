import { useState } from 'react'

function Demo() {
  const [activeTab, setActiveTab] = useState('overview')

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'explore', label: 'Explore' },
    { id: 'suggestions', label: 'Suggestions' }
  ]

  const renderDemoContent = () => {
    switch (activeTab) {
      case 'overview':
        return (
          <div className="demo-visual">
            <div className="viz-grid">
              <div className="viz-card">
                <div className="viz-title">Convergence Trend</div>
                <div className="viz-bar">
                  <div className="viz-bar-fill" style={{ width: '78%' }}></div>
                </div>
                <div className="viz-value" style={{ color: 'var(--color-green)' }}>+0.23 ▲</div>
              </div>
              <div className="viz-card">
                <div className="viz-title">Exploration Coverage</div>
                <div className="viz-bar">
                  <div className="viz-bar-fill" style={{ width: '67%' }}></div>
                </div>
                <div className="viz-value" style={{ color: 'var(--color-yellow)' }}>0.67 ⚠</div>
              </div>
              <div className="viz-card">
                <div className="viz-title">Noise Estimate</div>
                <div className="viz-bar">
                  <div className="viz-bar-fill" style={{ width: '15%' }}></div>
                </div>
                <div className="viz-value" style={{ color: 'var(--color-green)' }}>0.04 ✓</div>
              </div>
            </div>
            <div style={{ 
              marginTop: '24px', 
              padding: '16px', 
              background: 'var(--color-surface)', 
              borderRadius: 'var(--radius)',
              border: '1px solid var(--color-border)'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                <span style={{ fontSize: '0.875rem', fontWeight: 600 }}>Campaign Health</span>
                <span style={{ 
                  padding: '4px 12px', 
                  background: 'var(--color-green)', 
                  color: 'white',
                  borderRadius: '100px',
                  fontSize: '0.75rem'
                }}>
                  Healthy
                </span>
              </div>
              <p style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)' }}>
                Convergence trending positive. Exploration coverage adequate. 
                No anomalies detected in last 5 iterations.
              </p>
            </div>
          </div>
        )
      case 'explore':
        return (
          <div className="demo-visual">
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '16px' }}>
              <div className="viz-card">
                <div className="viz-title">Parameter Importance</div>
                <div style={{ marginTop: '12px' }}>
                  {['Temperature', 'Pressure', 'Concentration', 'Catalyst Type'].map((param, i) => (
                    <div key={param} style={{ display: 'flex', alignItems: 'center', marginBottom: '8px' }}>
                      <span style={{ width: '100px', fontSize: '0.75rem' }}>{param}</span>
                      <div style={{ flex: 1, height: '8px', background: 'var(--color-bg)', borderRadius: '4px', margin: '0 8px' }}>
                        <div style={{ width: `${[85, 62, 45, 23][i]}%`, height: '100%', background: 'var(--gradient-primary)', borderRadius: '4px' }}></div>
                      </div>
                      <span style={{ fontSize: '0.75rem', fontWeight: 600 }}>{[85, 62, 45, 23][i]}%</span>
                    </div>
                  ))}
                </div>
              </div>
              <div className="viz-card">
                <div className="viz-title">Local Optima Map</div>
                <div style={{ 
                  height: '120px', 
                  background: 'var(--color-bg)', 
                  borderRadius: 'var(--radius)',
                  position: 'relative',
                  marginTop: '12px'
                }}>
                  {/* Scatter plot simulation */}
                  {[
                    { x: 20, y: 30, r: 6 },
                    { x: 45, y: 50, r: 8 },
                    { x: 70, y: 25, r: 5 },
                    { x: 35, y: 70, r: 7 },
                    { x: 80, y: 60, r: 6 },
                  ].map((point, i) => (
                    <div key={i} style={{
                      position: 'absolute',
                      left: `${point.x}%`,
                      top: `${point.y}%`,
                      width: point.r * 2,
                      height: point.r * 2,
                      background: i === 1 ? 'var(--color-green)' : 'var(--color-primary)',
                      borderRadius: '50%',
                      transform: 'translate(-50%, -50%)',
                      opacity: i === 1 ? 1 : 0.6
                    }}></div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )
      case 'suggestions':
        return (
          <div className="demo-visual">
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              {[
                { temp: 320, press: 2.5, yield: 0.84, unc: 0.05, novelty: 0.92 },
                { temp: 285, press: 3.2, yield: 0.81, unc: 0.07, novelty: 0.88 },
                { temp: 340, press: 1.8, yield: 0.79, unc: 0.06, novelty: 0.85 },
              ].map((sugg, i) => (
                <div key={i} style={{
                  display: 'flex',
                  alignItems: 'center',
                  padding: '16px',
                  background: i === 0 ? 'var(--color-primary-subtle)' : 'var(--color-surface)',
                  borderRadius: 'var(--radius)',
                  border: `1px solid ${i === 0 ? 'var(--color-primary)' : 'var(--color-border)'}`,
                }}>
                  <div style={{ width: '40px', fontWeight: 700, color: 'var(--color-primary)' }}>#{i + 1}</div>
                  <div style={{ flex: 1, display: 'flex', gap: '24px', fontSize: '0.875rem' }}>
                    <span>T: {sugg.temp}K</span>
                    <span>P: {sugg.press}bar</span>
                    <span style={{ color: 'var(--color-green)' }}>Yield: {Math.round(sugg.yield * 100)}%</span>
                  </div>
                  <div style={{ display: 'flex', gap: '16px', fontSize: '0.75rem' }}>
                    <span>±{sugg.unc}</span>
                    <span style={{ color: 'var(--color-purple)' }}>Novelty: {sugg.novelty}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )
      default:
        return null
    }
  }

  return (
    <section id="demo" className="section">
      <div className="container">
        <div className="section-header">
          <span className="section-tag">Live Demo</span>
          <h2 className="section-title">
            See It In Action
          </h2>
          <p className="section-subtitle">
            Explore the Workspace with interactive tabs. 
            Every visualization updates in real-time as your campaign progresses.
          </p>
        </div>

        <div className="demo-container">
          <div className="demo-tabs">
            {tabs.map(tab => (
              <button
                key={tab.id}
                className={`demo-tab ${activeTab === tab.id ? 'active' : ''}`}
                onClick={() => setActiveTab(tab.id)}
              >
                {tab.label}
              </button>
            ))}
          </div>
          
          <div className="demo-content">
            {renderDemoContent()}
            
            <div style={{ 
              display: 'flex', 
              gap: '16px', 
              justifyContent: 'center',
              marginTop: '24px',
              flexWrap: 'wrap'
            }}>
              <button 
                onClick={() => {
                  const element = document.getElementById('playground')
                  if (element) element.scrollIntoView({ behavior: 'smooth' })
                }}
                className="btn btn-primary"
              >
                🚀 Try Interactive Demo
              </button>
              <a 
                href="https://raw.githubusercontent.com/SissiFeng/Nexus/main/data/gollum/c2_yield_data.csv"
                target="_blank"
                rel="noopener noreferrer"
                className="btn btn-secondary"
              >
                📁 Download Sample CSV
              </a>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

export default Demo
