import { Upload, AlertCircle, Sparkles, ArrowRight } from 'lucide-react'

function DemoVideos() {
  const demos = [
    {
      id: 'upload',
      icon: Upload,
      title: 'Upload Your Data',
      description: 'Drag & drop CSV files. Automatic column detection and type inference.',
      duration: '0:15',
      placeholder: '📁 CSV Upload',
      color: '#3d5af1'
    },
    {
      id: 'diagnose',
      icon: AlertCircle,
      title: 'Real-Time Diagnostics',
      description: '17 health signals with traffic-light indicators catch problems early.',
      duration: '0:15',
      placeholder: '🔴 Red Flag Alert',
      color: '#dc3545'
    },
    {
      id: 'suggest',
      icon: Sparkles,
      title: 'AI-Powered Suggestions',
      description: 'Get next experiments with uncertainty estimates and novelty scores.',
      duration: '0:15',
      placeholder: '✨ Smart Suggestions',
      color: '#0fa968'
    }
  ]

  return (
    <section id="demos" className="section" style={{ background: 'var(--color-surface)' }}>
      <div className="container">
        <div className="section-header">
          <span className="section-tag">Watch Demo</span>
          <h2 className="section-title">
            From Upload to Insights in 45 Seconds
          </h2>
          <p className="section-subtitle">
            See how Nexus transforms raw data into actionable optimization strategies
          </p>
        </div>

        {/* Video Demo Grid */}
        <div className="demo-videos-grid">
          {demos.map((demo, index) => (
            <div key={demo.id} className="demo-video-card">
              {/* Video Placeholder */}
              <div 
                className="video-placeholder"
                style={{ 
                  background: `linear-gradient(135deg, ${demo.color}15 0%, ${demo.color}08 100%)`,
                  borderColor: `${demo.color}30`
                }}
              >
                <div className="video-placeholder-content">
                  <demo.icon size={48} style={{ color: demo.color }} />
                  <span className="video-placeholder-text">{demo.placeholder}</span>
                  <span className="video-duration">{demo.duration}</span>
                </div>
                
                {/* Play Button Overlay */}
                <div className="video-play-overlay">
                  <div className="play-button">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
                      <polygon points="5 3 19 12 5 21 5 3" />
                    </svg>
                  </div>
                </div>
              </div>
              
              {/* Step Number */}
              <div className="demo-step-number" style={{ background: demo.color }}>
                {index + 1}
              </div>
              
              {/* Card Content */}
              <div className="demo-video-content">
                <h3 className="demo-video-title">{demo.title}</h3>
                <p className="demo-video-desc">{demo.description}</p>
              </div>
              
              {/* Arrow Connector (except last) */}
              {index < demos.length - 1 && (
                <div className="demo-arrow">
                  <ArrowRight size={24} />
                </div>
              )}
            </div>
          ))}
        </div>

        {/* GitHub Codespaces CTA */}
        <div className="codespaces-cta">
          <div className="codespaces-content">
            <div className="codespaces-badge">
              <svg viewBox="0 0 16 16" width="16" height="16" fill="currentColor">
                <path d="M4.25 2A2.25 2.25 0 0 0 2 4.25v7.5A2.25 2.25 0 0 0 4.25 14h7.5A2.25 2.25 0 0 0 14 11.75v-7.5A2.25 2.25 0 0 0 11.75 2h-7.5Zm0 1.5h7.5a.75.75 0 0 1 .75.75v7.5a.75.75 0 0 1-.75.75h-7.5a.75.75 0 0 1-.75-.75v-7.5a.75.75 0 0 1 .75-.75ZM5 5.75A.75.75 0 0 1 5.75 5h.5a.75.75 0 0 1 0 1.5h-.5A.75.75 0 0 1 5 5.75Zm3 0A.75.75 0 0 1 8.75 5h.5a.75.75 0 0 1 0 1.5h-.5A.75.75 0 0 1 8 5.75Zm-3 3A.75.75 0 0 1 5.75 8h.5a.75.75 0 0 1 0 1.5h-.5A.75.75 0 0 1 5 8.75Zm3 0A.75.75 0 0 1 8.75 8h.5a.75.75 0 0 1 0 1.5h-.5A.75.75 0 0 1 8 8.75Zm3 0A.75.75 0 0 1 11.75 8h.5a.75.75 0 0 1 0 1.5h-.5A.75.75 0 0 1 11 8.75ZM5 11.75a.75.75 0 0 1 .75-.75h4.5a.75.75 0 0 1 0 1.5h-4.5a.75.75 0 0 1-.75-.75Z"/>
              </svg>
              Try in Browser
            </div>
            <h3 className="codespaces-title">No Installation Required</h3>
            <p className="codespaces-desc">
              Launch Nexus instantly in GitHub Codespaces — free tier includes 120 core-hours/month
            </p>
            <a 
              href="https://codespaces.new/SissiFeng/Nexus?quickstart=1"
              target="_blank"
              rel="noopener noreferrer"
              className="codespaces-button"
            >
              <svg viewBox="0 0 16 16" width="20" height="20" fill="currentColor">
                <path d="M4.25 2A2.25 2.25 0 0 0 2 4.25v7.5A2.25 2.25 0 0 0 4.25 14h7.5A2.25 2.25 0 0 0 14 11.75v-7.5A2.25 2.25 0 0 0 11.75 2h-7.5Zm0 1.5h7.5a.75.75 0 0 1 .75.75v7.5a.75.75 0 0 1-.75.75h-7.5a.75.75 0 0 1-.75-.75v-7.5a.75.75 0 0 1 .75-.75ZM5 5.75A.75.75 0 0 1 5.75 5h.5a.75.75 0 0 1 0 1.5h-.5A.75.75 0 0 1 5 5.75Zm3 0A.75.75 0 0 1 8.75 5h.5a.75.75 0 0 1 0 1.5h-.5A.75.75 0 0 1 8 5.75Zm-3 3A.75.75 0 0 1 5.75 8h.5a.75.75 0 0 1 0 1.5h-.5A.75.75 0 0 1 5 8.75Zm3 0A.75.75 0 0 1 8.75 8h.5a.75.75 0 0 1 0 1.5h-.5A.75.75 0 0 1 8 8.75Zm3 0A.75.75 0 0 1 11.75 8h.5a.75.75 0 0 1 0 1.5h-.5A.75.75 0 0 1 11 8.75ZM5 11.75a.75.75 0 0 1 .75-.75h4.5a.75.75 0 0 1 0 1.5h-4.5a.75.75 0 0 1-.75-.75Z"/>
              </svg>
              Open in GitHub Codespaces
            </a>
            <div className="codespaces-note">
              <span className="codespaces-dot"></span>
              Ready in ~2 minutes • No credit card required
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

export default DemoVideos
