import { Share2, Globe, Shield, Users, Database, Sparkles } from 'lucide-react'

function HubSection() {
  const features = [
    {
      icon: Share2,
      title: 'One-Click Share',
      description: 'Share your optimization campaigns instantly with a single click. Get a permanent link to share with collaborators.',
      color: '#3d5af1'
    },
    {
      icon: Database,
      title: 'FAIR Metadata',
      description: 'Automatic FAIR-compliant metadata generation. Schema.org JSON-LD and DataCite XML formats for DOI registration.',
      color: '#0fa968'
    },
    {
      icon: Shield,
      title: 'Access Control',
      description: 'Private, unlisted, public, or collaborative visibility. Granular permissions: view, fork, comment, edit.',
      color: '#d4940a'
    },
    {
      icon: Users,
      title: 'Collaborative',
      description: 'Invite specific collaborators with different access levels. Fork shared campaigns to your workspace.',
      color: '#dc3545'
    }
  ]

  const fairPrinciples = [
    { letter: 'F', title: 'Findable', desc: 'DOI-style identifiers, searchable metadata, keywords' },
    { letter: 'A', title: 'Accessible', desc: 'Retrievable via standard protocols, clear access levels' },
    { letter: 'I', title: 'Interoperable', desc: 'Schema.org JSON-LD, DataCite XML, formal parameter specs' },
    { letter: 'R', title: 'Reusable', desc: 'Rich metadata, clear licensing (MIT, Apache, CC-BY)' }
  ]

  return (
    <section id="hub" className="section" style={{ background: 'var(--color-bg)' }}>
      <div className="container">
        <div className="section-header">
          <span className="section-tag" style={{ 
            background: 'linear-gradient(135deg, #3d5af1 0%, #6c4cf0 100%)',
            color: 'white'
          }}>
            <Sparkles size={14} style={{ marginRight: '6px', display: 'inline' }} />
            New: Nexus Hub
          </span>
          <h2 className="section-title">
            Share Your Research
          </h2>
          <p className="section-subtitle">
            Cloud platform for sharing optimization campaigns with FAIR-compliant metadata
          </p>
        </div>

        {/* Main Hub Card */}
        <div style={{
          background: 'linear-gradient(135deg, #1a1f36 0%, #0f1320 100%)',
          borderRadius: 'var(--radius-lg)',
          padding: '48px',
          marginBottom: '48px',
          position: 'relative',
          overflow: 'hidden'
        }}>
          {/* Background decoration */}
          <div style={{
            position: 'absolute',
            top: '-50%',
            right: '-20%',
            width: '600px',
            height: '600px',
            background: 'radial-gradient(circle, rgba(61,90,241,0.15) 0%, transparent 70%)',
            pointerEvents: 'none'
          }} />

          <div style={{ position: 'relative', zIndex: 1 }}>
            <div style={{
              display: 'grid',
              gridTemplateColumns: '1fr 1fr',
              gap: '48px',
              alignItems: 'center'
            }}>
              <div>
                <h3 style={{
                  fontSize: '2rem',
                  color: 'white',
                  marginBottom: '16px'
                }}>
                  Nexus Hub
                </h3>
                <p style={{
                  color: '#9aa5b8',
                  fontSize: '1rem',
                  lineHeight: 1.6,
                  marginBottom: '24px'
                }}>
                  Share your optimization campaigns with the world. Generate FAIR-compliant 
                  metadata, get permanent links, and collaborate with other researchers. 
                  All with a single click.
                </p>
                
                <div style={{ display: 'flex', gap: '12px', marginBottom: '24px' }}>
                  <span style={{
                    padding: '6px 14px',
                    background: 'rgba(15, 169, 104, 0.2)',
                    color: '#0fa968',
                    borderRadius: '100px',
                    fontSize: '0.75rem',
                    fontWeight: 600
                  }}>
                    Free Tier Available
                  </span>
                  <span style={{
                    padding: '6px 14px',
                    background: 'rgba(61, 90, 241, 0.2)',
                    color: '#3d5af1',
                    borderRadius: '100px',
                    fontSize: '0.75rem',
                    fontWeight: 600
                  }}>
                    FAIR Compliant
                  </span>
                </div>

                <div style={{ display: 'flex', gap: '12px' }}>
                  <a 
                    href="https://hub.nexus.dev" 
                    target="_blank"
                    rel="noopener noreferrer"
                    className="btn"
                    style={{
                      background: '#3d5af1',
                      color: 'white',
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '8px'
                    }}
                  >
                    <Globe size={18} />
                    Visit Nexus Hub
                  </a>
                  <button 
                    className="btn"
                    style={{
                      background: 'transparent',
                      border: '1px solid rgba(255,255,255,0.2)',
                      color: 'white'
                    }}
                    onClick={() => {
                      const element = document.getElementById('features')
                      if (element) element.scrollIntoView({ behavior: 'smooth' })
                    }}
                  >
                    Learn More
                  </button>
                </div>
              </div>

              {/* Code Example */}
              <div style={{
                background: 'rgba(0,0,0,0.3)',
                borderRadius: 'var(--radius)',
                padding: '24px',
                fontFamily: 'var(--font-mono)',
                fontSize: '0.8rem'
              }}>
                <div style={{ 
                  display: 'flex', 
                  gap: '8px', 
                  marginBottom: '16px',
                  paddingBottom: '12px',
                  borderBottom: '1px solid rgba(255,255,255,0.1)'
                }}>
                  <div style={{ width: '12px', height: '12px', borderRadius: '50%', background: '#ff5f57' }} />
                  <div style={{ width: '12px', height: '12px', borderRadius: '50%', background: '#febc2e' }} />
                  <div style={{ width: '12px', height: '12px', borderRadius: '50%', background: '#28c840' }} />
                  <span style={{ marginLeft: 'auto', color: '#5d6b7e', fontSize: '0.7rem' }}>
                    POST /api/hub/share/one-click
                  </span>
                </div>
                <pre style={{ margin: 0, color: '#e8ecf2', lineHeight: 1.6 }}>
                  <code>{`{
  "hub_id": "hub_a7f2b9c1",
  "share_url": "https://hub.nexus.dev/c/hub_a7f2b9c1",
  "fair_metadata": {
    "@context": "https://schema.org",
    "@type": "Dataset",
    "name": "Catalyst Optimization",
    "creator": {
      "name": "Jane Doe",
      "orcid": "0000-0001-2345..."
    },
    "license": "MIT"
  }
}`}</code>
                </pre>
              </div>
            </div>
          </div>
        </div>

        {/* Features Grid */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(4, 1fr)',
          gap: '24px',
          marginBottom: '64px'
        }}>
          {features.map((feature, index) => (
            <div key={index} style={{
              padding: '28px',
              background: 'var(--color-surface)',
              borderRadius: 'var(--radius-lg)',
              border: '1px solid var(--color-border)',
              transition: 'all 0.2s'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.borderColor = feature.color
              e.currentTarget.style.transform = 'translateY(-4px)'
              e.currentTarget.style.boxShadow = `0 12px 32px ${feature.color}10`
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.borderColor = 'var(--color-border)'
              e.currentTarget.style.transform = 'translateY(0)'
              e.currentTarget.style.boxShadow = 'none'
            }}
            >
              <div style={{
                width: '48px',
                height: '48px',
                background: `${feature.color}15`,
                borderRadius: 'var(--radius)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                marginBottom: '16px'
              }}>
                <feature.icon size={24} style={{ color: feature.color }} />
              </div>
              <h4 style={{ fontSize: '1.125rem', marginBottom: '8px' }}>
                {feature.title}
              </h4>
              <p style={{
                fontSize: '0.875rem',
                color: 'var(--color-text-muted)',
                lineHeight: 1.5
              }}>
                {feature.description}
              </p>
            </div>
          ))}
        </div>

        {/* FAIR Principles */}
        <div style={{
          background: 'var(--color-surface)',
          borderRadius: 'var(--radius-lg)',
          padding: '32px',
          border: '1px solid var(--color-border)'
        }}>
          <h3 style={{
            fontSize: '1.25rem',
            textAlign: 'center',
            marginBottom: '32px'
          }}>
            FAIR Principles Compliant
          </h3>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(4, 1fr)',
            gap: '24px'
          }}>
            {fairPrinciples.map((principle, index) => (
              <div key={index} style={{ textAlign: 'center' }}>
                <div style={{
                  width: '56px',
                  height: '56px',
                  margin: '0 auto 12px',
                  background: 'var(--gradient-primary)',
                  borderRadius: '50%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'white',
                  fontSize: '1.5rem',
                  fontWeight: 700
                }}>
                  {principle.letter}
                </div>
                <h5 style={{ fontSize: '1rem', marginBottom: '4px' }}>
                  {principle.title}
                </h5>
                <p style={{
                  fontSize: '0.75rem',
                  color: 'var(--color-text-muted)'
                }}>
                  {principle.desc}
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* Free Tier Info */}
        <div style={{
          marginTop: '32px',
          display: 'grid',
          gridTemplateColumns: 'repeat(4, 1fr)',
          gap: '16px'
        }}>
          {[
            { value: '10', label: 'Campaigns' },
            { value: '1 GB', label: 'Storage' },
            { value: '10K', label: 'API calls/month' },
            { value: '20', label: 'Collaborators' }
          ].map((item, index) => (
            <div key={index} style={{
              padding: '20px',
              background: 'var(--color-surface)',
              borderRadius: 'var(--radius)',
              border: '1px solid var(--color-border)',
              textAlign: 'center'
            }}>
              <div style={{
                fontSize: '1.75rem',
                fontWeight: 700,
                color: 'var(--color-primary)',
                marginBottom: '4px'
              }}>
                {item.value}
              </div>
              <div style={{
                fontSize: '0.75rem',
                color: 'var(--color-text-muted)'
              }}>
                {item.label}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

export default HubSection
