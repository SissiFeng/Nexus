import { useState, useEffect } from 'react'

interface NavigationProps {
  theme: 'light' | 'dark'
  toggleTheme: () => void
}

function Navigation({ theme, toggleTheme }: NavigationProps) {
  const [scrolled, setScrolled] = useState(false)

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 50)
    }

    window.addEventListener('scroll', handleScroll)
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  const scrollToSection = (id: string) => {
    const element = document.getElementById(id)
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' })
    }
  }

  return (
    <nav className="nav" style={{
      boxShadow: scrolled ? 'var(--shadow-md)' : 'none',
      background: scrolled ? 'rgba(240, 242, 245, 0.95)' : 'rgba(240, 242, 245, 0.9)'
    }}>
      <div className="container nav-content">
        <a href="#" className="nav-brand">
          <span className="nav-brand-icon">◆</span>
          Nexus
        </a>

        <div className="nav-links">
          <button onClick={() => scrollToSection('problem')} className="nav-link">
            Problem
          </button>
          <button onClick={() => scrollToSection('solution')} className="nav-link">
            Solution
          </button>
          <button onClick={() => scrollToSection('features')} className="nav-link">
            Features
          </button>
          <button onClick={() => scrollToSection('playground')} className="nav-link" style={{ color: 'var(--color-primary)', fontWeight: 600 }}>
            ▶ Try It
          </button>
          <button onClick={() => scrollToSection('demo')} className="nav-link">
            Demo
          </button>
        </div>

        <div className="nav-right">
          <button onClick={toggleTheme} className="theme-toggle" aria-label="Toggle theme">
            {theme === 'light' ? '🌙' : '☀️'}
          </button>
          <a 
            href="https://github.com/SissiFeng/Nexus" 
            target="_blank" 
            rel="noopener noreferrer"
            className="btn btn-secondary"
          >
            GitHub
          </a>
        </div>
      </div>
    </nav>
  )
}

export default Navigation
