function Footer() {
  const currentYear = new Date().getFullYear()

  return (
    <footer className="footer">
      <div className="container">
        <div className="footer-content">
          <div>
            <div className="footer-brand">
              <span style={{ color: 'var(--color-primary)', fontSize: '1.5rem' }}>◆</span>
              Nexus
            </div>
            <p className="footer-desc">
              Intelligent Optimization Platform for Scientific Experiments. 
              Transform your black-box optimization into a transparent, 
              trusted research partner.
            </p>
          </div>

          <div>
            <h4 className="footer-title">Product</h4>
            <div className="footer-links">
              <a href="#features" className="footer-link">Features</a>
              <a href="#demo" className="footer-link">Demo</a>
              <a href="#usecases" className="footer-link">Use Cases</a>
              <a href="https://github.com/SissiFeng/Nexus/blob/main/README.md" target="_blank" rel="noopener noreferrer" className="footer-link">Documentation</a>
            </div>
          </div>

          <div>
            <h4 className="footer-title">Resources</h4>
            <div className="footer-links">
              <a href="https://github.com/SissiFeng/Nexus/tree/main/demo-datasets" target="_blank" rel="noopener noreferrer" className="footer-link">Demo Datasets</a>
              <a href="https://github.com/SissiFeng/Nexus/blob/main/CLAUDE.md" target="_blank" rel="noopener noreferrer" className="footer-link">Development Guide</a>
              <a href="https://github.com/SissiFeng/Nexus/issues" target="_blank" rel="noopener noreferrer" className="footer-link">Issues</a>
              <a href="https://github.com/SissiFeng/Nexus/discussions" target="_blank" rel="noopener noreferrer" className="footer-link">Discussions</a>
            </div>
          </div>

          <div>
            <h4 className="footer-title">Connect</h4>
            <div className="footer-links">
              <a href="https://github.com/SissiFeng/Nexus" target="_blank" rel="noopener noreferrer" className="footer-link">GitHub</a>
              <a href="https://github.com/SissiFeng/Nexus/stargazers" target="_blank" rel="noopener noreferrer" className="footer-link">Star the Project</a>
              <a href="https://github.com/SissiFeng/Nexus/fork" target="_blank" rel="noopener noreferrer" className="footer-link">Fork & Contribute</a>
            </div>
          </div>
        </div>

        <div className="footer-bottom">
          <div>
            © {currentYear} Nexus. Built with Claude Code. MIT License.
          </div>
          <div className="footer-social">
            <a href="https://github.com/SissiFeng/Nexus" target="_blank" rel="noopener noreferrer" className="social-link">
              GitHub
            </a>
            <span>|</span>
            <span style={{ color: 'rgba(255, 255, 255, 0.5)' }}>
              Made with ❤️ for Science
            </span>
          </div>
        </div>
      </div>
    </footer>
  )
}

export default Footer
