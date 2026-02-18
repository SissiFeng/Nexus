import { useState } from 'react'
import { Share2, Globe, Lock, Users, Copy, Check, ExternalLink } from 'lucide-react'

interface ShareToHubProps {
  campaignId: string
  campaignName: string
  onShare?: (result: ShareResult) => void
}

interface ShareResult {
  success: boolean
  hubId?: string
  shareUrl?: string
  error?: string
}

interface ShareOptions {
  title: string
  description: string
  keywords: string
  visibility: 'private' | 'unlisted' | 'public' | 'collaborative'
  license: string
}

export function ShareToHub({ campaignId, campaignName, onShare }: ShareToHubProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [isSharing, setIsSharing] = useState(false)
  const [shareResult, setShareResult] = useState<ShareResult | null>(null)
  const [copied, setCopied] = useState(false)
  
  const [options, setOptions] = useState<ShareOptions>({
    title: campaignName,
    description: '',
    keywords: 'optimization, bayesian',
    visibility: 'unlisted',
    license: 'MIT',
  })

  const handleOneClickShare = async () => {
    setIsSharing(true)
    try {
      const response = await fetch('/api/hub/share/one-click', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ campaign_id: campaignId }),
      })
      
      const result = await response.json()
      setShareResult(result)
      onShare?.(result)
    } catch (error) {
      setShareResult({ success: false, error: 'Network error' })
    } finally {
      setIsSharing(false)
    }
  }

  const handleCustomShare = async () => {
    setIsSharing(true)
    try {
      const response = await fetch('/api/hub/share', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          campaign_data: { campaign_id: campaignId },
          owner_info: { name: 'Nexus User' },
          sharing_options: {
            ...options,
            keywords: options.keywords.split(',').map(k => k.trim()),
          },
        }),
      })
      
      const result = await response.json()
      setShareResult(result)
      onShare?.(result)
    } catch (error) {
      setShareResult({ success: false, error: 'Network error' })
    } finally {
      setIsSharing(false)
    }
  }

  const copyToClipboard = async (text: string) => {
    await navigator.clipboard.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const visibilityIcons = {
    private: <Lock size={16} />,
    unlisted: <Users size={16} />,
    public: <Globe size={16} />,
    collaborative: <Users size={16} />,
  }

  const visibilityLabels = {
    private: 'Private - Only you',
    unlisted: 'Unlisted - Anyone with link',
    public: 'Public - Discoverable in Hub',
    collaborative: 'Collaborative - Specific people',
  }

  if (shareResult?.success) {
    return (
      <div className="share-success-card">
        <div className="share-success-header">
          <div className="share-success-icon">✓</div>
          <h3>Campaign Shared!</h3>
        </div>
        
        <div className="share-url-container">
          <input
            type="text"
            value={shareResult.shareUrl}
            readOnly
            className="share-url-input"
          />
          <button
            onClick={() => copyToClipboard(shareResult.shareUrl!)}
            className="btn btn-secondary"
          >
            {copied ? <Check size={16} /> : <Copy size={16} />}
            {copied ? 'Copied!' : 'Copy'}
          </button>
        </div>
        
        <div className="share-actions">
          <a
            href={shareResult.shareUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="btn btn-primary"
          >
            <ExternalLink size={16} />
            View on Hub
          </a>
          <button
            onClick={() => setShareResult(null)}
            className="btn btn-ghost"
          >
            Share Again
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="share-to-hub">
      {!isOpen ? (
        <button
          onClick={() => setIsOpen(true)}
          className="btn btn-primary btn-share"
        >
          <Share2 size={18} />
          Share to Hub
        </button>
      ) : (
        <div className="share-modal">
          <div className="share-modal-header">
            <h3>Share to Nexus Hub</h3>
            <button
              onClick={() => setIsOpen(false)}
              className="btn-close"
            >
              ×
            </button>
          </div>

          {shareResult?.error && (
            <div className="share-error">
              {shareResult.error}
            </div>
          )}

          {/* One-Click Share */}
          <div className="share-section">
            <button
              onClick={handleOneClickShare}
              disabled={isSharing}
              className="btn btn-primary btn-block"
            >
              {isSharing ? 'Sharing...' : '⚡ One-Click Share'}
            </button>
            <p className="share-hint">
              Instantly share with sensible defaults (unlisted, MIT license)
            </p>
          </div>

          <div className="share-divider">
            <span>or customize</span>
          </div>

          {/* Custom Options */}
          <div className="share-form">
            <div className="form-group">
              <label>Title</label>
              <input
                type="text"
                value={options.title}
                onChange={(e) => setOptions({ ...options, title: e.target.value })}
                className="form-input"
              />
            </div>

            <div className="form-group">
              <label>Description</label>
              <textarea
                value={options.description}
                onChange={(e) => setOptions({ ...options, description: e.target.value })}
                className="form-textarea"
                rows={3}
              />
            </div>

            <div className="form-group">
              <label>Keywords (comma-separated)</label>
              <input
                type="text"
                value={options.keywords}
                onChange={(e) => setOptions({ ...options, keywords: e.target.value })}
                className="form-input"
                placeholder="optimization, catalysis, ml"
              />
            </div>

            <div className="form-group">
              <label>Visibility</label>
              <div className="visibility-options">
                {(Object.keys(visibilityLabels) as Array<keyof typeof visibilityLabels>).map((v) => (
                  <label
                    key={v}
                    className={`visibility-option ${options.visibility === v ? 'active' : ''}`}
                  >
                    <input
                      type="radio"
                      name="visibility"
                      value={v}
                      checked={options.visibility === v}
                      onChange={(e) => setOptions({ ...options, visibility: e.target.value as any })}
                    />
                    <span className="visibility-icon">{visibilityIcons[v]}</span>
                    <span className="visibility-label">{visibilityLabels[v]}</span>
                  </label>
                ))}
              </div>
            </div>

            <div className="form-group">
              <label>License</label>
              <select
                value={options.license}
                onChange={(e) => setOptions({ ...options, license: e.target.value })}
                className="form-select"
              >
                <option value="MIT">MIT (Permissive)</option>
                <option value="Apache-2.0">Apache 2.0</option>
                <option value="CC-BY-4.0">CC BY 4.0</option>
                <option value="CC0-1.0">CC0 (Public Domain)</option>
              </select>
            </div>

            <button
              onClick={handleCustomShare}
              disabled={isSharing}
              className="btn btn-primary btn-block"
            >
              {isSharing ? 'Sharing...' : 'Share with Custom Settings'}
            </button>
          </div>

          {/* Free Tier Info */}
          <div className="share-tier-info">
            <p>
              <strong>Free Tier:</strong> 10 campaigns, 1 GB storage
            </p>
            <a href="/hub/pricing" className="link">
              Upgrade for more
            </a>
          </div>
        </div>
      )}

      <style>{`
        .share-to-hub {
          position: relative;
        }

        .btn-share {
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .share-modal {
          background: var(--color-surface);
          border: 1px solid var(--color-border);
          border-radius: var(--radius-lg);
          padding: 24px;
          width: 400px;
          max-width: 90vw;
          box-shadow: var(--shadow-lg);
        }

        .share-modal-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 20px;
        }

        .share-modal-header h3 {
          margin: 0;
          font-size: 1.25rem;
        }

        .btn-close {
          background: none;
          border: none;
          font-size: 1.5rem;
          cursor: pointer;
          color: var(--color-text-muted);
        }

        .share-error {
          background: #dc354515;
          color: #dc3545;
          padding: 12px;
          border-radius: var(--radius);
          margin-bottom: 16px;
          font-size: 0.875rem;
        }

        .share-section {
          margin-bottom: 20px;
        }

        .share-hint {
          font-size: 0.75rem;
          color: var(--color-text-muted);
          margin-top: 8px;
          text-align: center;
        }

        .share-divider {
          text-align: center;
          margin: 20px 0;
          position: relative;
        }

        .share-divider::before {
          content: '';
          position: absolute;
          top: 50%;
          left: 0;
          right: 0;
          height: 1px;
          background: var(--color-border);
        }

        .share-divider span {
          background: var(--color-surface);
          padding: 0 12px;
          position: relative;
          color: var(--color-text-muted);
          font-size: 0.875rem;
        }

        .share-form {
          display: flex;
          flex-direction: column;
          gap: 16px;
        }

        .form-group {
          display: flex;
          flex-direction: column;
          gap: 6px;
        }

        .form-group label {
          font-size: 0.875rem;
          font-weight: 500;
        }

        .form-input,
        .form-textarea,
        .form-select {
          padding: 10px 12px;
          border: 1px solid var(--color-border);
          border-radius: var(--radius);
          background: var(--color-bg);
          font-family: inherit;
          font-size: 0.875rem;
        }

        .form-input:focus,
        .form-textarea:focus,
        .form-select:focus {
          outline: none;
          border-color: var(--color-primary);
        }

        .visibility-options {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }

        .visibility-option {
          display: flex;
          align-items: center;
          gap: 10px;
          padding: 10px 12px;
          border: 1px solid var(--color-border);
          border-radius: var(--radius);
          cursor: pointer;
          transition: all 0.2s;
        }

        .visibility-option:hover {
          border-color: var(--color-primary);
        }

        .visibility-option.active {
          border-color: var(--color-primary);
          background: var(--color-primary-subtle);
        }

        .visibility-option input {
          display: none;
        }

        .visibility-icon {
          color: var(--color-text-muted);
        }

        .visibility-label {
          font-size: 0.875rem;
        }

        .btn-block {
          width: 100%;
          justify-content: center;
        }

        .share-tier-info {
          margin-top: 20px;
          padding-top: 16px;
          border-top: 1px solid var(--color-border);
          display: flex;
          justify-content: space-between;
          align-items: center;
          font-size: 0.75rem;
          color: var(--color-text-muted);
        }

        .link {
          color: var(--color-primary);
          text-decoration: none;
        }

        .link:hover {
          text-decoration: underline;
        }

        /* Success State */
        .share-success-card {
          background: var(--color-surface);
          border: 1px solid var(--color-green);
          border-radius: var(--radius-lg);
          padding: 24px;
          text-align: center;
        }

        .share-success-header {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 12px;
          margin-bottom: 20px;
        }

        .share-success-icon {
          width: 40px;
          height: 40px;
          background: var(--color-green);
          color: white;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 1.25rem;
          font-weight: bold;
        }

        .share-success-header h3 {
          margin: 0;
          color: var(--color-green);
        }

        .share-url-container {
          display: flex;
          gap: 8px;
          margin-bottom: 20px;
        }

        .share-url-input {
          flex: 1;
          padding: 10px 12px;
          border: 1px solid var(--color-border);
          border-radius: var(--radius);
          background: var(--color-bg);
          font-family: var(--font-mono);
          font-size: 0.75rem;
        }

        .share-actions {
          display: flex;
          gap: 12px;
          justify-content: center;
        }

        .btn-ghost {
          background: transparent;
          border: 1px solid var(--color-border);
          color: var(--color-text);
        }

        .btn-ghost:hover {
          background: var(--color-bg);
        }
      `}</style>
    </div>
  )
}
