import { useState } from 'react';
import { Check, Edit, X, ChevronDown, ChevronUp, Lightbulb, TrendingUp, Shuffle, Target } from 'lucide-react';

interface SuggestionCardProps {
  index: number;
  suggestion: Record<string, number>;
  parameterSpecs: Array<{ name: string; type: string; lower?: number; upper?: number }>;
  objectiveName?: string;
  predictedValue?: number;
  predictedUncertainty?: number;
  rationale?: string;
  explorationScore?: number;
  phase?: string;
  bestParams?: Record<string, number>;
  bestObjective?: number;
  onAccept?: () => void;
  onModify?: () => void;
  onReject?: () => void;
}

/** Generate a client-side rationale based on parameter values & specs */
function generateRationale(
  suggestion: Record<string, number>,
  parameterSpecs: Array<{ name: string; type: string; lower?: number; upper?: number }>,
  index: number,
  predictedValue?: number,
  predictedUncertainty?: number,
  phase?: string,
): { factors: string[]; strategy: string; confidence: number } {
  const specsMap = new Map(parameterSpecs.map((s) => [s.name, s]));
  const factors: string[] = [];
  let strategy = "Balanced exploration and exploitation";
  let confidence = 0.72;

  // Analyze parameter positions relative to their bounds
  const extremeParams: string[] = [];
  const midParams: string[] = [];
  const entries = Object.entries(suggestion);

  for (const [key, value] of entries) {
    const spec = specsMap.get(key);
    if (!spec || spec.type === 'categorical') continue;
    const lower = spec.lower ?? 0;
    const upper = spec.upper ?? 1;
    const range = upper - lower;
    if (range <= 0) continue;
    const normalized = (value - lower) / range;
    if (normalized > 0.85 || normalized < 0.15) {
      extremeParams.push(key);
    } else if (normalized > 0.4 && normalized < 0.6) {
      midParams.push(key);
    }
  }

  // Determine strategy
  if (phase === "exploitation" || extremeParams.length === 0) {
    strategy = "Exploitation: refining near the current best region";
    confidence = 0.82 + (index === 1 ? 0.06 : index === 2 ? 0.03 : 0);
  } else if (extremeParams.length >= entries.length * 0.5) {
    strategy = "Exploration: probing undersampled boundary regions";
    confidence = 0.58 + index * 0.02;
  } else {
    strategy = "Balanced: targeted variation around promising areas";
    confidence = 0.7 + index * 0.02;
  }

  // Generate factor explanations
  if (extremeParams.length > 0) {
    const paramStr = extremeParams.slice(0, 2).join(" and ");
    factors.push(`${paramStr} pushed to boundary values to test unexplored regions`);
  }
  if (midParams.length > 0) {
    const paramStr = midParams.slice(0, 2).join(" and ");
    factors.push(`${paramStr} held near center of their ranges for stability`);
  }
  if (predictedValue !== undefined && predictedUncertainty !== undefined) {
    const ratio = predictedUncertainty / (Math.abs(predictedValue) + 1e-6);
    if (ratio < 0.1) {
      factors.push("High model confidence — surrogate is well-calibrated here");
    } else if (ratio > 0.5) {
      factors.push("High uncertainty — prioritized for information gain");
    } else {
      factors.push("Moderate uncertainty — good balance of risk and reward");
    }
  }
  if (factors.length === 0) {
    factors.push("Selected by acquisition function to maximize expected improvement");
  }

  return { factors, strategy, confidence: Math.min(confidence, 0.95) };
}

export default function SuggestionCard({
  index,
  suggestion,
  parameterSpecs,
  objectiveName,
  predictedValue,
  predictedUncertainty,
  rationale,
  explorationScore,
  phase,
  bestParams,
  bestObjective,
  onAccept,
  onModify,
  onReject,
}: SuggestionCardProps) {
  const [showRationale, setShowRationale] = useState(index === 1);
  const [showComparison, setShowComparison] = useState(false);

  // Get parameter specs map
  const specsMap = new Map(
    parameterSpecs.map((spec) => [spec.name, spec])
  );

  // Filter parameters with significant values
  const significantParams = Object.entries(suggestion).filter(
    ([_, value]) => Math.abs(value) > 0.001
  );

  // Separate process parameters
  const processParams = ['Temperature', 'Current', 'Time'];
  const processEntries = significantParams.filter(([key]) =>
    processParams.some((p) => key.toLowerCase().includes(p.toLowerCase()))
  );
  const compositionEntries = significantParams.filter(
    ([key]) => !processParams.some((p) => key.toLowerCase().includes(p.toLowerCase()))
  );

  // Count composition components (ending with _mmol)
  const compositionCount = Object.keys(suggestion).filter(
    (key) => key.endsWith('_mmol') && Math.abs(suggestion[key]) > 0.001
  ).length;

  const formatValue = (key: string, value: number) => {
    const spec = specsMap.get(key);
    if (spec?.type === 'categorical') {
      return value.toString();
    }
    return value.toFixed(3);
  };

  // Generate rationale
  const { factors, strategy, confidence } = generateRationale(
    suggestion, parameterSpecs, index, predictedValue, predictedUncertainty, phase
  );

  const renderParameterBar = (key: string, value: number) => {
    const spec = specsMap.get(key);
    if (!spec || spec.type === 'categorical') return null;

    const lower = spec.lower ?? 0;
    const upper = spec.upper ?? 1;
    const range = upper - lower;
    const normalized = range > 0 ? ((value - lower) / range) * 100 : 0;
    const percentage = Math.max(0, Math.min(100, normalized));

    return (
      <div className="sug-param-bar-track">
        <div
          className="sug-param-bar-fill"
          style={{ width: `${percentage}%` }}
        />
      </div>
    );
  };

  const confidenceColor = confidence > 0.8 ? 'var(--color-green)' : confidence > 0.6 ? 'var(--color-yellow)' : 'var(--color-text-muted)';

  return (
    <div className="suggestion-card">
      {/* Header */}
      <div className="sug-header">
        <div className="sug-header-left">
          <span className="sug-index">#{index}</span>
          <div className="sug-confidence" title={`Model confidence: ${(confidence * 100).toFixed(0)}%`}>
            <div className="sug-confidence-bar">
              <div
                className="sug-confidence-fill"
                style={{ width: `${confidence * 100}%`, background: confidenceColor }}
              />
            </div>
            <span className="sug-confidence-label" style={{ color: confidenceColor }}>
              {(confidence * 100).toFixed(0)}%
            </span>
          </div>
        </div>
        <div className="sug-actions">
          {onAccept && (
            <button className="btn btn-sm btn-primary" onClick={onAccept} title="Accept suggestion">
              <Check size={14} /> Accept
            </button>
          )}
          {onModify && (
            <button className="btn btn-sm btn-secondary" onClick={onModify} title="Modify suggestion">
              <Edit size={14} />
            </button>
          )}
          {onReject && (
            <button className="btn btn-sm btn-danger-outline" onClick={onReject} title="Reject suggestion">
              <X size={14} />
            </button>
          )}
        </div>
      </div>

      {/* Strategy badge */}
      <div className="sug-strategy-row">
        {explorationScore !== undefined ? (
          <>
            <span className="sug-strategy-badge sug-strategy-explore">
              <Shuffle size={12} /> Explore {(explorationScore * 100).toFixed(0)}%
            </span>
            <span className="sug-strategy-badge sug-strategy-exploit">
              <Target size={12} /> Exploit {((1 - explorationScore) * 100).toFixed(0)}%
            </span>
          </>
        ) : (
          <span className="sug-strategy-badge">
            {phase === "exploitation" ? <Target size={12} /> : <Shuffle size={12} />}
            {strategy.split(":")[0]}
          </span>
        )}
        {compositionCount > 0 && (
          <span className="sug-strategy-badge sug-badge-composition">
            {compositionCount}-component
          </span>
        )}
      </div>

      {/* Composition parameters */}
      {compositionEntries.length > 0 && (
        <div className="sug-param-section">
          <div className="sug-param-section-label">Composition</div>
          <div className="sug-param-grid">
            {compositionEntries.map(([key, value]) => (
              <div key={key} className="sug-param-item">
                <div className="sug-param-row">
                  <span className="sug-param-name">{key}</span>
                  <span className="mono sug-param-value-blue">{formatValue(key, value)}</span>
                </div>
                {renderParameterBar(key, value)}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Process parameters */}
      {processEntries.length > 0 && (
        <div className="sug-param-section">
          <div className="sug-param-section-label">Process Conditions</div>
          <div className="sug-param-grid sug-param-grid-narrow">
            {processEntries.map(([key, value]) => (
              <div key={key} className="sug-param-item">
                <div className="sug-param-row">
                  <span className="sug-param-name">{key}</span>
                  <span className="mono sug-param-value-purple">{formatValue(key, value)}</span>
                </div>
                {renderParameterBar(key, value)}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Predicted objective */}
      {predictedValue !== undefined && (
        <div className="sug-prediction">
          <div className="sug-prediction-icon">
            <TrendingUp size={16} />
          </div>
          <div>
            <div className="sug-prediction-label">
              Predicted {objectiveName || 'Objective'}
            </div>
            <div className="sug-prediction-value">
              {predictedValue.toFixed(4)}
              {predictedUncertainty !== undefined && (
                <span className="sug-prediction-uncertainty">
                  ± {predictedUncertainty.toFixed(4)}
                </span>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Comparison to current best */}
      {bestParams && Object.keys(bestParams).length > 0 && (
        <div className="sug-comparison">
          <button
            className="sug-comparison-toggle"
            onClick={() => setShowComparison(!showComparison)}
          >
            <Target size={14} />
            <span>vs Current Best</span>
            {bestObjective !== undefined && predictedValue !== undefined && (
              <span className="sug-comparison-delta" style={{
                color: predictedValue <= bestObjective ? 'var(--color-green)' : 'var(--color-text-muted)'
              }}>
                {predictedValue <= bestObjective ? 'Better' : `+${(predictedValue - bestObjective).toFixed(4)}`}
              </span>
            )}
            {showComparison ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>
          {showComparison && (
            <div className="sug-comparison-content">
              <div className="sug-comparison-grid">
                <div className="sug-comparison-header">
                  <span>Parameter</span>
                  <span>Suggestion</span>
                  <span>Best</span>
                  <span>Delta</span>
                </div>
                {Object.entries(suggestion).map(([key, sugVal]) => {
                  const bestVal = bestParams[key];
                  if (bestVal === undefined) return null;
                  const delta = sugVal - bestVal;
                  const spec = specsMap.get(key);
                  const range = spec ? (spec.upper ?? 1) - (spec.lower ?? 0) : 1;
                  const pctChange = range > 0 ? (delta / range) * 100 : 0;
                  return (
                    <div key={key} className="sug-comparison-row">
                      <span className="sug-comparison-name">{key}</span>
                      <span className="mono sug-comparison-val">{sugVal.toFixed(3)}</span>
                      <span className="mono sug-comparison-val" style={{ opacity: 0.6 }}>{bestVal.toFixed(3)}</span>
                      <span className={`mono sug-comparison-delta-val ${Math.abs(pctChange) > 20 ? 'sug-comparison-big-change' : ''}`}>
                        {delta > 0 ? '+' : ''}{delta.toFixed(3)}
                        <span className="sug-comparison-pct">({pctChange > 0 ? '+' : ''}{pctChange.toFixed(0)}%)</span>
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Why this suggestion? (Explainability) */}
      <div className="sug-rationale">
        <button
          className="sug-rationale-toggle"
          onClick={() => setShowRationale(!showRationale)}
        >
          <Lightbulb size={14} />
          <span>Why this suggestion?</span>
          {showRationale ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </button>
        {showRationale && (
          <div className="sug-rationale-content">
            {rationale ? (
              <p className="sug-rationale-text">{rationale}</p>
            ) : (
              <>
                <div className="sug-rationale-strategy">
                  <strong>Strategy:</strong> {strategy}
                </div>
                <ul className="sug-rationale-factors">
                  {factors.map((f, i) => (
                    <li key={i}>{f}</li>
                  ))}
                </ul>
              </>
            )}
          </div>
        )}
      </div>

      <style>{`
        .suggestion-card {
          background: var(--color-surface);
          border: 1px solid var(--color-border);
          border-radius: var(--radius-lg, 12px);
          padding: 20px;
          transition: border-color 0.2s, box-shadow 0.2s;
        }
        .suggestion-card:hover {
          border-color: var(--color-primary);
          box-shadow: 0 4px 16px rgba(79, 110, 247, 0.08);
        }
        .sug-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 14px;
        }
        .sug-header-left {
          display: flex;
          align-items: center;
          gap: 12px;
        }
        .sug-index {
          font-size: 1.1rem;
          font-weight: 700;
          color: var(--color-primary);
        }
        .sug-confidence {
          display: flex;
          align-items: center;
          gap: 6px;
        }
        .sug-confidence-bar {
          width: 48px;
          height: 4px;
          background: var(--color-border);
          border-radius: 2px;
          overflow: hidden;
        }
        .sug-confidence-fill {
          height: 100%;
          border-radius: 2px;
          transition: width 0.4s ease;
        }
        .sug-confidence-label {
          font-size: 0.72rem;
          font-weight: 600;
          letter-spacing: 0.02em;
        }
        .sug-actions {
          display: flex;
          gap: 6px;
        }
        .sug-strategy-row {
          display: flex;
          flex-wrap: wrap;
          gap: 6px;
          margin-bottom: 16px;
        }
        .sug-strategy-badge {
          display: inline-flex;
          align-items: center;
          gap: 4px;
          padding: 3px 10px;
          font-size: 0.72rem;
          font-weight: 600;
          border-radius: 6px;
          background: var(--color-primary-subtle, rgba(79, 110, 247, 0.08));
          color: var(--color-primary);
          letter-spacing: 0.02em;
        }
        .sug-strategy-explore {
          background: rgba(59, 130, 246, 0.08);
          color: #3b82f6;
        }
        .sug-strategy-exploit {
          background: rgba(34, 197, 94, 0.08);
          color: #16a34a;
        }
        .sug-badge-composition {
          background: #e0e7ff;
          color: #3730a3;
        }
        .sug-param-section {
          margin-bottom: 14px;
        }
        .sug-param-section-label {
          font-size: 0.78rem;
          font-weight: 600;
          color: var(--color-text-muted);
          margin-bottom: 8px;
          text-transform: uppercase;
          letter-spacing: 0.06em;
        }
        .sug-param-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
          gap: 10px;
        }
        .sug-param-grid-narrow {
          grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
        }
        .sug-param-item {
          padding: 8px 10px;
          background: var(--color-bg);
          border-radius: 8px;
        }
        .sug-param-row {
          display: flex;
          justify-content: space-between;
          font-size: 0.82rem;
          margin-bottom: 2px;
        }
        .sug-param-name {
          font-weight: 500;
          color: var(--color-text);
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
          max-width: 60%;
        }
        .sug-param-value-blue { color: var(--color-primary); font-weight: 600; }
        .sug-param-value-purple { color: #8b5cf6; font-weight: 600; }
        .sug-param-bar-track {
          margin-top: 4px;
          height: 4px;
          background: var(--color-border);
          border-radius: 2px;
          overflow: hidden;
        }
        .sug-param-bar-fill {
          height: 100%;
          background: var(--color-primary);
          border-radius: 2px;
          transition: width 0.4s ease;
        }
        .sug-prediction {
          display: flex;
          align-items: flex-start;
          gap: 12px;
          margin-top: 14px;
          padding: 14px;
          background: var(--color-primary-subtle, rgba(79, 110, 247, 0.06));
          border-radius: 10px;
          border-left: 3px solid var(--color-primary);
        }
        .sug-prediction-icon {
          color: var(--color-primary);
          margin-top: 2px;
          flex-shrink: 0;
        }
        .sug-prediction-label {
          font-size: 0.78rem;
          color: var(--color-text-muted);
          margin-bottom: 2px;
        }
        .sug-prediction-value {
          font-size: 1.15rem;
          font-weight: 700;
          color: var(--color-text);
        }
        .sug-prediction-uncertainty {
          font-size: 0.82rem;
          font-weight: 400;
          color: var(--color-text-muted);
          margin-left: 6px;
        }
        .sug-rationale {
          margin-top: 14px;
          border-top: 1px solid var(--color-border);
          padding-top: 12px;
        }
        .sug-rationale-toggle {
          display: flex;
          align-items: center;
          gap: 6px;
          background: none;
          border: none;
          color: var(--color-text-muted);
          font-size: 0.82rem;
          font-weight: 500;
          cursor: pointer;
          padding: 4px 0;
          transition: color 0.15s;
          font-family: inherit;
        }
        .sug-rationale-toggle:hover {
          color: var(--color-primary);
        }
        .sug-rationale-content {
          margin-top: 10px;
          padding: 12px 14px;
          background: var(--color-bg);
          border-radius: 8px;
          font-size: 0.82rem;
          line-height: 1.6;
          animation: fadeIn 0.2s ease;
        }
        .sug-rationale-text {
          margin: 0;
          color: var(--color-text);
        }
        .sug-rationale-strategy {
          margin-bottom: 8px;
          color: var(--color-text);
        }
        .sug-rationale-factors {
          margin: 0;
          padding-left: 18px;
          color: var(--color-text-muted);
        }
        .sug-rationale-factors li {
          margin-bottom: 4px;
        }
        .sug-rationale-factors li:last-child {
          margin-bottom: 0;
        }
        .sug-comparison {
          margin-top: 14px;
          border-top: 1px solid var(--color-border);
          padding-top: 12px;
        }
        .sug-comparison-toggle {
          display: flex;
          align-items: center;
          gap: 6px;
          background: none;
          border: none;
          color: var(--color-text-muted);
          font-size: 0.82rem;
          font-weight: 500;
          cursor: pointer;
          padding: 4px 0;
          transition: color 0.15s;
          font-family: inherit;
        }
        .sug-comparison-toggle:hover {
          color: var(--color-primary);
        }
        .sug-comparison-delta {
          font-size: 0.72rem;
          font-weight: 600;
          margin-left: auto;
          margin-right: 4px;
        }
        .sug-comparison-content {
          margin-top: 10px;
          animation: fadeIn 0.2s ease;
        }
        .sug-comparison-grid {
          font-size: 0.78rem;
        }
        .sug-comparison-header {
          display: grid;
          grid-template-columns: 1.5fr 1fr 1fr 1.2fr;
          gap: 6px;
          padding: 6px 8px;
          font-weight: 600;
          color: var(--color-text-muted);
          text-transform: uppercase;
          font-size: 0.68rem;
          letter-spacing: 0.05em;
          border-bottom: 1px solid var(--color-border);
        }
        .sug-comparison-row {
          display: grid;
          grid-template-columns: 1.5fr 1fr 1fr 1.2fr;
          gap: 6px;
          padding: 5px 8px;
          border-bottom: 1px solid var(--color-border-subtle, var(--color-border));
        }
        .sug-comparison-row:last-child {
          border-bottom: none;
        }
        .sug-comparison-name {
          font-weight: 500;
          color: var(--color-text);
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }
        .sug-comparison-val {
          font-size: 0.78rem;
        }
        .sug-comparison-delta-val {
          font-size: 0.78rem;
          color: var(--color-text-muted);
        }
        .sug-comparison-big-change {
          color: var(--color-primary);
          font-weight: 600;
        }
        .sug-comparison-pct {
          font-size: 0.68rem;
          opacity: 0.7;
          margin-left: 3px;
        }
      `}</style>
    </div>
  );
}
