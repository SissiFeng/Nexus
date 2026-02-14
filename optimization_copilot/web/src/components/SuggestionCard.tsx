import { Check, Edit, X } from 'lucide-react';

interface SuggestionCardProps {
  index: number;
  suggestion: Record<string, number>;
  parameterSpecs: Array<{ name: string; type: string; lower?: number; upper?: number }>;
  objectiveName?: string;
  predictedValue?: number;
  predictedUncertainty?: number;
  onAccept?: () => void;
  onModify?: () => void;
  onReject?: () => void;
}

export default function SuggestionCard({
  index,
  suggestion,
  parameterSpecs,
  objectiveName,
  predictedValue,
  predictedUncertainty,
  onAccept,
  onModify,
  onReject,
}: SuggestionCardProps) {
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

  const renderParameterBar = (key: string, value: number) => {
    const spec = specsMap.get(key);
    if (!spec || spec.type === 'categorical') return null;

    const lower = spec.lower ?? 0;
    const upper = spec.upper ?? 1;
    const range = upper - lower;
    const normalized = range > 0 ? ((value - lower) / range) * 100 : 0;
    const percentage = Math.max(0, Math.min(100, normalized));

    return (
      <div
        style={{
          marginTop: '4px',
          height: '6px',
          background: '#f1f5f9',
          borderRadius: '3px',
          overflow: 'hidden',
        }}
      >
        <div
          style={{
            width: `${percentage}%`,
            height: '100%',
            background: '#3b82f6',
            transition: 'width 0.3s ease',
          }}
        />
      </div>
    );
  };

  return (
    <div className="card" style={{ marginBottom: '16px' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
        <h3 style={{ fontSize: '1rem', fontWeight: 600, margin: 0 }}>
          Suggestion #{index}
        </h3>
        <div style={{ display: 'flex', gap: '8px' }}>
          {onAccept && (
            <button
              className="btn btn-sm btn-primary"
              onClick={onAccept}
              title="Accept suggestion"
            >
              <Check size={16} />
            </button>
          )}
          {onModify && (
            <button
              className="btn btn-sm btn-secondary"
              onClick={onModify}
              title="Modify suggestion"
            >
              <Edit size={16} />
            </button>
          )}
          {onReject && (
            <button
              className="btn btn-sm btn-danger"
              onClick={onReject}
              title="Reject suggestion"
            >
              <X size={16} />
            </button>
          )}
        </div>
      </div>

      {/* Composition badge */}
      {compositionCount > 0 && (
        <div style={{ marginBottom: '12px' }}>
          <span className="badge" style={{ background: '#e0e7ff', color: '#3730a3' }}>
            {compositionCount}-Component System
          </span>
        </div>
      )}

      {/* Composition parameters */}
      {compositionEntries.length > 0 && (
        <div style={{ marginBottom: '16px' }}>
          <div style={{ fontSize: '0.85rem', fontWeight: 600, color: '#718096', marginBottom: '8px' }}>
            Composition
          </div>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
              gap: '12px',
            }}
          >
            {compositionEntries.map(([key, value]) => (
              <div key={key}>
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    fontSize: '0.85rem',
                    marginBottom: '2px',
                  }}
                >
                  <span style={{ fontWeight: 500 }}>{key}</span>
                  <span className="mono" style={{ color: '#3b82f6' }}>
                    {formatValue(key, value)}
                  </span>
                </div>
                {renderParameterBar(key, value)}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Process parameters */}
      {processEntries.length > 0 && (
        <div style={{ marginBottom: '16px' }}>
          <div style={{ fontSize: '0.85rem', fontWeight: 600, color: '#718096', marginBottom: '8px' }}>
            Process Conditions
          </div>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))',
              gap: '12px',
            }}
          >
            {processEntries.map(([key, value]) => (
              <div key={key}>
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    fontSize: '0.85rem',
                    marginBottom: '2px',
                  }}
                >
                  <span style={{ fontWeight: 500 }}>{key}</span>
                  <span className="mono" style={{ color: '#8b5cf6' }}>
                    {formatValue(key, value)}
                  </span>
                </div>
                {renderParameterBar(key, value)}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Predicted objective */}
      {predictedValue !== undefined && (
        <div
          style={{
            marginTop: '12px',
            padding: '12px',
            background: '#f8f9fa',
            borderRadius: '6px',
            borderLeft: '3px solid #3b82f6',
          }}
        >
          <div style={{ fontSize: '0.85rem', color: '#718096', marginBottom: '4px' }}>
            Predicted {objectiveName || 'Objective'}
          </div>
          <div style={{ fontSize: '1.1rem', fontWeight: 600 }}>
            {predictedValue.toFixed(4)}
            {predictedUncertainty !== undefined && (
              <span style={{ fontSize: '0.85rem', fontWeight: 400, color: '#718096', marginLeft: '8px' }}>
                ± {predictedUncertainty.toFixed(4)}
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
