import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { CheckCircle2, Circle } from "lucide-react";
import FileUpload from "../components/FileUpload";
import ColumnMapper, { type ColumnMapping } from "../components/ColumnMapper";
import DataQualityReport from "../components/DataQualityReport";

// Main NewCampaign Component
export default function NewCampaign() {
  const navigate = useNavigate();
  const [currentStep, setCurrentStep] = useState(1);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // Step 1 state
  const [columns, setColumns] = useState<string[]>([]);
  const [rows, setRows] = useState<Record<string, string>[]>([]);
  const [fileName, setFileName] = useState("");

  // Step 2 state
  const [campaignName, setCampaignName] = useState("");
  const [description, setDescription] = useState("");
  const [batchSize, setBatchSize] = useState(5);
  const [explorationWeight, setExplorationWeight] = useState(0.5);
  const [mapping, setMapping] = useState<ColumnMapping | null>(null);

  const handleDataParsed = (
    parsedColumns: string[],
    parsedRows: Record<string, string>[],
    parsedFileName: string
  ) => {
    setColumns(parsedColumns);
    setRows(parsedRows);
    setFileName(parsedFileName);
    const defaultName = parsedFileName.replace(/\.(csv|tsv|json|jsonl|xlsx|xls)$/i, "");
    setCampaignName(defaultName);
  };

  const handleMappingComplete = (completedMapping: ColumnMapping) => {
    setMapping(completedMapping);
  };

  const handleCreateCampaign = async () => {
    if (!mapping) return;

    setLoading(true);
    setError(null);

    try {
      const response = await fetch("/api/campaigns/from-upload", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: campaignName,
          description,
          data: rows,
          mapping,
          batch_size: batchSize,
          exploration_weight: explorationWeight,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || "Failed to create campaign");
      }

      const result = await response.json();
      const campaignId = result.campaign_id || result.id;

      navigate(`/workspace/${campaignId}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create campaign");
    } finally {
      setLoading(false);
    }
  };

  const canGoToStep2 = columns.length > 0 && rows.length > 0;
  const canGoToStep3 = mapping !== null;
  const sampleRows = rows.slice(0, 5);

  return (
    <div className="page">
      <div className="page-header">
        <h1>Create New Campaign</h1>
      </div>

      {error && <div className="error-banner">{error}</div>}

      {/* Step Indicator */}
      <div className="step-indicator">
        <div className={`step-item ${currentStep >= 1 ? "active" : ""}`}>
          {currentStep > 1 ? (
            <CheckCircle2 className="step-icon" size={24} />
          ) : (
            <Circle className="step-icon" size={24} />
          )}
          <span className="step-label">Upload Data</span>
          <span className="step-description">Upload your CSV file with experimental results</span>
        </div>
        <div className="step-line" />
        <div className={`step-item ${currentStep >= 2 ? "active" : ""}`}>
          {currentStep > 2 ? (
            <CheckCircle2 className="step-icon" size={24} />
          ) : (
            <Circle className="step-icon" size={24} />
          )}
          <span className="step-label">Map Columns</span>
          <span className="step-description">Tell us which columns are parameters and which are objectives</span>
        </div>
        <div className="step-line" />
        <div className={`step-item ${currentStep >= 3 ? "active" : ""}`}>
          <Circle className="step-icon" size={24} />
          <span className="step-label">Review & Create</span>
          <span className="step-description">Review data quality and create your campaign</span>
        </div>
      </div>

      {/* Step 1: Upload */}
      {currentStep === 1 && (
        <div className="card">
          <h2>Upload Data</h2>
          <FileUpload onDataParsed={handleDataParsed} />
          {fileName && (
            <div className="step-actions">
              <button
                className="btn btn-primary"
                onClick={() => setCurrentStep(2)}
                disabled={!canGoToStep2}
              >
                Next
              </button>
            </div>
          )}
        </div>
      )}

      {/* Step 2: Configure */}
      {currentStep === 2 && (
        <div className="card">
          <h2>Configure Campaign</h2>
          <div className="config-section">
            <ColumnMapper
              columns={columns}
              sampleRows={sampleRows}
              onMappingComplete={handleMappingComplete}
            />
          </div>

          <div className="config-section">
            <h3>Campaign Settings</h3>
            <div className="form-grid">
              <label>
                Campaign Name
                <input
                  type="text"
                  className="input"
                  value={campaignName}
                  onChange={(e) => setCampaignName(e.target.value)}
                />
              </label>
              <label>
                Description
                <textarea
                  className="input"
                  rows={3}
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                />
              </label>
              <label>
                Batch Size
                <input
                  type="number"
                  className="input"
                  min="1"
                  max="20"
                  value={batchSize}
                  onChange={(e) => setBatchSize(parseInt(e.target.value))}
                />
              </label>
              <label>
                Exploration Weight ({explorationWeight.toFixed(2)})
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={explorationWeight}
                  onChange={(e) => setExplorationWeight(parseFloat(e.target.value))}
                />
                <div className="slider-labels">
                  <span>Exploit</span>
                  <span>Explore</span>
                </div>
              </label>
            </div>
          </div>

          <div className="step-actions">
            <button className="btn btn-secondary" onClick={() => setCurrentStep(1)}>
              Back
            </button>
            <button
              className="btn btn-primary"
              onClick={() => setCurrentStep(3)}
              disabled={!canGoToStep3}
            >
              Next
            </button>
          </div>
        </div>
      )}

      {/* Step 3: Review */}
      {currentStep === 3 && mapping && (
        <div className="card">
          <h2>Review & Launch</h2>

          <div className="review-section">
            <h3>Campaign Summary</h3>
            <div className="stats-row">
              <div className="stat-card">
                <div className="stat-label">Parameters</div>
                <div className="stat-value">{mapping.parameters.length}</div>
              </div>
              <div className="stat-card">
                <div className="stat-label">Objectives</div>
                <div className="stat-value">{mapping.objectives.length}</div>
              </div>
              <div className="stat-card">
                <div className="stat-label">Observations</div>
                <div className="stat-value">{rows.length}</div>
              </div>
              <div className="stat-card">
                <div className="stat-label">Batch Size</div>
                <div className="stat-value">{batchSize}</div>
              </div>
            </div>
          </div>

          <div className="review-section">
            <h3>Parameters</h3>
            <div className="review-list">
              {mapping.parameters.map((param) => (
                <div key={param.name} className="review-item">
                  <strong>{param.name}</strong>
                  <span className="badge">{param.type}</span>
                  {param.type === "continuous" && (
                    <span className="mono">
                      [{param.lower} – {param.upper}]
                    </span>
                  )}
                </div>
              ))}
            </div>
          </div>

          <div className="review-section">
            <h3>Objectives</h3>
            <div className="review-list">
              {mapping.objectives.map((obj) => (
                <div key={obj.name} className="review-item">
                  <strong>{obj.name}</strong>
                  <span className={`badge badge-${obj.direction}`}>
                    {obj.direction}
                  </span>
                </div>
              ))}
            </div>
          </div>

          <DataQualityReport
            data={rows}
            parameters={mapping.parameters}
            objectives={mapping.objectives}
          />

          <div className="step-actions">
            <button className="btn btn-secondary" onClick={() => setCurrentStep(2)}>
              Back
            </button>
            <button
              className="btn btn-primary"
              onClick={handleCreateCampaign}
              disabled={loading}
            >
              {loading ? "Creating..." : "Create Campaign"}
            </button>
          </div>
        </div>
      )}

      <style>{`
        .step-indicator {
          display: flex;
          align-items: center;
          justify-content: center;
          margin-bottom: 32px;
          padding: 24px;
          background: var(--color-surface);
          border: 1px solid var(--color-border);
          border-radius: var(--radius);
        }

        .step-item {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 8px;
          color: var(--color-text-muted);
          transition: color 0.2s;
        }

        .step-item.active {
          color: var(--color-primary);
        }

        .step-icon {
          stroke-width: 2;
        }

        .step-label {
          font-size: 0.85rem;
          font-weight: 500;
        }

        .step-description {
          font-size: 0.72rem;
          color: var(--color-text-muted);
          text-align: center;
          max-width: 160px;
          line-height: 1.3;
        }

        .step-item.active .step-description {
          color: var(--color-primary);
          opacity: 0.8;
        }

        .step-line {
          width: 80px;
          height: 2px;
          background: var(--color-border);
          margin: 0 16px;
        }

        .step-actions {
          display: flex;
          gap: 12px;
          justify-content: flex-end;
          margin-top: 24px;
          padding-top: 24px;
          border-top: 1px solid var(--color-border);
        }

        .config-section {
          margin-bottom: 32px;
          padding-bottom: 24px;
          border-bottom: 1px solid var(--color-border);
        }

        .config-section:last-of-type {
          border-bottom: none;
        }

        .config-section h3 {
          font-size: 1rem;
          font-weight: 600;
          margin-bottom: 16px;
        }

        .form-grid {
          display: grid;
          gap: 16px;
          grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        }

        .form-grid label {
          display: flex;
          flex-direction: column;
          gap: 6px;
          font-size: 0.9rem;
          font-weight: 500;
        }

        .slider-labels {
          display: flex;
          justify-content: space-between;
          font-size: 0.75rem;
          color: var(--color-text-muted);
          margin-top: 4px;
        }

        .review-section {
          margin-bottom: 24px;
        }

        .review-section h3 {
          font-size: 1rem;
          font-weight: 600;
          margin-bottom: 12px;
        }

        .review-list {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }

        .review-item {
          display: flex;
          align-items: center;
          gap: 12px;
          padding: 8px 12px;
          background: var(--color-bg);
          border-radius: var(--radius);
          font-size: 0.9rem;
        }

        .badge-maximize {
          background: #dcfce7;
          color: #166534;
        }

        .badge-minimize {
          background: #fef9c3;
          color: #854d0e;
        }

        @media (max-width: 768px) {
          .step-indicator {
            padding: 16px;
          }

          .step-line {
            width: 40px;
            margin: 0 8px;
          }

          .step-label {
            font-size: 0.75rem;
          }

          .step-description {
            display: none;
          }

          .form-grid {
            grid-template-columns: 1fr;
          }
        }
      `}</style>
    </div>
  );
}
