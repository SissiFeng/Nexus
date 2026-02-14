import { useState, useMemo, useEffect } from "react";
import { AlertCircle } from "lucide-react";

export interface ParamConfig {
  name: string;
  type: "continuous" | "categorical";
  lower?: number;
  upper?: number;
}

export interface ObjConfig {
  name: string;
  direction: "minimize" | "maximize";
}

export interface ColumnMapping {
  parameters: ParamConfig[];
  objectives: ObjConfig[];
  metadata: string[];
  ignored: string[];
}

interface ColumnMapperProps {
  columns: string[];
  sampleRows: Record<string, string>[];
  onMappingComplete: (mapping: ColumnMapping) => void;
}

type ColumnRole = "parameter" | "objective" | "metadata" | "ignore";

interface ColumnState {
  role: ColumnRole;
  paramType?: "continuous" | "categorical";
  objDirection?: "minimize" | "maximize";
  lower?: number;
  upper?: number;
}

export default function ColumnMapper({
  columns,
  sampleRows,
  onMappingComplete,
}: ColumnMapperProps) {
  const [columnStates, setColumnStates] = useState<
    Record<string, ColumnState>
  >({});

  // Auto-detect column types and roles
  const columnAnalysis = useMemo(() => {
    const analysis: Record<
      string,
      {
        isNumeric: boolean;
        min: number;
        max: number;
        distinctValues: string[];
        suggestedRole: ColumnRole;
        suggestedParamType?: "continuous" | "categorical";
        suggestedObjDirection?: "minimize" | "maximize";
      }
    > = {};

    columns.forEach((col) => {
      const values = sampleRows
        .map((row) => row[col])
        .filter((v) => v !== undefined && v !== null && v !== "");

      // Check if numeric
      const numericValues = values
        .map((v) => parseFloat(v))
        .filter((v) => !isNaN(v));
      const isNumeric =
        numericValues.length > 0 && numericValues.length === values.length;

      // Get distinct values
      const distinctValues = Array.from(new Set(values)).slice(0, 3);

      // Calculate min/max for numeric columns
      const min = isNumeric ? Math.min(...numericValues) : 0;
      const max = isNumeric ? Math.max(...numericValues) : 0;

      // Auto-detect role based on column name
      const lowerName = col.toLowerCase();
      let suggestedRole: ColumnRole = "metadata";
      let suggestedParamType: "continuous" | "categorical" | undefined;
      let suggestedObjDirection: "minimize" | "maximize" | undefined;

      // Objective indicators
      if (
        lowerName.includes("yield") ||
        lowerName.includes("objective") ||
        lowerName.includes("target") ||
        lowerName.includes("score") ||
        lowerName.includes("loss") ||
        lowerName.includes("error") ||
        lowerName.includes("lsv")
      ) {
        suggestedRole = "objective";
        suggestedObjDirection =
          lowerName.includes("loss") ||
          lowerName.includes("error") ||
          lowerName.includes("lsv")
            ? "minimize"
            : "maximize";
      }
      // Metadata indicators
      else if (
        lowerName.includes("iteration") ||
        lowerName.includes("well") ||
        lowerName.includes("reactor") ||
        lowerName.includes("id") ||
        lowerName.includes("substrate") ||
        lowerName.includes("type") ||
        lowerName.includes("date") ||
        lowerName.includes("time") ||
        lowerName.includes("name")
      ) {
        suggestedRole = "metadata";
      }
      // Parameter indicators
      else if (isNumeric) {
        suggestedRole = "parameter";
        suggestedParamType = "continuous";
      } else {
        // Text columns could be categorical parameters or metadata
        suggestedRole = "parameter";
        suggestedParamType = "categorical";
      }

      analysis[col] = {
        isNumeric,
        min,
        max,
        distinctValues,
        suggestedRole,
        suggestedParamType,
        suggestedObjDirection,
      };
    });

    return analysis;
  }, [columns, sampleRows]);

  // Initialize column states with auto-detected values
  useEffect(() => {
    const initialStates: Record<string, ColumnState> = {};
    columns.forEach((col) => {
      const analysis = columnAnalysis[col];
      initialStates[col] = {
        role: analysis.suggestedRole,
        paramType: analysis.suggestedParamType,
        objDirection: analysis.suggestedObjDirection,
        lower: analysis.isNumeric ? analysis.min : undefined,
        upper: analysis.isNumeric ? analysis.max : undefined,
      };
    });
    setColumnStates(initialStates);
  }, [columns, columnAnalysis]);

  const updateColumnState = (
    col: string,
    updates: Partial<ColumnState>
  ) => {
    setColumnStates((prev) => ({
      ...prev,
      [col]: { ...prev[col], ...updates },
    }));
  };

  const handleRoleChange = (col: string, role: ColumnRole) => {
    const analysis = columnAnalysis[col];
    const updates: Partial<ColumnState> = { role };

    if (role === "parameter") {
      updates.paramType = analysis.isNumeric ? "continuous" : "categorical";
      if (analysis.isNumeric) {
        updates.lower = analysis.min;
        updates.upper = analysis.max;
      }
    } else if (role === "objective") {
      updates.objDirection = "maximize";
    }

    updateColumnState(col, updates);
  };

  const handleConfirm = () => {
    const mapping: ColumnMapping = {
      parameters: [],
      objectives: [],
      metadata: [],
      ignored: [],
    };

    Object.entries(columnStates).forEach(([col, state]) => {
      switch (state.role) {
        case "parameter":
          mapping.parameters.push({
            name: col,
            type: state.paramType || "continuous",
            lower: state.lower,
            upper: state.upper,
          });
          break;
        case "objective":
          mapping.objectives.push({
            name: col,
            direction: state.objDirection || "maximize",
          });
          break;
        case "metadata":
          mapping.metadata.push(col);
          break;
        case "ignore":
          mapping.ignored.push(col);
          break;
      }
    });

    onMappingComplete(mapping);
  };

  const hasParameters = Object.values(columnStates).some(
    (s) => s.role === "parameter"
  );
  const hasObjectives = Object.values(columnStates).some(
    (s) => s.role === "objective"
  );

  return (
    <div className="column-mapper-container">
      <div className="mapper-header">
        <h2>Map Column Roles</h2>
        <p className="mapper-description">
          Configure how each column should be used in optimization. Auto-detected
          roles can be adjusted.
        </p>
      </div>

      {(!hasParameters || !hasObjectives) && (
        <div className="mapper-warning">
          <AlertCircle size={20} />
          <span>
            {!hasParameters && !hasObjectives
              ? "You need at least one parameter and one objective to proceed."
              : !hasParameters
              ? "You need at least one parameter to proceed."
              : "You need at least one objective to proceed."}
          </span>
        </div>
      )}

      <div className="table-wrapper">
        <table className="data-table column-mapper-table">
          <thead>
            <tr>
              <th>Column Name</th>
              <th>Sample Values</th>
              <th>Type</th>
              <th>Role</th>
              <th>Configuration</th>
            </tr>
          </thead>
          <tbody>
            {columns.map((col) => {
              const analysis = columnAnalysis[col];
              const state = columnStates[col] || {};

              return (
                <tr key={col}>
                  <td className="column-name">{col}</td>
                  <td className="sample-values">
                    {analysis.distinctValues.join(", ")}
                  </td>
                  <td>
                    <span
                      className={`type-badge ${
                        analysis.isNumeric ? "numeric" : "text"
                      }`}
                    >
                      {analysis.isNumeric ? "Numeric" : "Text"}
                    </span>
                  </td>
                  <td>
                    <select
                      className="role-select"
                      value={state.role}
                      onChange={(e) =>
                        handleRoleChange(col, e.target.value as ColumnRole)
                      }
                    >
                      <option value="parameter">Parameter</option>
                      <option value="objective">Objective</option>
                      <option value="metadata">Metadata</option>
                      <option value="ignore">Ignore</option>
                    </select>
                  </td>
                  <td className="config-cell">
                    {state.role === "parameter" && (
                      <div className="param-config">
                        <select
                          className="param-type-select"
                          value={state.paramType}
                          onChange={(e) =>
                            updateColumnState(col, {
                              paramType: e.target.value as
                                | "continuous"
                                | "categorical",
                            })
                          }
                        >
                          <option value="continuous">Continuous</option>
                          <option value="categorical">Categorical</option>
                        </select>
                        {state.paramType === "continuous" && (
                          <div className="bounds-inputs">
                            <input
                              type="number"
                              className="bound-input"
                              placeholder="Min"
                              value={state.lower ?? ""}
                              onChange={(e) =>
                                updateColumnState(col, {
                                  lower: parseFloat(e.target.value),
                                })
                              }
                            />
                            <span className="bounds-separator">to</span>
                            <input
                              type="number"
                              className="bound-input"
                              placeholder="Max"
                              value={state.upper ?? ""}
                              onChange={(e) =>
                                updateColumnState(col, {
                                  upper: parseFloat(e.target.value),
                                })
                              }
                            />
                          </div>
                        )}
                      </div>
                    )}
                    {state.role === "objective" && (
                      <div className="obj-config">
                        <select
                          className="obj-direction-select"
                          value={state.objDirection}
                          onChange={(e) =>
                            updateColumnState(col, {
                              objDirection: e.target.value as
                                | "minimize"
                                | "maximize",
                            })
                          }
                        >
                          <option value="maximize">Maximize</option>
                          <option value="minimize">Minimize</option>
                        </select>
                      </div>
                    )}
                    {(state.role === "metadata" || state.role === "ignore") && (
                      <span className="config-placeholder">-</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="mapper-summary">
        <div className="summary-stats">
          <div className="summary-stat">
            <span className="summary-badge badge-parameter">
              {Object.values(columnStates).filter((s) => s.role === "parameter")
                .length}
            </span>
            <span className="summary-label">Parameters</span>
          </div>
          <div className="summary-stat">
            <span className="summary-badge badge-objective">
              {Object.values(columnStates).filter((s) => s.role === "objective")
                .length}
            </span>
            <span className="summary-label">Objectives</span>
          </div>
          <div className="summary-stat">
            <span className="summary-badge badge-metadata">
              {Object.values(columnStates).filter((s) => s.role === "metadata")
                .length}
            </span>
            <span className="summary-label">Metadata</span>
          </div>
          <div className="summary-stat">
            <span className="summary-badge badge-ignored">
              {Object.values(columnStates).filter((s) => s.role === "ignore")
                .length}
            </span>
            <span className="summary-label">Ignored</span>
          </div>
        </div>
      </div>

      <div className="mapper-actions">
        <button
          className="btn btn-primary"
          onClick={handleConfirm}
          disabled={!hasParameters || !hasObjectives}
        >
          Confirm Mapping
        </button>
      </div>

      <style>{`
        .column-mapper-container {
          width: 100%;
        }

        .mapper-header {
          margin-bottom: 24px;
        }

        .mapper-header h2 {
          font-size: 1.3rem;
          font-weight: 700;
          margin-bottom: 8px;
        }

        .mapper-description {
          font-size: 0.9rem;
          color: var(--color-text-muted);
        }

        .mapper-warning {
          display: flex;
          align-items: center;
          gap: 12px;
          background: #fef9c3;
          color: #854d0e;
          padding: 12px 16px;
          border-radius: var(--radius);
          margin-bottom: 20px;
          font-size: 0.9rem;
        }

        .column-mapper-table {
          font-size: 0.88rem;
        }

        .column-name {
          font-weight: 600;
          color: var(--color-text);
        }

        .sample-values {
          color: var(--color-text-muted);
          font-size: 0.82rem;
          max-width: 200px;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }

        .type-badge {
          display: inline-block;
          padding: 2px 8px;
          border-radius: 10px;
          font-size: 0.75rem;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.03em;
        }

        .type-badge.numeric {
          background: #dbeafe;
          color: #1e40af;
        }

        .type-badge.text {
          background: #fef3c7;
          color: #92400e;
        }

        .role-select,
        .param-type-select,
        .obj-direction-select {
          padding: 6px 10px;
          border: 1px solid var(--color-border);
          border-radius: var(--radius);
          font-size: 0.85rem;
          font-family: inherit;
          background: var(--color-surface);
          color: var(--color-text);
          cursor: pointer;
          outline: none;
          transition: border-color 0.15s;
        }

        .role-select:focus,
        .param-type-select:focus,
        .obj-direction-select:focus {
          border-color: var(--color-primary);
        }

        .role-select {
          min-width: 120px;
        }

        .config-cell {
          min-width: 250px;
        }

        .param-config,
        .obj-config {
          display: flex;
          gap: 8px;
          align-items: center;
        }

        .bounds-inputs {
          display: flex;
          gap: 6px;
          align-items: center;
        }

        .bound-input {
          width: 80px;
          padding: 6px 8px;
          border: 1px solid var(--color-border);
          border-radius: var(--radius);
          font-size: 0.82rem;
          font-family: var(--font-mono);
          outline: none;
          transition: border-color 0.15s;
        }

        .bound-input:focus {
          border-color: var(--color-primary);
        }

        .bounds-separator {
          color: var(--color-text-muted);
          font-size: 0.82rem;
        }

        .config-placeholder {
          color: var(--color-text-muted);
          font-size: 0.9rem;
        }

        .mapper-summary {
          background: var(--color-bg);
          border: 1px solid var(--color-border);
          border-radius: var(--radius);
          padding: 20px;
          margin: 24px 0;
        }

        .summary-stats {
          display: flex;
          gap: 32px;
          justify-content: center;
          flex-wrap: wrap;
        }

        .summary-stat {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 8px;
        }

        .summary-badge {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          width: 48px;
          height: 48px;
          border-radius: 50%;
          font-size: 1.2rem;
          font-weight: 700;
        }

        .summary-badge.badge-parameter {
          background: #dcfce7;
          color: #166534;
        }

        .summary-badge.badge-objective {
          background: #dbeafe;
          color: #1e40af;
        }

        .summary-badge.badge-metadata {
          background: #fef3c7;
          color: #92400e;
        }

        .summary-badge.badge-ignored {
          background: #f1f5f9;
          color: #475569;
        }

        .summary-label {
          font-size: 0.85rem;
          color: var(--color-text-muted);
          font-weight: 500;
        }

        .mapper-actions {
          display: flex;
          justify-content: flex-end;
        }

        @media (max-width: 768px) {
          .column-mapper-table {
            font-size: 0.8rem;
          }

          .config-cell {
            min-width: 200px;
          }

          .param-config {
            flex-direction: column;
            align-items: flex-start;
          }

          .summary-stats {
            gap: 20px;
          }
        }
      `}</style>
    </div>
  );
}
