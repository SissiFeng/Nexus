import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, FileText, X } from "lucide-react";
import Papa from "papaparse";

interface FileUploadProps {
  onDataParsed: (
    columns: string[],
    rows: Record<string, string>[],
    fileName: string
  ) => void;
}

interface ParsedFile {
  fileName: string;
  fileSize: string;
  rowCount: number;
  columns: string[];
  previewRows: Record<string, string>[];
  fullData: Record<string, string>[];
}

export default function FileUpload({ onDataParsed }: FileUploadProps) {
  const [parsedFile, setParsedFile] = useState<ParsedFile | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [warning, setWarning] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const parseCSV = (file: File): Promise<ParsedFile> => {
    return new Promise((resolve, reject) => {
      Papa.parse(file, {
        header: true,
        skipEmptyLines: true,
        complete: (results) => {
          const data = results.data as Record<string, string>[];
          if (!data || data.length === 0) {
            reject(new Error("No data found in file"));
            return;
          }

          const columns = Object.keys(data[0]);
          if (columns.length === 0) {
            reject(new Error("No columns found in file"));
            return;
          }

          resolve({
            fileName: file.name,
            fileSize: formatFileSize(file.size),
            rowCount: data.length,
            columns,
            previewRows: data.slice(0, 10),
            fullData: data,
          });
        },
        error: (error) => {
          reject(error);
        },
      });
    });
  };

  const handleFileDrop = useCallback(
    async (acceptedFiles: File[]) => {
      if (acceptedFiles.length === 0) return;

      const file = acceptedFiles[0];
      setError(null);
      setWarning(null);
      setIsLoading(true);

      // File size checks
      const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB
      const LARGE_FILE_SIZE = 10 * 1024 * 1024; // 10MB
      if (file.size > MAX_FILE_SIZE) {
        setError("File too large. Maximum size is 50MB.");
        setIsLoading(false);
        return;
      }
      if (file.size > LARGE_FILE_SIZE) {
        setWarning("Large file detected. Parsing may take a moment...");
      }

      try {
        const ext = file.name.split(".").pop()?.toLowerCase();

        if (ext === "xlsx" || ext === "xls") {
          // For Excel files, we'll just show a message for now
          setError(
            "Excel files are not yet fully supported. Please convert to CSV or TSV format."
          );
          setIsLoading(false);
          return;
        }

        if (ext === "json" || ext === "jsonl") {
          // Handle JSON/JSONL files
          const text = await file.text();
          const lines = ext === "jsonl" ? text.trim().split("\n") : [text];
          const data: Record<string, string>[] = [];

          for (const line of lines) {
            try {
              const obj = JSON.parse(line);
              data.push(obj);
            } catch {
              setError("Invalid JSON format");
              setIsLoading(false);
              return;
            }
          }

          if (data.length === 0) {
            setError("No data found in JSON file");
            setIsLoading(false);
            return;
          }

          const columns = Object.keys(data[0]);
          const parsed: ParsedFile = {
            fileName: file.name,
            fileSize: formatFileSize(file.size),
            rowCount: data.length,
            columns,
            previewRows: data.slice(0, 10),
            fullData: data,
          };

          setParsedFile(parsed);
          if (parsed.rowCount > 100000) {
            setWarning(`Very large dataset (${parsed.rowCount.toLocaleString()} rows). Consider sampling to improve performance.`);
          }
        } else {
          // Handle CSV/TSV files
          const parsed = await parseCSV(file);
          setParsedFile(parsed);
          if (parsed.rowCount > 100000) {
            setWarning(`Very large dataset (${parsed.rowCount.toLocaleString()} rows). Consider sampling to improve performance.`);
          }
        }
      } catch (err) {
        setError(
          err instanceof Error ? err.message : "Failed to parse file"
        );
      } finally {
        setIsLoading(false);
      }
    },
    []
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop: handleFileDrop,
    accept: {
      "text/csv": [".csv"],
      "text/tab-separated-values": [".tsv"],
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [
        ".xlsx",
      ],
      "application/vnd.ms-excel": [".xls"],
      "application/json": [".json", ".jsonl"],
    },
    multiple: false,
  });

  const handleConfirm = () => {
    if (!parsedFile) return;
    onDataParsed(parsedFile.columns, parsedFile.fullData, parsedFile.fileName);
  };

  const handleClear = () => {
    setParsedFile(null);
    setError(null);
    setWarning(null);
  };

  return (
    <div className="file-upload-container">
      {error && (
        <div className="error-banner">
          <strong>Error:</strong> {error}
        </div>
      )}

      {warning && !error && (
        <div className="warning-banner">
          <strong>Warning:</strong> {warning}
        </div>
      )}

      {!parsedFile ? (
        <div
          {...getRootProps()}
          className={`dropzone ${isDragActive ? "dropzone-active" : ""}`}
        >
          <input {...getInputProps()} />
          <Upload className="dropzone-icon" size={48} />
          <p className="dropzone-text">
            {isDragActive
              ? "Drop your file here..."
              : "Drag files here or click to browse"}
          </p>
          <p className="dropzone-subtext">
            Supported formats: CSV, TSV, JSON, JSONL, Excel (XLSX/XLS)
          </p>
          {isLoading && <p className="dropzone-loading">Loading...</p>}
        </div>
      ) : (
        <div className="file-preview-card">
          <div className="file-preview-header">
            <div className="file-preview-info">
              <FileText size={24} className="file-icon" />
              <div>
                <div className="file-name">{parsedFile.fileName}</div>
                <div className="file-meta">
                  {parsedFile.fileSize} • {parsedFile.rowCount} rows •{" "}
                  {parsedFile.columns.length} columns
                </div>
              </div>
            </div>
            <button
              className="btn-icon"
              onClick={handleClear}
              title="Remove file"
            >
              <X size={20} />
            </button>
          </div>

          <div className="file-preview-section">
            <h3 className="preview-section-title">Data Preview</h3>
            <div className="table-wrapper">
              <table className="data-table">
                <thead>
                  <tr>
                    {parsedFile.columns.map((col) => (
                      <th key={col}>{col}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {parsedFile.previewRows.map((row, idx) => (
                    <tr key={idx}>
                      {parsedFile.columns.map((col) => (
                        <td key={col} className="mono">
                          {row[col] ?? "-"}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {parsedFile.rowCount > 10 && (
              <p className="preview-note">
                Showing first 10 of {parsedFile.rowCount} rows
              </p>
            )}
          </div>

          <div className="file-preview-actions">
            <button className="btn btn-primary" onClick={handleConfirm}>
              Continue to Column Mapping
            </button>
          </div>
        </div>
      )}

      <style>{`
        .file-upload-container {
          width: 100%;
        }

        .dropzone {
          border: 2px dashed var(--color-border);
          border-radius: var(--radius);
          padding: 48px 24px;
          text-align: center;
          cursor: pointer;
          transition: all 0.2s;
          background: var(--color-surface);
        }

        .dropzone:hover {
          border-color: var(--color-primary);
          background: var(--color-bg);
        }

        .dropzone-active {
          border-color: var(--color-primary);
          background: var(--color-upload-active-bg);
        }

        .dropzone-icon {
          color: var(--color-text-muted);
          margin-bottom: 16px;
        }

        .dropzone-text {
          font-size: 1rem;
          font-weight: 500;
          color: var(--color-text);
          margin-bottom: 8px;
        }

        .dropzone-subtext {
          font-size: 0.85rem;
          color: var(--color-text-muted);
        }

        .dropzone-loading {
          margin-top: 16px;
          color: var(--color-primary);
          font-weight: 500;
        }

        .file-preview-card {
          background: var(--color-surface);
          border: 1px solid var(--color-border);
          border-radius: var(--radius);
          padding: 24px;
          box-shadow: var(--shadow-sm);
        }

        .file-preview-header {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          margin-bottom: 24px;
          padding-bottom: 16px;
          border-bottom: 1px solid var(--color-border);
        }

        .file-preview-info {
          display: flex;
          gap: 12px;
          align-items: flex-start;
        }

        .file-icon {
          color: var(--color-primary);
          flex-shrink: 0;
        }

        .file-name {
          font-size: 1.1rem;
          font-weight: 600;
          color: var(--color-text);
          margin-bottom: 4px;
        }

        .file-meta {
          font-size: 0.85rem;
          color: var(--color-text-muted);
        }

        .btn-icon {
          background: none;
          border: none;
          padding: 4px;
          cursor: pointer;
          color: var(--color-text-muted);
          border-radius: var(--radius);
          transition: all 0.15s;
        }

        .btn-icon:hover {
          background: var(--color-bg);
          color: var(--color-text);
        }

        .file-preview-section {
          margin-bottom: 24px;
        }

        .preview-section-title {
          font-size: 0.9rem;
          font-weight: 600;
          color: var(--color-text-muted);
          text-transform: uppercase;
          letter-spacing: 0.04em;
          margin-bottom: 12px;
        }

        .preview-note {
          margin-top: 12px;
          font-size: 0.85rem;
          color: var(--color-text-muted);
          text-align: center;
        }

        .warning-banner {
          background: var(--color-warning-bg);
          color: var(--color-warning-text);
          padding: 12px 16px;
          border-radius: var(--radius);
          margin-bottom: 16px;
          font-size: 0.9rem;
        }

        .file-preview-actions {
          display: flex;
          gap: 12px;
          justify-content: flex-end;
        }
      `}</style>
    </div>
  );
}
