import { useState } from "react";

interface AuditEntry {
  hash: string;
  prev_hash: string;
  event: string;
  timestamp: number;
  details: string;
}

interface AuditTrailProps {
  entries: AuditEntry[];
}

function AuditEntryRow({ entry }: { entry: AuditEntry }) {
  const [expanded, setExpanded] = useState(false);
  const date = new Date(entry.timestamp * 1000);

  return (
    <div className="audit-entry">
      <div
        className="audit-entry-header"
        onClick={() => setExpanded(!expanded)}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") setExpanded(!expanded);
        }}
      >
        <span className="audit-expand">{expanded ? "\u25BC" : "\u25B6"}</span>
        <span className="audit-hash mono" title={entry.hash}>
          {entry.hash.slice(0, 8)}
        </span>
        <span className="audit-event">{entry.event}</span>
        <span className="audit-time">{date.toLocaleString()}</span>
        <span className="audit-chain mono" title={`prev: ${entry.prev_hash}`}>
          &larr; {entry.prev_hash.slice(0, 8)}
        </span>
      </div>
      {expanded && (
        <div className="audit-entry-details">
          <div className="audit-detail-row">
            <span className="detail-label">Hash:</span>
            <span className="mono">{entry.hash}</span>
          </div>
          <div className="audit-detail-row">
            <span className="detail-label">Previous:</span>
            <span className="mono">{entry.prev_hash}</span>
          </div>
          <div className="audit-detail-row">
            <span className="detail-label">Details:</span>
            <span>{entry.details}</span>
          </div>
        </div>
      )}
    </div>
  );
}

export default function AuditTrail({ entries }: AuditTrailProps) {
  if (entries.length === 0) {
    return <p className="empty-state">No audit entries.</p>;
  }

  return (
    <div className="audit-trail">
      <div className="audit-header">
        <span>{entries.length} audit entries (hash-chained)</span>
      </div>
      {entries.map((entry, i) => (
        <AuditEntryRow key={`${entry.hash}-${i}`} entry={entry} />
      ))}
    </div>
  );
}
