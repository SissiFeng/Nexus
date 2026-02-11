import { useParams } from "react-router-dom";
import { useEffect, useState } from "react";
import { fetchAuditLog, type AuditEntry } from "../api";
import AuditTrail from "../components/AuditTrail";

export default function Reports() {
  const { id } = useParams<{ id: string }>();
  const [entries, setEntries] = useState<AuditEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!id) return;
    setLoading(true);
    fetchAuditLog(id)
      .then((data) => {
        setEntries(data);
        setError(null);
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : "Failed to load audit log");
      })
      .finally(() => setLoading(false));
  }, [id]);

  return (
    <div className="page">
      <h1>Compliance Report</h1>
      <p className="campaign-id mono">Campaign: {id}</p>

      {error && <div className="error-banner">{error}</div>}

      {loading ? (
        <div className="loading">Loading audit log...</div>
      ) : entries.length === 0 ? (
        <div className="empty-state">
          <p>No audit entries found for this campaign.</p>
        </div>
      ) : (
        <AuditTrail entries={entries} />
      )}
    </div>
  );
}
