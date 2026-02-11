import { useState, useMemo } from "react";

interface Trial {
  id: string;
  parameters: Record<string, number>;
  kpis: Record<string, number>;
  status: string;
}

interface TrialTableProps {
  trials: Trial[];
}

type SortKey = { col: string; type: "param" | "kpi" | "field" };
type SortDir = "asc" | "desc";

export default function TrialTable({ trials }: TrialTableProps) {
  const [sortKey, setSortKey] = useState<SortKey | null>(null);
  const [sortDir, setSortDir] = useState<SortDir>("asc");

  const paramKeys = useMemo(() => {
    const keys = new Set<string>();
    trials.forEach((t) => Object.keys(t.parameters).forEach((k) => keys.add(k)));
    return Array.from(keys);
  }, [trials]);

  const kpiKeys = useMemo(() => {
    const keys = new Set<string>();
    trials.forEach((t) => Object.keys(t.kpis).forEach((k) => keys.add(k)));
    return Array.from(keys);
  }, [trials]);

  const handleSort = (col: string, type: "param" | "kpi" | "field") => {
    if (sortKey?.col === col && sortKey?.type === type) {
      setSortDir((prev) => (prev === "asc" ? "desc" : "asc"));
    } else {
      setSortKey({ col, type });
      setSortDir("asc");
    }
  };

  const sortedTrials = useMemo(() => {
    if (!sortKey) return trials;

    return [...trials].sort((a, b) => {
      let valA: number | string;
      let valB: number | string;

      if (sortKey.type === "param") {
        valA = a.parameters[sortKey.col] ?? 0;
        valB = b.parameters[sortKey.col] ?? 0;
      } else if (sortKey.type === "kpi") {
        valA = a.kpis[sortKey.col] ?? 0;
        valB = b.kpis[sortKey.col] ?? 0;
      } else {
        valA = sortKey.col === "status" ? a.status : a.id;
        valB = sortKey.col === "status" ? b.status : b.id;
      }

      if (typeof valA === "number" && typeof valB === "number") {
        return sortDir === "asc" ? valA - valB : valB - valA;
      }
      const cmp = String(valA).localeCompare(String(valB));
      return sortDir === "asc" ? cmp : -cmp;
    });
  }, [trials, sortKey, sortDir]);

  const sortIndicator = (col: string, type: "param" | "kpi" | "field") => {
    if (sortKey?.col !== col || sortKey?.type !== type) return "";
    return sortDir === "asc" ? " \u25B2" : " \u25BC";
  };

  if (trials.length === 0) {
    return <p className="empty-state">No trials to display.</p>;
  }

  return (
    <div className="table-wrapper">
      <table className="data-table trial-table">
        <thead>
          <tr>
            <th
              className="sortable"
              onClick={() => handleSort("id", "field")}
            >
              Trial ID{sortIndicator("id", "field")}
            </th>
            {paramKeys.map((k) => (
              <th
                key={`p-${k}`}
                className="sortable"
                onClick={() => handleSort(k, "param")}
              >
                {k}{sortIndicator(k, "param")}
              </th>
            ))}
            {kpiKeys.map((k) => (
              <th
                key={`k-${k}`}
                className="sortable"
                onClick={() => handleSort(k, "kpi")}
              >
                {k}{sortIndicator(k, "kpi")}
              </th>
            ))}
            <th
              className="sortable"
              onClick={() => handleSort("status", "field")}
            >
              Status{sortIndicator("status", "field")}
            </th>
          </tr>
        </thead>
        <tbody>
          {sortedTrials.map((trial) => (
            <tr key={trial.id}>
              <td className="mono">{trial.id.slice(0, 8)}</td>
              {paramKeys.map((k) => (
                <td key={`p-${k}`} className="mono">
                  {trial.parameters[k]?.toPrecision(4) ?? "-"}
                </td>
              ))}
              {kpiKeys.map((k) => (
                <td key={`k-${k}`} className="mono">
                  {trial.kpis[k]?.toPrecision(4) ?? "-"}
                </td>
              ))}
              <td>
                <span className={`badge badge-${trial.status}`}>
                  {trial.status}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
