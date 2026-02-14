import type { LucideIcon } from "lucide-react";

interface EmptyStateProps {
  icon: LucideIcon;
  title: string;
  description: string;
  actionLabel?: string;
  onAction?: () => void;
}

export default function EmptyState({
  icon: Icon,
  title,
  description,
  actionLabel,
  onAction,
}: EmptyStateProps) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: "48px 24px",
      }}
    >
      <div
        style={{
          maxWidth: "420px",
          width: "100%",
          textAlign: "center",
        }}
      >
        <div
          style={{
            width: "64px",
            height: "64px",
            borderRadius: "50%",
            background: "var(--color-badge-completed-bg)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            margin: "0 auto 20px",
          }}
        >
          <Icon size={28} style={{ color: "var(--color-primary)" }} />
        </div>
        <h3
          style={{
            fontSize: "1.15rem",
            fontWeight: 600,
            color: "var(--color-text)",
            margin: "0 0 8px",
          }}
        >
          {title}
        </h3>
        <p
          style={{
            fontSize: "0.9rem",
            color: "var(--color-text-muted)",
            margin: "0 0 24px",
            lineHeight: 1.6,
          }}
        >
          {description}
        </p>
        {actionLabel && onAction && (
          <button
            onClick={onAction}
            className="btn btn-primary"
            style={{
              padding: "10px 24px",
              fontSize: "0.9rem",
              fontWeight: 500,
              borderRadius: "8px",
            }}
          >
            {actionLabel}
          </button>
        )}
      </div>
    </div>
  );
}
