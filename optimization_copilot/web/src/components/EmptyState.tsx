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
            background: "#f0f4ff",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            margin: "0 auto 20px",
          }}
        >
          <Icon size={28} color="#6366f1" />
        </div>
        <h3
          style={{
            fontSize: "1.15rem",
            fontWeight: 600,
            color: "#1a202c",
            margin: "0 0 8px",
          }}
        >
          {title}
        </h3>
        <p
          style={{
            fontSize: "0.9rem",
            color: "#718096",
            margin: "0 0 24px",
            lineHeight: 1.6,
          }}
        >
          {description}
        </p>
        {actionLabel && onAction && (
          <button
            onClick={onAction}
            style={{
              padding: "10px 24px",
              fontSize: "0.9rem",
              fontWeight: 500,
              color: "#ffffff",
              background: "#6366f1",
              border: "none",
              borderRadius: "8px",
              cursor: "pointer",
              transition: "background 0.15s",
            }}
            onMouseOver={(e) =>
              ((e.target as HTMLButtonElement).style.background = "#4f46e5")
            }
            onMouseOut={(e) =>
              ((e.target as HTMLButtonElement).style.background = "#6366f1")
            }
          >
            {actionLabel}
          </button>
        )}
      </div>
    </div>
  );
}
