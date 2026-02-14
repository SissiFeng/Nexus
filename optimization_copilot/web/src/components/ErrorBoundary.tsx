import { Component, type ReactNode, type ErrorInfo } from "react";

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

export default class ErrorBoundary extends Component<
  ErrorBoundaryProps,
  ErrorBoundaryState
> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    console.error("ErrorBoundary caught an error:", error, errorInfo);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            minHeight: "300px",
            padding: "32px",
          }}
        >
          <div
            style={{
              maxWidth: "480px",
              width: "100%",
              background: "var(--color-surface)",
              borderRadius: "12px",
              border: "1px solid var(--color-border)",
              boxShadow: "var(--shadow-md)",
              padding: "40px 32px",
              textAlign: "center",
            }}
          >
            <div
              style={{
                width: "56px",
                height: "56px",
                borderRadius: "50%",
                background: "var(--color-error-banner-bg)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                margin: "0 auto 20px",
                fontSize: "28px",
              }}
            >
              &#9888;
            </div>
            <h2
              style={{
                fontSize: "1.25rem",
                fontWeight: 600,
                color: "var(--color-text)",
                margin: "0 0 8px",
              }}
            >
              Something went wrong
            </h2>
            <p
              style={{
                fontSize: "0.9rem",
                color: "var(--color-text-muted)",
                margin: "0 0 24px",
                lineHeight: 1.5,
              }}
            >
              {this.state.error?.message || "An unexpected error occurred."}
            </p>
            <button
              onClick={this.handleReset}
              className="btn btn-primary"
              style={{
                padding: "10px 24px",
                fontSize: "0.9rem",
                fontWeight: 500,
                borderRadius: "8px",
              }}
            >
              Try Again
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
