import { useState, useCallback, createContext, useContext } from "react";
import { Check, AlertTriangle, Info, X } from "lucide-react";

type ToastType = "success" | "error" | "info" | "warning";

interface Toast {
  id: number;
  message: string;
  type: ToastType;
}

interface ToastContextValue {
  toast: (message: string, type?: ToastType) => void;
}

const ToastContext = createContext<ToastContextValue>({ toast: () => {} });

export function useToast() {
  return useContext(ToastContext);
}

let nextId = 0;

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const addToast = useCallback((message: string, type: ToastType = "success") => {
    const id = ++nextId;
    setToasts((prev) => [...prev, { id, message, type }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 3500);
  }, []);

  const dismiss = useCallback((id: number) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const icons: Record<ToastType, React.ReactNode> = {
    success: <Check size={15} />,
    error: <AlertTriangle size={15} />,
    warning: <AlertTriangle size={15} />,
    info: <Info size={15} />,
  };

  return (
    <ToastContext.Provider value={{ toast: addToast }}>
      {children}
      {toasts.length > 0 && (
        <div className="toast-container">
          {toasts.map((t) => (
            <div key={t.id} className={`toast-item toast-${t.type}`}>
              <span className="toast-icon">{icons[t.type]}</span>
              <span className="toast-msg">{t.message}</span>
              <button className="toast-close" onClick={() => dismiss(t.id)}>
                <X size={13} />
              </button>
            </div>
          ))}
        </div>
      )}
      <style>{`
        .toast-container {
          position: fixed;
          bottom: 20px;
          right: 20px;
          z-index: 9999;
          display: flex;
          flex-direction: column-reverse;
          gap: 8px;
          pointer-events: none;
        }
        .toast-item {
          display: flex;
          align-items: center;
          gap: 10px;
          padding: 10px 14px;
          border-radius: 8px;
          font-size: 0.82rem;
          font-weight: 500;
          font-family: var(--font-sans);
          box-shadow: 0 6px 20px rgba(0,0,0,0.12), 0 2px 6px rgba(0,0,0,0.06);
          animation: toastSlideIn 0.25s ease;
          pointer-events: auto;
          min-width: 220px;
          max-width: 380px;
        }
        .toast-success {
          background: #ecfdf5;
          border: 1px solid #6ee7a0;
          color: #065f46;
        }
        .toast-error {
          background: #fef2f2;
          border: 1px solid #fca5a5;
          color: #991b1b;
        }
        .toast-warning {
          background: #fffbeb;
          border: 1px solid #fcd34d;
          color: #92400e;
        }
        .toast-info {
          background: #eff6ff;
          border: 1px solid #93c5fd;
          color: #1e40af;
        }
        [data-theme="dark"] .toast-success {
          background: #052e16;
          border-color: #166534;
          color: #86efac;
        }
        [data-theme="dark"] .toast-error {
          background: #450a0a;
          border-color: #991b1b;
          color: #fca5a5;
        }
        [data-theme="dark"] .toast-warning {
          background: #451a03;
          border-color: #92400e;
          color: #fcd34d;
        }
        [data-theme="dark"] .toast-info {
          background: #172554;
          border-color: #1e40af;
          color: #93c5fd;
        }
        .toast-icon {
          flex-shrink: 0;
          display: flex;
        }
        .toast-msg {
          flex: 1;
          line-height: 1.4;
        }
        .toast-close {
          flex-shrink: 0;
          background: none;
          border: none;
          color: inherit;
          opacity: 0.5;
          cursor: pointer;
          padding: 2px;
          display: flex;
        }
        .toast-close:hover {
          opacity: 1;
        }
        @keyframes toastSlideIn {
          from {
            transform: translateX(30px);
            opacity: 0;
          }
          to {
            transform: translateX(0);
            opacity: 1;
          }
        }
      `}</style>
    </ToastContext.Provider>
  );
}
