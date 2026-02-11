import { useEffect, useRef, useState, useCallback } from "react";

export interface WsMessage {
  type: string;
  payload: unknown;
  timestamp: number;
}

type ConnectionStatus = "connecting" | "connected" | "disconnected" | "error";

export function useWebSocket(campaignId?: string) {
  const [messages, setMessages] = useState<WsMessage[]>([]);
  const [status, setStatus] = useState<ConnectionStatus>("disconnected");
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout>>();

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host;
    const path = campaignId ? `/api/ws/${campaignId}` : "/api/ws";
    const url = `${protocol}//${host}${path}`;

    setStatus("connecting");
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus("connected");
    };

    ws.onmessage = (event) => {
      try {
        const msg: WsMessage = JSON.parse(event.data);
        setMessages((prev) => [...prev, msg]);
      } catch {
        // ignore malformed messages
      }
    };

    ws.onerror = () => {
      setStatus("error");
    };

    ws.onclose = () => {
      setStatus("disconnected");
      wsRef.current = null;
      // attempt reconnect after 3 seconds
      reconnectTimer.current = setTimeout(connect, 3000);
    };
  }, [campaignId]);

  useEffect(() => {
    connect();

    return () => {
      clearTimeout(reconnectTimer.current);
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [connect]);

  const clearMessages = useCallback(() => {
    setMessages([]);
  }, []);

  return { messages, status, clearMessages };
}
