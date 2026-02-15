import { useState, useCallback } from "react";
import type { ChatMsg } from "../components/ChatMessage";

const BASE_URL = "/api";

interface ChatResponse {
  reply: string;
  role?: "agent" | "suggestion" | "system";
  metadata?: ChatMsg["metadata"];
}

export function useChat(campaignId: string) {
  const [messages, setMessages] = useState<ChatMsg[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const sendMessage = useCallback(
    async (text: string) => {
      const isWelcome = text === "";

      if (!isWelcome && !text.trim()) return;

      setIsLoading(true);
      setError(null);

      if (!isWelcome) {
        const userMessage: ChatMsg = {
          id: `user-${Date.now()}`,
          role: "user",
          content: text,
          timestamp: Date.now(),
        };

        setMessages((prev) => [...prev, userMessage]);
      }

      try {
        const response = await fetch(`${BASE_URL}/chat/${campaignId}`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ message: text }),
        });

        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(errorText || `Request failed: ${response.statusText}`);
        }

        const data: ChatResponse = await response.json();

        const agentMessage: ChatMsg = {
          id: `agent-${Date.now()}`,
          role: data.metadata?.suggestions ? "suggestion" : "agent",
          content: data.reply,
          timestamp: Date.now(),
          metadata: data.metadata,
        };

        setMessages((prev) => [...prev, agentMessage]);
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : "Unknown error";
        setError(errorMessage);

        const systemMessage: ChatMsg = {
          id: `system-${Date.now()}`,
          role: "system",
          content: `Error: ${errorMessage}`,
          timestamp: Date.now(),
        };

        setMessages((prev) => [...prev, systemMessage]);
      } finally {
        setIsLoading(false);
      }
    },
    [campaignId]
  );

  return {
    messages,
    sendMessage,
    isLoading,
    error,
  };
}
