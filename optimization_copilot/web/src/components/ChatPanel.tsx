import { useState, useRef, useEffect } from "react";
import {
  X,
  Send,
  Beaker,
  Activity,
  HelpCircle,
  Target,
  Download,
  MessageCircle,
  Lightbulb,
} from "lucide-react";
import { ChatMessage } from "./ChatMessage";
import { useChat } from "../hooks/useChat";

interface ChatPanelProps {
  campaignId: string;
  isOpen: boolean;
  onToggle: () => void;
}

const QUICK_ACTIONS = [
  { text: "Discover insights from data", icon: Lightbulb, key: "discover" },
  { text: "Suggest next parameter values", icon: Beaker, key: "suggest" },
  { text: "Show diagnostics", icon: Activity, key: "diagnostics" },
  { text: "Why did you recommend this?", icon: HelpCircle, key: "why" },
  { text: "Focus on specific region", icon: Target, key: "focus" },
  { text: "Export results", icon: Download, key: "export" },
];

export function ChatPanel({ campaignId, isOpen, onToggle }: ChatPanelProps) {
  const [inputText, setInputText] = useState("");
  const { messages, sendMessage, isLoading } = useChat(campaignId);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const welcomeSent = useRef(false);

  useEffect(() => {
    if (messages.length === 0 && !welcomeSent.current) {
      welcomeSent.current = true;
      sendMessage("");
    }
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async () => {
    if (!inputText.trim() || isLoading) return;
    const text = inputText;
    setInputText("");
    await sendMessage(text);
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleQuickAction = (text: string) => {
    setInputText(text);
    inputRef.current?.focus();
  };

  if (!isOpen) {
    return (
      <button
        className="chat-panel-fab"
        onClick={onToggle}
        aria-label="Open chat"
      >
        <MessageCircle size={24} />
      </button>
    );
  }

  return (
    <div className="chat-panel">
      <div className="chat-panel-header">
        <div className="chat-panel-title">
          <MessageCircle size={16} />
          <span>Campaign {campaignId.slice(0, 8)}</span>
        </div>
        <button
          className="chat-panel-close"
          onClick={onToggle}
          aria-label="Close chat"
        >
          <X size={18} />
        </button>
      </div>

      <div className="chat-panel-messages">
        {messages.map((msg) => (
          <ChatMessage key={msg.id} message={msg} />
        ))}
        {isLoading && (
          <div className="chat-loading">
            <div className="chat-loading-dots">
              <span />
              <span />
              <span />
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="chat-panel-input-section">
        <div className="chat-quick-actions">
          {QUICK_ACTIONS.map((action) => (
            <button
              key={action.key}
              className="chat-quick-action-chip"
              onClick={() => handleQuickAction(action.text)}
              disabled={isLoading}
            >
              <action.icon size={12} />
              <span>{action.text.split(" ")[0]}</span>
            </button>
          ))}
        </div>

        <div className="chat-panel-input-wrapper">
          <textarea
            ref={inputRef}
            className="chat-panel-input"
            placeholder="Ask about your optimization..."
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyPress={handleKeyPress}
            disabled={isLoading}
            rows={1}
          />
          <button
            className="chat-panel-send-btn"
            onClick={handleSend}
            disabled={!inputText.trim() || isLoading}
            aria-label="Send message"
          >
            <Send size={18} />
          </button>
        </div>
      </div>
    </div>
  );
}
