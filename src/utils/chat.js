import { DEFAULT_MESSAGE, STORAGE_KEYS } from "../constants/chat";

export const createSession = (messages = [DEFAULT_MESSAGE]) => {
  const id = Date.now();
  return { id, title: "New Session", messages, createdAt: id };
};

export const getInitialChatState = () => {
  const fallbackSession = createSession();

  try {
    const savedSessions = JSON.parse(localStorage.getItem(STORAGE_KEYS.CHAT_SESSIONS) || "[]");
    if (Array.isArray(savedSessions) && savedSessions.length > 0) {
      return {
        sessions: savedSessions,
        currentSessionId: savedSessions[0].id,
        messages: savedSessions[0].messages,
      };
    }
  } catch (error) {
    console.warn("Failed to parse stored chat sessions. Resetting to default.", error);
  }

  return {
    sessions: [fallbackSession],
    currentSessionId: fallbackSession.id,
    messages: fallbackSession.messages,
  };
};

export const stripImagesFromHistory = (history = []) =>
  history.map((item) => {
    const rest = { ...item };
    delete rest.images;
    return rest;
  });

export const updateLastAiMessage = (messages, updater) => {
  const nextMessages = [...messages];
  const lastIndex = nextMessages.length - 1;
  if (lastIndex < 0 || nextMessages[lastIndex].role !== "AI") return nextMessages;

  nextMessages[lastIndex] = updater(nextMessages[lastIndex]);
  return nextMessages;
};

export const cleanFinalText = (rawText) => {
  if (!rawText) return "";

  const lines = rawText.split(/\r?\n/);
  const filteredLines = lines.filter((line) => {
    if (line.includes("🧠") && line.includes("分析")) return false;
    if (line.includes("⚡") && line.includes("生成")) return false;
    if (line.includes("🔗") && line.includes("調閱")) return false;
    if (line.includes("🤔") && line.includes("聯想")) return false;
    if (line.includes("✨") && line.includes("關鍵字")) return false;
    if (line.includes("🚀") && line.includes("檢索")) return false;
    if (line.includes("📄") && line.includes("觸發")) return false;
    if (line.includes("========")) return false;
    return true;
  });

  return filteredLines.join("\n").replace(/\n{3,}/g, "\n\n").trim();
};

export const parseSourcesHeader = (headerValue) => {
  if (!headerValue) return [];
  try {
    const parsed = JSON.parse(headerValue);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
};
