import { useEffect, useRef, useState } from "react";

import { STORAGE_KEYS } from "../constants/chat";
import { createSession, getInitialChatState } from "../utils/chat";

// 🚀 新增：資料清洗過濾器
// 在存入 localStorage 之前，把肥大的圖片 Base64 拔掉，只保留純文字紀錄
const cleanSessionsForStorage = (sessions) => {
  return sessions.map((session) => ({
    ...session,
    messages: session.messages.map((msg) => {
      // 將 images (或任何你用來存圖片的 key) 解構出來丟掉，只回傳乾淨的 cleanMsg
      const { images, image_url, ...cleanMsg } = msg;
      return cleanMsg;
    }),
  }));
};

export const useSessions = () => {
  const initialChatStateRef = useRef(null);
  if (!initialChatStateRef.current) {
    initialChatStateRef.current = getInitialChatState();
  }

  const [sessions, setSessions] = useState(initialChatStateRef.current.sessions);
  const [currentSessionId, setCurrentSessionId] = useState(initialChatStateRef.current.currentSessionId);
  const [messages, setMessages] = useState(initialChatStateRef.current.messages);
  const currentSessionIdRef = useRef(currentSessionId);

  useEffect(() => {
    currentSessionIdRef.current = currentSessionId;
  }, [currentSessionId]);

  useEffect(() => {
    if (!currentSessionId) return;

    setSessions((prevSessions) => {
      const updatedSessions = prevSessions.map((session) => {
        if (session.id === currentSessionId) {
          let newTitle = session.title;
          const firstUserMsg = messages.find((m) => m.role === "User");
          if (firstUserMsg && session.title === "New Session") {
            newTitle = firstUserMsg.content.slice(0, 15) + (firstUserMsg.content.length > 15 ? "..." : "");
          }
          return { ...session, messages, title: newTitle };
        }
        return session;
      });

      // 🚀 修改：存檔前套用清洗過濾器
      localStorage.setItem(STORAGE_KEYS.CHAT_SESSIONS, JSON.stringify(cleanSessionsForStorage(updatedSessions)));
      return updatedSessions;
    });
  }, [messages, currentSessionId]);

  const syncMessagesBySession = (sessionId, updater) => {
    setSessions((prevSessions) =>
      prevSessions.map((session) =>
        session.id === sessionId
          ? { ...session, messages: updater(session.messages) }
          : session
      )
    );

    if (currentSessionIdRef.current === sessionId) {
      setMessages((prevMessages) => updater(prevMessages));
    }
  };

  const createNewChat = () => {
    const newSession = createSession();
    setSessions((prev) => [newSession, ...prev]);
    setCurrentSessionId(newSession.id);
    setMessages(newSession.messages);
  };

  const switchSession = (sessionId) => {
    const targetSession = sessions.find((s) => s.id === sessionId);
    if (!targetSession) return;

    setSessions((prevSessions) => {
      const updatedSessions = prevSessions.map((session) =>
        session.id === currentSessionId ? { ...session, messages } : session
      );
      // 🚀 修改：存檔前套用清洗過濾器
      localStorage.setItem(STORAGE_KEYS.CHAT_SESSIONS, JSON.stringify(cleanSessionsForStorage(updatedSessions)));
      return updatedSessions;
    });

    setCurrentSessionId(sessionId);
    setMessages(targetSession.messages);
  };

  const deleteSession = (event, sessionId) => {
    event.stopPropagation();
    const newSessions = sessions.filter((session) => session.id !== sessionId);
    setSessions(newSessions);

    // 🚀 修改：存檔前套用清洗過濾器
    localStorage.setItem(STORAGE_KEYS.CHAT_SESSIONS, JSON.stringify(cleanSessionsForStorage(newSessions)));

    if (sessionId !== currentSessionId) return;

    if (newSessions.length > 0) {
      setCurrentSessionId(newSessions[0].id);
      setMessages(newSessions[0].messages);
      return;
    }

    const initialSession = createSession();
    setSessions([initialSession]);
    setCurrentSessionId(initialSession.id);
    setMessages(initialSession.messages);
  };

  return {
    sessions,
    setSessions,
    currentSessionId,
    setCurrentSessionId,
    currentSessionIdRef,
    messages,
    setMessages,
    syncMessagesBySession,
    createNewChat,
    switchSession,
    deleteSession,
  };
};