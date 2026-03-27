import { useCallback, useEffect, useRef, useState } from "react";

import {
  cleanFinalText,
  parseSourcesHeader,
  stripImagesFromHistory,
  updateLastAiMessage,
} from "../utils/chat";
import { sendChatStream } from "../services/chatApi";

export const useChatStream = ({
  input,
  setInput,
  messages,
  selectedModel,
  currentSessionId,
  currentSessionIdRef,
  syncMessagesBySession,
  setErrorModal,
}) => {
  const [loadingSessionId, setLoadingSessionId] = useState(null);
  const [chatImages, setChatImages] = useState([]);
  const chatImageInputRef = useRef(null);
  const abortControllerRef = useRef(null);
  const typingIntervalRef = useRef(null);

  const clearTypingInterval = useCallback(() => {
    if (typingIntervalRef.current) {
      clearInterval(typingIntervalRef.current);
      typingIntervalRef.current = null;
    }
  }, []);

  const abortActiveStream = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    clearTypingInterval();
    setLoadingSessionId(null);
  }, [clearTypingInterval]);

  useEffect(() => {
    return () => {
      abortActiveStream();
    };
  }, [abortActiveStream]);

  const processImageFile = (file) =>
    new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => {
        const result = String(reader.result || "");
        const base64 = result.split(",")[1];
        resolve({
          id: Date.now() + Math.random(),
          url: result,
          base64,
          name: file.name,
        });
      };
      reader.onerror = (error) => reject(error);
    });

  const handleChatImageSelect = async (event) => {
    const files = Array.from(event.target.files || []);
    if (files.length === 0) return;

    try {
      const processedImages = await Promise.all(files.map(processImageFile));
      setChatImages((prevImages) => [...prevImages, ...processedImages]);
    } catch (error) {
      console.error("Failed to process chat images.", error);
      setErrorModal?.({ show: true, message: "Image processing failed" });
    } finally {
      event.target.value = "";
    }
  };

  const removeChatImage = (index) => {
    setChatImages((prevImages) => prevImages.filter((_, i) => i !== index));
  };

  const handleStop = () => {
    abortActiveStream();

    const targetSessionId = loadingSessionId ?? currentSessionIdRef.current;
    if (targetSessionId == null) return;

    syncMessagesBySession(targetSessionId, (prevMessages) =>
      updateLastAiMessage(prevMessages, (lastMessage) => ({
        ...lastMessage,
        isTyping: false,
      }))
    );
  };

  const handleSendMessage = async (event) => {
    if (event && event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
    } else if (event && (event.key !== "Enter" || event.shiftKey)) {
      return;
    }

    if (!input.trim() && chatImages.length === 0) return;

    const targetSessionId = currentSessionIdRef.current ?? currentSessionId;
    if (targetSessionId == null) return;

    const userMessage = {
      role: "User",
      content: input,
      images: chatImages.map((img) => img.url),
    };
    const aiPlaceholder = { role: "AI", content: "", sources: [], isTyping: true };

    syncMessagesBySession(targetSessionId, (prevMessages) => [
      ...prevMessages,
      userMessage,
      aiPlaceholder,
    ]);

    const imagesPayload = chatImages.map((img) => img.base64);
    const cleanHistory = stripImagesFromHistory(messages);

    setChatImages([]);
    setInput("");
    setLoadingSessionId(targetSessionId);

    const controller = new AbortController();
    abortControllerRef.current = controller;

    let fullRawText = "";
    let displayedText = "";

    try {
      const response = await sendChatStream({
        query: userMessage.content,
        modelName: selectedModel,
        history: cleanHistory,
        images: imagesPayload,
        signal: controller.signal,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Network error: ${response.status} - ${errorText}`);
      }

      const sources = parseSourcesHeader(response.headers.get("X-Sources"));
      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error("Empty response stream.");
      }

      const decoder = new TextDecoder();
      let isStreamDone = false;

      clearTypingInterval();
      typingIntervalRef.current = setInterval(() => {
        const remainingChars = fullRawText.length - displayedText.length;

        if (remainingChars > 0) {
          const dynamicChunk = Math.max(2, Math.floor(remainingChars / 20));
          displayedText = fullRawText.slice(0, displayedText.length + dynamicChunk);

          syncMessagesBySession(targetSessionId, (prevMessages) =>
            updateLastAiMessage(prevMessages, (lastMessage) => ({
              ...lastMessage,
              content: displayedText,
              sources,
              isTyping: true,
            }))
          );
          return;
        }

        if (!isStreamDone) return;

        clearTypingInterval();
        setLoadingSessionId(null);
        syncMessagesBySession(targetSessionId, (prevMessages) =>
          updateLastAiMessage(prevMessages, (lastMessage) => ({
            ...lastMessage,
            content: cleanFinalText(lastMessage.content),
            isTyping: false,
          }))
        );
      }, 50);

      for (;;) {
        const { value, done } = await reader.read();
        if (done) {
          fullRawText += decoder.decode();
          isStreamDone = true;
          break;
        }
        fullRawText += decoder.decode(value, { stream: true });
      }
    } catch (error) {
      if (error.name !== "AbortError") {
        console.error("Stream request failed:", error);
        setErrorModal?.({ show: true, message: "Stream connection failed" });
      }

      setLoadingSessionId(null);
      clearTypingInterval();

      syncMessagesBySession(targetSessionId, (prevMessages) =>
        updateLastAiMessage(prevMessages, (lastMessage) => ({
          ...lastMessage,
          content: cleanFinalText(displayedText || fullRawText || lastMessage.content),
          isTyping: false,
        }))
      );
    } finally {
      abortControllerRef.current = null;
    }
  };

  return {
    loadingSessionId,
    chatImages,
    setChatImages,
    chatImageInputRef,
    handleChatImageSelect,
    removeChatImage,
    handleSendMessage,
    handleStop,
    abortActiveStream,
  };
};
