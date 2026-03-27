import React, { useEffect, useRef, useState } from "react";
import { AnimatePresence } from "framer-motion";

import { STORAGE_KEYS } from "./constants/chat";
import { createSession } from "./utils/chat";
import CyberpunkNeonBackground from "./components/chat/CyberpunkNeonBackground";
import ConnectionErrorModal from "./components/chat/ConnectionErrorModal";
import SidebarPanel from "./components/chat/SidebarPanel";
import ChatWorkspace from "./components/chat/ChatWorkspace";
import FileViewerModal from "./components/chat/FileViewerModal";
import {
  fetchBackendStatus,
  fetchModels as fetchModelsApi,
  resetSystem,
} from "./services/chatApi";
import { useSessions } from "./hooks/useSessions";
import { useFileManager } from "./hooks/useFileManager";
import { useChatStream } from "./hooks/useChatStream";

function App() {
  const {
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
  } = useSessions();

  const [input, setInput] = useState("");
  const [availableModels, setAvailableModels] = useState(["gemma3:27b"]);
  const [selectedModel, setSelectedModel] = useState("gemma3:27b");
  const [errorModal, setErrorModal] = useState({ show: false, message: "" });

  const fileInputRef = useRef(null);
  const chatEndRef = useRef(null);
  const textareaRef = useRef(null);

  const {
    fileList,
    viewingFile,
    setViewingFile,
    viewContent,
    loadingFiles,
    filesToUpload,
    uploadStatus,
    fetchFileList,
    handleDeleteFile,
    handleViewFile,
    handleFileSelect,
    removeFile,
    handleUpload,
    resetFileManager,
  } = useFileManager({
    currentSessionIdRef,
    syncMessagesBySession,
    setErrorModal,
  });

  const {
    loadingSessionId,
    chatImages,
    setChatImages,
    chatImageInputRef,
    handleChatImageSelect,
    removeChatImage,
    handleSendMessage,
    handleStop,
    abortActiveStream,
  } = useChatStream({
    input,
    setInput,
    messages,
    selectedModel,
    currentSessionId,
    currentSessionIdRef,
    syncMessagesBySession,
    setErrorModal,
  });

  useEffect(() => {
    setChatImages([]);
  }, [currentSessionId, setChatImages]);

  useEffect(() => {
    const checkBackendStatus = async () => {
      try {
        const status = await fetchBackendStatus();
        const currentBootTime = status.boot_time;
        const savedBootTime = localStorage.getItem(STORAGE_KEYS.BACKEND_BOOT_TIME);

        if (!savedBootTime) {
          localStorage.setItem(STORAGE_KEYS.BACKEND_BOOT_TIME, currentBootTime);
          return;
        }

        if (savedBootTime !== currentBootTime) {
          localStorage.removeItem(STORAGE_KEYS.CHAT_SESSIONS);

          const rebootSession = createSession([
            {
              role: "AI",
              content:
                "🔄 **System Rebooted**\nBackend restarted successfully. Your conversation has been reset.",
            },
          ]);

          setSessions([rebootSession]);
          setCurrentSessionId(rebootSession.id);
          setMessages(rebootSession.messages);
          resetFileManager();

          localStorage.setItem(STORAGE_KEYS.BACKEND_BOOT_TIME, currentBootTime);
        }
      } catch {
        // Ignore transient status polling errors.
      }
    };

    checkBackendStatus();
    const radarInterval = setInterval(checkBackendStatus, 5000);

    return () => clearInterval(radarInterval);
  }, [setSessions, setCurrentSessionId, setMessages, resetFileManager]);

  useEffect(() => {
    const loadModels = async () => {
      try {
        const modelData = await fetchModelsApi();
        const rawModels = Array.isArray(modelData?.models) ? modelData.models : [];
        const modelNames = rawModels
          .map((model) => model.name)
          .filter((name) => !name.toLowerCase().includes("embed"));

        if (modelNames.length === 0) return;

        setAvailableModels(modelNames);
        setSelectedModel((prevModel) =>
          modelNames.includes(prevModel) ? prevModel : modelNames[0]
        );
      } catch (error) {
        console.error("Failed to load model list.", error);
      }
    };

    loadModels();
  }, []);

  useEffect(() => {
    requestAnimationFrame(() => {
      if (!chatEndRef.current) return;
      chatEndRef.current.scrollIntoView({ behavior: "smooth", block: "end" });
    });
  }, [messages, loadingSessionId]);

  useEffect(() => {
    if (!textareaRef.current) return;

    textareaRef.current.style.height = "auto";
    textareaRef.current.style.height = `${Math.min(
      textareaRef.current.scrollHeight,
      120
    )}px`;
  }, [input]);

  const handleReset = async () => {
    if (
      !window.confirm(
        "Are you sure you want to reset the backend and clear all conversations and files?"
      )
    ) {
      return;
    }

    abortActiveStream();

    try {
      await resetSystem();

      const resetSession = createSession([
        {
          role: "AI",
          content:
            "🧹 **System Cleared**\nAll conversations and uploaded files have been reset.",
        },
      ]);

      setSessions([resetSession]);
      setCurrentSessionId(resetSession.id);
      setMessages(resetSession.messages);
      localStorage.removeItem(STORAGE_KEYS.CHAT_SESSIONS);

      resetFileManager();
      setChatImages([]);
    } catch (error) {
      console.error(error);
      setErrorModal({ show: true, message: "Reset failed" });
    }
  };

  return (
    <div className="flex items-center justify-center h-screen w-screen bg-[#0f0c29] font-sans overflow-hidden relative selection:bg-fuchsia-500 selection:text-white text-slate-800">
      <CyberpunkNeonBackground />
      <div className="relative z-10 w-[90vw] h-[90vh] max-w-[1400px] flex rounded-[40px] overflow-hidden shadow-[0_0_50px_rgba(217,70,239,0.2)] border border-white/20 bg-white/10 backdrop-blur-2xl">
        <SidebarPanel
          sessions={sessions}
          currentSessionId={currentSessionId}
          switchSession={switchSession}
          deleteSession={deleteSession}
          createNewChat={createNewChat}
          selectedModel={selectedModel}
          setSelectedModel={setSelectedModel}
          availableModels={availableModels}
          fileInputRef={fileInputRef}
          handleFileSelect={handleFileSelect}
          filesToUpload={filesToUpload}
          removeFile={removeFile}
          handleUpload={handleUpload}
          uploadStatus={uploadStatus}
          fileList={fileList}
          loadingFiles={loadingFiles}
          fetchFileList={fetchFileList}
          handleViewFile={handleViewFile}
          handleDeleteFile={handleDeleteFile}
          handleReset={handleReset}
        />
        <ChatWorkspace
          currentSessionId={currentSessionId}
          messages={messages}
          loadingSessionId={loadingSessionId}
          chatEndRef={chatEndRef}
          chatImages={chatImages}
          removeChatImage={removeChatImage}
          chatImageInputRef={chatImageInputRef}
          handleChatImageSelect={handleChatImageSelect}
          textareaRef={textareaRef}
          input={input}
          setInput={setInput}
          handleSendMessage={handleSendMessage}
          handleStop={handleStop}
        />
      </div>
      <AnimatePresence>
        {errorModal.show && (
          <ConnectionErrorModal
            message={errorModal.message}
            onClose={() => setErrorModal({ show: false, message: "" })}
          />
        )}
      </AnimatePresence>
      <FileViewerModal
        viewingFile={viewingFile}
        viewContent={viewContent}
        onClose={() => setViewingFile(null)}
      />
    </div>
  );
}

export default App;
