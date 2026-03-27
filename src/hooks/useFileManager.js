import { useCallback, useEffect, useRef, useState } from "react";

import {
  deleteFile,
  fetchFiles,
  getFileViewUrl,
  uploadFiles,
} from "../services/chatApi";

export const useFileManager = ({
  currentSessionIdRef,
  syncMessagesBySession,
  setErrorModal,
}) => {
  const [fileList, setFileList] = useState([]);
  const [viewingFile, setViewingFile] = useState(null);
  const [viewContent] = useState("");
  const [loadingFiles, setLoadingFiles] = useState(false);
  const [filesToUpload, setFilesToUpload] = useState([]);
  const [uploadStatus, setUploadStatus] = useState("");
  const uploadStatusTimerRef = useRef(null);

  const appendAiMessage = useCallback(
    (content) => {
      const sessionId = currentSessionIdRef?.current;
      if (sessionId == null || typeof syncMessagesBySession !== "function") return;

      syncMessagesBySession(sessionId, (prevMessages) => [
        ...prevMessages,
        { role: "AI", content },
      ]);
    },
    [currentSessionIdRef, syncMessagesBySession]
  );

  const fetchFileList = useCallback(async () => {
    setLoadingFiles(true);
    try {
      const data = await fetchFiles();
      setFileList(Array.isArray(data?.files) ? data.files : []);
    } catch (error) {
      console.error("Failed to fetch file list.", error);
    } finally {
      setLoadingFiles(false);
    }
  }, []);

  useEffect(() => {
    fetchFileList();
  }, [fetchFileList]);

  useEffect(() => {
    return () => {
      if (uploadStatusTimerRef.current) {
        clearTimeout(uploadStatusTimerRef.current);
      }
    };
  }, []);

  const handleDeleteFile = async (event, filename) => {
    event?.stopPropagation?.();
    if (!window.confirm(`Are you sure you want to delete "${filename}"?`)) return;

    try {
      await deleteFile(filename);
      await fetchFileList();
      appendAiMessage(`🗑️ **File Removed**\n\nDeleted \`${filename}\`.`);
    } catch (error) {
      alert("Delete failed.");
      console.error(error);
    }
  };

  const handleViewFile = (filename) => {
    const fileUrl = getFileViewUrl(filename);
    window.open(fileUrl, "_blank", "noopener,noreferrer");
  };

  const handleFileSelect = (event) => {
    const selected = Array.from(event.target.files || []);
    if (selected.length === 0) return;

    setFilesToUpload((prevFiles) => [...prevFiles, ...selected]);
    event.target.value = "";
  };

  const removeFile = (index) => {
    setFilesToUpload((prevFiles) => prevFiles.filter((_, i) => i !== index));
  };

  const handleUpload = async () => {
    if (filesToUpload.length === 0) return;
    setUploadStatus("📡 Uploading...");

    const formData = new FormData();
    filesToUpload.forEach((file) => formData.append("files", file));

    try {
      const { processed } = await uploadFiles(formData);
      setUploadStatus("✅ Sync complete");
      await fetchFileList();
      appendAiMessage(
        `📦 **Knowledge Synced**\n\nProcessed **${
          Array.isArray(processed) ? processed.length : 0
        }** file(s).`
      );

      setFilesToUpload([]);
      if (uploadStatusTimerRef.current) {
        clearTimeout(uploadStatusTimerRef.current);
      }
      uploadStatusTimerRef.current = setTimeout(() => setUploadStatus(""), 3000);
    } catch (error) {
      console.error(error);
      setUploadStatus("");
      setErrorModal?.({ show: true, message: "Upload failed" });
    }
  };

  const resetFileManager = useCallback(() => {
    setUploadStatus("");
    setFilesToUpload([]);
    setFileList([]);
  }, []);

  return {
    fileList,
    setFileList,
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
  };
};
