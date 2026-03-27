import { motion, AnimatePresence } from "framer-motion";
import {
  Plus,
  Archive,
  MessageCircle,
  X,
  Upload,
  FileText,
  RefreshCw,
  Trash2,
  Sparkles,
} from "lucide-react";

const LABELS = {
  proVersion: "PRO VERSION",
  newChat: "New Chat",
  modelEngine: "Model Engine",
  modelAvailable: "Available",
  sessionLogs: "Session Logs",
  dataInjection: "Data Injection",
  uploadFiles: "Upload Files",
  uploadStart: "Start Upload",
  emptyKnowledge: "No knowledge files yet",
  knowledgeBase: "Knowledge Base",
  refresh: "Refresh",
  delete: "Delete",
  purge: "Purge System",
};

const SidebarPanel = ({
  sessions,
  currentSessionId,
  switchSession,
  deleteSession,
  createNewChat,
  selectedModel,
  setSelectedModel,
  availableModels,
  fileInputRef,
  handleFileSelect,
  filesToUpload,
  removeFile,
  handleUpload,
  uploadStatus,
  fileList,
  loadingFiles,
  fetchFileList,
  handleViewFile,
  handleDeleteFile,
  handleReset,
}) => {
  return (
    <div className="w-80 min-w-[300px] bg-slate-900/60 backdrop-blur-xl flex flex-col p-7 text-white relative border-r border-white/10">
      <div className="mb-9 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-11 h-11 rounded-2xl bg-gradient-to-br from-fuchsia-600 via-purple-600 to-cyan-600 flex items-center justify-center shadow-[0_0_20px_rgba(217,70,239,0.5)]">
            <Sparkles size={20} className="text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-fuchsia-400 to-cyan-400">
              Chroma AI
            </h1>
            <p className="text-[11px] text-slate-300 font-medium tracking-wide">
              {LABELS.proVersion}
            </p>
          </div>
        </div>
        <motion.button
          whileHover={{ scale: 1.1, rotate: 90 }}
          whileTap={{ scale: 0.9 }}
          onClick={createNewChat}
          className="p-2 rounded-full bg-slate-800 border border-white/10 hover:border-cyan-400 text-cyan-400 hover:text-white transition-colors"
          title={LABELS.newChat}
        >
          <Plus size={18} />
        </motion.button>
      </div>

      <div className="flex-1 overflow-y-auto custom-scrollbar pr-2 space-y-9">
        <div className="space-y-3">
          <label className="text-[11px] font-bold text-cyan-300 tracking-wide pl-1">
            {LABELS.modelEngine}
          </label>
          <div className="relative group">
            <div className="absolute inset-0 bg-gradient-to-r from-fuchsia-600 to-cyan-600 rounded-2xl blur opacity-20 group-hover:opacity-50 transition-opacity" />
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="relative w-full bg-slate-800/80 text-white text-[13px] rounded-2xl px-4 py-3.5 border border-white/10 focus:outline-none focus:ring-1 focus:ring-cyan-400 appearance-none cursor-pointer font-medium hover:bg-slate-700 transition-colors"
            >
              {availableModels.map((model) => (
                <option key={model} value={model}>
                  {model} ({LABELS.modelAvailable})
                </option>
              ))}
            </select>
            <div className="absolute right-4 top-4 text-cyan-400 pointer-events-none text-xs">
              {"\u25BE"}
            </div>
          </div>
        </div>

        <div className="space-y-3">
          <label className="text-[11px] font-bold text-violet-300 tracking-wide pl-1 flex items-center gap-2">
            <Archive size={11} /> {LABELS.sessionLogs}
          </label>
          <div className="space-y-2.5">
            {sessions.map((session) => (
              <motion.div
                key={session.id}
                onClick={() => switchSession(session.id)}
                className={`group relative flex items-center gap-3 p-3.5 rounded-xl cursor-pointer transition-all border ${
                  currentSessionId === session.id
                    ? "bg-slate-800/90 border-fuchsia-500/50 shadow-[0_0_15px_rgba(217,70,239,0.15)]"
                    : "bg-slate-800/30 border-transparent hover:bg-slate-800/60 hover:border-white/10"
                }`}
              >
                {currentSessionId === session.id && (
                  <div className="absolute left-0 w-1 h-6 bg-gradient-to-b from-fuchsia-500 to-cyan-500 rounded-r-full" />
                )}
                <MessageCircle
                  size={14}
                  className={currentSessionId === session.id ? "text-fuchsia-400" : "text-slate-500"}
                />
                <div className="flex-1 min-w-0">
                  <p
                    className={`text-[12px] font-medium truncate ${
                      currentSessionId === session.id
                        ? "text-white"
                        : "text-slate-300 group-hover:text-slate-100"
                    }`}
                  >
                    {session.title || LABELS.newChat}
                  </p>
                  <p className="text-[10px] text-slate-400 truncate mt-0.5">
                    {new Date(session.createdAt || Date.now()).toLocaleTimeString()}
                  </p>
                </div>
                <button
                  onClick={(e) => deleteSession(e, session.id)}
                  className="opacity-0 group-hover:opacity-100 p-1.5 hover:bg-red-500/20 rounded-md text-slate-500 hover:text-red-400 transition-all"
                  title={LABELS.delete}
                >
                  <X size={12} />
                </button>
              </motion.div>
            ))}
          </div>
        </div>

        <div className="space-y-4">
          <label className="text-[11px] font-bold text-fuchsia-300 tracking-wide pl-1">
            {LABELS.dataInjection}
          </label>
          <motion.div
            whileHover={{ scale: 1.02, backgroundColor: "rgba(30, 41, 59, 0.8)" }}
            whileTap={{ scale: 0.98 }}
            onClick={() => fileInputRef.current.click()}
            className="relative overflow-hidden bg-slate-800/50 border border-fuchsia-500/30 border-dashed rounded-3xl px-6 pt-7 pb-5 text-center cursor-pointer group transition-colors hover:border-fuchsia-400"
          >
            <div className="w-12 h-12 bg-slate-700/50 rounded-full flex items-center justify-center mx-auto mb-3 group-hover:bg-fuchsia-500/20 group-hover:text-fuchsia-300 transition-all text-slate-400">
              <Upload size={20} />
            </div>
            <p className="text-[15px] font-semibold text-slate-300 group-hover:text-white transition-colors">
              {LABELS.uploadFiles}
            </p>
            <div className="flex justify-center gap-1.5 mt-3">
              <span className="text-[9px] bg-slate-900/80 px-2 py-0.5 rounded text-orange-300 border border-orange-500/20">
                PDF
              </span>
              <span className="text-[9px] bg-slate-900/80 px-2 py-0.5 rounded text-cyan-300 border border-blue-500/20">
                DOCX
              </span>
              <span className="text-[9px] bg-slate-900/80 px-2 py-0.5 rounded text-purple-300 border border-purple-500/20">
                JPG
              </span>
              <span className="text-[9px] bg-slate-900/80 px-2 py-0.5 rounded text-emerald-300 border border-emerald-500/20">
                XLSX
              </span>
            </div>
            <input
              ref={fileInputRef}
              type="file"
              className="hidden"
              onChange={handleFileSelect}
              multiple
              accept=".pdf,.docx,.txt,.xlsx,.csv,.jpg,.jpeg,.png,.webp"
            />
          </motion.div>

          <AnimatePresence>
            {filesToUpload.length > 0 && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                className="space-y-3 overflow-hidden"
              >
                <div className="flex flex-wrap gap-2 pt-1">
                  {filesToUpload.map((file, idx) => (
                    <div
                      key={idx}
                      className="flex items-center gap-2 bg-slate-800/80 px-3 py-1.5 rounded-full text-[10px] border border-white/10"
                    >
                      <span className="truncate max-w-[80px] text-slate-300">{file.name}</span>
                      <button
                        onClick={() => removeFile(idx)}
                        className="text-slate-500 hover:text-white"
                        title={LABELS.delete}
                      >
                        <X size={10} />
                      </button>
                    </div>
                  ))}
                </div>
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={handleUpload}
                  className="w-full py-3 rounded-2xl bg-gradient-to-r from-fuchsia-600 via-violet-600 to-cyan-600 text-white text-sm font-bold shadow-[0_0_20px_rgba(217,70,239,0.3)]"
                >
                  {LABELS.uploadStart}
                </motion.button>
              </motion.div>
            )}
          </AnimatePresence>

          {uploadStatus && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="text-center text-sm font-semibold text-cyan-200 bg-cyan-900/30 py-2.5 rounded-xl border border-cyan-500/30"
            >
              {uploadStatus}
            </motion.div>
          )}

          <div className="mt-1 rounded-2xl border border-emerald-400/35 bg-emerald-500/10 px-3 py-3.5 shadow-[0_0_20px_rgba(16,185,129,0.15)]">
            <div className="flex items-center justify-between mb-2 px-1">
              <label className="text-[12px] font-extrabold text-emerald-300 tracking-wide flex items-center gap-2">
                <FileText size={12} /> {LABELS.knowledgeBase}
              </label>
              <div className="flex items-center gap-2">
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/20 border border-emerald-300/20 text-emerald-200 font-semibold">
                  {fileList.length}
                </span>
                <button
                  onClick={fetchFileList}
                  className="text-slate-400 hover:text-emerald-300 transition-colors"
                  title={LABELS.refresh}
                >
                  <RefreshCw size={12} className={loadingFiles ? "animate-spin" : ""} />
                </button>
              </div>
            </div>

            <div className="flex-1 overflow-y-auto custom-scrollbar space-y-1.5 pr-1 max-h-[190px]">
              {fileList.length === 0 ? (
                <p className="text-center text-[11px] text-slate-300/80 py-4 italic">
                  {LABELS.emptyKnowledge}
                </p>
              ) : (
                <AnimatePresence>
                  {fileList.map((file) => (
                    <motion.div
                      key={file}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, height: 0 }}
                      className="group flex items-center justify-between p-2.5 rounded-lg bg-slate-900/40 border border-transparent hover:border-emerald-400/40 hover:bg-slate-900/70 transition-all cursor-pointer"
                      onClick={() => handleViewFile(file)}
                    >
                      <div className="flex items-center gap-2 min-w-0">
                        <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 shadow-[0_0_6px_#34d399]" />
                        <span className="text-[12px] text-slate-100 truncate font-mono max-w-[150px]" title={file}>
                          {file}
                        </span>
                      </div>

                      <button
                        onClick={(e) => handleDeleteFile(e, file)}
                        className="opacity-0 group-hover:opacity-100 p-1.5 rounded hover:bg-red-500/20 text-slate-400 hover:text-red-400 transition-all"
                        title={LABELS.delete}
                      >
                        <Trash2 size={12} />
                      </button>
                    </motion.div>
                  ))}
                </AnimatePresence>
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="pt-5 mt-auto">
        <button
          onClick={handleReset}
          className="w-full py-3.5 flex items-center justify-center gap-2 text-red-300 border border-red-500/35 hover:bg-red-500/10 hover:text-red-200 hover:border-red-400 rounded-2xl transition-all text-sm font-bold shadow-[0_0_10px_rgba(239,68,68,0.1)] hover:shadow-[0_0_20px_rgba(239,68,68,0.2)]"
        >
          <Trash2 size={14} /> {LABELS.purge}
        </button>
      </div>
    </div>
  );
};

export default SidebarPanel;
