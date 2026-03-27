import { motion, AnimatePresence } from "framer-motion";
import {
  Bot,
  User,
  X,
  Square,
  Send,
  Image as ImageIcon,
  Paperclip,
} from "lucide-react";

import MarkdownRenderer from "./MarkdownRenderer";
import ThinkingBubble from "./ThinkingBubble";

const ChatWorkspace = ({
  currentSessionId,
  messages,
  loadingSessionId,
  chatEndRef,
  chatImages,
  removeChatImage,
  chatImageInputRef,
  handleChatImageSelect,
  textareaRef,
  input,
  setInput,
  handleSendMessage,
  handleStop,
}) => {
  return (
    <div className="flex-1 flex flex-col relative bg-white/60 backdrop-blur-3xl border-l border-white/50">
      <div className="h-16 border-b border-white/50 flex items-center px-8 bg-white/40 backdrop-blur-md justify-between z-20">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse shadow-[0_0_10px_#34d399]" />
          <span className="text-sm font-bold text-slate-600">Connection Secure</span>
          <span className="text-xs text-slate-400 ml-2 px-2 py-0.5 bg-white/50 rounded-full border border-white/20">
            ID: {currentSessionId ? currentSessionId.toString().slice(-6) : "Unknown"}
          </span>
        </div>
        <div className="flex gap-2">
          <div className="w-3 h-3 rounded-full bg-red-400/50" />
          <div className="w-3 h-3 rounded-full bg-yellow-400/50" />
          <div className="w-3 h-3 rounded-full bg-green-400/50" />
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-8 space-y-8 scroll-smooth relative z-10">
        {messages.map((msg, index) => {
          if (msg.role === "AI" && msg.isTyping) return null;

          return (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 15, scale: 0.98 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              className={`flex ${msg.role === "User" ? "justify-end" : "justify-start"}`}
            >
              <div className={`flex gap-3 max-w-[85%] ${msg.role === "User" ? "flex-row-reverse" : "flex-row"}`}>
                <div
                  className={`w-10 h-10 rounded-2xl flex-shrink-0 flex items-center justify-center shadow-md ${
                    msg.role === "User"
                      ? "bg-gradient-to-tr from-sky-400 to-cyan-400 text-white shadow-sky-500/40"
                      : "bg-white text-cyan-600 shadow-slate-200/50"
                  }`}
                >
                  {msg.role === "User" ? <User size={18} /> : <Bot size={22} />}
                </div>
                <div className={`flex flex-col ${msg.role === "User" ? "items-end" : "items-start"} min-w-0`}>
                  <div
                    className={`p-5 rounded-3xl shadow-sm backdrop-blur-xl border relative overflow-hidden ${
                      msg.role === "User"
                        ? "bg-gradient-to-br from-sky-500 to-blue-600 text-white rounded-br-none border-white/20 shadow-[0_5px_15px_rgba(14,165,233,0.3)]"
                        : "bg-white/80 text-slate-800 rounded-bl-none border-white/60 shadow-lg shadow-cyan-500/5"
                    }`}
                  >
                    {msg.content && (
                      <div
                        className={`prose max-w-none text-sm leading-relaxed ${
                          msg.role === "User" ? "prose-invert text-white" : "prose-slate"
                        }`}
                      >
                        <MarkdownRenderer content={msg.content} />
                      </div>
                    )}

                    {msg.images && msg.images.length > 0 && (
                      <div className="flex flex-wrap gap-2 mb-3">
                        {msg.images.map((imgUrl, i) => (
                          <img
                            key={i}
                            src={imgUrl}
                            alt="uploaded"
                            className="max-w-[200px] max-h-[200px] rounded-lg border border-white/20"
                          />
                        ))}
                      </div>
                    )}

                    {!msg.isTyping && msg.sources && msg.sources.length > 0 && (
                      <div className="mt-4 pt-3 border-t border-white/20 flex flex-wrap gap-2">
                        {msg.sources.map((src, i) => (
                          <span
                            key={i}
                            className="text-[10px] px-2 py-1 rounded-md bg-white/20 border border-white/10 text-white/90 flex items-center gap-1 font-bold"
                          >
                            <Paperclip size={10} /> {src}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </motion.div>
          );
        })}

        {loadingSessionId !== null && loadingSessionId === currentSessionId && (
          <ThinkingBubble content={messages.find((m) => m.role === "AI" && m.isTyping)?.content || ""} />
        )}

        <div ref={chatEndRef} />
      </div>

      <div className="p-8 pt-2 z-20">
        <div className="relative max-w-4xl mx-auto">
          <div className="absolute -inset-1 bg-gradient-to-r from-fuchsia-500 via-violet-500 to-cyan-500 rounded-full opacity-30 blur-md group-focus-within:opacity-60 transition-opacity duration-500" />

          <AnimatePresence>
            {chatImages.length > 0 && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 10 }}
                className="absolute bottom-full mb-3 left-0 flex gap-2"
              >
                {chatImages.map((img, index) => (
                  <div key={img.id} className="relative group">
                    <img
                      src={img.url}
                      alt="preview"
                      className="h-16 w-16 object-cover rounded-xl border-2 border-cyan-400 shadow-lg"
                    />
                    <button
                      onClick={() => removeChatImage(index)}
                      className="absolute -top-1 -right-1 bg-red-500 text-white rounded-full p-0.5 shadow-md hover:bg-red-600 transition-colors"
                    >
                      <X size={10} />
                    </button>
                  </div>
                ))}
              </motion.div>
            )}
          </AnimatePresence>

          <div className="relative flex items-center gap-3 bg-white/90 backdrop-blur-2xl rounded-3xl p-2 pl-4 shadow-[0_10px_30px_-5px_rgba(6,182,212,0.2)] border border-white">
            <input
              type="file"
              ref={chatImageInputRef}
              className="hidden"
              onChange={handleChatImageSelect}
              multiple
              accept="image/jpeg,image/png,image/webp,image/gif"
            />

            <motion.button
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              onClick={() => chatImageInputRef.current.click()}
              className="p-2 text-slate-400 hover:text-cyan-500 hover:bg-cyan-500/10 rounded-full transition-colors self-end mb-1"
              title="Upload image"
            >
              <ImageIcon size={22} />
            </motion.button>

            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleSendMessage}
              placeholder="Type your question..."
              rows={1}
              className="flex-1 bg-transparent text-slate-700 text-base focus:outline-none placeholder-slate-400 font-medium resize-none py-3 max-h-[120px]"
              disabled={false}
            />

            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() =>
                loadingSessionId !== null && loadingSessionId === currentSessionId
                  ? handleStop()
                  : handleSendMessage()
              }
              disabled={!input.trim() && chatImages.length === 0 && loadingSessionId !== null && loadingSessionId !== currentSessionId}
              className={`p-3 rounded-full text-white shadow-lg transition-all self-end ${
                loadingSessionId !== null && loadingSessionId === currentSessionId
                  ? "bg-gradient-to-r from-red-500 to-orange-500 hover:shadow-red-500/30 cursor-pointer"
                  : "bg-gradient-to-r from-fuchsia-600 to-cyan-600 hover:shadow-cyan-500/30 disabled:opacity-50 disabled:cursor-not-allowed"
              }`}
            >
              {loadingSessionId !== null && loadingSessionId === currentSessionId ? (
                <Square size={20} className="fill-current animate-pulse" />
              ) : (
                <Send size={20} />
              )}
            </motion.button>
          </div>
        </div>
        <p className="text-center text-[10px] text-slate-400 mt-3 font-medium opacity-60">
          Secure connection established | v9.0
        </p>
      </div>
    </div>
  );
};

export default ChatWorkspace;
