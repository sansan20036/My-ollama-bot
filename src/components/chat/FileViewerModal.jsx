import { AnimatePresence, motion } from "framer-motion";
import { FileText, X } from "lucide-react";

const FileViewerModal = ({ viewingFile, viewContent, onClose }) => {
  return (
    <AnimatePresence>
      {viewingFile && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4"
          onClick={onClose}
        >
          <motion.div
            initial={{ scale: 0.9, y: 20 }}
            animate={{ scale: 1, y: 0 }}
            exit={{ scale: 0.9, y: 20 }}
            className="bg-slate-900 border border-cyan-500/50 w-full max-w-3xl max-h-[80vh] rounded-2xl shadow-[0_0_50px_rgba(6,182,212,0.2)] flex flex-col overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between p-4 border-b border-white/10 bg-slate-800/50">
              <div className="flex items-center gap-2 text-cyan-400">
                <FileText size={20} />
                <h3 className="font-bold text-lg truncate max-w-md">{viewingFile}</h3>
              </div>
              <button
                onClick={onClose}
                className="p-2 hover:bg-white/10 rounded-full transition-colors text-slate-400 hover:text-white"
              >
                <X size={20} />
              </button>
            </div>

            <div className="flex-1 overflow-y-auto p-6 custom-scrollbar bg-[#0f0c29]">
              <pre className="whitespace-pre-wrap font-mono text-sm text-slate-300 leading-relaxed">
                {viewContent}
              </pre>
            </div>

            <div className="p-3 border-t border-white/10 bg-slate-800/50 text-right text-xs text-slate-500">
              File Content Viewer v1.0
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default FileViewerModal;
