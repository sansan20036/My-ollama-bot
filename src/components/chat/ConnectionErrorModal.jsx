import { motion } from "framer-motion";
import { WifiOff } from "lucide-react";

const ConnectionErrorModal = ({ message, onClose }) => (
  <motion.div
    initial={{ opacity: 0 }}
    animate={{ opacity: 1 }}
    exit={{ opacity: 0 }}
    className="absolute inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-md p-6"
  >
    <motion.div
      initial={{ scale: 0.9, y: 20 }}
      animate={{ scale: 1, y: 0 }}
      exit={{ scale: 0.9, y: 20 }}
      className="bg-slate-900 rounded-[32px] shadow-2xl p-8 max-w-sm w-full border border-red-500/50 text-center relative overflow-hidden"
    >
      <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-red-500 to-orange-500" />
      <div className="w-20 h-20 bg-red-900/30 rounded-full flex items-center justify-center mx-auto mb-6 text-red-500 shadow-[0_0_20px_rgba(239,68,68,0.2)]">
        <WifiOff size={32} />
      </div>
      <h3 className="text-xl font-black text-white mb-2">System Offline</h3>
      <p className="text-sm text-slate-400 mb-8 leading-relaxed font-mono">
        Unable to connect to Chroma AI.
        <br />
        <span className="text-xs opacity-50 text-red-400">
          Error: {message}
        </span>
      </p>
      <button
        onClick={onClose}
        className="w-full py-3 rounded-2xl bg-gradient-to-r from-red-600 to-orange-600 text-white font-bold hover:shadow-[0_0_20px_rgba(239,68,68,0.4)] transition-all active:scale-95"
      >
        Retry Connection
      </button>
    </motion.div>
  </motion.div>
);

export default ConnectionErrorModal;
