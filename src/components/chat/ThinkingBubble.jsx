import { useEffect, useState } from "react";
import { Bot, Loader2, Zap } from "lucide-react";

const GhostTypewriter = ({ content }) => {
  return (
    <div className="mt-3 p-3 bg-black/80 rounded-lg border-l-2 border-cyan-400 font-mono relative overflow-hidden animate-fade-in backdrop-blur-sm shadow-inner">
      <div className="absolute inset-0 pointer-events-none bg-gradient-to-b from-transparent via-cyan-400/10 to-transparent h-full w-full animate-scanline-fast" />

      <div className="flex items-start gap-2 relative z-10">
        <span className="text-cyan-400 font-bold animate-pulse text-xs mt-0.5">[{">"}]</span>
        <p className="text-[12px] leading-relaxed text-cyan-50/90 break-all whitespace-pre-wrap font-mono">
          {content}
          <span className="inline-block w-2 h-4 bg-cyan-400 ml-1 animate-blink shadow-[0_0_8px_#22d3ee] align-middle" />
        </p>
      </div>
    </div>
  );
};

const ThinkingBubble = ({ content }) => {
  const [timer, setTimer] = useState(0.0);

  useEffect(() => {
    const startTime = Date.now();

    const interval = setInterval(() => {
      const elapsedMilliseconds = Date.now() - startTime;
      setTimer((elapsedMilliseconds / 1000).toFixed(1));
    }, 100);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex gap-4 mb-6 animate-fade-in pl-2 max-w-[90%]">
      <div className="w-10 h-10 rounded-2xl bg-white text-cyan-600 shadow-slate-200/50 flex items-center justify-center flex-shrink-0">
        <Bot size={22} />
      </div>

      <div className="flex-1">
        <div className="bg-white/95 backdrop-blur-xl rounded-2xl rounded-tl-none p-5 shadow-2xl border border-white/50 relative overflow-hidden group">
          <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-fuchsia-500 via-cyan-500 to-transparent opacity-50" />

          <div className="flex flex-col gap-4">
            <div className="flex items-center gap-3 text-xs font-bold tracking-widest text-fuchsia-600">
              <Loader2 size={14} className="animate-spin" />
              <span>Establishing secure channel...</span>
              <span className="ml-auto font-mono text-slate-400 flex items-center gap-1">
                <Zap size={10} className="text-yellow-500 fill-yellow-500" />
                {timer}s
              </span>
            </div>

            <div className="h-px w-full bg-slate-200 relative overflow-hidden">
              <div className="absolute top-0 left-0 h-full w-1/3 bg-cyan-400/50 blur-[2px] animate-shimmer" />
            </div>

            <div className="flex flex-col gap-2">
              <div className="flex items-center gap-2 text-xs font-bold tracking-widest text-cyan-600 animate-pulse">
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-500" />
                Decoding stream...
              </div>

              {content && (
                <div className="mt-2">
                  <GhostTypewriter content={content} />
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ThinkingBubble;
