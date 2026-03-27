const CyberpunkNeonBackground = () => (
  <div className="absolute inset-0 z-0 overflow-hidden bg-[#0f0c29]">
    <div className="absolute inset-0 bg-gradient-to-b from-[#0f0c29] via-[#302b63] to-[#24243e] opacity-80" />
    <div className="absolute top-[-10%] left-[-10%] w-[60vw] h-[60vw] bg-fuchsia-600/30 rounded-full blur-[120px] mix-blend-screen animate-pulse-slow" />
    <div className="absolute top-[10%] right-[-10%] w-[50vw] h-[50vw] bg-cyan-500/30 rounded-full blur-[120px] mix-blend-screen animate-pulse-slow animation-delay-2000" />
    <div
      className="absolute bottom-0 left-[-50%] right-[-50%] h-[50vh] perspective-grid-container"
      style={{ transform: "perspective(500px) rotateX(60deg)" }}
    >
      <div className="absolute inset-0 bg-[linear-gradient(rgba(255,0,255,0.3)_2px,transparent_2px),linear-gradient(90deg,rgba(0,255,255,0.3)_2px,transparent_2px)] bg-[size:60px_60px] animate-grid-move shadow-[0_0_20px_rgba(255,0,255,0.5)]" />
      <div className="absolute top-0 left-0 right-0 h-[100px] bg-gradient-to-b from-cyan-400/50 to-transparent blur-xl" />
    </div>
    <div className="absolute inset-0 pointer-events-none">
      <div className="absolute top-[20%] left-0 w-full h-[2px] bg-gradient-to-r from-transparent via-cyan-400 to-transparent opacity-50 animate-scanline" />
    </div>
    <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMDAlIiBoZWlnaHQ9IjEwMCUiPjxmaWx0ZXIgaWQ9Im4iPjxmZVR1cmJ1bGVuY2UgdHlwZT0iZnJhY3RhbE5vaXNlIiBiYXNlRnJlcXVlbmN5PSIwLjUiIG51bU9jdGF2ZXM9IjMiIHN0aXRjaFRpbGVzPSJzdGl0Y2giLz48L2ZpbHRlcj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSJ0cmFuc3BhcmVudCIvPjxyZWN0IHdpZHRoPSIxMDAlIiBoZWlnaHQ9IjEwMCUiIGZpbGw9IiNmZmZmZmYiIG9wYWNpdHk9IjAuNSIgZmlsdGVyPSJ1cmwoI24pIi8+PC9zdmc+')] opacity-10 mix-blend-overlay" />
  </div>
);

export default CyberpunkNeonBackground;
