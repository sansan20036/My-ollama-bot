import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { motion, AnimatePresence } from "framer-motion";
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/cjs/styles/prism';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
import { DndContext, closestCenter, KeyboardSensor, PointerSensor, useSensor, useSensors } from '@dnd-kit/core';
import { arrayMove, SortableContext, sortableKeyboardCoordinates, horizontalListSortingStrategy, useSortable } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';

// 圖表套件
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

import {
  Send, Upload, Trash2, Bot, User, Loader2, Paperclip, X, Sparkles,
  Clock, WifiOff, Plus, Archive, MessageCircle, Square, ExternalLink, BarChart2, Zap,
  Image as ImageIcon, // 圖片圖示

  // 新增下面這三個：
  FileText,   // 用於顯示文件圖示
  Eye,        // 用於檢視按鈕 (眼睛)
  RefreshCw   // 用於重新整理列表 (旋轉箭頭)

} from "lucide-react";

const API_CONFIG = {
  BASE_URL: "http://127.0.0.1:8000/api/models",
  TYPING_SPEED: 50,      // 打字速度 (毫秒)
  CHUNK_SIZE: 1,         // 每次吐幾個字
  TIMEOUT: 500,          // 停止打字後的延遲
};


// UI 元件區域 
const CyberpunkNeonBackground = () => (
  <div className="absolute inset-0 z-0 overflow-hidden bg-[#0f0c29]">
    <div className="absolute inset-0 bg-gradient-to-b from-[#0f0c29] via-[#302b63] to-[#24243e] opacity-80" />
    <div className="absolute top-[-10%] left-[-10%] w-[60vw] h-[60vw] bg-fuchsia-600/30 rounded-full blur-[120px] mix-blend-screen animate-pulse-slow" />
    <div className="absolute top-[10%] right-[-10%] w-[50vw] h-[50vw] bg-cyan-500/30 rounded-full blur-[120px] mix-blend-screen animate-pulse-slow animation-delay-2000" />
    <div className="absolute bottom-0 left-[-50%] right-[-50%] h-[50vh] perspective-grid-container"
      style={{ transform: 'perspective(500px) rotateX(60deg)' }}>
      <div className="absolute inset-0 bg-[linear-gradient(rgba(255,0,255,0.3)_2px,transparent_2px),linear-gradient(90deg,rgba(0,255,255,0.3)_2px,transparent_2px)] bg-[size:60px_60px] animate-grid-move shadow-[0_0_20px_rgba(255,0,255,0.5)]" />
      <div className="absolute top-0 left-0 right-0 h-[100px] bg-gradient-to-b from-cyan-400/50 to-transparent blur-xl" />
    </div>
    <div className="absolute inset-0 pointer-events-none">
      <div className="absolute top-[20%] left-0 w-full h-[2px] bg-gradient-to-r from-transparent via-cyan-400 to-transparent opacity-50 animate-scanline" />
    </div>
    <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMDAlIiBoZWlnaHQ9IjEwMCUiPjxmaWx0ZXIgaWQ9Im4iPjxmZVR1cmJ1bGVuY2UgdHlwZT0iZnJhY3RhbE5vaXNlIiBiYXNlRnJlcXVlbmN5PSIwLjUiIG51bU9jdGF2ZXM9IjMiIHN0aXRjaFRpbGVzPSJzdGl0Y2giLz48L2ZpbHRlcj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSJ0cmFuc3BhcmVudCIvPjxyZWN0IHdpZHRoPSIxMDAlIiBoZWlnaHQ9IjEwMCUiIGZpbGw9IiNmZmZmZmYiIG9wYWNpdHk9IjAuNSIgZmlsdGVyPSJ1cmwoI24pIi8+PC9zdmc+')] opacity-10 mix-blend-overlay" />
  </div>
);

// GhostTypewriter (駭客解碼器 - 內嵌樣式版)
const GhostTypewriter = ({ content }) => {
  return (
    <div className="mt-3 p-3 bg-black/80 rounded-lg border-l-2 border-cyan-400 font-mono relative overflow-hidden animate-fade-in backdrop-blur-sm shadow-inner">
      {/* 快速掃描線 */}
      <div className="absolute inset-0 pointer-events-none bg-gradient-to-b from-transparent via-cyan-400/10 to-transparent h-full w-full animate-scanline-fast" />

      <div className="flex items-start gap-2 relative z-10">
        <span className="text-cyan-400 font-bold animate-pulse text-xs mt-0.5">[{'>'}]</span>
        <p className="text-[12px] leading-relaxed text-cyan-50/90 break-all whitespace-pre-wrap font-mono">
          {content}
          <span className="inline-block w-2 h-4 bg-cyan-400 ml-1 animate-blink shadow-[0_0_8px_#22d3ee] align-middle" />
        </p>
      </div>
    </div>
  );
};

// 可拖曳的表頭元件 (處理拖曳動畫與樣式)
const SortableHeader = ({ id, children }) => {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({ id });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    cursor: isDragging ? 'grabbing' : 'grab',
    backgroundColor: isDragging ? 'rgba(6, 182, 212, 0.1)' : undefined,
    opacity: isDragging ? 0.3 : 1,
    border: isDragging ? '1px dashed #22d3ee' : undefined,
    zIndex: isDragging ? 999 : 'auto',
  };

  return (
    <th
      ref={setNodeRef}
      style={style}
      {...attributes}
      {...listeners}
      className="px-6 py-4 font-semibold select-none relative hover:bg-white/5 transition-colors group whitespace-nowrap"
    >
      {children}
      <span className="absolute right-2 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-50 text-[10px] text-cyan-400">⋮⋮</span>
    </th>
  );
};

// 思考泡泡 (整合了解碼文字流)
const ThinkingBubble = ({ content }) => {
  const [timer, setTimer] = useState(0.0);

  useEffect(() => {
    // 1. 記下開始計時的「真實時間」(毫秒)
    const startTime = Date.now();

    const interval = setInterval(() => {
      // 2. 用「現在的真實時間」減去「開始時間」，算出真正經過的毫秒數
      const elapsedMilliseconds = Date.now() - startTime;

      // 3. 換算成秒數，並保留小數點後 1 位 (例如 1.2)
      setTimer((elapsedMilliseconds / 1000).toFixed(1));
    }, 100);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex gap-4 mb-6 animate-fade-in pl-2 max-w-[90%]">
      {/* AI 頭像 */}
      <div className="w-10 h-10 rounded-2xl bg-white text-cyan-600 shadow-slate-200/50 flex items-center justify-center flex-shrink-0">
        <Bot size={22} />
      </div>

      <div className="flex-1">
        <div className="bg-white/95 backdrop-blur-xl rounded-2xl rounded-tl-none p-5 shadow-2xl border border-white/50 relative overflow-hidden group">
          {/* 頂部裝飾條 */}
          <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-fuchsia-500 via-cyan-500 to-transparent opacity-50" />

          <div className="flex flex-col gap-4">
            {/*  正常回覆狀態：連線中... */}
            <div className="flex items-center gap-3 text-xs font-bold tracking-widest text-fuchsia-600">
              <Loader2 size={14} className="animate-spin" />
              <span>ESTABLISHING SECURE CONNECTION...</span>
              <span className="ml-auto font-mono text-slate-400 flex items-center gap-1">
                <Zap size={10} className="text-yellow-500 fill-yellow-500" />
                {timer}s
              </span>
            </div>

            {/* 分隔線 */}
            <div className="h-px w-full bg-slate-200 relative overflow-hidden">
              <div className="absolute top-0 left-0 h-full w-1/3 bg-cyan-400/50 blur-[2px] animate-shimmer" />
            </div>

            <div className="flex flex-col gap-2">
              {/* 狀態文字 */}
              <div className="flex items-center gap-2 text-xs font-bold tracking-widest text-cyan-600 animate-pulse">
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-500" />
                DECODING STREAM...
              </div>

              {/* 特效需求：駭客解碼文字流 (放在下方) */}
              {/* 只有當有內容時才顯示這個帥氣的黑色解碼框 */}
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

// 圖表渲染元件
const ChartRenderer = ({ data, type, title }) => {
  const COLORS = ['#06b6d4', '#d946ef', '#8b5cf6', '#f59e0b', '#10b981'];

  return (
    <div className="my-6 p-4 bg-slate-900/90 border border-slate-700/50 rounded-xl shadow-lg backdrop-blur-md">
      {title && (
        <div className="flex items-center gap-2 mb-4 border-b border-white/10 pb-2">
          <BarChart2 size={16} className="text-cyan-400" />
          <span className="text-xs font-bold text-cyan-300 uppercase tracking-widest">{title}</span>
        </div>
      )}
      <div className="h-[250px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          {type === 'pie' ? (
            <PieChart>
              <Pie
                data={data}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={80}
                fill="#8884d8"
                paddingAngle={5}
                dataKey="value"
                label
              >
                {data.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} stroke="rgba(0,0,0,0.5)" />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', color: '#fff' }}
                itemStyle={{ color: '#fff' }}
              />
              <Legend />
            </PieChart>
          ) : (
            <BarChart data={data}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
              <XAxis dataKey="name" stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} />
              <YAxis stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} />
              <Tooltip
                cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', color: '#fff' }}
              />
              <Bar dataKey="value" fill="url(#colorGradient)" radius={[4, 4, 0, 0]}>
                {data.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          )}
        </ResponsiveContainer>
      </div>
    </div>
  );
};

// 新增：支援 D&D 排序的表格容器
// 完整覆蓋 DraggableTable 元件
const DraggableTable = ({ children }) => {
  const childrenArray = React.Children.toArray(children);
  const thead = childrenArray.find(c => c.type === 'thead');
  const tbody = childrenArray.find(c => c.type === 'tbody');

  const extractText = (node) => {
    if (typeof node === 'string') return node;
    if (Array.isArray(node)) return node.map(extractText).join('');
    if (node && node.props && node.props.children) return extractText(node.props.children);
    return 'Column';
  };

  const initialHeaders = React.Children.map(thead?.props?.children?.props?.children, child => {
    return extractText(child);
  }) || [];

  const [columns, setColumns] = useState(() => {
    const saved = localStorage.getItem('tableColumnOrder');
    if (saved) {
      const savedCols = JSON.parse(saved);
      if (savedCols.length === initialHeaders.length && savedCols.every(c => initialHeaders.includes(c))) {
        return savedCols;
      }
    }
    return initialHeaders;
  });

  useEffect(() => {
    if (initialHeaders.length > 0 && JSON.stringify(initialHeaders) !== JSON.stringify(columns)) {
      setColumns(initialHeaders);
    }
  }, [thead]);

  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 8 } }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates })
  );

  const handleDragEnd = (event) => {
    const { active, over } = event;
    if (active.id !== over.id) {
      setColumns((items) => {
        const oldIndex = items.indexOf(active.id);
        const newIndex = items.indexOf(over.id);
        const newOrder = arrayMove(items, oldIndex, newIndex);
        localStorage.setItem('tableColumnOrder', JSON.stringify(newOrder));
        return newOrder;
      });
    }
  };

  const originalHeaderIndexMap = initialHeaders.reduce((acc, col, idx) => ({ ...acc, [col]: idx }), {});

  return (
    <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={handleDragEnd}>
      {/* 關鍵修改 1：容器層級限制
         1. display: grid -> 這是防止 flex item 被子元素撐開的絕招。
         2. w-full max-w-full -> 強制寬度不超過父層 (對話氣泡)。
         3. my-6 -> 上下邊距。
      */}
      <div className="grid w-full max-w-full my-6">

        {/* 關鍵修改 2：滾動視窗層
           1. overflow-auto -> 同時開啟 X 軸與 Y 軸滾動。
           2. max-h-[500px] -> 限制高度，超過出 Y 軸卷軸。
           3. w-full -> 繼承 grid 的寬度。
           4. rounded/border/shadow -> 樣式美化。
        */}
        <div className="w-full overflow-auto max-h-[500px] rounded-xl border border-slate-700/50 shadow-lg bg-slate-900/90 custom-scrollbar">

          {/* 關鍵修改 3：表格實體
             1. w-max -> 這是核心！"Width Max Content"。
                它告訴表格：「你的寬度 = 所有欄位加起來的總寬度」。
                因為外層 div 限制了寬度，所以當 table > div 時，卷軸就會出現。
          */}
          <table className="w-max min-w-full text-left text-sm border-separate border-spacing-0">

            {/* Sticky Header: 固定在容器頂部 */}
            <thead className="sticky top-0 z-20 bg-slate-900 text-cyan-300 font-bold uppercase tracking-wider text-xs shadow-md">
              <SortableContext items={columns} strategy={horizontalListSortingStrategy}>
                <tr>
                  {columns.map((col) => (
                    <SortableHeader key={col} id={col}>
                      {/* 強制不換行，撐開寬度 */}
                      <span className="whitespace-nowrap px-2">{col}</span>
                    </SortableHeader>
                  ))}
                </tr>
              </SortableContext>
            </thead>

            <tbody className="text-slate-300 divide-y divide-white/5">
              {React.Children.map(tbody?.props?.children, (row) => {
                const cells = React.Children.toArray(row.props.children);
                return (
                  <tr className="hover:bg-white/5 transition-colors duration-200">
                    {columns.map((col, newIndex) => {
                      const originalIndex = originalHeaderIndexMap[col];
                      return cells[originalIndex] ? (
                        // 強制每個儲存格都不換行
                        React.cloneElement(cells[originalIndex], {
                          className: "px-6 py-4 whitespace-nowrap"
                        })
                      ) : (
                        <td key={newIndex} className="px-6 py-4 whitespace-nowrap text-slate-500 italic">-</td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </DndContext>
  );
};

const MarkdownRenderer = ({ content }) => {
  const copyToClipboard = (code) => {
    navigator.clipboard.writeText(code);
  };

  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm, remarkMath]}
      rehypePlugins={[rehypeKatex]}
      components={{

        // 1. 程式碼區塊 & 圖表渲染 (Chart & Code)
        code({ node, inline, className, children, ...props }) {
          const match = /language-(\w+)/.exec(className || '');
          const codeString = String(children).replace(/\n$/, '');
          const safeStyle = vscDarkPlus || {};

          // 處理 JSON 圖表
          if (!inline && match && match[1] === 'json-chart') {
            try {
              if (!codeString || codeString.trim().length === 0) return null;
              const chartData = JSON.parse(codeString);
              return <ChartRenderer data={chartData.data} type={chartData.type} title={chartData.title} />;
            } catch (e) {
              return (
                <div className="text-red-400 text-xs bg-red-900/20 p-2 rounded border border-red-500/30 font-mono">
                  Chart Rendering Error: Invalid JSON Format
                </div>
              );
            }
          }

          // 一般程式碼區塊
          return !inline && match ? (
            <div className="rounded-xl overflow-hidden my-4 shadow-lg border border-white/20 bg-slate-900 group relative z-10">
              <div className="bg-slate-800/50 px-4 py-2 text-[10px] text-slate-400 flex justify-between items-center border-b border-white/5">
                <span className="font-mono uppercase tracking-widest">{match[1]}</span>
                <button
                  onClick={() => copyToClipboard(codeString)}
                  className="hover:text-white transition-colors flex items-center gap-1 active:scale-95"
                >
                  <Paperclip size={10} /> Copy
                </button>
              </div>
              <SyntaxHighlighter
                style={safeStyle}
                language={match[1]}
                PreTag="div"
                customStyle={{ margin: 0, padding: '1.5rem', background: 'transparent', fontSize: '13px', lineHeight: '1.6' }}
                {...props}
              >
                {codeString}
              </SyntaxHighlighter>
            </div>
          ) : (
            <code className="bg-pink-100/50 text-pink-600 px-1.5 py-0.5 rounded font-mono text-[0.9em] font-bold" {...props}>
              {children}
            </code>
          );
        },

        // 2. 表格 (Draggable Table)
        table: DraggableTable,
        td: ({ children }) => <td className="px-6 py-4 whitespace-nowrap">{children}</td>,

        // 3. 段落 (P) - 這裡加入了「出處高亮」功能
        p: ({ children }) => {
          // 1. 判斷是否為「純文字」或「純文字陣列」
          const isPureString = typeof children === 'string';
          const isStringArray = Array.isArray(children) && children.every(c => typeof c === 'string');

          // 2. 如果包含非文字的元素（例如 LaTeX 公式物件、圖片、粗體等），直接回傳原始內容，不做處理
          if (!isPureString && !isStringArray) {
            return <p className="mb-4 last:mb-0 leading-7">{children}</p>;
          }

          // 3. 只有確認是純文字，才執行原本的「出處高亮」邏輯
          const text = Array.isArray(children) ? children.join('') : String(children);
          const parts = text.split(/(\(出處:.*?\))/g);

          return (
            <p className="mb-4 last:mb-0 leading-7">
              {parts.map((part, index) => {
                if (part.startsWith('(出處:') && part.endsWith(')')) {
                  const content = part.replace(/[()]/g, '');
                  return (
                    <span key={index} className="inline-flex items-center gap-1 mx-1 text-cyan-400 text-xs font-bold tracking-wide select-none hover:text-cyan-300 transition-colors cursor-help hover:underline underline-offset-2">
                      <Paperclip size={8} />
                      {content}
                    </span>
                  );
                }
                return part;
              })}
            </p>
          );
        },
        // 4. 其他基本標籤樣式
        a: ({ children, href }) => (
          <a href={href} target="_blank" rel="noopener noreferrer" className="text-cyan-400 hover:text-cyan-300 hover:underline underline-offset-4 decoration-dashed inline-flex items-center gap-1 transition-colors">
            {children} <ExternalLink size={10} />
          </a>
        ),
        h1: ({ children }) => <h1 className="text-2xl font-black mb-6 mt-4 text-transparent bg-clip-text bg-gradient-to-r from-slate-700 to-slate-900 border-b border-slate-200/50 pb-2">{children}</h1>,
        h2: ({ children }) => <h2 className="text-xl font-bold mb-4 mt-6 text-slate-800 flex items-center gap-2"><span className="w-1 h-6 bg-cyan-500 rounded-full inline-block" />{children}</h2>,
        h3: ({ children }) => <h3 className="text-lg font-bold mb-3 mt-4 text-slate-700">{children}</h3>,
        ul: ({ children }) => <ul className="list-disc pl-5 space-y-2 mb-4 marker:text-cyan-500">{children}</ul>,
        ol: ({ children }) => <ol className="list-decimal pl-5 space-y-2 mb-4 marker:text-fuchsia-500 font-bold">{children}</ol>,
        li: ({ children }) => <li className="pl-1 font-normal">{children}</li>,
        blockquote: ({ children }) => (
          <blockquote className="border-l-4 border-fuchsia-400 bg-fuchsia-50/50 pl-4 py-2 my-4 rounded-r italic text-slate-600">
            {children}
          </blockquote>
        ),
      }}
    >
      {content
        .replace(/\\\[/g, '$$')  // 把 \[ 換成 $$ (區塊公式)
        .replace(/\\\]/g, '$$')
        .replace(/\\\(/g, '$')   // 把 \( 換成 $ (行內公式)
        .replace(/\\\)/g, '$')
      }
    </ReactMarkdown>
  );
};

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
      <h3 className="text-xl font-black text-white mb-2">SYSTEM OFFLINE</h3>
      <p className="text-sm text-slate-400 mb-8 leading-relaxed font-mono">
        Connection to Chroma AI failed.<br />
        <span className="text-xs opacity-50 text-red-400">Error: {message}</span>
      </p>
      <button
        onClick={onClose}
        className="w-full py-3 rounded-2xl bg-gradient-to-r from-red-600 to-orange-600 text-white font-bold hover:shadow-[0_0_20px_rgba(239,68,68,0.4)] transition-all active:scale-95"
      >
        RETRY CONNECTION
      </button>
    </motion.div>
  </motion.div>
);

function App() {
  const API_BASE = "http://127.0.0.1:8000/api";
  const defaultMessage = {
    role: "AI",
    content: "💠 **SYSTEM ONLINE.**\n\nThere!我是 Chroma AI，請上傳資料以開始駭入分析。💾"
  };

  // 加上這個共用的初始 ID
  const initId = Date.now();

  const [sessions, setSessions] = useState(() => {
    const saved = localStorage.getItem("chatSessions");
    return saved ? JSON.parse(saved) : [{ id: initId, title: "New Session", messages: [defaultMessage], createdAt: initId }];
  });

  const [currentSessionId, setCurrentSessionId] = useState(() => {
    const savedSessions = JSON.parse(localStorage.getItem("chatSessions") || "[]");
    // 如果找不到紀錄，就用上面宣告的 initId，不要回傳 null！
    return savedSessions.length > 0 ? savedSessions[0].id : initId;
  });

  const [messages, setMessages] = useState(() => {
    const savedSessions = JSON.parse(localStorage.getItem("chatSessions") || "[]");
    return savedSessions.length > 0 ? savedSessions[0].messages : [defaultMessage];
  });

  const [input, setInput] = useState("");
  const [loadingSessionId, setLoadingSessionId] = useState(null);
  const currentSessionIdRef = useRef(currentSessionId);
  const [uploadStatus, setUploadStatus] = useState("");
  const [availableModels, setAvailableModels] = useState(["gemma3:27b"]);
  const [selectedModel, setSelectedModel] = useState("gemma3:27b");
  const [filesToUpload, setFilesToUpload] = useState([]);
  const [errorModal, setErrorModal] = useState({ show: false, message: "" });

  const [fileList, setFileList] = useState([]);          // 儲存從後端抓來的檔案清單
  const [viewingFile, setViewingFile] = useState(null);  // 目前正在檢視哪個檔案 (null 代表沒開)
  const [viewContent, setViewContent] = useState("");    // 該檔案的文字內容
  const [loadingFiles, setLoadingFiles] = useState(false); // 是否正在讀取列表

  // 檔案管理邏輯 (File Management Logic)

  // 1. 抓取檔案列表
  const fetchFileList = async () => {
    setLoadingFiles(true);
    try {
      // 呼叫後端 GET /files
      const response = await axios.get(`${API_BASE}/files`);
      setFileList(response.data.files);
    } catch (error) {
      console.error("無法取得檔案列表", error);
    } finally {
      setLoadingFiles(false);
    }
  };

  // 2. 刪除檔案
  const handleDeleteFile = async (e, filename) => {
    e.stopPropagation(); // 防止誤觸其他點擊事件
    if (!window.confirm(`確定要永久刪除 "${filename}" 嗎？`)) return;

    try {
      await axios.delete(`${API_BASE}/files/${encodeURIComponent(filename)}`);

      // 刪除成功後，重新抓取列表以更新畫面
      await fetchFileList();

      // (選擇性) 讓 AI 在對話框通知使用者
      setMessages(prev => [...prev, {
        role: "AI",
        content: `🗑️ **FILE DELETED**\n\n已移除檔案核心：\`${filename}\``
      }]);
    } catch (error) {
      alert("刪除失敗，請檢查後端連線");
      console.error(error);
    }
  };

  // 3. 檢視檔案 (直接開新分頁)
  const handleViewFile = (filename) => {
    // 透過 encodeURIComponent 處理中文檔名
    const fileUrl = `${API_BASE}/files/${encodeURIComponent(filename)}/view`;

    //  在新分頁打開該網址
    window.open(fileUrl, '_blank');
  };

  // 4. 初始化：畫面載入時自動抓一次列表
  useEffect(() => {
    fetchFileList();
  }, []);

  useEffect(() => {
    currentSessionIdRef.current = currentSessionId;
  }, [currentSessionId]);

  // 新增：聊天圖片相關狀態
  const chatImageInputRef = useRef(null);
  const [chatImages, setChatImages] = useState([]);

  const fileInputRef = useRef(null);
  const chatEndRef = useRef(null);
  const textareaRef = useRef(null);
  const abortControllerRef = useRef(null);
  const typingIntervalRef = useRef(null);

  //  狀態同步守護神：精準區分「F5 重新整理」與「後端重啟」
  useEffect(() => {
    const checkBackendStatus = async () => {
      try {
        // 1. 去問後端現在的「開機身分證」
        const response = await axios.get(`${API_BASE}/status`);
        const currentBootTime = response.data.boot_time;

        // 2. 拿出前端上次記下來的「開機身分證」
        const savedBootTime = localStorage.getItem("backendBootTime");

        // 狀況 A：第一次開啟網頁，先記下目前的後端身分證，並保留預設畫面
        if (!savedBootTime) {
          localStorage.setItem("backendBootTime", currentBootTime);
          return;
        }

        // 狀況 B：核彈觸發！發現後端重啟了 (身分證號碼不一樣)
        if (savedBootTime !== currentBootTime) {
          console.log("⚠️ 偵測到後端已重新啟動！前端同步清空舊資料...");

          // 1. 清除瀏覽器記憶
          localStorage.removeItem("chatSessions");

          //  2. 建立全新的乾淨對話
          const initSession = {
            id: Date.now(),
            title: "New Session",
            messages: [{ role: "AI", content: "♻️ **SYSTEM REBOOTED**\n偵測到伺服器已重啟，請重新上傳檔案。" }],
            createdAt: Date.now()
          };

          // 3. 強制洗掉畫面上的所有狀態
          setSessions([initSession]);
          setCurrentSessionId(initSession.id);
          setMessages(initSession.messages);
          setFileList([]); // 左側檔案清單歸零

          // 4. 更新身分證，避免重複觸發
          localStorage.setItem("backendBootTime", currentBootTime);
        }

        // 狀況 C：身分證一樣 (代表只是單純按 F5 重新整理)
        // 什麼都不用做！React 的 useState 會自動從 localStorage 把紀錄好好地讀出來！

      } catch (error) {
        // 伺服器關閉中，連不到是很正常的，略過不處理
      }
    };

    // 網頁剛載入 (或按 F5) 時，立刻執行第一次檢查
    checkBackendStatus();

    // 啟動背景雷達：每 5 秒自動檢查一次 (處理網頁放著不動，後端突然重啟的狀況)
    const radarInterval = setInterval(checkBackendStatus, 5000);

    // 當元件卸載時，關閉雷達
    return () => clearInterval(radarInterval);
  }, []);

  // 自動抓取模型清單 (修正版：自動過濾 embedding 模型)
  useEffect(() => {
    const fetchModels = async () => {
      try {
        // 1. 呼叫 API
        const response = await axios.get(`${API_BASE}/models`);

        // 2. 取得原始資料
        const rawModels = response.data.models;

        // 3. 關鍵修正：過濾掉名字包含 "embed" 的模型
        const modelNames = rawModels
          .map(model => model.name)
          .filter(name => !name.toLowerCase().includes('embed')); // 👈 加上這行過濾器

        if (modelNames && modelNames.length > 0) {
          setAvailableModels(modelNames);

          // 防呆：如果當前選中的模型是被過濾掉的 (例如 nomic-embed)，自動切換回正常的第一個模型
          if (!modelNames.includes(selectedModel)) {
            setSelectedModel(modelNames[0]);
          }
        }
      } catch (error) {
        console.error("無法獲取模型清單，使用預設值", error);
      }
    };
    fetchModels();
  }, []); // 空陣列代表只在掛載時執行一次

  //  修正後的自動捲動 (依賴 messages 和 loadingSessionId)
  useEffect(() => {
    requestAnimationFrame(() => {
      if (!chatEndRef.current) return;
      chatEndRef.current.scrollIntoView({ behavior: "smooth", block: "end" });
    });
  }, [messages, loadingSessionId]);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 120)}px`;
    }
  }, [input]);

  // Session sync logic
  useEffect(() => {
    if (!currentSessionId) return;

    setSessions(prevSessions => {
      const updatedSessions = prevSessions.map(session => {
        if (session.id === currentSessionId) {
          let newTitle = session.title;
          const firstUserMsg = messages.find(m => m.role === "User");
          if (firstUserMsg && session.title === "New Session") {
            newTitle = firstUserMsg.content.slice(0, 15) + (firstUserMsg.content.length > 15 ? "..." : "");
          }
          return { ...session, messages: messages, title: newTitle };
        }
        return session;
      });
      localStorage.setItem("chatSessions", JSON.stringify(updatedSessions));
      return updatedSessions;
    });
  }, [messages, currentSessionId]);

  const createNewChat = () => {

    // 建立一個全新的 Session 物件
    const newSession = {
      id: Date.now(),
      title: "New Session",
      messages: [defaultMessage], // 每個新對話都有自己的歡迎詞
      createdAt: Date.now()
    };

    // 更新 Session 列表：把新的放在最上面，但保留舊的 (prev)
    setSessions(prev => [newSession, ...prev]);

    // 切換視角到這個新的 Session
    setCurrentSessionId(newSession.id);

    // 更新目前的訊息視窗為這個新 Session 的訊息 (也就是只有歡迎詞)
    setMessages(newSession.messages);

    // 清空圖片暫存
    setChatImages([]);
  };

  const switchSession = (sessionId) => {
    // 1. 先把「當前」的對話紀錄存回 sessions 陣列 (Auto-save)
    // 雖然 useEffect 已經有做 sync，但手動切換時再防呆一次
    setSessions(prevSessions => prevSessions.map(s =>
      s.id === currentSessionId ? { ...s, messages: messages } : s
    ));

    // 2. 找出目標 Session
    const targetSession = sessions.find(s => s.id === sessionId);

    if (targetSession) {
      // 3. 切換 ID
      setCurrentSessionId(sessionId);
      // 4. 載入目標 Session 的訊息到畫面上
      setMessages(targetSession.messages);
      // 5. 清空圖片暫存 (因為換房間了)
      setChatImages([]);
    }
  };

  const deleteSession = (e, sessionId) => {
    e.stopPropagation();
    const newSessions = sessions.filter(s => s.id !== sessionId);
    setSessions(newSessions);
    localStorage.setItem("chatSessions", JSON.stringify(newSessions));
    if (sessionId === currentSessionId) {
      if (newSessions.length > 0) {
        setCurrentSessionId(newSessions[0].id);
        setMessages(newSessions[0].messages);
      } else {
        const initialSession = { id: Date.now(), title: "New Session", messages: [defaultMessage], createdAt: Date.now() };
        setSessions([initialSession]);
        setCurrentSessionId(initialSession.id);
        setMessages([defaultMessage]);
      }
    }
  };

  const handleFileSelect = (e) => {
    const selected = Array.from(e.target.files);
    setFilesToUpload((prev) => [...prev, ...selected]);
    e.target.value = "";
  };

  const removeFile = (index) => {
    setFilesToUpload(prev => prev.filter((_, i) => i !== index));
  };

  const handleUpload = async () => {
    if (filesToUpload.length === 0) return;
    setUploadStatus("🚀 UPLOADING...");

    const formData = new FormData();
    filesToUpload.forEach((file) => formData.append("files", file));

    try {
      const response = await axios.post(`${API_BASE}/upload`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      const { processed } = response.data;
      setUploadStatus(`✨ SYNC COMPLETE`);

      // 關鍵修改：上傳成功後，立刻重新抓取檔案列表
      fetchFileList();

      setMessages(prev => [...prev, {
        role: "AI",
        content: `🎉 **DATA INJECTED**\n\n成功載入 **${processed ? processed.length : 0}** 份文件核心。`
      }]);

      setFilesToUpload([]);
      setTimeout(() => setUploadStatus(""), 3000);

    } catch (error) {
      console.error(error);
      setUploadStatus("");
      setErrorModal({ show: true, message: "Upload Gateway Error" });
    }
  };

  const handleReset = async () => {
    if (!window.confirm("確定要清空所有對話紀錄與知識庫嗎？此動作無法復原。")) return; // 建議加上防呆確認

    // A. 切斷網路連線
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    // B. 殺死打字機迴圈
    if (typingIntervalRef.current) {
      clearInterval(typingIntervalRef.current);
      typingIntervalRef.current = null;
    }
    // C. 清除讀取狀態
    setLoadingSessionId(null);

    try {
      // 呼叫後端重置 API
      await axios.post(`${API_BASE}/reset`); // 建議改用 API_BASE 變數，比較乾淨
      // 如果你的環境沒有 API_BASE，就維持 "http://127.0.0.1:8000/api/reset"

      const resetMsg = { role: "AI", content: "🧹 **SYSTEM PURGED**\n記憶體與資料庫已強制格式化。" };

      // 1. 重置對話
      setMessages([resetMsg]);
      setSessions([{ id: Date.now(), title: "New Session", messages: [resetMsg], createdAt: Date.now() }]);
      localStorage.removeItem("chatSessions");

      // 2. 重置上傳區塊
      setUploadStatus("");
      setFilesToUpload([]);

      // 3. 【關鍵修改】同步清空左側檔案列表
      setFileList([]);
      // 這樣使用者就不會看到「幽靈檔案」，也不會誤點導致 404 錯誤

    } catch (error) {
      console.error(error);
      setErrorModal({ show: true, message: "Reset Protocol Failed" });
    }
  };

  const handleStop = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setLoadingSessionId(null);
    setMessages((prev) => {
      const newMessages = [...prev];
      const lastIndex = newMessages.length - 1;
      if (lastIndex >= 0) {
        newMessages[lastIndex] = {
          ...newMessages[lastIndex],
          isTyping: false
        };
      }
      return newMessages;
    });
  };

  // 新增：處理圖片轉 Base64
  const processImageFile = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => {
        const result = reader.result.toString();
        // 移除 data URL header 取得純 base64 字串
        const base64 = result.split(',')[1];
        resolve({
          id: Date.now() + Math.random(),
          url: result,    // 用於前端預覽
          base64: base64, // 用於後端發送
          name: file.name
        });
      };
      reader.onerror = error => reject(error);
    });
  };

  const handleChatImageSelect = async (e) => {
    const files = Array.from(e.target.files);
    if (files.length === 0) return;
    const processedImages = await Promise.all(files.map(processImageFile));
    setChatImages(prev => [...prev, ...processedImages]);
    e.target.value = "";
  };

  const removeChatImage = (index) => {
    setChatImages(prev => prev.filter((_, i) => i !== index));
  };

  const handleSendMessage = async (e) => {
    // 1. 按鍵判斷
    if (e && e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
    } else if (e && (e.key !== "Enter" || e.shiftKey)) {
      return;
    }

    // 檢查是否有輸入
    if (!input.trim() && chatImages.length === 0) return;

    const targetSessionId = currentSessionId;
    const userMsgContent = input;

    // 2. 準備訊息物件
    const userMessage = {
      role: "User",
      content: userMsgContent,
      images: chatImages.map(img => img.url)
    };
    const aiPlaceholder = { role: "AI", content: "", sources: [], isTyping: true };

    // 3. 更新畫面 (Session 資料庫 & 前景)
    setSessions(prev => prev.map(session => {
      if (session.id === targetSessionId) {
        return { ...session, messages: [...session.messages, userMessage, aiPlaceholder] };
      }
      return session;
    }));

    if (currentSessionIdRef.current === targetSessionId) {
      setMessages(prev => [...prev, userMessage, aiPlaceholder]);
    }

    // 4. 準備發送 payload
    const imagesPayload = chatImages.map(img => img.base64);
    setChatImages([]);
    setInput("");
    setLoadingSessionId(targetSessionId); // 開始轉圈圈

    const cleanHistory = messages.map(msg => {
      // 解構賦值，把 images 抽出來，其餘的屬性放進 rest
      const { images, ...rest } = msg;
      return rest;
    });

    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      const response = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: userMsgContent,
          model_name: selectedModel,
          history: cleanHistory, // <--- 使用清理過的純文字歷史紀錄
          images: imagesPayload  // <--- 只傳送當次的新圖片
        }),
        signal: controller.signal,
      });

      if (!response.ok) {
        // 增加更詳細的錯誤訊息處理
        const errorText = await response.text();
        throw new Error(`Network error: ${response.status} - ${errorText}`);
      }

      const sourcesHeader = response.headers.get("X-Sources");
      const sources = sourcesHeader ? JSON.parse(sourcesHeader) : [];

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      // 變數準備：生產者與消費者共享的變數
      let fullRawText = "";   // 【倉庫】後端傳來的所有文字
      let displayedText = ""; // 【貨架】目前顯示在螢幕上的文字
      let isStreamDone = false; // 標記：網路傳輸是否結束


      // 消費者：打字機特效迴圈 (每 30ms 執行一次)
      typingIntervalRef.current = setInterval(() => {
        // 算出還有多少字沒顯示
        const remainingChars = fullRawText.length - displayedText.length;

        if (remainingChars > 0) {
          // 動態加速邏輯：
          // 如果剩餘字數很多 (>100)，每次就多吐一點 (例如剩 500 字，除以 20，一次吐 25 字)
          // 如果剩餘字數很少，就維持最少 2 個字，保持打字感
          // 這樣既能快速顯示長文，又能保留結尾的打字特效
          const dynamicChunk = Math.max(2, Math.floor(remainingChars / 20));

          const nextChunk = fullRawText.slice(displayedText.length, displayedText.length + dynamicChunk);
          displayedText += nextChunk;

          // 更新 Session 資料庫 (背景)
          setSessions(prevSessions => prevSessions.map(session => {
            if (session.id === targetSessionId) {
              const newMsgs = [...session.messages];
              const lastIdx = newMsgs.length - 1;
              if (lastIdx >= 0 && newMsgs[lastIdx].role === "AI") {
                newMsgs[lastIdx] = {
                  ...newMsgs[lastIdx],
                  content: displayedText,
                  sources: sources,
                  isTyping: true
                };
              }
              return { ...session, messages: newMsgs };
            }
            return session;
          }));

          // 更新目前畫面 (前景)
          if (currentSessionIdRef.current === targetSessionId) {
            setMessages(prev => {
              const newMessages = [...prev];
              const lastIndex = newMessages.length - 1;
              if (lastIndex >= 0 && newMessages[lastIndex].role === "AI") {
                newMessages[lastIndex] = {
                  ...newMessages[lastIndex],
                  content: displayedText,
                  sources: sources,
                  isTyping: true
                };
              }
              return newMessages;
            });
          }
        }
        // 停止條件
        else if (isStreamDone) {
          clearInterval(typingIntervalRef.current);
          setLoadingSessionId(null);

          const cleanFinalText = (rawText) => {
            if (!rawText) return "";

            // 1. 將所有文字依換行符號強制切成一行一行 (免疫所有換行陷阱)
            const lines = rawText.split(/\r?\n/);

            // 2. 制定嚴格的「黑名單特徵」進行過濾
            const filteredLines = lines.filter(line => {
              // 只要這行同時包含「特定符號」與「特定動作」，無論中間隔什麼字，整行拔除！
              if (line.includes('🧠') && line.includes('分析')) return false;
              if (line.includes('⚡') && line.includes('生成')) return false;
              if (line.includes('🔗') && line.includes('調閱')) return false;
              if (line.includes('🤔') && line.includes('聯想')) return false;
              if (line.includes('✨') && line.includes('關鍵字')) return false;
              if (line.includes('🚀') && line.includes('檢索')) return false;
              if (line.includes('📄') && line.includes('觸發')) return false;
              if (line.includes('========')) return false;

              return true; // 沒有中槍的行，就乖乖保留
            });

            // 3. 重新組合文字，並把拔除後留下的連續空白行壓扁
            return filteredLines.join('\n').replace(/\n{3,}/g, '\n\n').trim();
          };

          const finalizeMessage = (msgs) => {
            const newMsgs = [...msgs];
            const lastIdx = newMsgs.length - 1;
            if (lastIdx >= 0) {
              newMsgs[lastIdx] = {
                ...newMsgs[lastIdx],
                // 關鍵點：在這裡把文字洗乾淨後，再存入最終對話框！
                content: cleanFinalText(newMsgs[lastIdx].content),
                isTyping: false
              };
            }
            return newMsgs;
          };

          setSessions(prev => prev.map(s => s.id === targetSessionId ? { ...s, messages: finalizeMessage(s.messages) } : s));
          if (currentSessionIdRef.current === targetSessionId) {
            setMessages(prev => finalizeMessage(prev));
          }
        }
      }, 50);

      //  生產者：網路接收迴圈 (負責填滿倉庫)
      while (true) {
        const { value, done } = await reader.read();
        if (done) {
          isStreamDone = true; // 告訴打字機：貨補完了，印完就可以下班了
          break;
        }
        // 只負責解碼並塞入 fullRawText，不負責更新 UI
        fullRawText += decoder.decode(value, { stream: true });
      }

    } catch (error) {
      if (error.name !== 'AbortError') {
        console.error("❌ 串流錯誤:", error);
        setErrorModal({ show: true, message: "Stream Connection Failed" });
      }
      setLoadingSessionId(null);
    } finally {
      abortControllerRef.current = null;
    }
  };

  return (
    <div className="flex items-center justify-center h-screen w-screen bg-[#0f0c29] font-sans overflow-hidden relative selection:bg-fuchsia-500 selection:text-white text-slate-800">
      <CyberpunkNeonBackground />
      <div className="relative z-10 w-[90vw] h-[90vh] max-w-[1400px] flex rounded-[40px] overflow-hidden shadow-[0_0_50px_rgba(217,70,239,0.2)] border border-white/20 bg-white/10 backdrop-blur-2xl">
        <div className="w-80 min-w-[300px] bg-slate-900/60 backdrop-blur-xl flex flex-col p-6 text-white relative border-r border-white/10">
          <div className="mb-8 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-2xl bg-gradient-to-br from-fuchsia-600 via-purple-600 to-cyan-600 flex items-center justify-center shadow-[0_0_20px_rgba(217,70,239,0.5)]">
                <Sparkles size={20} className="text-white" />
              </div>
              <div>
                <h1 className="text-lg font-bold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-fuchsia-400 to-cyan-400">Chroma AI</h1>
                <p className="text-[10px] text-slate-400 font-medium tracking-wider">PRO VERSION</p>
              </div>
            </div>
            <motion.button
              whileHover={{ scale: 1.1, rotate: 90 }}
              whileTap={{ scale: 0.9 }}
              onClick={createNewChat}
              className="p-2 rounded-full bg-slate-800 border border-white/10 hover:border-cyan-400 text-cyan-400 hover:text-white transition-colors"
              title="New Connection"
            >
              <Plus size={18} />
            </motion.button>
          </div>
          <div className="flex-1 overflow-y-auto custom-scrollbar pr-2 space-y-8">
            <div className="space-y-3">
              <label className="text-[10px] font-bold text-cyan-400 uppercase tracking-widest pl-1">Model Engine</label>
              <div className="relative group">
                <div className="absolute inset-0 bg-gradient-to-r from-fuchsia-600 to-cyan-600 rounded-2xl blur opacity-20 group-hover:opacity-50 transition-opacity" />
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="relative w-full bg-slate-800/80 text-white text-sm rounded-2xl px-4 py-3 border border-white/10 focus:outline-none focus:ring-1 focus:ring-cyan-400 appearance-none cursor-pointer font-medium hover:bg-slate-700 transition-colors"
                >
                  {/* 修改這裡：用 map 把後端抓來的清單印出來 */}
                  {availableModels.map((model) => (
                    <option key={model} value={model}>
                      {model} (遠端)
                    </option>
                  ))}
                </select>
                <div className="absolute right-4 top-3.5 text-cyan-400 pointer-events-none text-xs">▼</div>
              </div>
            </div>
            <div className="space-y-3">
              <label className="text-[10px] font-bold text-violet-400 uppercase tracking-widest pl-1 flex items-center gap-2">
                <Archive size={10} /> Mission Logs
              </label>
              <div className="space-y-2">
                {sessions.map((session) => (
                  <motion.div
                    key={session.id}
                    onClick={() => switchSession(session.id)}
                    className={`group relative flex items-center gap-3 p-3 rounded-xl cursor-pointer transition-all border ${currentSessionId === session.id
                      ? "bg-slate-800/90 border-fuchsia-500/50 shadow-[0_0_15px_rgba(217,70,239,0.15)]"
                      : "bg-slate-800/30 border-transparent hover:bg-slate-800/60 hover:border-white/10"
                      }`}
                  >
                    {currentSessionId === session.id && (
                      <div className="absolute left-0 w-1 h-6 bg-gradient-to-b from-fuchsia-500 to-cyan-500 rounded-r-full" />
                    )}
                    <MessageCircle size={14} className={currentSessionId === session.id ? "text-fuchsia-400" : "text-slate-500"} />
                    <div className="flex-1 min-w-0">
                      <p className={`text-xs font-medium truncate ${currentSessionId === session.id ? "text-white" : "text-slate-400 group-hover:text-slate-300"}`}>
                        {session.title || "New Session"}
                      </p>
                      <p className="text-[10px] text-slate-200 truncate mt-0.5">
                        {new Date(session.createdAt || Date.now()).toLocaleTimeString()}
                      </p>
                    </div>
                    <button
                      onClick={(e) => deleteSession(e, session.id)}
                      className="opacity-0 group-hover:opacity-100 p-1.5 hover:bg-red-500/20 rounded-md text-slate-500 hover:text-red-400 transition-all"
                    >
                      <X size={12} />
                    </button>
                  </motion.div>
                ))}
              </div>
            </div>
            <div className="space-y-3">
              <label className="text-[10px] font-bold text-fuchsia-400 uppercase tracking-widest pl-1">Data Injection</label>
              <motion.div
                whileHover={{ scale: 1.02, backgroundColor: "rgba(30, 41, 59, 0.8)" }}
                whileTap={{ scale: 0.98 }}
                onClick={() => fileInputRef.current.click()}
                className="relative overflow-hidden bg-slate-800/50 border border-fuchsia-500/30 border-dashed rounded-3xl p-6 text-center cursor-pointer group transition-colors pb-4 hover:border-fuchsia-400"
              >
                <div className="w-12 h-12 bg-slate-700/50 rounded-full flex items-center justify-center mx-auto mb-3 group-hover:bg-fuchsia-500/20 group-hover:text-fuchsia-300 transition-all text-slate-400">
                  <Upload size={20} />
                </div>
                <p className="text-sm font-semibold text-slate-300 group-hover:text-white transition-colors">Upload Files</p>
                <div className="flex justify-center gap-1.5 mt-3">
                  <span className="text-[9px] bg-slate-900/80 px-2 py-0.5 rounded text-orange-300 border border-orange-500/20">PDF</span>
                  <span className="text-[9px] bg-slate-900/80 px-2 py-0.5 rounded text-cyan-300 border border-blue-500/20">DOCX</span>
                  <span className="text-[9px] bg-slate-900/80 px-2 py-0.5 rounded text-purple-300 border border-purple-500/20">JPG</span>
                  <span className="text-[9px] bg-slate-900/80 px-2 py-0.5 rounded text-emerald-300 border border-emerald-500/20">XLSX</span>
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
                  <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }} className="space-y-3 overflow-hidden">
                    <div className="flex flex-wrap gap-2 pt-1">
                      {filesToUpload.map((file, idx) => (
                        <div key={idx} className="flex items-center gap-2 bg-slate-800/80 px-3 py-1.5 rounded-full text-[10px] border border-white/10">
                          <span className="truncate max-w-[80px] text-slate-300">{file.name}</span>
                          <button onClick={() => removeFile(idx)} className="text-slate-500 hover:text-white"><X size={10} /></button>
                        </div>
                      ))}
                    </div>
                    <motion.button
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      onClick={handleUpload}
                      className="w-full py-3 rounded-2xl bg-gradient-to-r from-fuchsia-600 via-violet-600 to-cyan-600 text-white text-xs font-bold shadow-[0_0_20px_rgba(217,70,239,0.3)]"
                    >
                      INITIALIZE UPLOAD
                    </motion.button>
                  </motion.div>
                )}
              </AnimatePresence>
              {uploadStatus && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-center text-xs font-medium text-cyan-300 bg-cyan-900/30 py-2 rounded-xl border border-cyan-500/30">
                  {uploadStatus}
                </motion.div>
              )}
              <div className="mt-4 pt-4 border-t border-white/10">
                <div className="flex items-center justify-between mb-2 px-1">
                  <label className="text-[10px] font-bold text-emerald-400 uppercase tracking-widest flex items-center gap-2">
                    <FileText size={10} /> Knowledge Base ({fileList.length})
                  </label>
                  <button onClick={fetchFileList} className="text-slate-500 hover:text-cyan-400 transition-colors" title="Refresh">
                    <RefreshCw size={10} className={loadingFiles ? "animate-spin" : ""} />
                  </button>
                </div>

                <div className="flex-1 overflow-y-auto custom-scrollbar space-y-1 pr-1 max-h-[150px]">
                  {fileList.length === 0 ? (
                    <p className="text-center text-[10px] text-slate-600 py-4 italic">No data injected yet.</p>
                  ) : (
                    <AnimatePresence>
                      {fileList.map((file) => (
                        <motion.div
                          key={file}
                          initial={{ opacity: 0, x: -10 }}
                          animate={{ opacity: 1, x: 0 }}
                          exit={{ opacity: 0, height: 0 }}
                          className="group flex items-center justify-between p-2 rounded-lg bg-slate-800/30 border border-transparent hover:border-emerald-500/30 hover:bg-slate-800/80 transition-all cursor-pointer"
                          onClick={() => handleViewFile(file)}
                        >
                          <div className="flex items-center gap-2 min-w-0">
                            <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 shadow-[0_0_5px_#10b981]" />
                            <span className="text-[11px] text-slate-300 truncate font-mono max-w-[140px]" title={file}>
                              {file}
                            </span>
                          </div>

                          <button
                            onClick={(e) => handleDeleteFile(e, file)}
                            className="opacity-0 group-hover:opacity-100 p-1.5 rounded hover:bg-red-500/20 text-slate-500 hover:text-red-400 transition-all"
                            title="Delete"
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
          <div className="pt-4 mt-auto">
            <button
              onClick={handleReset}
              className="w-full py-3 flex items-center justify-center gap-2 text-red-400 border border-red-500/30 hover:bg-red-500/10 hover:text-red-300 hover:border-red-400 rounded-2xl transition-all text-xs font-bold shadow-[0_0_10px_rgba(239,68,68,0.1)] hover:shadow-[0_0_20px_rgba(239,68,68,0.2)]"
            >
              <Trash2 size={14} /> PURGE SYSTEM
            </button>
          </div>
        </div>
        <div className="flex-1 flex flex-col relative bg-white/60 backdrop-blur-3xl border-l border-white/50">
          <div className="h-16 border-b border-white/50 flex items-center px-8 bg-white/40 backdrop-blur-md justify-between z-20">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse shadow-[0_0_10px_#34d399]" />
              <span className="text-sm font-bold text-slate-600">SUCCESS RUN</span>
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
              //  關鍵修改：如果是 AI 且正在打字，直接 return null (隱藏)，交給底部的 ThinkingBubble 顯示
              if (msg.role === "AI" && msg.isTyping) return null;

              return (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 15, scale: 0.98 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  className={`flex ${msg.role === "User" ? "justify-end" : "justify-start"}`}
                >
                  <div className={`flex gap-3 max-w-[85%] ${msg.role === "User" ? "flex-row-reverse" : "flex-row"}`}>
                    <div className={`w-10 h-10 rounded-2xl flex-shrink-0 flex items-center justify-center shadow-md ${msg.role === "User"
                      ? "bg-gradient-to-tr from-sky-400 to-cyan-400 text-white shadow-sky-500/40"
                      : "bg-white text-cyan-600 shadow-slate-200/50"
                      }`}>
                      {msg.role === "User" ? <User size={18} /> : <Bot size={22} />}
                    </div>
                    <div className={`flex flex-col ${msg.role === "User" ? "items-end" : "items-start"} min-w-0`}>
                      <div className={`p-5 rounded-3xl shadow-sm backdrop-blur-xl border relative overflow-hidden ${msg.role === "User"
                        ? "bg-gradient-to-br from-sky-500 to-blue-600 text-white rounded-br-none border-white/20 shadow-[0_5px_15px_rgba(14,165,233,0.3)]"
                        : "bg-white/80 text-slate-800 rounded-bl-none border-white/60 shadow-lg shadow-cyan-500/5"
                        }`}>

                        {/*  內容顯示區 */}
                        {msg.content && (
                          <div className={`prose max-w-none text-sm leading-relaxed ${msg.role === "User" ? "prose-invert text-white" : "prose-slate"}`}>
                            <MarkdownRenderer content={msg.content} />
                          </div>
                        )}

                        {msg.images && msg.images.length > 0 && (
                          <div className="flex flex-wrap gap-2 mb-3">
                            {msg.images.map((imgUrl, i) => (
                              <img key={i} src={imgUrl} alt="uploaded" className="max-w-[200px] max-h-[200px] rounded-lg border border-white/20" />
                            ))}
                          </div>
                        )}

                        {!msg.isTyping && msg.sources && msg.sources.length > 0 && (
                          <div className="mt-4 pt-3 border-t border-white/20 flex flex-wrap gap-2">
                            {msg.sources.map((src, i) => (
                              <span key={i} className="text-[10px] px-2 py-1 rounded-md bg-white/20 border border-white/10 text-white/90 flex items-center gap-1 font-bold">
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
              <ThinkingBubble
                content={messages.find(m => m.role === 'AI' && m.isTyping)?.content || ""}
              />
            )}

            <div ref={chatEndRef} />
          </div>
          {/* 輸入框區域 */}
          <div className="p-8 pt-2 z-20">
            <div className="relative max-w-4xl mx-auto">
              <div className="absolute -inset-1 bg-gradient-to-r from-fuchsia-500 via-violet-500 to-cyan-500 rounded-full opacity-30 blur-md group-focus-within:opacity-60 transition-opacity duration-500" />

              {/* 新增：圖片預覽區 (放在輸入框容器上方) */}
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
                        <img src={img.url} alt="preview" className="h-16 w-16 object-cover rounded-xl border-2 border-cyan-400 shadow-lg" />
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
                {/* 1. 隱藏的圖片選擇器 */}
                <input
                  type="file"
                  ref={chatImageInputRef}
                  className="hidden"
                  onChange={handleChatImageSelect}
                  multiple
                  accept="image/jpeg,image/png,image/webp,image/gif"
                />

                {/* 2. 圖片上傳按鈕 (這顆就是你截圖裡不見的按鈕！) */}
                <motion.button
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                  onClick={() => chatImageInputRef.current.click()}
                  className="p-2 text-slate-400 hover:text-cyan-500 hover:bg-cyan-500/10 rounded-full transition-colors self-end mb-1"
                  title="Upload Image"
                >
                  <ImageIcon size={22} />
                </motion.button>
                <textarea
                  ref={textareaRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleSendMessage}
                  placeholder="Enter command..."
                  rows={1}
                  className="flex-1 bg-transparent text-slate-700 text-base focus:outline-none placeholder-slate-400 font-medium resize-none py-3 max-h-[120px]"
                  disabled={false}
                />
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  // 加上 loadingSessionId !== null 的檢查
                  onClick={() => (loadingSessionId !== null && loadingSessionId === currentSessionId) ? handleStop() : handleSendMessage()}
                  disabled={!input.trim() && chatImages.length === 0 && loadingSessionId !== null && loadingSessionId !== currentSessionId}
                  className={`p-3 rounded-full text-white shadow-lg transition-all self-end ${(loadingSessionId !== null && loadingSessionId === currentSessionId)
                    ? "bg-gradient-to-r from-red-500 to-orange-500 hover:shadow-red-500/30 cursor-pointer"
                    : "bg-gradient-to-r from-fuchsia-600 to-cyan-600 hover:shadow-cyan-500/30 disabled:opacity-50 disabled:cursor-not-allowed"
                    }`}
                >
                  {(loadingSessionId !== null && loadingSessionId === currentSessionId) ? (
                    <Square size={20} className="fill-current animate-pulse" />
                  ) : (
                    <Send size={20} />
                  )}
                </motion.button>
              </div>
            </div>
            <p className="text-center text-[10px] text-slate-400 mt-3 font-medium opacity-60">
              SECURE CONNECTION ESTABLISHED • v9.0
            </p>
          </div>
        </div>
      </div>
      <AnimatePresence>
        {viewingFile && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4"
            onClick={() => setViewingFile(null)}
          >
            <motion.div
              initial={{ scale: 0.9, y: 20 }}
              animate={{ scale: 1, y: 0 }}
              exit={{ scale: 0.9, y: 20 }}
              className="bg-slate-900 border border-cyan-500/50 w-full max-w-3xl max-h-[80vh] rounded-2xl shadow-[0_0_50px_rgba(6,182,212,0.2)] flex flex-col overflow-hidden"
              onClick={e => e.stopPropagation()}
            >
              {/* Header */}
              <div className="flex items-center justify-between p-4 border-b border-white/10 bg-slate-800/50">
                <div className="flex items-center gap-2 text-cyan-400">
                  <FileText size={20} />
                  <h3 className="font-bold text-lg truncate max-w-md">{viewingFile}</h3>
                </div>
                <button
                  onClick={() => setViewingFile(null)}
                  className="p-2 hover:bg-white/10 rounded-full transition-colors text-slate-400 hover:text-white"
                >
                  <X size={20} />
                </button>
              </div>

              {/* Content */}
              <div className="flex-1 overflow-y-auto p-6 custom-scrollbar bg-[#0f0c29]">
                <pre className="whitespace-pre-wrap font-mono text-sm text-slate-300 leading-relaxed">
                  {viewContent}
                </pre>
              </div>

              {/* Footer */}
              <div className="p-3 border-t border-white/10 bg-slate-800/50 text-right text-xs text-slate-500">
                SOURCE CONTENT VIEWER v1.0
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
export default App;