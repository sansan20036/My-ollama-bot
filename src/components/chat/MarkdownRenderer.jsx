import React, { useCallback, useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { vscDarkPlus } from "react-syntax-highlighter/dist/cjs/styles/prism";
import {
  DndContext,
  closestCenter,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors
} from "@dnd-kit/core";
import {
  arrayMove,
  SortableContext,
  sortableKeyboardCoordinates,
  horizontalListSortingStrategy,
  useSortable
} from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell
} from "recharts";
import { Paperclip, ExternalLink, BarChart2 } from "lucide-react";

import { STORAGE_KEYS } from "../../constants/chat";

// 可拖曳的表頭元件 (處理拖曳動畫與樣式)
const SortableHeader = ({ id, children }) => {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({ id });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    cursor: isDragging ? "grabbing" : "grab",
    backgroundColor: isDragging ? "rgba(6, 182, 212, 0.1)" : undefined,
    opacity: isDragging ? 0.3 : 1,
    border: isDragging ? "1px dashed #22d3ee" : undefined,
    zIndex: isDragging ? 999 : "auto",
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

// 圖表渲染元件
const ChartRenderer = ({ data, type, title }) => {
  const COLORS = ["#06b6d4", "#d946ef", "#8b5cf6", "#f59e0b", "#10b981"];

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
          {type === "pie" ? (
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
                contentStyle={{ backgroundColor: "#1e293b", borderColor: "#334155", color: "#fff" }}
                itemStyle={{ color: "#fff" }}
              />
              <Legend />
            </PieChart>
          ) : (
            <BarChart data={data}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
              <XAxis dataKey="name" stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} />
              <YAxis stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} />
              <Tooltip
                cursor={{ fill: "rgba(255,255,255,0.05)" }}
                contentStyle={{ backgroundColor: "#1e293b", borderColor: "#334155", color: "#fff" }}
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
  const thead = childrenArray.find((c) => c.type === "thead");
  const tbody = childrenArray.find((c) => c.type === "tbody");

  const extractText = useCallback((node) => {
    if (typeof node === "string") return node;
    if (Array.isArray(node)) return node.map(extractText).join("");
    if (node && node.props && node.props.children) return extractText(node.props.children);
    return "Column";
  }, []);

  const initialHeaders = useMemo(
    () =>
      React.Children.map(thead?.props?.children?.props?.children, (child) => {
        return extractText(child);
      }) || [],
    [thead, extractText]
  );

  const [columns, setColumns] = useState(() => {
    const saved = localStorage.getItem(STORAGE_KEYS.TABLE_COLUMN_ORDER);
    if (saved) {
      const savedCols = JSON.parse(saved);
      if (savedCols.length === initialHeaders.length && savedCols.every((c) => initialHeaders.includes(c))) {
        return savedCols;
      }
    }
    return initialHeaders;
  });

  useEffect(() => {
    if (initialHeaders.length > 0 && JSON.stringify(initialHeaders) !== JSON.stringify(columns)) {
      setColumns(initialHeaders);
    }
  }, [initialHeaders, columns]);

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
        localStorage.setItem(STORAGE_KEYS.TABLE_COLUMN_ORDER, JSON.stringify(newOrder));
        return newOrder;
      });
    }
  };

  const originalHeaderIndexMap = initialHeaders.reduce((acc, col, idx) => ({ ...acc, [col]: idx }), {});

  return (
    <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={handleDragEnd}>
      <div className="grid w-full max-w-full my-6">
        <div className="w-full overflow-auto max-h-[500px] rounded-xl border border-slate-700/50 shadow-lg bg-slate-900/90 custom-scrollbar">
          <table className="w-max min-w-full text-left text-sm border-separate border-spacing-0">
            <thead className="sticky top-0 z-20 bg-slate-900 text-cyan-300 font-bold uppercase tracking-wider text-xs shadow-md">
              <SortableContext items={columns} strategy={horizontalListSortingStrategy}>
                <tr>
                  {columns.map((col) => (
                    <SortableHeader key={col} id={col}>
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
        code({ inline, className, children, ...props }) {
          const match = /language-(\w+)/.exec(className || "");
          const codeString = String(children).replace(/\n$/, "");
          const safeStyle = vscDarkPlus || {};

          // 處理 JSON 圖表
          if (!inline && match && match[1] === "json-chart") {
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
                customStyle={{ margin: 0, padding: "1.5rem", background: "transparent", fontSize: "13px", lineHeight: "1.6" }}
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
          const isPureString = typeof children === "string";
          const isStringArray = Array.isArray(children) && children.every((c) => typeof c === "string");

          if (!isPureString && !isStringArray) {
            return <p className="mb-4 last:mb-0 leading-7">{children}</p>;
          }

          const text = Array.isArray(children) ? children.join("") : String(children);
          const parts = text.split(/(\(出處:.*?\))/g);

          return (
            <p className="mb-4 last:mb-0 leading-7">
              {parts.map((part, index) => {
                if (part.startsWith("(出處:") && part.endsWith(")")) {
                  const content = part.replace(/[()]/g, "");
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
        .replace(/\\\[/g, "$$")
        .replace(/\\\]/g, "$$")
        .replace(/\\\(/g, "$")
        .replace(/\\\)/g, "$")
      }
    </ReactMarkdown>
  );
};

export default MarkdownRenderer;
