import { useState, useCallback, useRef, useEffect } from "react";

const API = "http://localhost:5000/api";

function DropZone({ label, items, onDrop, onRemove, color, icon, processing }) {
  const [dragging, setDragging] = useState(false);
  const ref = useRef(null);

  const readFiles = useCallback(async (files) => {
    const results = [];
    for (const file of files) {
      try { results.push({ fileName: file.name, text: await file.text() }); }
      catch (e) { console.error(e); }
    }
    if (results.length > 0) onDrop(results);
  }, [onDrop]);

  const handleDrop = useCallback((e) => {
    e.preventDefault(); setDragging(false);
    const files = [...e.dataTransfer.files].filter(f =>
      /\.(txt|csv|md|text|json|tsv)$/i.test(f.name) || f.type.startsWith("text/")
    );
    if (files.length === 0) {
      const t = e.dataTransfer.getData("text/plain");
      if (t) onDrop([{ fileName: "pasted.txt", text: t }]);
      return;
    }
    readFiles(files);
  }, [onDrop, readFiles]);

  return (
    <div style={{ flex: 1, minWidth: 0 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10, fontSize: 14, fontWeight: 700, color: "#1e293b" }}>
        <span style={{ width: 26, height: 26, borderRadius: 6, display: "flex", alignItems: "center", justifyContent: "center", background: color + "15", color, fontSize: 14 }}>{icon}</span>
        {label}
        <span style={{ marginLeft: "auto", fontSize: 11, fontWeight: 600, color: "#94a3b8", background: "#f1f5f9", padding: "2px 8px", borderRadius: 10 }}>{items.length}件</span>
      </div>
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        onClick={() => ref.current?.click()}
        style={{
          border: `2px dashed ${dragging ? color : "#d1d5db"}`, borderRadius: 12,
          padding: items.length > 0 ? 10 : 36, minHeight: items.length > 0 ? 50 : 130,
          background: dragging ? color + "06" : "#fafbfc", cursor: "pointer",
          transition: "all 0.2s", display: "flex", flexDirection: "column",
          alignItems: items.length > 0 ? "stretch" : "center",
          justifyContent: items.length > 0 ? "flex-start" : "center", gap: 6,
        }}
      >
        <input ref={ref} type="file" multiple accept=".txt,.csv,.md,.text,.json,.tsv" style={{ display: "none" }}
          onChange={(e) => readFiles([...e.target.files])} />
        {items.length === 0 ? (
          <>
            <div style={{ fontSize: 28, opacity: 0.25, marginBottom: 6 }}>{icon}</div>
            <div style={{ fontSize: 13, color: "#64748b", fontWeight: 600 }}>{label}をドロップ</div>
            <div style={{ fontSize: 11, color: "#a1a1aa" }}>.txt .csv .md（クリックで選択）</div>
          </>
        ) : items.map((item, i) => (
          <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, padding: "7px 10px", background: "#fff", borderRadius: 8, border: "1px solid #e5e7eb" }}>
            <div style={{ width: 6, height: 6, borderRadius: "50%", background: item.extracted ? "#10b981" : item.error ? "#ef4444" : "#f59e0b", flexShrink: 0 }} />
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: "#1e293b", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {item.extracted?.name || item.fileName}
              </div>
              <div style={{ fontSize: 10, color: "#94a3b8", display: "flex", flexWrap: "wrap", gap: 3, marginTop: 2 }}>
                {item.extracted ? (
                  <>
                    <span>{item.extracted.skills?.length || 0}スキル</span>
                    {item.extracted._extraction_method && (
                      <span style={{ padding: "0 5px", borderRadius: 3, background: item.extracted._extraction_method === "qwen" ? "#eff6ff" : "#f5f3ff", color: item.extracted._extraction_method === "qwen" ? "#2563eb" : "#7c3aed", fontWeight: 600 }}>
                        {item.extracted._extraction_method === "qwen" ? "Qwen" : "Rule"}
                      </span>
                    )}
                    {item.extracted.skills?.slice(0, 3).map(s =>
                      <span key={s.name} style={{ padding: "0 5px", borderRadius: 3, background: color + "12", color, fontWeight: 600 }}>{s.name}</span>
                    )}
                  </>
                ) : item.error ? (
                  <span style={{ color: "#ef4444" }}>{item.error}</span>
                ) : "解析中..."}
              </div>
            </div>
            <button onClick={(e) => { e.stopPropagation(); onRemove(i); }}
              style={{ background: "none", border: "none", cursor: "pointer", color: "#cbd5e1", fontSize: 15, padding: "2px 4px", borderRadius: 4, lineHeight: 1 }}>×</button>
          </div>
        ))}
      </div>
    </div>
  );
}

function MatchCard({ match, rank }) {
  const [open, setOpen] = useState(false);
  const rc = match.score >= 85 ? "#10b981" : match.score >= 70 ? "#f59e0b" : match.score >= 50 ? "#6366f1" : "#94a3b8";
  return (
    <div onClick={() => setOpen(!open)} style={{ background: "#fff", borderRadius: 10, padding: "12px 16px", border: "1px solid #e5e7eb", cursor: "pointer", transition: "box-shadow 0.15s", boxShadow: open ? "0 2px 12px rgba(0,0,0,0.07)" : "none" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <span style={{ width: 26, height: 26, borderRadius: "50%", display: "flex", alignItems: "center", justifyContent: "center", background: "#f1f5f9", fontSize: 12, fontWeight: 700, color: "#475569", flexShrink: 0 }}>{rank}</span>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: 13, fontWeight: 700 }}>{match.project_name} × {match.engineer_name}</div>
          <div style={{ display: "flex", gap: 3, flexWrap: "wrap", marginTop: 3 }}>
            {match.direct_matches?.map(s => <span key={s} style={{ padding: "1px 6px", borderRadius: 3, fontSize: 10, fontWeight: 600, background: "#dbeafe", color: "#1d4ed8" }}>{s}</span>)}
            {match.indirect_matches?.slice(0, 2).map((m, i) => <span key={i} style={{ padding: "1px 6px", borderRadius: 3, fontSize: 10, fontWeight: 600, background: "#fef3c7", color: "#92400e" }}>{m.via}→{m.target}</span>)}
            {match.missing_skills?.slice(0, 2).map(s => <span key={s} style={{ padding: "1px 6px", borderRadius: 3, fontSize: 10, fontWeight: 600, background: "#fef2f2", color: "#dc2626", textDecoration: "line-through" }}>{s}</span>)}
          </div>
        </div>
        <div style={{ textAlign: "right", flexShrink: 0 }}>
          <div style={{ fontSize: 22, fontWeight: 800, color: rc }}>{match.score}%</div>
          <span style={{ fontSize: 10, fontWeight: 700, padding: "1px 8px", borderRadius: 8, background: rc + "15", color: rc }}>{match.rank}</span>
        </div>
      </div>
      {open && (
        <div style={{ marginTop: 12, paddingTop: 12, borderTop: "1px solid #f1f5f9" }}>
          <div style={{ display: "flex", gap: 14 }}>
            {[{ l: "スキル", v: match.skill_score, c: "#2563eb" }, { l: "単価", v: match.price_score, c: "#10b981" }, { l: "勤務地", v: match.location_score, c: "#f59e0b" }].map(b => (
              <div key={b.l} style={{ flex: 1 }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 3 }}>
                  <span style={{ color: "#64748b", fontWeight: 600 }}>{b.l}</span><span style={{ fontWeight: 700, color: b.c }}>{b.v}%</span>
                </div>
                <div style={{ height: 5, background: "#f1f5f9", borderRadius: 3 }}><div style={{ width: `${b.v}%`, height: "100%", borderRadius: 3, background: b.c }} /></div>
              </div>
            ))}
          </div>
          <div style={{ marginTop: 10, fontSize: 11, color: "#64748b", display: "flex", gap: 12, flexWrap: "wrap" }}>
            {match.engineer_price && <span>単価: {match.engineer_price}万円</span>}
            {match.price_range && <span>案件: {match.price_range}</span>}
            {match.remote && <span>{match.remote}</span>}
            {match.engineer_available && <span>参画: {match.engineer_available}</span>}
          </div>
          {match.indirect_matches?.length > 0 && (
            <div style={{ marginTop: 8, fontSize: 11, color: "#7c3aed", background: "#f5f3ff", padding: "6px 10px", borderRadius: 6 }}>
              グラフ間接マッチ: {match.indirect_matches.map(m => `${m.via} →[${(m.weight * 100).toFixed(0)}%]→ ${m.target}`).join(", ")}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function TextModal({ show, onClose, onSubmit, type }) {
  const [text, setText] = useState("");
  if (!show) return null;
  return (
    <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.35)", zIndex: 100, display: "flex", alignItems: "center", justifyContent: "center" }} onClick={onClose}>
      <div onClick={e => e.stopPropagation()} style={{ background: "#fff", borderRadius: 14, padding: 24, width: "90%", maxWidth: 560, boxShadow: "0 8px 32px rgba(0,0,0,0.12)" }}>
        <h3 style={{ fontSize: 15, fontWeight: 700, marginBottom: 12 }}>{type === "engineer" ? "スキルシート" : "求人票"}テキスト入力</h3>
        <textarea value={text} onChange={e => setText(e.target.value)}
          placeholder={type === "engineer"
            ? "氏名：田中太郎\nスキル：Java, Spring Boot, AWS\n経験：5年\n希望単価：80万円\n勤務地：東京"
            : "案件名：EC基盤刷新\n必須スキル：Java / Spring Boot / AWS\n単価：75〜85万円\n勤務地：東京"}
          style={{ width: "100%", height: 180, padding: 12, borderRadius: 8, border: "1px solid #e5e7eb", fontSize: 13, resize: "vertical", fontFamily: "inherit", lineHeight: 1.6 }} />
        <div style={{ display: "flex", gap: 8, justifyContent: "flex-end", marginTop: 12 }}>
          <button onClick={onClose} style={{ padding: "7px 18px", borderRadius: 8, border: "1px solid #e5e7eb", background: "#fff", fontSize: 13, cursor: "pointer", fontWeight: 600 }}>キャンセル</button>
          <button onClick={() => { if (text.trim()) { onSubmit(text); setText(""); onClose(); } }}
            style={{ padding: "7px 18px", borderRadius: 8, border: "none", background: "#1e293b", color: "#fff", fontSize: 13, cursor: "pointer", fontWeight: 600 }}>追加</button>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [engineers, setEngineers] = useState([]);
  const [projects, setProjects] = useState([]);
  const [matches, setMatches] = useState([]);
  const [processing, setProcessing] = useState(false);
  const [tab, setTab] = useState("upload");
  const [modal, setModal] = useState(null);
  const [health, setHealth] = useState(null);
  const [graphStats, setGraphStats] = useState(null);
  const [useQwen, setUseQwen] = useState(true);

  // ヘルスチェック
  useEffect(() => {
    fetch(`${API}/health`).then(r => r.json()).then(setHealth).catch(() => setHealth({ status: "error", neo4j: "disconnected", qwen_api: "unknown" }));
  }, []);

  const extractAndRegister = useCallback(async (files, type) => {
    setProcessing(true);
    const setter = type === "engineer" ? setEngineers : setProjects;
    for (const file of files) {
      // プレースホルダー追加
      setter(prev => [...prev, { fileName: file.fileName, text: file.text, extracted: null }]);
      try {
        const res = await fetch(`${API}/extract`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: file.text, type, use_api: useQwen }),
        });
        const extracted = await res.json();

        // グラフ登録
        const regRes = await fetch(`${API}/register`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ data: extracted, type }),
        });
        const reg = await regRes.json();
        extracted.node_id = reg.node_id;

        setter(prev => prev.map(item =>
          item.fileName === file.fileName && !item.extracted
            ? { ...item, extracted }
            : item
        ));
      } catch (e) {
        setter(prev => prev.map(item =>
          item.fileName === file.fileName && !item.extracted
            ? { ...item, error: "接続エラー: Flask起動確認" }
            : item
        ));
      }
    }
    // グラフ統計更新
    try {
      const st = await fetch(`${API}/graph/stats`).then(r => r.json());
      setGraphStats(st);
    } catch (e) {}
    setProcessing(false);
  }, [useQwen]);

  const runMatching = useCallback(async () => {
    setProcessing(true);
    try {
      const res = await fetch(`${API}/matching`);
      const data = await res.json();
      setMatches(data);
      setTab("results");
    } catch (e) {
      alert("マッチング失敗: Flaskサーバーを確認してください");
    }
    setProcessing(false);
  }, []);

  const clearAll = useCallback(async () => {
    try { await fetch(`${API}/clear`, { method: "POST" }); } catch (e) {}
    setEngineers([]); setProjects([]); setMatches([]);
    setGraphStats(null);
  }, []);

  const loadDemo = useCallback(() => {
    const demoEng = [
      { fileName: "佐藤健一.txt", text: "氏名：佐藤 健一\nスキル：Java, Spring Boot, AWS, Docker, PostgreSQL\n経験：8年\n希望単価：80万円\n勤務地希望：東京、リモート可\n参画可能：2026年5月" },
      { fileName: "鈴木麻衣.txt", text: "氏名：鈴木 麻衣\nスキル：React, TypeScript, Next.js, Node.js, JavaScript\n経験：5年\n希望単価：78万円\n勤務地希望：フルリモート\n参画可能：即日" },
      { fileName: "田中亮.txt", text: "氏名：田中 亮\nスキル：Python, Azure, ETL, SQL, Docker, 機械学習\n経験：7年\n希望単価：90万円\n勤務地希望：横浜、リモート\n参画可能：2026年6月" },
      { fileName: "高橋さくら.txt", text: "氏名：高橋 さくら\nスキル：Python, 機械学習, PyTorch, AWS, Docker, Kubernetes\n経験：6年\n希望単価：85万円\n勤務地希望：東京\n参画可能：2026年5月" },
    ];
    const demoProj = [
      { fileName: "EC基盤刷新.txt", text: "案件名：大手小売向けEC基盤刷新\n必須スキル：Java, Spring Boot, AWS\n歓迎：Docker, PostgreSQL\n単価：75〜85万円\n勤務地：東京・リモート併用\n開始：2026年5月" },
      { fileName: "Webアプリ開発.txt", text: "案件名：通信会社向けWebアプリ開発\n必須スキル：React, TypeScript, Node.js\n歓迎：Next.js, AWS\n単価：70〜85万円\n勤務地：新宿・フルリモート可\n開始：2026年5月" },
      { fileName: "データ分析基盤.txt", text: "案件名：製造業向けデータ分析基盤構築\n必須スキル：Python, ETL, Azure\n歓迎：機械学習, Docker\n単価：80〜95万円\n勤務地：横浜・リモート併用\n開始：2026年6月" },
      { fileName: "医療AI.txt", text: "案件名：医療データ分析PF\n必須スキル：Python, 機械学習, AWS\n歓迎：PyTorch, Docker, Kubernetes\n単価：85〜100万円\n勤務地：東京・リモート併用\n開始：2026年7月" },
    ];
    extractAndRegister(demoEng, "engineer");
    extractAndRegister(demoProj, "project");
  }, [extractAndRegister]);

  const hasData = engineers.some(e => e.extracted) && projects.some(p => p.extracted);

  return (
    <div style={{ fontFamily: "'Noto Sans JP','Hiragino Sans',system-ui,sans-serif", color: "#1e293b", maxWidth: 940, margin: "0 auto", padding: "18px 14px" }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", marginBottom: 18 }}>
        <div>
          <h1 style={{ fontSize: 20, fontWeight: 800, margin: 0, letterSpacing: -0.5 }}>Skill Matcher</h1>
          <p style={{ fontSize: 12, color: "#64748b", margin: "3px 0 0" }}>Qwen API + Neo4j グラフマイニング</p>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
          {/* Backend status */}
          <div style={{ display: "flex", gap: 6, fontSize: 10, color: "#94a3b8" }}>
            <span style={{ display: "flex", alignItems: "center", gap: 3 }}>
              <span style={{ width: 6, height: 6, borderRadius: "50%", background: health?.neo4j === "connected" ? "#10b981" : "#ef4444" }} />
              Neo4j
            </span>
            <span style={{ display: "flex", alignItems: "center", gap: 3 }}>
              <span style={{ width: 6, height: 6, borderRadius: "50%", background: health?.qwen_api?.includes("configured") ? "#10b981" : "#f59e0b" }} />
              Qwen
            </span>
          </div>
          {/* Toggle */}
          <label style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 11, color: "#64748b", cursor: "pointer", userSelect: "none" }}>
            <div style={{ width: 34, height: 18, borderRadius: 9, padding: 2, background: useQwen ? "#2563eb" : "#cbd5e1", transition: "background 0.2s", cursor: "pointer" }}
              onClick={() => setUseQwen(!useQwen)}>
              <div style={{ width: 14, height: 14, borderRadius: "50%", background: "#fff", transform: useQwen ? "translateX(16px)" : "translateX(0)", transition: "transform 0.2s", boxShadow: "0 1px 3px rgba(0,0,0,0.15)" }} />
            </div>
            {useQwen ? "Qwen API" : "ルール抽出"}
          </label>
        </div>
      </div>

      {/* Graph stats bar */}
      {graphStats && (
        <div style={{ display: "flex", gap: 16, marginBottom: 14, padding: "8px 14px", background: "#f0fdf4", borderRadius: 8, border: "1px solid #bbf7d0", fontSize: 11, color: "#166534" }}>
          <span>Neo4j グラフ:</span>
          <span>要員 {graphStats.engineers}</span>
          <span>案件 {graphStats.projects}</span>
          <span>スキル {graphStats.skills}</span>
          <span>HAS_SKILL {graphStats.has_skill_edges}</span>
          <span>REQUIRES {graphStats.requires_edges}</span>
          <span>RELATED_TO {graphStats.related_to_edges}</span>
        </div>
      )}

      {/* Tabs */}
      <div style={{ display: "flex", gap: 2, marginBottom: 16 }}>
        {[{ id: "upload", label: "データ入力" }, { id: "results", label: "マッチング結果" }].map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: "8px 18px", borderRadius: "7px 7px 0 0", border: "none", cursor: "pointer",
            fontSize: 12, fontWeight: 600, background: tab === t.id ? "#1e293b" : "#f1f5f9",
            color: tab === t.id ? "#fff" : "#64748b", transition: "all 0.15s",
          }}>{t.label}
            {t.id === "results" && matches.length > 0 && <span style={{ marginLeft: 5, padding: "1px 6px", borderRadius: 6, fontSize: 10, background: "rgba(255,255,255,0.2)" }}>{matches.length}</span>}
          </button>
        ))}
      </div>

      {tab === "upload" && (
        <>
          <div style={{ display: "flex", gap: 16, marginBottom: 16 }}>
            <DropZone label="スキルシート" items={engineers} onDrop={(f) => extractAndRegister(f, "engineer")}
              onRemove={(i) => setEngineers(prev => prev.filter((_, idx) => idx !== i))} color="#2563eb" icon="👤" processing={processing} />
            <DropZone label="求人データ" items={projects} onDrop={(f) => extractAndRegister(f, "project")}
              onRemove={(i) => setProjects(prev => prev.filter((_, idx) => idx !== i))} color="#dc2626" icon="📋" processing={processing} />
          </div>

          <div style={{ display: "flex", gap: 8, marginBottom: 14 }}>
            <button onClick={() => setModal("engineer")} style={{ flex: 1, padding: 9, borderRadius: 8, border: "1px dashed #93c5fd", background: "#eff6ff", color: "#2563eb", fontSize: 12, fontWeight: 600, cursor: "pointer" }}>+ スキルシート手入力</button>
            <button onClick={() => setModal("project")} style={{ flex: 1, padding: 9, borderRadius: 8, border: "1px dashed #fca5a5", background: "#fef2f2", color: "#dc2626", fontSize: 12, fontWeight: 600, cursor: "pointer" }}>+ 求人データ手入力</button>
          </div>

          <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 14, padding: "10px 14px", background: "#f8fafc", borderRadius: 8, border: "1px solid #e5e7eb" }}>
            <span style={{ fontSize: 12, fontWeight: 700, color: "#475569" }}>デモ:</span>
            <button onClick={loadDemo} style={{ padding: "5px 14px", borderRadius: 6, border: "1px solid #e5e7eb", background: "#fff", color: "#475569", fontSize: 12, fontWeight: 600, cursor: "pointer" }}>サンプルデータ読込</button>
            <button onClick={clearAll} style={{ padding: "5px 14px", borderRadius: 6, border: "1px solid #fecaca", background: "#fff", color: "#dc2626", fontSize: 12, fontWeight: 600, cursor: "pointer" }}>全クリア</button>
          </div>

          <button onClick={runMatching} disabled={!hasData || processing} style={{
            width: "100%", padding: 13, borderRadius: 10, border: "none",
            background: hasData && !processing ? "#1e293b" : "#e5e7eb",
            color: hasData && !processing ? "#fff" : "#9ca3af",
            fontSize: 14, fontWeight: 700, cursor: hasData && !processing ? "pointer" : "default",
          }}>
            {processing ? "処理中..." : `グラフマッチング実行 (${engineers.filter(e => e.extracted).length}名 × ${projects.filter(p => p.extracted).length}件)`}
          </button>
        </>
      )}

      {tab === "results" && (
        matches.length > 0 ? (
          <>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
              <div style={{ fontSize: 12, color: "#64748b" }}>
                {matches.length}件 （最有力: {matches.filter(m => m.score >= 85).length} / 有力: {matches.filter(m => m.score >= 70 && m.score < 85).length}）
              </div>
              <button onClick={() => setTab("upload")} style={{ padding: "5px 12px", borderRadius: 6, border: "1px solid #e5e7eb", background: "#fff", fontSize: 11, fontWeight: 600, cursor: "pointer", color: "#64748b" }}>戻る</button>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {matches.map((m, i) => <MatchCard key={`${m.project_id}-${m.engineer_id}`} match={m} rank={i + 1} />)}
            </div>
          </>
        ) : (
          <div style={{ textAlign: "center", padding: "50px 20px", color: "#94a3b8" }}>
            <div style={{ fontSize: 36, marginBottom: 10, opacity: 0.25 }}>📊</div>
            <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 4 }}>まだマッチング結果がありません</div>
            <div style={{ fontSize: 12 }}>データ入力タブでスキルシートと求人データを追加してください</div>
          </div>
        )
      )}

      <TextModal show={modal !== null} onClose={() => setModal(null)}
        onSubmit={(text) => extractAndRegister([{ fileName: `手入力_${Date.now()}.txt`, text }], modal)} type={modal} />
    </div>
  );
}
