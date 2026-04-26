"""
LightRAG スキルマッチング
═══════════════════════════
本家 LightRAG (HKUDS) のアーキテクチャに準拠した実装。

パイプライン:
  1. テキスト → チャンク分割
  2. LLM エンティティ＆リレーション抽出
  3. LLM エンティティ重複排除 (dedup)
  4. ベクトル埋め込み (sentence-transformers)
  5. Neo4j グラフ + hnswlib ベクトルストア に格納
  6. クエリ → LLM キーワード抽出 → dual-level retrieval
     - low-level: エンティティ名でベクトル検索 → グラフ隣接ノード取得
     - high-level: トピックキーワードでベクトル検索 → 関連サブグラフ取得
  7. 検索結果をコンテキストとして LLM で回答生成

依存:
  pip install neo4j requests sentence-transformers hnswlib numpy

Qwen API なしでも動作（ルールベースにフォールバック）
"""

import os
import json
import re
import hashlib
import time
import numpy as np
from collections import defaultdict
from typing import Optional
from neo4j import GraphDatabase
import requests

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

# LLM 設定（優先順位: Ollama → Qwen API → None）
# Ollama: ローカル、無料、API キー不要
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1/chat/completions")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")

# Qwen API: クラウド、無料枠あり
QWEN_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "YOUR_API_KEY_HERE")
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
QWEN_MODEL = "qwen-plus"

# Neo4j
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_AUTH = (
    os.environ.get("NEO4J_USER", "neo4j"),
    os.environ.get("NEO4J_PASS", "unko1234"),
)

EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIM = 384
VECTOR_INDEX_PATH = "./lightrag_vectors"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

os.makedirs(VECTOR_INDEX_PATH, exist_ok=True)

# LLM バックエンドの自動検出
_llm_backend = None

def _detect_llm_backend():
    global _llm_backend
    if _llm_backend is not None:
        return _llm_backend

    # 1. Ollama をチェック
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            _llm_backend = "ollama"
            print(f"  LLM: Ollama 検出 (models: {', '.join(models[:5])})")
            return _llm_backend
    except Exception:
        pass

    # 2. Qwen API キーをチェック
    if QWEN_API_KEY != "YOUR_API_KEY_HERE":
        _llm_backend = "qwen"
        print(f"  LLM: Qwen API ({QWEN_MODEL})")
        return _llm_backend

    # 3. LLM なし
    _llm_backend = "none"
    print("  LLM: なし（ルールベースフォールバック）")
    return _llm_backend


# ──────────────────────────────────────────────
# LLM wrapper
# ──────────────────────────────────────────────

def call_llm(system: str, user: str, temperature: float = 0.1) -> Optional[str]:
    backend = _detect_llm_backend()

    if backend == "ollama":
        return _call_ollama(system, user, temperature)
    elif backend == "qwen":
        return _call_qwen(system, user, temperature)
    else:
        return None


def _call_ollama(system: str, user: str, temperature: float) -> Optional[str]:
    try:
        resp = requests.post(OLLAMA_BASE_URL, json={
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "stream": False,
        }, timeout=120)  # Ollama は初回ロードが遅い
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  Ollama error: {e}")
        return None


def _call_qwen(system: str, user: str, temperature: float) -> Optional[str]:
    try:
        resp = requests.post(QWEN_BASE_URL, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {QWEN_API_KEY}",
        }, json={
            "model": QWEN_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": 3000,
        }, timeout=60)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  Qwen error: {e}")
        return None


def parse_json_response(text: str) -> dict:
    if text is None:
        return {}
    text = re.sub(r"```json\s*|```\s*", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


# ──────────────────────────────────────────────
# 1. Embedding model
# ──────────────────────────────────────────────

class EmbeddingModel:
    """sentence-transformers ベースの埋め込みモデル"""

    def __init__(self):
        self.model = None
        self._fallback = False

    def _load(self):
        if self.model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            print(f"  Embedding model: {EMBED_MODEL}")
            self.model = SentenceTransformer(EMBED_MODEL)
        except ImportError:
            print("  sentence-transformers 未インストール → TF-IDF fallback")
            self._fallback = True

    def encode(self, texts: list[str]) -> np.ndarray:
        self._load()
        if self._fallback:
            return self._tfidf_encode(texts)
        return self.model.encode(texts, normalize_embeddings=True)

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def _tfidf_encode(self, texts: list[str]) -> np.ndarray:
        """簡易TF-IDF (フォールバック)"""
        from collections import Counter
        vocab = {}
        for t in texts:
            for w in self._tokenize(t):
                if w not in vocab:
                    vocab[w] = len(vocab)
        dim = min(len(vocab), EMBED_DIM)
        if dim == 0:
            return np.zeros((len(texts), EMBED_DIM), dtype=np.float32)
        vecs = np.zeros((len(texts), EMBED_DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            counts = Counter(self._tokenize(t))
            for w, c in counts.items():
                if w in vocab and vocab[w] < EMBED_DIM:
                    vecs[i, vocab[w]] = c
            norm = np.linalg.norm(vecs[i])
            if norm > 0:
                vecs[i] /= norm
        return vecs

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        tokens = re.findall(r"[a-zA-Z0-9\.\+\#]+|[\u3040-\u9fff]+", text)
        result = []
        for t in tokens:
            if re.match(r"[\u3040-\u9fff]+", t):
                result.extend(t[i:i+2] for i in range(len(t)-1))
            else:
                result.append(t)
        return result


embedder = EmbeddingModel()


# ──────────────────────────────────────────────
# 2. Vector store (hnswlib)
# ──────────────────────────────────────────────

class VectorStore:
    """hnswlib ベースのベクトルストア（フォールバック: numpy brute-force）"""

    def __init__(self, name: str, dim: int = EMBED_DIM):
        self.name = name
        self.dim = dim
        self.vectors = []
        self.metadata = []
        self.index = None
        self._use_hnsw = False

    def _try_init_hnsw(self, max_elements: int):
        try:
            import hnswlib
            self.index = hnswlib.Index(space="cosine", dim=self.dim)
            self.index.init_index(max_elements=max(max_elements, 100),
                                  M=16, ef_construction=200)
            self.index.set_ef(50)
            self._use_hnsw = True
        except ImportError:
            self._use_hnsw = False

    def add(self, vectors: np.ndarray, metadata_list: list[dict]):
        start_id = len(self.vectors)
        for i, (vec, meta) in enumerate(zip(vectors, metadata_list)):
            self.vectors.append(vec.astype(np.float32))
            self.metadata.append(meta)

        if self._use_hnsw and self.index is not None:
            ids = list(range(start_id, start_id + len(vectors)))
            self.index.add_items(vectors.astype(np.float32), ids)
        elif not self._use_hnsw and len(self.vectors) > 0:
            self._try_init_hnsw(len(self.vectors) + 100)
            if self._use_hnsw:
                all_vecs = np.array(self.vectors, dtype=np.float32)
                self.index.add_items(all_vecs, list(range(len(all_vecs))))

    def search(self, query_vec: np.ndarray, k: int = 10) -> list[dict]:
        if len(self.vectors) == 0:
            return []

        if self._use_hnsw and self.index is not None:
            ids, distances = self.index.knn_query(
                query_vec.reshape(1, -1).astype(np.float32), k=min(k, len(self.vectors))
            )
            results = []
            for idx, dist in zip(ids[0], distances[0]):
                if idx < len(self.metadata):
                    results.append({
                        **self.metadata[idx],
                        "_score": float(1 - dist),
                    })
            return results
        else:
            # numpy brute-force
            mat = np.array(self.vectors, dtype=np.float32)
            qv = query_vec.astype(np.float32)
            scores = mat @ qv
            top_k = min(k, len(scores))
            top_ids = np.argsort(-scores)[:top_k]
            return [{
                **self.metadata[i],
                "_score": float(scores[i]),
            } for i in top_ids if scores[i] > 0.1]

    def clear(self):
        self.vectors = []
        self.metadata = []
        self.index = None
        self._use_hnsw = False


# ──────────────────────────────────────────────
# 3. Entity extraction + deduplication
# ──────────────────────────────────────────────

EXTRACT_PROMPT = """テキストからエンティティ(ノード)とリレーション(エッジ)を抽出してください。
スキルシート/求人票のドメインに特化し、以下の型を使用:

ノード型: Person, Skill, Experience, Project, Industry, Location, Certification, Methodology
エッジ型: HAS_SKILL, HAS_EXPERIENCE, USED_IN, RELATED_TO, REQUIRES, LOCATED_IN, WORKS_IN, PREFERS

重要ルール:
- スキル名は正式名称に統一 (TS→TypeScript, k8s→Kubernetes)
- 経験文脈も Experience ノードで抽出 (例: "ECサイト構築" → Experience)
- 業界経験は Industry ノード (例: "金融系" → Industry:金融)
- スキル間の暗黙の関連も RELATED_TO で抽出

JSONのみ出力:
{"entities": [{"id": "...", "type": "...", "name": "...", "description": "説明文"}],
 "relations": [{"source": "...", "target": "...", "type": "...", "description": "関係の説明"}]}"""

DEDUP_PROMPT = """以下のエンティティリストの中から、同一のものを統合してください。
表記揺れ (Spring Boot / SpringBoot / スプリングブート) や
同義語 (ML / 機械学習 / Machine Learning) をマージします。

入力エンティティ:
{entities}

JSONのみ出力:
{"merge_groups": [
  {"canonical_id": "正規ID", "canonical_name": "正規名",
   "merged_ids": ["統合されるID1", "統合されるID2"]}
]}

統合不要ならmerge_groupsを空配列にしてください。"""


def extract_entities_from_text(text: str, doc_type: str) -> dict:
    """Phase 1: LLM でエンティティ抽出"""
    type_label = "スキルシート" if doc_type == "engineer" else "求人票"
    content = call_llm(
        EXTRACT_PROMPT,
        f"以下の{type_label}からエンティティとリレーションを抽出:\n\n{text[:4000]}"
    )
    result = parse_json_response(content)
    if result and result.get("entities"):
        return result
    return _rule_extract(text, doc_type)


def deduplicate_entities(entities: list[dict]) -> dict:
    """Phase 2: LLM でエンティティ重複排除"""
    if len(entities) < 3:
        return {"merge_groups": []}

    entity_summary = "\n".join(
        f"- {e['id']}: {e.get('type', '?')} / {e.get('name', '?')}"
        for e in entities[:50]
    )
    content = call_llm(DEDUP_PROMPT.replace("{entities}", entity_summary), "")
    return parse_json_response(content) or {"merge_groups": []}


def _rule_extract(text: str, doc_type: str) -> dict:
    """ルールベースフォールバック"""
    skill_db = {
        "language": ["Java","Python","C#","TypeScript","JavaScript","Go","Rust",
                     "SQL","Kotlin","Swift","PHP","Ruby","Scala"],
        "framework": ["Spring Boot","React","Next\\.js","Vue\\.js","Angular",
                      "Node\\.js","Express","Django","Flask","FastAPI","\\.NET",
                      "Rails","Laravel","Flutter","jQuery","Tailwind"],
        "cloud": ["AWS","Azure","GCP"],
        "infra": ["Docker","Kubernetes","Terraform","Ansible","Jenkins","Linux","Nginx"],
        "database": ["PostgreSQL","MySQL","SQL Server","Oracle","MongoDB","Redis",
                     "Neo4j","Elasticsearch","BigQuery","Snowflake"],
        "data": ["ETL","Spark","Kafka","Airflow","Pandas","NumPy","Tableau","Power BI"],
        "ai": ["機械学習","深層学習","自然言語処理","画像認識","PyTorch","TensorFlow",
               "scikit-learn","OpenCV","LLM","RAG","生成AI","LangChain","BERT"],
        "management": ["Git","GitHub","JIRA","スクラム","アジャイル"],
    }

    entities, relations = [], []
    seen = set()
    uid = hashlib.md5(text[:80].encode()).hexdigest()[:8]
    main_type = "Person" if doc_type == "engineer" else "Project"

    # name
    name = None
    if doc_type == "engineer":
        m = re.search(r"(?:氏名|名前)[：:\s]*([^\n,、]{2,10})", text)
        if m: name = m.group(1).strip()
    else:
        m = re.search(r"(?:案件名|PJ名|プロジェクト)[：:\s]*([^\n]{3,40})", text)
        if m: name = m.group(1).strip()

    main_id = f"{main_type.lower()}_{uid}"
    entities.append({"id": main_id, "type": main_type,
                     "name": name or (main_type), "description": text[:100]})

    for cat, patterns in skill_db.items():
        for p in patterns:
            clean = p.replace("\\.", ".").replace("\\", "")
            if re.search(r"\b" + p + r"\b|" + p, text, re.IGNORECASE) and clean not in seen:
                seen.add(clean)
                sid = f"skill_{clean.lower().replace(' ','_').replace('.','_')}"
                entities.append({"id": sid, "type": "Skill",
                                "name": clean, "description": f"{cat} skill"})
                rtype = "HAS_SKILL" if doc_type == "engineer" else "REQUIRES"
                relations.append({"source": main_id, "target": sid,
                                 "type": rtype, "description": ""})

    # locations
    for loc in set(re.findall(
        r"(?:東京|大阪|名古屋|横浜|品川|新宿|渋谷|福岡|リモート|フルリモート|在宅)", text)):
        lid = f"loc_{loc}"
        if loc not in seen:
            seen.add(loc)
            entities.append({"id": lid, "type": "Location",
                            "name": loc, "description": ""})
            rtype = "PREFERS" if doc_type == "engineer" else "LOCATED_IN"
            relations.append({"source": main_id, "target": lid,
                             "type": rtype, "description": ""})

    # price, available
    price_min = price_max = None
    m = re.search(r"(\d{2,3})\s*[~〜～ー−]\s*(\d{2,3})\s*万", text)
    if m: price_min, price_max = int(m.group(1)), int(m.group(2))
    else:
        m = re.search(r"(?:単価|希望単価)[：:\s]*(\d{2,3})\s*万", text)
        if m: price_min = int(m.group(1))

    available = None
    m = re.search(r"(?:参画可能|開始)[：:\s]*(即日|20\d{2}[年/]\d{1,2}月?)", text)
    if m: available = m.group(1)

    # store metadata on main entity
    entities[0]["price_min"] = price_min
    entities[0]["price_max"] = price_max
    entities[0]["available"] = available

    return {"entities": entities, "relations": relations}


# ──────────────────────────────────────────────
# 4. LightRAG Core
# ──────────────────────────────────────────────

class LightRAG:
    """LightRAG 本体

    Internal stores:
      - Neo4j: グラフ (entities + relations)
      - entity_store: エンティティのベクトルインデックス
      - relation_store: リレーションのベクトルインデックス
      - chunk_store: 原文チャンクのベクトルインデックス
    """

    def __init__(self):
        self.driver = None
        self.entity_store = VectorStore("entities")
        self.relation_store = VectorStore("relations")
        self.chunk_store = VectorStore("chunks")
        self.entity_map = {}  # id -> {name, type, description, ...}
        self.dedup_map = {}   # old_id -> canonical_id

    def connect(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        self.driver.verify_connectivity()
        self._ensure_constraints()
        print("  LightRAG: Neo4j connected")

    def _ensure_constraints(self):
        with self.driver.session() as s:
            try:
                s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:LR_Entity) REQUIRE n.eid IS UNIQUE")
            except Exception:
                pass

    # ──── Insert pipeline ────

    def insert(self, text: str, doc_type: str, doc_id: str = None) -> dict:
        """テキストを LightRAG に投入

        1. チャンク分割
        2. 各チャンクからエンティティ＆リレーション抽出
        3. 重複排除
        4. ベクトル埋め込み
        5. Neo4j + ベクトルストアに格納
        """
        doc_id = doc_id or hashlib.md5(text[:200].encode()).hexdigest()[:12]
        stats = {"chunks": 0, "entities": 0, "relations": 0, "dedup_merges": 0}

        # Step 1: Chunk
        chunks = self._chunk_text(text)
        stats["chunks"] = len(chunks)

        # Step 2: Extract from each chunk
        all_entities = []
        all_relations = []

        for chunk in chunks:
            extracted = extract_entities_from_text(chunk, doc_type)
            all_entities.extend(extracted.get("entities", []))
            all_relations.extend(extracted.get("relations", []))

        if not all_entities:
            return stats

        # Step 3: Dedup
        dedup = deduplicate_entities(all_entities)
        for group in dedup.get("merge_groups", []):
            canonical = group["canonical_id"]
            for old_id in group.get("merged_ids", []):
                self.dedup_map[old_id] = canonical
            stats["dedup_merges"] += len(group.get("merged_ids", []))

        # Apply dedup to relations
        for rel in all_relations:
            rel["source"] = self.dedup_map.get(rel["source"], rel["source"])
            rel["target"] = self.dedup_map.get(rel["target"], rel["target"])

        # Deduplicate entity list
        seen_ids = set()
        unique_entities = []
        for e in all_entities:
            eid = self.dedup_map.get(e["id"], e["id"])
            e["id"] = eid
            if eid not in seen_ids:
                seen_ids.add(eid)
                unique_entities.append(e)
        all_entities = unique_entities

        # Step 4: Embed
        entity_texts = [
            f"{e.get('type','')}: {e.get('name','')}. {e.get('description','')}"
            for e in all_entities
        ]
        entity_vecs = embedder.encode(entity_texts)

        relation_texts = [
            f"{r.get('source','')} -[{r.get('type','')}]-> {r.get('target','')}. {r.get('description','')}"
            for r in all_relations
        ]
        if relation_texts:
            relation_vecs = embedder.encode(relation_texts)
        else:
            relation_vecs = np.array([])

        chunk_vecs = embedder.encode(chunks)

        # Step 5: Store

        # 5a: Vector store
        entity_meta = [{"eid": e["id"], "name": e.get("name",""),
                        "type": e.get("type",""), "doc_id": doc_id}
                       for e in all_entities]
        self.entity_store.add(entity_vecs, entity_meta)

        if len(relation_vecs) > 0:
            rel_meta = [{"source": r["source"], "target": r["target"],
                        "type": r["type"], "doc_id": doc_id}
                       for r in all_relations]
            self.relation_store.add(relation_vecs, rel_meta)

        chunk_meta = [{"text": c, "doc_id": doc_id, "doc_type": doc_type,
                       "idx": i} for i, c in enumerate(chunks)]
        self.chunk_store.add(chunk_vecs, chunk_meta)

        # 5b: Neo4j
        self._store_to_neo4j(all_entities, all_relations, doc_id, doc_type)

        # 5c: Local map
        for e in all_entities:
            self.entity_map[e["id"]] = e

        stats["entities"] = len(all_entities)
        stats["relations"] = len(all_relations)
        return stats

    def _chunk_text(self, text: str) -> list[str]:
        """テキストをチャンク分割"""
        text = text.strip()
        if len(text) <= CHUNK_SIZE:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            chunk = text[start:end]
            # 文の途中で切れないように調整
            if end < len(text):
                for sep in ["\n\n", "\n", "。", ".", " "]:
                    last = chunk.rfind(sep)
                    if last > CHUNK_SIZE // 2:
                        end = start + last + len(sep)
                        chunk = text[start:end]
                        break
            chunks.append(chunk.strip())
            start = end - CHUNK_OVERLAP
        return [c for c in chunks if c]

    def _store_to_neo4j(self, entities, relations, doc_id, doc_type):
        with self.driver.session() as sess:
            for e in entities:
                props = {
                    "eid": e["id"], "name": e.get("name", ""),
                    "type": e.get("type", ""), "description": e.get("description", ""),
                    "doc_id": doc_id, "doc_type": doc_type,
                }
                # optional properties
                for k in ["price_min", "price_max", "available", "category"]:
                    if e.get(k) is not None:
                        props[k] = e[k]

                sess.run("""
                    MERGE (n:LR_Entity {eid: $eid})
                    SET n += $props
                """, eid=e["id"], props=props)

            for r in relations:
                sess.run(f"""
                    MATCH (a:LR_Entity {{eid: $src}})
                    MATCH (b:LR_Entity {{eid: $tgt}})
                    MERGE (a)-[rel:{r['type']}]->(b)
                    SET rel.description = $desc, rel.doc_id = $doc_id
                """, src=r["source"], tgt=r["target"],
                     desc=r.get("description", ""), doc_id=doc_id)

    # ──── Retrieval pipeline (dual-level) ────

    KEYWORD_PROMPT = """クエリからキーワードを2種類抽出してください。

low_level: 具体的なエンティティ名 (人名、スキル名、会社名、技術名)
high_level: 抽象的なトピック/テーマ (業界、分野、技術領域、経験の種類)

JSONのみ:
{"low_level": ["Spring Boot", "AWS", "佐藤健一"],
 "high_level": ["バックエンド開発", "クラウドネイティブ", "EC系システム"]}"""

    def query(self, query_text: str, mode: str = "hybrid", top_k: int = 10) -> dict:
        """LightRAG クエリ

        mode:
          "low"    - エンティティレベル検索のみ
          "high"   - トピックレベル検索のみ
          "hybrid" - 両方を組み合わせ（推奨）

        Returns:
          {"context": str, "entities": list, "answer": str}
        """
        # Step 1: キーワード抽出
        keywords = self._extract_keywords(query_text)
        low_keywords = keywords.get("low_level", [])
        high_keywords = keywords.get("high_level", [])

        context_parts = []
        retrieved_entities = set()

        # Step 2a: Low-level retrieval (entity-specific)
        if mode in ("low", "hybrid") and low_keywords:
            for kw in low_keywords:
                kw_vec = embedder.encode_single(kw)

                # ベクトル検索でエンティティ取得
                hits = self.entity_store.search(kw_vec, k=top_k)
                for hit in hits:
                    eid = hit.get("eid", "")
                    if eid and eid not in retrieved_entities:
                        retrieved_entities.add(eid)
                        # グラフ隣接ノードも取得
                        neighbors = self._get_neighbors(eid)
                        context_parts.append(
                            f"[Entity: {hit.get('name','')} ({hit.get('type','')})] "
                            f"score={hit.get('_score',0):.2f}\n"
                            f"  Neighbors: {', '.join(neighbors[:8])}"
                        )

        # Step 2b: High-level retrieval (topic-based)
        if mode in ("high", "hybrid") and high_keywords:
            for kw in high_keywords:
                kw_vec = embedder.encode_single(kw)

                # リレーションベクトル検索
                rel_hits = self.relation_store.search(kw_vec, k=top_k)
                for hit in rel_hits:
                    src = hit.get("source", "")
                    tgt = hit.get("target", "")
                    context_parts.append(
                        f"[Relation: {src} -[{hit.get('type','')}]-> {tgt}] "
                        f"score={hit.get('_score',0):.2f}"
                    )
                    retrieved_entities.add(src)
                    retrieved_entities.add(tgt)

                # チャンクベクトル検索
                chunk_hits = self.chunk_store.search(kw_vec, k=3)
                for hit in chunk_hits:
                    text_snippet = hit.get("text", "")[:200]
                    context_parts.append(f"[Chunk] {text_snippet}")

        # Step 3: グラフからサブグラフ情報を補強
        if retrieved_entities:
            graph_context = self._get_subgraph_context(list(retrieved_entities)[:20])
            context_parts.append(graph_context)

        context = "\n".join(context_parts)

        # Step 4: LLM 回答生成
        answer = self._generate_answer(query_text, context)

        return {
            "context": context,
            "entities": list(retrieved_entities),
            "keywords": keywords,
            "answer": answer,
        }

    def _extract_keywords(self, query: str) -> dict:
        content = call_llm(self.KEYWORD_PROMPT, f"クエリ: {query}")
        result = parse_json_response(content)
        if result and (result.get("low_level") or result.get("high_level")):
            return result
        # フォールバック: 単純トークン化
        tokens = re.findall(r"[A-Za-z\.\+\#]+|[\u3040-\u9fff]{2,}", query)
        return {"low_level": tokens[:5], "high_level": [query]}

    def _get_neighbors(self, eid: str) -> list[str]:
        """グラフから1ホップ隣接ノードを取得"""
        if self.driver is None:
            return []
        with self.driver.session() as sess:
            result = sess.run("""
                MATCH (n:LR_Entity {eid: $eid})-[r]-(m:LR_Entity)
                RETURN m.name AS name, m.type AS type, type(r) AS rel
                LIMIT 15
            """, eid=eid).data()
        return [f"{r['name']}({r['rel']})" for r in result]

    def _get_subgraph_context(self, eids: list[str]) -> str:
        """複数エンティティ周辺のサブグラフをテキスト化"""
        if self.driver is None:
            return ""
        with self.driver.session() as sess:
            result = sess.run("""
                MATCH (n:LR_Entity)-[r]-(m:LR_Entity)
                WHERE n.eid IN $eids OR m.eid IN $eids
                RETURN n.name AS src_name, n.type AS src_type,
                       type(r) AS rel, r.description AS rel_desc,
                       m.name AS tgt_name, m.type AS tgt_type
                LIMIT 30
            """, eids=eids).data()

        if not result:
            return ""

        lines = ["[Graph subgraph]"]
        for r in result:
            lines.append(
                f"  ({r['src_name']}:{r['src_type']}) "
                f"-[{r['rel']}]-> "
                f"({r['tgt_name']}:{r['tgt_type']})"
            )
        return "\n".join(lines)

    ANSWER_PROMPT = """あなたはIT人材マッチングの専門家です。
以下のナレッジグラフから取得した情報をもとに、質問に回答してください。

情報がグラフに含まれていない場合は「情報が不足しています」と回答してください。
マッチング評価の場合は、スコア(0-100)、ランク(最有力/有力/候補/要検討)、
理由、強み、不足点を含めてください。"""

    def _generate_answer(self, query: str, context: str) -> str:
        content = call_llm(
            self.ANSWER_PROMPT,
            f"検索結果:\n{context}\n\n質問: {query}"
        )
        return content or "LLM 未設定のため回答を生成できません。検索結果を参照してください。"

    # ──── Matching (skill matching specific) ────

    def match_all(self) -> list[dict]:
        """全案件 × 全要員のマッチング"""
        with self.driver.session() as sess:
            projects = sess.run("""
                MATCH (p:LR_Entity) WHERE p.type = 'Project'
                RETURN p.eid AS eid, p.name AS name
            """).data()

            engineers = sess.run("""
                MATCH (e:LR_Entity) WHERE e.type = 'Person'
                RETURN e.eid AS eid, e.name AS name
            """).data()

        if not projects or not engineers:
            return []

        all_matches = []
        for proj in projects:
            query = f"案件「{proj['name']}」に最適な要員を評価してください"
            result = self.query(query, mode="hybrid")

            # LLM の回答をパースしてマッチ結果を構造化
            matches = self._parse_matching_answer(
                result["answer"], proj, engineers
            )
            all_matches.extend(matches)

        all_matches.sort(key=lambda x: x.get("score", 0), reverse=True)
        return all_matches

    def match_project(self, project_eid: str) -> list[dict]:
        """特定案件に対するマッチング"""
        with self.driver.session() as sess:
            proj = sess.run("""
                MATCH (p:LR_Entity {eid: $eid})
                OPTIONAL MATCH (p)-[:REQUIRES]->(s:LR_Entity)
                RETURN p.name AS name, p.eid AS eid,
                       p.price_min AS price_min, p.price_max AS price_max,
                       collect(s.name) AS skills
            """, eid=project_eid).single()

            engineers = sess.run("""
                MATCH (e:LR_Entity) WHERE e.type = 'Person'
                OPTIONAL MATCH (e)-[:HAS_SKILL]->(s:LR_Entity)
                RETURN e.eid AS eid, e.name AS name,
                       e.price_min AS price, e.available AS available,
                       collect(s.name) AS skills
            """).data()

        if not proj or not engineers:
            return []

        proj_skills = set(s for s in (proj["skills"] or []) if s)
        if not proj_skills:
            return []

        # 必須スキルの埋め込みを事前計算
        proj_skill_vecs = {}
        for ps in proj_skills:
            proj_skill_vecs[ps] = embedder.encode_single(ps)

        results = []

        for eng in engineers:
            eng_skills = set(s for s in (eng["skills"] or []) if s)

            # 直接マッチ（名前完全一致）
            direct = proj_skills & eng_skills

            # ベクトル類似度による間接マッチ
            # 各必須スキルに対して最も近い要員スキルを1つだけ選ぶ（1:1）
            missing = proj_skills - direct
            used_eng_skills = set()
            indirect = []

            if missing and (eng_skills - direct):
                # 要員スキルの埋め込みを事前計算
                eng_skill_vecs = {}
                for es in (eng_skills - direct):
                    eng_skill_vecs[es] = embedder.encode_single(es)

                # 全ペアの類似度を計算してから、貪欲に最良マッチを選択
                pairs = []
                for ps in missing:
                    for es, es_vec in eng_skill_vecs.items():
                        sim = float(np.dot(proj_skill_vecs[ps], es_vec))
                        pairs.append((sim, ps, es))

                # 類似度降順でソートし、1:1 で割り当て
                pairs.sort(key=lambda x: -x[0])
                matched_proj = set()
                matched_eng = set()

                for sim, ps, es in pairs:
                    if sim < 0.75:  # 閾値: 0.75
                        break
                    if ps in matched_proj or es in matched_eng:
                        continue
                    matched_proj.add(ps)
                    matched_eng.add(es)
                    indirect.append({
                        "required": ps, "has": es,
                        "similarity": round(sim, 2),
                    })

            # スコア計算
            # 直接マッチ: 1.0点、間接マッチ: similarity × 0.6 点
            skill_points = len(direct)
            for ind in indirect:
                skill_points += ind["similarity"] * 0.6
            skill_score = skill_points / len(proj_skills)
            skill_score = min(skill_score, 1.0)

            # 単価スコア
            price_score = 0.5
            eng_price = eng.get("price")
            p_min = proj.get("price_min")
            p_max = proj.get("price_max")
            if eng_price and p_min and p_max:
                if p_min <= eng_price <= p_max:
                    price_score = 1.0
                elif eng_price < p_min:
                    price_score = max(0.2, 1.0 - (p_min - eng_price) / 20)
                else:
                    price_score = max(0.2, 1.0 - (eng_price - p_max) / 20)

            total = int(skill_score * 60 + price_score * 20 + 0.5 * 20)
            total = min(total, 100)
            rank = ("最有力" if total >= 80 else "有力" if total >= 60
                    else "候補" if total >= 40 else "要検討")

            truly_missing = list(
                proj_skills - direct - {i["required"] for i in indirect}
            )

            results.append({
                "project_id": project_eid,
                "project_name": proj["name"],
                "engineer_id": eng["eid"],
                "engineer_name": eng["name"],
                "score": total,
                "rank": rank,
                "direct_matches": list(direct),
                "indirect_matches": indirect,
                "missing_skills": truly_missing,
                "engineer_price": eng_price,
                "engineer_available": eng.get("available"),
                "skill_score": int(skill_score * 100),
                "price_score": int(price_score * 100),
                "location_score": 50,
                "method": "lightrag",
            })

        results.sort(key=lambda x: x["score"], reverse=True)

        # LLM reasoning (上位のみ)
        if results[:3]:
            self._add_llm_reasoning(results[:3], proj["name"], proj_skills)

        return results

    def _add_llm_reasoning(self, top_matches, proj_name, proj_skills):
        """上位マッチにLLM推論の理由を追加"""
        summary = f"案件「{proj_name}」(必須: {', '.join(proj_skills)})\n\n"
        for m in top_matches:
            summary += (f"- {m['engineer_name']}: "
                       f"直接={', '.join(m['direct_matches'])} / "
                       f"間接={len(m['indirect_matches'])}件 / "
                       f"不足={', '.join(m['missing_skills'])}\n")

        content = call_llm(
            "各マッチングについて1-2文の理由を述べてください。JSONで: "
            '{"reasons": [{"name": "要員名", "reason": "理由"}]}',
            summary
        )
        reasons = parse_json_response(content)
        for r in reasons.get("reasons", []):
            for m in top_matches:
                if m["engineer_name"] == r.get("name"):
                    m["reasoning"] = r.get("reason", "")

    def _parse_matching_answer(self, answer, proj, engineers):
        """LLM 回答からマッチング結果をパース（ベストエフォート）"""
        matches = []
        for eng in engineers:
            score = 50  # default
            if eng["name"] in answer:
                if "最有力" in answer: score = 90
                elif "有力" in answer: score = 75
                elif "候補" in answer: score = 55

            matches.append({
                "project_id": proj["eid"],
                "project_name": proj["name"],
                "engineer_id": eng["eid"],
                "engineer_name": eng["name"],
                "score": score,
                "rank": ("最有力" if score >= 80 else "有力" if score >= 60
                        else "候補" if score >= 40 else "要検討"),
                "reasoning": "",
                "method": "lightrag",
            })
        return matches

    # ──── Clear ────

    def clear(self):
        self.entity_store.clear()
        self.relation_store.clear()
        self.chunk_store.clear()
        self.entity_map.clear()
        self.dedup_map.clear()
        if self.driver:
            with self.driver.session() as sess:
                sess.run("MATCH (n:LR_Entity) DETACH DELETE n")

    def stats(self) -> dict:
        s = {"entities_vectorized": len(self.entity_store.vectors),
             "relations_vectorized": len(self.relation_store.vectors),
             "chunks_vectorized": len(self.chunk_store.vectors)}
        if self.driver:
            with self.driver.session() as sess:
                r = sess.run("""
                    OPTIONAL MATCH (n:LR_Entity)
                    WITH count(n) AS nodes
                    OPTIONAL MATCH (:LR_Entity)-[r]->(:LR_Entity)
                    RETURN nodes, count(r) AS edges
                """).single()
                s["neo4j_nodes"] = r["nodes"]
                s["neo4j_edges"] = r["edges"]
        return s


# ──────────────────────────────────────────────
# 5. Flask integration
# ──────────────────────────────────────────────

def register_lightrag_routes(app):
    from flask import jsonify, request

    rag = LightRAG()

    @app.before_request
    def ensure_connected():
        if rag.driver is None:
            try:
                rag.connect()
            except Exception as e:
                pass

    @app.route("/api/lightrag/insert", methods=["POST"])
    def lr_insert():
        data = request.json
        text = data.get("text", "")
        doc_type = data.get("type", "engineer")
        if not text.strip():
            return jsonify({"error": "empty text"}), 400
        doc_id = hashlib.md5(text[:200].encode()).hexdigest()[:12]
        stats = rag.insert(text, doc_type, doc_id)
        return jsonify({"doc_id": doc_id, "stats": stats})

    @app.route("/api/lightrag/query", methods=["POST"])
    def lr_query():
        data = request.json
        query_text = data.get("query", "")
        mode = data.get("mode", "hybrid")
        if not query_text.strip():
            return jsonify({"error": "empty query"}), 400
        result = rag.query(query_text, mode=mode)
        return jsonify(result)

    @app.route("/api/lightrag/matching", methods=["GET"])
    def lr_matching():
        project_id = request.args.get("project_id", "")
        if project_id:
            matches = rag.match_project(project_id)
        else:
            # 全案件
            with rag.driver.session() as sess:
                projects = sess.run("""
                    MATCH (p:LR_Entity) WHERE p.type = 'Project'
                    RETURN p.eid AS eid
                """).data()
            all_matches = []
            for p in projects:
                all_matches.extend(rag.match_project(p["eid"]))
            all_matches.sort(key=lambda x: x["score"], reverse=True)
            matches = all_matches
        return jsonify(matches)

    @app.route("/api/lightrag/stats", methods=["GET"])
    def lr_stats():
        return jsonify(rag.stats())

    @app.route("/api/lightrag/clear", methods=["POST"])
    def lr_clear():
        rag.clear()
        return jsonify({"status": "cleared"})

    print("  LightRAG エンドポイント登録:")
    print("    POST /api/lightrag/insert")
    print("    POST /api/lightrag/query")
    print("    GET  /api/lightrag/matching")
    return rag


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  LightRAG Skill Matching - Test")
    print("=" * 60)

    rag = LightRAG()
    rag.connect()

    # テストデータ
    eng1 = """氏名：佐藤 健一
Java/Spring Boot を中心に8年のバックエンド開発経験。
直近3年はAWS上でマイクロサービス設計・構築に従事。
EC系サービスのAPI設計、決済連携の実績あり。
Docker/Kubernetes でのCI/CD構築も経験。PostgreSQL, Redis。
希望単価：80万円、勤務地：東京・リモート可、参画可能：2026年5月"""

    eng2 = """氏名：鈴木 麻衣
React/TypeScript でのフロントエンド開発5年。Next.js, Node.js。
SPA設計、コンポーネント設計、デザインシステム構築が得意。
AWS (S3, CloudFront) でのデプロイ経験あり。
希望単価：78万円、勤務地：フルリモート、参画可能：即日"""

    proj1 = """案件名：大手小売向けEC基盤刷新
Spring BootベースのAPI Gateway + マイクロサービス設計・開発。
AWS ECS/EKS上での運用前提。
必須スキル：Java, Spring Boot, AWS
歓迎：Docker, Kubernetes, マイクロサービス設計経験
単価：75〜85万円、勤務地：東京・リモート併用、開始：2026年5月"""

    proj2 = """案件名：通信会社向けWebアプリ開発
顧客向けポータルの新規開発。React/TypeScript。
必須スキル：React, TypeScript, Node.js
歓迎：Next.js, AWS
単価：70〜85万円、勤務地：新宿・フルリモート可、開始：2026年5月"""

    print("\n[1] Insert engineers...")
    print(f"  eng1: {rag.insert(eng1, 'engineer', 'eng_sato')}")
    print(f"  eng2: {rag.insert(eng2, 'engineer', 'eng_suzuki')}")

    print("\n[2] Insert projects...")
    print(f"  proj1: {rag.insert(proj1, 'project', 'proj_ec')}")
    print(f"  proj2: {rag.insert(proj2, 'project', 'proj_web')}")

    print(f"\n[3] Stats: {rag.stats()}")

    print("\n[4] Query test...")
    result = rag.query("Java と Spring Boot の経験がある要員は？", mode="hybrid")
    print(f"  Keywords: {result['keywords']}")
    print(f"  Entities found: {len(result['entities'])}")
    print(f"  Answer: {result['answer'][:200]}")

    print("\n[5] Matching...")
    matches = rag.match_project("proj_ec") or rag.match_project("project_ec")
    if not matches:
        # try finding project IDs
        with rag.driver.session() as s:
            pids = s.run("MATCH (p:LR_Entity) WHERE p.type='Project' RETURN p.eid, p.name").data()
            print(f"  Available projects: {pids}")
            if pids:
                matches = rag.match_project(pids[0]["p.eid"])

    for m in matches[:5]:
        print(f"  {m['engineer_name']}: {m['score']}% ({m['rank']}) "
              f"direct={m.get('direct_matches',[])} "
              f"indirect={len(m.get('indirect_matches',[]))}")

    print("\n  Done!")