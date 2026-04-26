"""
Skill Matching Backend
══════════════════════
Flask + Qwen API + Neo4j グラフマイニング

機能:
  1. Qwen API でスキルシート/求人票からエンティティ抽出
  2. 抽出結果を Neo4j ナレッジグラフに投入
  3. グラフマイニング（Jaccard, 2ホップ RELATED_TO, 勤務地 NEARBY）でマッチング

起動:
  python app_qwen.py

API:
  POST /api/extract          - テキストからエンティティ抽出 (Qwen API)
  POST /api/upload           - ファイルアップロード + 抽出
  POST /api/graph/register   - 抽出済みデータをグラフに登録
  GET  /api/matching          - グラフマイニングでマッチング実行
  GET  /api/graph/stats       - グラフ統計
  POST /api/clear             - グラフクリア
"""

import os
import json
import re
import traceback
from flask import Flask, jsonify, request
from flask_cors import CORS
from neo4j import GraphDatabase
import requests

app = Flask(__name__)
CORS(app)

# ──────────────────────────────────────────────
# 設定
# ──────────────────────────────────────────────

# LLM 設定（優先順位: Ollama → Qwen API → ルールベース）
# Ollama: ローカル、無料、API キー不要
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1/chat/completions")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")

# Qwen API (DashScope OpenAI 互換エンドポイント)
QWEN_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "YOUR_API_KEY_HERE")
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
QWEN_MODEL = "qwen-plus"

def _check_ollama():
    """Ollama が起動しているかチェック"""
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False

_ollama_available = None
def is_ollama_available():
    global _ollama_available
    if _ollama_available is None:
        _ollama_available = _check_ollama()
    return _ollama_available

# Neo4j
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_AUTH = (
    os.environ.get("NEO4J_USER", "neo4j"),
    os.environ.get("NEO4J_PASS", ""),
)

driver = None

def get_driver():
    global driver
    if driver is None:
        driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        driver.verify_connectivity()
        print(f"  Neo4j 接続成功: {NEO4J_URI}")
        _ensure_constraints()
    return driver


def _ensure_constraints():
    """初回起動時に制約とスキル知識グラフを構築"""
    with get_driver().session() as sess:
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Skill) REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Engineer) REQUIRE e.eid IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Project) REQUIRE p.pid IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (l:Location) REQUIRE l.name IS UNIQUE",
        ]
        for c in constraints:
            try:
                sess.run(c)
            except Exception:
                pass

        # スキル知識グラフ (RELATED_TO) がなければ構築
        count = sess.run("MATCH (s:Skill) RETURN count(s) AS c").single()["c"]
        if count == 0:
            print("  スキル知識グラフを初期構築中...")
            _build_skill_knowledge_graph(sess)
            print("  スキル知識グラフ構築完了")


def _build_skill_knowledge_graph(sess):
    """スキルノードと RELATED_TO リレーションを作成"""
    skills = [
        ("Java", "language"), ("Python", "language"), ("C#", "language"),
        ("TypeScript", "language"), ("JavaScript", "language"), ("Go", "language"),
        ("Rust", "language"), ("SQL", "language"), ("Kotlin", "language"),
        ("Swift", "language"), ("PHP", "language"), ("Ruby", "language"),
        ("Scala", "language"), ("R", "language"), ("Dart", "language"),
        ("Spring Boot", "framework"), ("Spring", "framework"),
        ("React", "framework"), ("Next.js", "framework"), ("Vue.js", "framework"),
        ("Angular", "framework"), ("Node.js", "framework"), ("Express", "framework"),
        ("Django", "framework"), ("Flask", "framework"), ("FastAPI", "framework"),
        (".NET", "framework"), ("Rails", "framework"), ("Laravel", "framework"),
        ("Nuxt", "framework"), ("Svelte", "framework"), ("Flutter", "framework"),
        ("jQuery", "framework"), ("Tailwind", "framework"), ("Bootstrap", "framework"),
        ("AWS", "cloud"), ("Azure", "cloud"), ("GCP", "cloud"),
        ("Docker", "infra"), ("Kubernetes", "infra"), ("Terraform", "infra"),
        ("Ansible", "infra"), ("Jenkins", "infra"), ("GitHub Actions", "infra"),
        ("Linux", "infra"), ("Nginx", "infra"),
        ("PostgreSQL", "database"), ("MySQL", "database"), ("SQL Server", "database"),
        ("Oracle", "database"), ("MongoDB", "database"), ("Redis", "database"),
        ("DynamoDB", "database"), ("Neo4j", "database"), ("Elasticsearch", "database"),
        ("BigQuery", "database"), ("Snowflake", "database"), ("SQLite", "database"),
        ("ETL", "data"), ("Spark", "data"), ("Kafka", "data"),
        ("Airflow", "data"), ("Tableau", "data"), ("Power BI", "data"),
        ("Pandas", "data"), ("NumPy", "data"), ("dbt", "data"),
        ("機械学習", "ai"), ("深層学習", "ai"), ("自然言語処理", "ai"),
        ("画像認識", "ai"), ("PyTorch", "ai"), ("TensorFlow", "ai"),
        ("scikit-learn", "ai"), ("OpenCV", "ai"), ("LLM", "ai"),
        ("RAG", "ai"), ("生成AI", "ai"), ("LangChain", "ai"),
        ("BERT", "ai"), ("Transformer", "ai"),
        ("Git", "management"), ("GitHub", "management"), ("JIRA", "management"),
        ("Confluence", "management"), ("Backlog", "management"),
        ("スクラム", "management"), ("アジャイル", "management"),
    ]
    for name, cat in skills:
        sess.run(
            "MERGE (s:Skill {name: $name}) SET s.category = $cat",
            name=name, cat=cat,
        )

    # スキル間の関連性
    relations = [
        ("Java", "Spring Boot", 0.9), ("Java", "Spring", 0.85),
        ("Python", "Django", 0.7), ("Python", "Flask", 0.7),
        ("Python", "FastAPI", 0.7), ("Python", "ETL", 0.6),
        ("Python", "機械学習", 0.8), ("Python", "PyTorch", 0.7),
        ("Python", "TensorFlow", 0.7), ("Python", "Pandas", 0.8),
        ("Python", "NumPy", 0.8), ("Python", "scikit-learn", 0.7),
        ("TypeScript", "React", 0.85), ("TypeScript", "Next.js", 0.8),
        ("TypeScript", "Node.js", 0.7), ("TypeScript", "Vue.js", 0.7),
        ("TypeScript", "Angular", 0.7),
        ("JavaScript", "TypeScript", 0.9), ("JavaScript", "React", 0.8),
        ("JavaScript", "Node.js", 0.8), ("JavaScript", "Vue.js", 0.8),
        ("JavaScript", "jQuery", 0.6),
        ("C#", ".NET", 0.95), ("C#", "SQL Server", 0.7),
        (".NET", "SQL Server", 0.6), (".NET", "Azure", 0.6),
        ("Ruby", "Rails", 0.9), ("PHP", "Laravel", 0.85),
        ("Dart", "Flutter", 0.9), ("Swift", "iOS", 0.9),
        ("Kotlin", "Android", 0.85), ("Kotlin", "Spring Boot", 0.5),
        ("AWS", "Docker", 0.6), ("AWS", "Kubernetes", 0.5),
        ("AWS", "Terraform", 0.6), ("AWS", "DynamoDB", 0.6),
        ("Azure", "Docker", 0.5), ("Azure", "Kubernetes", 0.5),
        ("Azure", ".NET", 0.6), ("Azure", "ETL", 0.5),
        ("GCP", "Docker", 0.5), ("GCP", "Kubernetes", 0.6),
        ("GCP", "BigQuery", 0.7),
        ("Docker", "Kubernetes", 0.8), ("Docker", "Linux", 0.6),
        ("Kubernetes", "Terraform", 0.5),
        ("React", "Next.js", 0.85), ("Vue.js", "Nuxt", 0.85),
        ("SQL", "PostgreSQL", 0.7), ("SQL", "MySQL", 0.7),
        ("SQL", "SQL Server", 0.7), ("SQL", "Oracle", 0.6),
        ("ETL", "SQL", 0.6), ("ETL", "Spark", 0.6),
        ("ETL", "Airflow", 0.6), ("ETL", "Kafka", 0.5),
        ("機械学習", "PyTorch", 0.8), ("機械学習", "TensorFlow", 0.8),
        ("機械学習", "scikit-learn", 0.8), ("機械学習", "深層学習", 0.8),
        ("深層学習", "PyTorch", 0.9), ("深層学習", "TensorFlow", 0.9),
        ("自然言語処理", "BERT", 0.8), ("自然言語処理", "Transformer", 0.8),
        ("自然言語処理", "LLM", 0.8),
        ("LLM", "RAG", 0.7), ("LLM", "LangChain", 0.7),
        ("LLM", "生成AI", 0.9),
        ("Spring Boot", "AWS", 0.4), ("Spring Boot", "PostgreSQL", 0.5),
        ("Node.js", "Express", 0.8), ("Node.js", "MongoDB", 0.5),
        ("Git", "GitHub", 0.9), ("Git", "GitHub Actions", 0.6),
        ("Tableau", "Power BI", 0.7), ("Pandas", "NumPy", 0.8),
    ]
    for s1, s2, w in relations:
        sess.run("""
            MATCH (a:Skill {name: $s1}), (b:Skill {name: $s2})
            MERGE (a)-[r:RELATED_TO]->(b) SET r.weight = $w
            MERGE (b)-[r2:RELATED_TO]->(a) SET r2.weight = $w
        """, s1=s1, s2=s2, w=w)

    # 勤務地ノード
    locations = [
        ("東京", "関東"), ("品川", "関東"), ("新宿", "関東"),
        ("渋谷", "関東"), ("横浜", "関東"), ("千葉", "関東"),
        ("埼玉", "関東"), ("大阪", "関西"), ("京都", "関西"),
        ("名古屋", "中部"), ("福岡", "九州"), ("札幌", "北海道"),
        ("仙台", "東北"), ("広島", "中国"), ("神戸", "関西"),
        ("リモート", "全国"), ("フルリモート", "全国"), ("在宅", "全国"),
    ]
    for name, region in locations:
        sess.run(
            "MERGE (l:Location {name: $name}) SET l.region = $region",
            name=name, region=region,
        )

    # 近接リレーション
    nearby = [
        ("東京", "品川", 0.95), ("東京", "新宿", 0.95), ("東京", "渋谷", 0.95),
        ("東京", "横浜", 0.7), ("東京", "千葉", 0.6), ("東京", "埼玉", 0.6),
        ("品川", "新宿", 0.8), ("品川", "渋谷", 0.85), ("品川", "横浜", 0.6),
        ("新宿", "渋谷", 0.9), ("大阪", "京都", 0.7), ("大阪", "神戸", 0.7),
        ("リモート", "フルリモート", 1.0), ("リモート", "在宅", 1.0),
        ("フルリモート", "在宅", 1.0),
    ]
    # リモートは全ての場所と近い
    for loc_name, _ in locations:
        if loc_name not in ("リモート", "フルリモート", "在宅"):
            nearby.append(("リモート", loc_name, 0.9))
            nearby.append(("フルリモート", loc_name, 1.0))
            nearby.append(("在宅", loc_name, 0.9))

    for l1, l2, prox in nearby:
        sess.run("""
            MATCH (a:Location {name: $l1}), (b:Location {name: $l2})
            MERGE (a)-[r:NEARBY]->(b) SET r.proximity = $prox
            MERGE (b)-[r2:NEARBY]->(a) SET r2.proximity = $prox
        """, l1=l1, l2=l2, prox=prox)


# ──────────────────────────────────────────────
# Qwen API エンティティ抽出
# ──────────────────────────────────────────────

EXTRACTION_PROMPT = """あなたはITスキルシート・求人票の解析エキスパートです。
与えられたテキストから以下の情報をJSON形式で抽出してください。
レスポンスはJSON**のみ**を返してください。マークダウンのコードブロックや説明は不要です。

{
  "name": "人名または案件名（文字列）",
  "skills": [
    {"name": "スキル名", "category": "language|framework|cloud|infra|database|data|ai|management", "years": 経験年数(わかれば、なければnull)}
  ],
  "price_min": 単価下限(万円、数値、なければnull),
  "price_max": 単価上限(万円、数値、なければnull),
  "locations": ["勤務地1", "勤務地2"],
  "available": "参画可能時期（文字列、なければnull）",
  "remote": "リモート可|フルリモート|常駐|リモート併用|null",
  "experience_years": 総経験年数(数値、なければnull),
  "description": "概要を1〜2文で（文字列）"
}

スキル名は正式名称を使ってください（例: JavaScript → JavaScript, TS → TypeScript, k8s → Kubernetes）。"""


def extract_with_qwen(text: str, doc_type: str) -> dict:
    """LLM でエンティティ抽出（Ollama → Qwen API の優先順位）"""
    type_label = "スキルシート（エンジニア情報）" if doc_type == "engineer" else "求人票（案件情報）"

    messages = [
        {"role": "system", "content": EXTRACTION_PROMPT},
        {"role": "user", "content": f"以下の{type_label}テキストからエンティティを抽出してください:\n\n{text[:4000]}"},
    ]

    content = None

    # 1. Ollama を試行
    if is_ollama_available():
        try:
            resp = requests.post(OLLAMA_BASE_URL, json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "temperature": 0.1,
                "stream": False,
            }, timeout=120)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"  Ollama エラー: {e}")

    # 2. Qwen API を試行
    if content is None and QWEN_API_KEY != "YOUR_API_KEY_HERE":
        try:
            resp = requests.post(QWEN_BASE_URL, headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {QWEN_API_KEY}",
            }, json={
                "model": QWEN_MODEL,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 2000,
            }, timeout=30)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"  Qwen API エラー: {e}")

    # 3. パース
    if content:
        try:
            content = re.sub(r"```json\s*", "", content)
            content = re.sub(r"```\s*", "", content).strip()
            result = json.loads(content)
            result["_extraction_method"] = "ollama" if is_ollama_available() else "qwen"
            return result
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  JSON パースエラー: {e}")

    # 4. フォールバック
    return extract_with_rules(text, doc_type)


def extract_with_rules(text: str, doc_type: str) -> dict:
    """ルールベース（正規表現）でエンティティ抽出（フォールバック用）"""
    # スキル検出
    skill_patterns = {
        "language": ["Java", "Python", "C#", "TypeScript", "JavaScript", "Go", "Rust",
                     "SQL", "Kotlin", "Swift", "PHP", "Ruby", "Scala", "R", "Dart"],
        "framework": ["Spring Boot", "Spring", "React", "Next\\.js", "Vue\\.js", "Angular",
                      "Node\\.js", "Express", "Django", "Flask", "FastAPI", "\\.NET",
                      "Rails", "Laravel", "Flutter", "jQuery", "Tailwind", "Bootstrap"],
        "cloud": ["AWS", "Azure", "GCP"],
        "infra": ["Docker", "Kubernetes", "Terraform", "Ansible", "Jenkins", "Linux", "Nginx"],
        "database": ["PostgreSQL", "MySQL", "SQL Server", "Oracle", "MongoDB", "Redis",
                     "DynamoDB", "Neo4j", "Elasticsearch", "BigQuery", "Snowflake"],
        "data": ["ETL", "Spark", "Kafka", "Airflow", "Tableau", "Power BI", "Pandas", "NumPy", "dbt"],
        "ai": ["機械学習", "深層学習", "自然言語処理", "画像認識", "PyTorch", "TensorFlow",
               "scikit-learn", "OpenCV", "LLM", "RAG", "生成AI", "LangChain", "BERT", "Transformer"],
        "management": ["Git", "GitHub", "JIRA", "Confluence", "スクラム", "アジャイル"],
    }

    skills = []
    seen = set()
    for cat, patterns in skill_patterns.items():
        for p in patterns:
            clean_name = p.replace("\\.", ".").replace("\\", "")
            regex = re.compile(r"\b" + p + r"\b|" + p, re.IGNORECASE)
            if regex.search(text) and clean_name not in seen:
                seen.add(clean_name)
                skills.append({"name": clean_name, "category": cat, "years": None})

    # 単価
    price_min, price_max = None, None
    m = re.search(r"(\d{2,3})\s*[~〜～ー−]\s*(\d{2,3})\s*万", text)
    if m:
        price_min, price_max = int(m.group(1)), int(m.group(2))
    else:
        m = re.search(r"(?:単価|希望単価|月額)[：:\s]*(\d{2,3})\s*万", text)
        if m:
            price_min = int(m.group(1))
            price_max = price_min + 10

    # 勤務地
    loc_re = re.findall(
        r"(?:東京|大阪|名古屋|横浜|品川|新宿|渋谷|福岡|札幌|仙台|広島|京都|神戸|千葉|埼玉|リモート|フルリモート|在宅|常駐)",
        text,
    )
    locations = list(dict.fromkeys(loc_re))

    # 名前 / 案件名
    name = None
    if doc_type == "engineer":
        m = re.search(r"(?:氏名|名前|Name)[：:\s]*([^\n\r,、]{2,10})", text)
        if m:
            name = m.group(1).strip()
    else:
        m = re.search(r"(?:案件名|プロジェクト名|案件|PJ名)[：:\s]*([^\n\r]{3,40})", text)
        if m:
            name = m.group(1).strip()

    # リモート
    remote = None
    if "フルリモート" in text:
        remote = "フルリモート"
    elif "リモート併用" in text or "リモート可" in text:
        remote = "リモート併用"
    elif "常駐" in text:
        remote = "常駐"
    elif "リモート" in text or "在宅" in text:
        remote = "リモート可"

    # 参画可能時期
    available = None
    m = re.search(r"(?:参画可能|開始|稼働開始)[：:\s]*(即日|20\d{2}[年/]\d{1,2}月?)", text)
    if m:
        available = m.group(1)

    # 経験年数
    exp = None
    m = re.search(r"(?:経験|実務)[：:\s]*(\d{1,2})\s*年", text)
    if m:
        exp = int(m.group(1))

    return {
        "name": name,
        "skills": skills,
        "price_min": price_min,
        "price_max": price_max,
        "locations": locations,
        "available": available,
        "remote": remote,
        "experience_years": exp,
        "description": None,
        "_extraction_method": "rules",
    }


# ──────────────────────────────────────────────
# Neo4j グラフ登録
# ──────────────────────────────────────────────

_engineer_counter = 0
_project_counter = 0


def register_engineer(data: dict) -> str:
    """抽出結果を Engineer ノードとしてグラフに登録"""
    global _engineer_counter
    _engineer_counter += 1
    eid = f"E-{_engineer_counter:03d}"

    with get_driver().session() as sess:
        sess.run("""
            MERGE (e:Engineer {eid: $eid})
            SET e.name = $name,
                e.price = $price,
                e.available = $available,
                e.remote = $remote,
                e.experience_years = $exp,
                e.status = '提案可能'
        """,
            eid=eid, name=data.get("name") or f"要員{_engineer_counter}",
            price=data.get("price_min"),
            available=data.get("available"),
            remote=data.get("remote"),
            exp=data.get("experience_years"),
        )

        # スキルリレーション
        for skill in data.get("skills", []):
            sess.run("""
                MERGE (s:Skill {name: $skill_name})
                ON CREATE SET s.category = $cat
                WITH s
                MATCH (e:Engineer {eid: $eid})
                MERGE (e)-[r:HAS_SKILL]->(s)
                SET r.years = $years, r.level = $level
            """,
                eid=eid, skill_name=skill["name"],
                cat=skill.get("category", "other"),
                years=skill.get("years"),
                level="上級" if (skill.get("years") or 0) >= 5 else
                      "中級" if (skill.get("years") or 0) >= 2 else "初級",
            )

        # 勤務地リレーション
        for loc in data.get("locations", []):
            sess.run("""
                MERGE (l:Location {name: $loc})
                WITH l
                MATCH (e:Engineer {eid: $eid})
                MERGE (e)-[:PREFERS_LOCATION]->(l)
            """, eid=eid, loc=loc)

    return eid


def register_project(data: dict) -> str:
    """抽出結果を Project ノードとしてグラフに登録"""
    global _project_counter
    _project_counter += 1
    pid = f"P-{_project_counter:03d}"

    with get_driver().session() as sess:
        sess.run("""
            MERGE (p:Project {pid: $pid})
            SET p.name = $name,
                p.price_min = $price_min,
                p.price_max = $price_max,
                p.start = $available,
                p.remote = $remote,
                p.status = '募集中',
                p.description = $desc
        """,
            pid=pid, name=data.get("name") or f"案件{_project_counter}",
            price_min=data.get("price_min"),
            price_max=data.get("price_max"),
            available=data.get("available"),
            remote=data.get("remote"),
            desc=data.get("description"),
        )

        for skill in data.get("skills", []):
            sess.run("""
                MERGE (s:Skill {name: $skill_name})
                ON CREATE SET s.category = $cat
                WITH s
                MATCH (p:Project {pid: $pid})
                MERGE (p)-[r:REQUIRES]->(s)
                SET r.level = '必須', r.priority = 1.0
            """,
                pid=pid, skill_name=skill["name"],
                cat=skill.get("category", "other"),
            )

        for loc in data.get("locations", []):
            sess.run("""
                MERGE (l:Location {name: $loc})
                WITH l
                MATCH (p:Project {pid: $pid})
                MERGE (p)-[:LOCATED_IN]->(l)
            """, pid=pid, loc=loc)

    return pid


# ──────────────────────────────────────────────
# グラフマイニング マッチング
# ──────────────────────────────────────────────

def compute_graph_matching() -> list:
    """Neo4j グラフマイニングでマッチングスコアを計算

    3つのグラフパスを探索:
      1. 直接マッチ:  Engineer→HAS_SKILL→Skill←REQUIRES←Project
      2. 間接マッチ:  Engineer→HAS_SKILL→Skill→RELATED_TO→Skill←REQUIRES←Project
      3. 勤務地マッチ: Engineer→PREFERS_LOCATION→Location←(NEARBY)→Location←LOCATED_IN←Project
    """
    with get_driver().session() as sess:
        results = sess.run("""
            // 全 Engineer × Project ペア
            MATCH (e:Engineer), (p:Project)

            // 1. 直接スキルマッチ
            OPTIONAL MATCH (e)-[:HAS_SKILL]->(s:Skill)<-[:REQUIRES]-(p)
            WITH e, p, collect(DISTINCT s.name) AS direct_matches

            // 案件の全必須スキル
            OPTIONAL MATCH (p)-[:REQUIRES]->(rs:Skill)
            WITH e, p, direct_matches, collect(DISTINCT rs.name) AS required_skills

            // 2. 間接スキルマッチ（RELATED_TO 経由、直接マッチ除外）
            OPTIONAL MATCH (e)-[:HAS_SKILL]->(es:Skill)-[rel:RELATED_TO]->(rs2:Skill)<-[:REQUIRES]-(p)
            WHERE NOT es.name IN direct_matches AND NOT rs2.name IN direct_matches
            WITH e, p, direct_matches, required_skills,
                 collect(DISTINCT {via: es.name, target: rs2.name, weight: rel.weight}) AS indirect_raw

            // 重複除去（target ごとに最高 weight のみ）
            WITH e, p, direct_matches, required_skills,
                 [x IN indirect_raw WHERE x.target IS NOT NULL | x] AS indirect_matches

            // 3. 勤務地マッチ
            OPTIONAL MATCH (e)-[:PREFERS_LOCATION]->(el:Location)
            OPTIONAL MATCH (p)-[:LOCATED_IN]->(pl:Location)
            OPTIONAL MATCH (el)-[nb:NEARBY]->(pl)
            WITH e, p, direct_matches, required_skills, indirect_matches,
                 el, pl, nb,
                 CASE
                   WHEN el.name = pl.name THEN 1.0
                   WHEN nb IS NOT NULL THEN nb.proximity
                   WHEN el IS NULL OR pl IS NULL THEN 0.5
                   ELSE 0.2
                 END AS loc_score

            // スコア計算
            WITH e, p, direct_matches, required_skills, indirect_matches, loc_score,
                 CASE WHEN size(required_skills) = 0 THEN 0
                      ELSE toFloat(size(direct_matches)) / size(required_skills)
                 END AS direct_ratio,
                 size([x IN indirect_matches WHERE x.weight >= 0.5]) AS indirect_count

            WITH e, p, direct_matches, required_skills, indirect_matches, loc_score,
                 direct_ratio, indirect_count,

                 // スキルスコア (直接マッチ + 間接マッチの半分)
                 CASE WHEN size(required_skills) = 0 THEN 0.5
                      ELSE toFloat(size(direct_matches) + indirect_count * 0.4) / size(required_skills)
                 END AS skill_score,

                 // 単価スコア
                 CASE
                   WHEN e.price IS NULL OR p.price_min IS NULL THEN 0.5
                   WHEN e.price >= p.price_min AND e.price <= p.price_max THEN 1.0
                   WHEN e.price < p.price_min THEN
                     CASE WHEN p.price_min - e.price <= 10 THEN 0.7 ELSE 0.3 END
                   ELSE
                     CASE WHEN e.price - p.price_max <= 10 THEN 0.6 ELSE 0.2 END
                 END AS price_score

            // 総合スコア
            WITH e, p, direct_matches, required_skills, indirect_matches,
                 toInteger((skill_score * 0.5 + price_score * 0.2 + loc_score * 0.3) * 100) AS total_score,
                 toInteger(skill_score * 100) AS skill_pct,
                 toInteger(price_score * 100) AS price_pct,
                 toInteger(loc_score * 100) AS loc_pct

            RETURN
                e.eid AS engineer_id,
                e.name AS engineer_name,
                e.price AS engineer_price,
                e.available AS engineer_available,
                p.pid AS project_id,
                p.name AS project_name,
                p.price_min AS price_min,
                p.price_max AS price_max,
                p.remote AS remote,
                direct_matches,
                required_skills,
                [x IN indirect_matches WHERE x.weight >= 0.5 | x] AS indirect_matches,
                total_score,
                skill_pct,
                price_pct,
                loc_pct
            ORDER BY total_score DESC
        """).data()

    matches = []
    for r in results:
        score = min(r["total_score"], 100)
        if score >= 85:
            rank = "最有力"
        elif score >= 70:
            rank = "有力"
        elif score >= 50:
            rank = "候補"
        else:
            rank = "要検討"

        missing = [s for s in r["required_skills"] if s not in r["direct_matches"]]

        matches.append({
            "engineer_id": r["engineer_id"],
            "engineer_name": r["engineer_name"],
            "engineer_price": r["engineer_price"],
            "engineer_available": r["engineer_available"],
            "project_id": r["project_id"],
            "project_name": r["project_name"],
            "price_range": f"{r['price_min']}〜{r['price_max']}万円" if r["price_min"] else None,
            "remote": r["remote"],
            "score": score,
            "rank": rank,
            "skill_score": r["skill_pct"],
            "price_score": r["price_pct"],
            "location_score": r["loc_pct"],
            "direct_matches": r["direct_matches"],
            "indirect_matches": r["indirect_matches"],
            "missing_skills": missing,
            "required_skills": r["required_skills"],
        })

    return matches


# ──────────────────────────────────────────────
# API エンドポイント
# ──────────────────────────────────────────────

@app.route("/api/extract", methods=["POST"])
def api_extract():
    """テキストからエンティティ抽出"""
    data = request.json
    text = data.get("text", "")
    doc_type = data.get("type", "engineer")
    use_api = data.get("use_api", True)

    if not text.strip():
        return jsonify({"error": "テキストが空です"}), 400

    if use_api and (is_ollama_available() or QWEN_API_KEY != "YOUR_API_KEY_HERE"):
        result = extract_with_qwen(text, doc_type)
    else:
        result = extract_with_rules(text, doc_type)

    return jsonify(result)


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """ファイルアップロード → 抽出 → グラフ登録"""
    if "file" not in request.files:
        return jsonify({"error": "ファイルがありません"}), 400

    file = request.files["file"]
    doc_type = request.form.get("type", "engineer")
    use_api = request.form.get("use_api", "true").lower() == "true"

    try:
        text = file.read().decode("utf-8")
    except UnicodeDecodeError:
        try:
            file.seek(0)
            text = file.read().decode("shift_jis")
        except Exception:
            return jsonify({"error": "ファイルの文字コードを読み取れません"}), 400

    # 抽出
    if use_api and (is_ollama_available() or QWEN_API_KEY != "YOUR_API_KEY_HERE"):
        extracted = extract_with_qwen(text, doc_type)
    else:
        extracted = extract_with_rules(text, doc_type)

    # グラフ登録
    if doc_type == "engineer":
        node_id = register_engineer(extracted)
    else:
        node_id = register_project(extracted)

    extracted["node_id"] = node_id
    return jsonify(extracted)


@app.route("/api/register", methods=["POST"])
def api_register():
    """抽出済みデータをグラフに登録"""
    data = request.json
    doc_type = data.get("type", "engineer")
    extracted = data.get("data", {})

    if doc_type == "engineer":
        node_id = register_engineer(extracted)
    else:
        node_id = register_project(extracted)

    return jsonify({"node_id": node_id, "type": doc_type})


@app.route("/api/matching", methods=["GET"])
def api_matching():
    """グラフマイニングでマッチング実行"""
    try:
        matches = compute_graph_matching()
        return jsonify(matches)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/graph/stats", methods=["GET"])
def api_graph_stats():
    """グラフ統計"""
    with get_driver().session() as sess:
        stats = sess.run("""
            OPTIONAL MATCH (e:Engineer) WITH count(e) AS engineers
            OPTIONAL MATCH (p:Project) WITH engineers, count(p) AS projects
            OPTIONAL MATCH (s:Skill) WITH engineers, projects, count(s) AS skills
            OPTIONAL MATCH ()-[r:HAS_SKILL]->() WITH engineers, projects, skills, count(r) AS has_skill
            OPTIONAL MATCH ()-[r:REQUIRES]->() WITH engineers, projects, skills, has_skill, count(r) AS requires
            OPTIONAL MATCH ()-[r:RELATED_TO]->() WITH engineers, projects, skills, has_skill, requires, count(r) AS related
            RETURN engineers, projects, skills, has_skill, requires, related
        """).single()

    return jsonify({
        "engineers": stats["engineers"],
        "projects": stats["projects"],
        "skills": stats["skills"],
        "has_skill_edges": stats["has_skill"],
        "requires_edges": stats["requires"],
        "related_to_edges": stats["related"],
    })


@app.route("/api/clear", methods=["POST"])
def api_clear():
    """Engineer/Project ノードをクリア（スキル知識グラフは保持）"""
    global _engineer_counter, _project_counter
    with get_driver().session() as sess:
        sess.run("MATCH (e:Engineer) DETACH DELETE e")
        sess.run("MATCH (p:Project) DETACH DELETE p")
    _engineer_counter = 0
    _project_counter = 0
    return jsonify({"message": "クリア完了"})


@app.route("/api/health", methods=["GET"])
def api_health():
    """ヘルスチェック"""
    try:
        get_driver().verify_connectivity()
        neo4j_ok = True
    except Exception:
        neo4j_ok = False

    qwen_ok = QWEN_API_KEY != "YOUR_API_KEY_HERE"
    ollama_ok = is_ollama_available()

    return jsonify({
        "status": "ok",
        "neo4j": "connected" if neo4j_ok else "disconnected",
        "ollama": f"connected ({OLLAMA_MODEL})" if ollama_ok else "not running",
        "qwen_api": "configured" if qwen_ok else "not configured",
        "llm": "ollama" if ollama_ok else "qwen" if qwen_ok else "rule-based only",
    })


# ──────────────────────────────────────────────
# フロントエンド HTML
# ──────────────────────────────────────────────

@app.route("/")
def index():
    return INDEX_HTML


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Skill Matcher - Qwen + Neo4j</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Noto Sans JP','Hiragino Sans',system-ui,sans-serif;color:#1e293b;background:#f8fafc}
.container{max-width:960px;margin:0 auto;padding:20px 16px}
h1{font-size:22px;font-weight:800;letter-spacing:-0.5px}
.subtitle{font-size:13px;color:#64748b;margin-top:3px}
.header{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:18px}
.status{display:flex;gap:8px;font-size:11px;color:#94a3b8;align-items:center}
.dot{width:7px;height:7px;border-radius:50%;display:inline-block}
.dot-ok{background:#10b981}.dot-warn{background:#f59e0b}.dot-err{background:#ef4444}
.tabs{display:flex;gap:2px;margin-bottom:16px}
.tab{padding:9px 20px;border-radius:8px 8px 0 0;border:none;cursor:pointer;font-size:13px;font-weight:600;background:#f1f5f9;color:#64748b;transition:.15s}
.tab.active{background:#1e293b;color:#fff}
.tab .badge{margin-left:5px;padding:1px 7px;border-radius:8px;font-size:10px;background:rgba(255,255,255,.2)}
.zones{display:flex;gap:18px;margin-bottom:16px}
.zone{flex:1;min-width:0}
.zone-label{display:flex;align-items:center;gap:8px;margin-bottom:10px;font-size:14px;font-weight:700}
.zone-icon{width:26px;height:26px;border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:14px}
.zone-count{margin-left:auto;font-size:11px;font-weight:600;color:#94a3b8;background:#f1f5f9;padding:2px 8px;border-radius:10px}
.droparea{border:2px dashed #d1d5db;border-radius:12px;padding:36px;min-height:130px;background:#fafbfc;cursor:pointer;transition:.2s;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:6px}
.droparea.has-items{padding:10px;min-height:50px;align-items:stretch;justify-content:flex-start;gap:6px}
.droparea.drag{border-color:#2563eb;background:#eff6ff08}
.droparea .placeholder{font-size:13px;color:#64748b;font-weight:600}
.droparea .hint{font-size:11px;color:#a1a1aa}
.file-item{display:flex;align-items:center;gap:8px;padding:7px 10px;background:#fff;border-radius:8px;border:1px solid #e5e7eb}
.file-dot{width:6px;height:6px;border-radius:50%;flex-shrink:0}
.file-name{font-size:12px;font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.file-meta{font-size:10px;color:#94a3b8;display:flex;flex-wrap:wrap;gap:3px;margin-top:2px}
.skill-tag{padding:0 5px;border-radius:3px;font-size:10px;font-weight:600}
.file-remove{background:none;border:none;cursor:pointer;color:#cbd5e1;font-size:15px;padding:2px 4px}
.btns{display:flex;gap:8px;margin-bottom:14px}
.btn{padding:9px 16px;border-radius:8px;font-size:12px;font-weight:600;cursor:pointer;border:none;transition:.15s}
.btn-add-eng{flex:1;border:1px dashed #93c5fd;background:#eff6ff;color:#2563eb}
.btn-add-proj{flex:1;border:1px dashed #fca5a5;background:#fef2f2;color:#dc2626}
.demo-bar{display:flex;gap:8px;align-items:center;margin-bottom:14px;padding:10px 14px;background:#f8fafc;border-radius:8px;border:1px solid #e5e7eb}
.demo-label{font-size:12px;font-weight:700;color:#475569}
.btn-demo{padding:5px 14px;border-radius:6px;border:1px solid #e5e7eb;background:#fff;color:#475569;font-size:12px;font-weight:600;cursor:pointer}
.btn-clear{padding:5px 14px;border-radius:6px;border:1px solid #fecaca;background:#fff;color:#dc2626;font-size:12px;font-weight:600;cursor:pointer}
.btn-match{width:100%;padding:14px;border-radius:10px;border:none;font-size:15px;font-weight:700;cursor:pointer;transition:.2s}
.btn-match.active{background:#1e293b;color:#fff}
.btn-match.disabled{background:#e5e7eb;color:#9ca3af;cursor:default}
.graph-bar{display:flex;gap:14px;margin-bottom:14px;padding:8px 14px;background:#f0fdf4;border-radius:8px;border:1px solid #bbf7d0;font-size:11px;color:#166534;flex-wrap:wrap}
.match-card{background:#fff;border-radius:10px;padding:12px 16px;border:1px solid #e5e7eb;cursor:pointer;transition:.15s;margin-bottom:6px}
.match-card:hover{box-shadow:0 2px 8px rgba(0,0,0,.06)}
.match-top{display:flex;align-items:center;gap:10px}
.match-rank{width:26px;height:26px;border-radius:50%;display:flex;align-items:center;justify-content:center;background:#f1f5f9;font-size:12px;font-weight:700;color:#475569;flex-shrink:0}
.match-name{font-size:13px;font-weight:700}
.match-tags{display:flex;gap:3px;flex-wrap:wrap;margin-top:3px}
.tag{padding:1px 6px;border-radius:3px;font-size:10px;font-weight:600}
.tag-match{background:#dbeafe;color:#1d4ed8}
.tag-indirect{background:#fef3c7;color:#92400e}
.tag-miss{background:#fef2f2;color:#dc2626;text-decoration:line-through}
.match-score{text-align:right;flex-shrink:0}
.match-pct{font-size:22px;font-weight:800}
.match-label{font-size:10px;font-weight:700;padding:1px 8px;border-radius:8px}
.match-detail{margin-top:12px;padding-top:12px;border-top:1px solid #f1f5f9}
.score-bars{display:flex;gap:14px}
.score-bar{flex:1}
.score-bar-header{display:flex;justify-content:space-between;font-size:11px;margin-bottom:3px}
.score-bar-label{color:#64748b;font-weight:600}
.score-bar-val{font-weight:700}
.bar-track{height:5px;background:#f1f5f9;border-radius:3px}
.bar-fill{height:100%;border-radius:3px;transition:width .3s}
.indirect-box{margin-top:8px;font-size:11px;color:#7c3aed;background:#f5f3ff;padding:6px 10px;border-radius:6px}
.match-meta{margin-top:8px;font-size:11px;color:#64748b;display:flex;gap:10px;flex-wrap:wrap}
.modal-bg{position:fixed;inset:0;background:rgba(0,0,0,.35);z-index:100;display:flex;align-items:center;justify-content:center}
.modal{background:#fff;border-radius:14px;padding:24px;width:90%;max-width:560px;box-shadow:0 8px 32px rgba(0,0,0,.12)}
.modal h3{font-size:15px;font-weight:700;margin-bottom:12px}
.modal textarea{width:100%;height:180px;padding:12px;border-radius:8px;border:1px solid #e5e7eb;font-size:13px;resize:vertical;font-family:inherit;line-height:1.6}
.modal-btns{display:flex;gap:8px;justify-content:flex-end;margin-top:12px}
.empty{text-align:center;padding:50px 20px;color:#94a3b8}
.empty-icon{font-size:36px;margin-bottom:10px;opacity:.25}
.toggle-track{width:34px;height:18px;border-radius:9px;padding:2px;cursor:pointer;transition:.2s;display:inline-block;position:relative}
.toggle-knob{width:14px;height:14px;border-radius:50%;background:#fff;transition:transform .2s;box-shadow:0 1px 3px rgba(0,0,0,.15)}
</style>
</head>
<body>
<div class="container" id="app"></div>
<script>
const API = '';  // same origin
let state = {
  engineers: [], projects: [], matches: [],
  tab: 'upload', processing: false,
  modal: null, useQwen: true,
  health: null, graphStats: null,
  engCounter: 0, projCounter: 0,
};

// ─── Render ───
function render() {
  const s = state;
  const hasData = s.engineers.some(e => e.extracted) && s.projects.some(p => p.extracted);
  let html = `
    <div class="header">
      <div><h1>Skill Matcher</h1><p class="subtitle">Qwen API + Neo4j グラフマイニング</p></div>
      <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
        <div class="status">
          <span><span class="dot ${s.health?.neo4j==='connected'?'dot-ok':'dot-err'}"></span> Neo4j</span>
          <span><span class="dot ${s.health?.ollama?.includes('connected')?'dot-ok':s.health?.qwen_api==='configured'?'dot-ok':'dot-warn'}"></span> ${s.health?.llm||'LLM'}</span>
        </div>
        <label style="display:flex;align-items:center;gap:5px;font-size:11px;color:#64748b;cursor:pointer;user-select:none" onclick="toggleQwen()">
          <div class="toggle-track" style="background:${s.useQwen?'#2563eb':'#cbd5e1'}">
            <div class="toggle-knob" style="transform:translateX(${s.useQwen?'16px':'0'})"></div>
          </div>
          ${s.useQwen?'Qwen API':'ルール抽出'}
        </label>
      </div>
    </div>`;

  if (s.graphStats) {
    const g = s.graphStats;
    html += `<div class="graph-bar">
      <span>Neo4j:</span><span>要員 ${g.engineers}</span><span>案件 ${g.projects}</span>
      <span>スキル ${g.skills}</span><span>HAS_SKILL ${g.has_skill_edges}</span>
      <span>REQUIRES ${g.requires_edges}</span><span>RELATED_TO ${g.related_to_edges}</span>
    </div>`;
  }

  html += `<div class="tabs">
    <button class="tab ${s.tab==='upload'?'active':''}" onclick="setTab('upload')">データ入力</button>
    <button class="tab ${s.tab==='results'?'active':''}" onclick="setTab('results')">マッチング結果${s.matches.length?`<span class="badge">${s.matches.length}</span>`:''}</button>
  </div>`;

  if (s.tab === 'upload') {
    html += renderUpload(hasData);
  } else {
    html += renderResults();
  }

  if (s.modal) html += renderModal();
  document.getElementById('app').innerHTML = html;
  setupDropListeners();
}

function renderUpload(hasData) {
  return `
    <div class="zones">
      ${renderZone('engineer', 'スキルシート', state.engineers, '#2563eb', '👤')}
      ${renderZone('project', '求人データ', state.projects, '#dc2626', '📋')}
    </div>
    <div class="btns">
      <button class="btn btn-add-eng" onclick="openModal('engineer')">+ スキルシート手入力</button>
      <button class="btn btn-add-proj" onclick="openModal('project')">+ 求人データ手入力</button>
    </div>
    <div class="demo-bar">
      <span class="demo-label">デモ:</span>
      <button class="btn-demo" onclick="loadDemo()">サンプルデータ読込</button>
      <button class="btn-clear" onclick="clearAll()">全クリア</button>
    </div>
    <div style="display:flex;gap:8px">
    <button class="btn-match ${hasData&&!state.processing?'active':'disabled'}" style="flex:1" onclick="${hasData&&!state.processing?'runMatching()':''}">
      ${state.processing?'処理中...': 'Rule-based'}
    </button>
    <button class="btn-match ${hasData&&!state.processing?'active':'disabled'}" style="flex:1;background:${hasData&&!state.processing?'#534AB7':'#e5e7eb'}" onclick="${hasData&&!state.processing?'runGraphRAG()':''}">
      ${state.processing?'処理中...': 'GraphRAG'}
    </button>
    <button class="btn-match ${hasData&&!state.processing?'active':'disabled'}" style="flex:1;background:${hasData&&!state.processing?'#0F6E56':'#e5e7eb'}" onclick="${hasData&&!state.processing?'runLightRAG()':''}">
      ${state.processing?'処理中...': 'LightRAG'}
    </button>
    </div>`;
}

function renderZone(type, label, items, color, icon) {
  const hasItems = items.length > 0;
  return `<div class="zone">
    <div class="zone-label">
      <span class="zone-icon" style="background:${color}15;color:${color}">${icon}</span>
      ${label}
      <span class="zone-count">${items.length}件</span>
    </div>
    <div class="droparea ${hasItems?'has-items':''}" id="drop-${type}" data-type="${type}">
      <input type="file" id="file-${type}" multiple accept=".txt,.csv,.md,.text,.json,.tsv" style="display:none" onchange="handleFileInput('${type}',this)">
      ${hasItems ? items.map((item,i) => `
        <div class="file-item">
          <div class="file-dot" style="background:${item.extracted?'#10b981':item.error?'#ef4444':'#f59e0b'}"></div>
          <div style="flex:1;min-width:0">
            <div class="file-name">${item.extracted?.name||item.fileName}</div>
            <div class="file-meta">
              ${item.extracted ? `
                <span>${item.extracted.skills?.length||0}スキル</span>
                <span class="skill-tag" style="background:${item.extracted._extraction_method==='qwen'?'#eff6ff':'#f5f3ff'};color:${item.extracted._extraction_method==='qwen'?'#2563eb':'#7c3aed'}">${item.extracted._extraction_method==='qwen'?'Qwen':'Rule'}</span>
                ${(item.extracted.skills||[]).slice(0,3).map(s=>`<span class="skill-tag" style="background:${color}12;color:${color}">${s.name}</span>`).join('')}
              ` : item.error ? `<span style="color:#ef4444">${item.error}</span>` : '解析中...'}
            </div>
          </div>
          <button class="file-remove" onclick="removeItem('${type}',${i})">×</button>
        </div>
      `).join('') : `
        <div style="font-size:28px;opacity:.25">${icon}</div>
        <div class="placeholder">${label}をドロップ</div>
        <div class="hint">.txt .csv .md（クリックで選択）</div>
      `}
    </div>
  </div>`;
}

function renderResults() {
  if (!state.matches.length) return `<div class="empty"><div class="empty-icon">📊</div><div style="font-size:14px;font-weight:600;margin-bottom:4px">マッチング結果がありません</div><div style="font-size:12px">データを追加してマッチングを実行してください</div></div>`;

  const top = state.matches.filter(m=>m.score>=85).length;
  const good = state.matches.filter(m=>m.score>=70&&m.score<85).length;

  let html = `<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
    <div style="font-size:12px;color:#64748b">${state.matches.length}件（最有力: ${top} / 有力: ${good}）</div>
    <button class="btn-demo" onclick="setTab('upload')">戻る</button>
  </div>`;

  state.matches.forEach((m,i) => {
    const rc = m.score>=85?'#10b981':m.score>=70?'#f59e0b':m.score>=50?'#6366f1':'#94a3b8';
    html += `<div class="match-card" onclick="toggleDetail(${i})">
      <div class="match-top">
        <span class="match-rank">${i+1}</span>
        <div style="flex:1;min-width:0">
          <div class="match-name">${m.project_name} × ${m.engineer_name}</div>
          <div class="match-tags">
            ${(m.direct_matches||[]).map(s=>`<span class="tag tag-match">${s}</span>`).join('')}
            ${(m.indirect_matches||[]).slice(0,2).map(x=>`<span class="tag tag-indirect">${x.via}→${x.target}</span>`).join('')}
            ${(m.missing_skills||[]).slice(0,2).map(s=>`<span class="tag tag-miss">${s}</span>`).join('')}
          </div>
        </div>
        <div class="match-score">
          <div class="match-pct" style="color:${rc}">${m.score}%</div>
          <span class="match-label" style="background:${rc}15;color:${rc}">${m.rank}</span>
        </div>
      </div>
      <div class="match-detail" id="detail-${i}" style="display:none">
        <div class="score-bars">
          ${[{l:'スキル',v:m.skill_score,c:'#2563eb'},{l:'単価',v:m.price_score,c:'#10b981'},{l:'勤務地',v:m.location_score,c:'#f59e0b'}].map(b=>`
            <div class="score-bar">
              <div class="score-bar-header"><span class="score-bar-label">${b.l}</span><span class="score-bar-val" style="color:${b.c}">${b.v}%</span></div>
              <div class="bar-track"><div class="bar-fill" style="width:${b.v}%;background:${b.c}"></div></div>
            </div>
          `).join('')}
        </div>
        ${(m.indirect_matches||[]).length?`<div class="indirect-box">グラフ間接マッチ: ${m.indirect_matches.map(x=>`${x.via} →[${Math.round(x.weight*100)}%]→ ${x.target}`).join(', ')}</div>`:''}
        <div class="match-meta">
          ${m.engineer_price?`<span>単価: ${m.engineer_price}万円</span>`:''}
          ${m.price_range?`<span>案件: ${m.price_range}</span>`:''}
          ${m.remote?`<span>${m.remote}</span>`:''}
          ${m.engineer_available?`<span>参画: ${m.engineer_available}</span>`:''}
          ${m.method?`<span style="padding:1px 6px;border-radius:4px;background:${m.method==='graphrag'?'#EEEDFE':'#f1f5f9'};color:${m.method==='graphrag'?'#534AB7':'#5F5E5A'};font-weight:600">${m.method}</span>`:''}
        </div>
        ${m.reasoning?`<div style="margin-top:8px;font-size:12px;color:var(--color-text-secondary,#475569);background:var(--color-background-secondary,#f8fafc);padding:8px 12px;border-radius:6px;line-height:1.6">${m.reasoning}</div>`:''}
        ${m.recommendation?`<div style="margin-top:6px;font-size:11px;color:#534AB7;font-weight:600">${m.recommendation}</div>`:''}
      </div>
    </div>`;
  });
  return html;
}

function renderModal() {
  const type = state.modal;
  const label = type==='engineer'?'スキルシート':'求人票';
  const ph = type==='engineer'
    ? '氏名：田中太郎\nスキル：Java, Spring Boot, AWS\n経験：5年\n希望単価：80万円\n勤務地：東京'
    : '案件名：EC基盤刷新\n必須スキル：Java / Spring Boot / AWS\n単価：75〜85万円\n勤務地：東京';
  return `<div class="modal-bg" onclick="closeModal()">
    <div class="modal" onclick="event.stopPropagation()">
      <h3>${label}テキスト入力</h3>
      <textarea id="modal-text" placeholder="${ph}"></textarea>
      <div class="modal-btns">
        <button class="btn-demo" onclick="closeModal()">キャンセル</button>
        <button class="btn" style="background:#1e293b;color:#fff;padding:7px 18px" onclick="submitModal()">追加</button>
      </div>
    </div>
  </div>`;
}

// ─── Actions ───
function setTab(t) { state.tab = t; render(); }
function toggleQwen() { state.useQwen = !state.useQwen; render(); }
function openModal(type) { state.modal = type; render(); }
function closeModal() { state.modal = null; render(); }

function toggleDetail(i) {
  const el = document.getElementById('detail-'+i);
  if (el) el.style.display = el.style.display === 'none' ? 'block' : 'none';
}

function removeItem(type, i) {
  if (type === 'engineer') state.engineers.splice(i, 1);
  else state.projects.splice(i, 1);
  render();
}

function submitModal() {
  const text = document.getElementById('modal-text')?.value?.trim();
  if (!text) return;
  const type = state.modal;
  state.modal = null;
  processTexts([{fileName: '手入力_'+Date.now()+'.txt', text}], type);
}

async function processTexts(files, type) {
  state.processing = true;
  const list = type === 'engineer' ? state.engineers : state.projects;

  for (const file of files) {
    list.push({fileName: file.fileName, text: file.text, extracted: null});
    render();
    try {
      const res = await fetch(API+'/api/extract', {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({text: file.text, type, use_api: state.useQwen}),
      });
      const extracted = await res.json();

      const reg = await fetch(API+'/api/register', {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({data: extracted, type}),
      });
      const regData = await reg.json();
      extracted.node_id = regData.node_id;

      const item = list.find(x => x.fileName === file.fileName && !x.extracted);
      if (item) item.extracted = extracted;
    } catch(e) {
      const item = list.find(x => x.fileName === file.fileName && !x.extracted);
      if (item) item.error = 'API接続エラー';
    }
    render();
  }

  try {
    const st = await fetch(API+'/api/graph/stats').then(r=>r.json());
    state.graphStats = st;
  } catch(e) {}

  state.processing = false;
  render();
}

async function runMatching() {
  state.processing = true; render();
  try {
    const res = await fetch(API+'/api/matching');
    state.matches = await res.json();
    state.matches.forEach(m => m.method = m.method || 'rule-based');
    state.tab = 'results';
  } catch(e) {
    alert('マッチング失敗');
  }
  state.processing = false; render();
}

async function runGraphRAG() {
  state.processing = true; render();
  try {
    // まず GraphRAG 用にデータを再登録
    for (const eng of state.engineers.filter(e => e.extracted)) {
      await fetch(API+'/api/graphrag/extract', {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({text: eng.text, type: 'engineer'}),
      });
    }
    for (const proj of state.projects.filter(p => p.extracted)) {
      await fetch(API+'/api/graphrag/extract', {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({text: proj.text, type: 'project'}),
      });
    }
    // GraphRAG マッチング
    const res = await fetch(API+'/api/graphrag/matching');
    const data = await res.json();
    state.matches = data.map(m => ({
      ...m,
      project_name: m.project_name || m.project_id,
      engineer_name: m.engineer_name,
      score: m.score,
      rank: m.rank,
      direct_matches: m.strengths || [],
      missing_skills: m.gaps || [],
      indirect_matches: [],
      skill_score: m.score,
      price_score: 50,
      location_score: 50,
      method: 'graphrag',
      reasoning: m.reasoning,
      recommendation: m.recommendation,
    }));
    state.tab = 'results';
  } catch(e) {
    console.error(e);
    alert('GraphRAG マッチング失敗: ' + e.message);
  }
  state.processing = false; render();
}

async function runLightRAG() {
  state.processing = true; render();
  try {
    await fetch(API+'/api/lightrag/clear', {method:'POST'});
    for (const eng of state.engineers.filter(e => e.text)) {
      await fetch(API+'/api/lightrag/insert', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({text: eng.text, type:'engineer'}),
      });
    }
    for (const proj of state.projects.filter(p => p.text)) {
      await fetch(API+'/api/lightrag/insert', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({text: proj.text, type:'project'}),
      });
    }
    const res = await fetch(API+'/api/lightrag/matching');
    const data = await res.json();
    state.matches = data.map(m => ({
      ...m,
      direct_matches: m.direct_matches || [],
      missing_skills: m.missing_skills || [],
      indirect_matches: (m.indirect_matches||[]).map(x=>({via:x.has,target:x.required,weight:x.similarity||0.5})),
      skill_score: m.score, price_score: 50, location_score: 50,
      method: 'lightrag',
    }));
    state.tab = 'results';
  } catch(e) {
    console.error(e);
    alert('LightRAG マッチング失敗: '+e.message);
  }
  state.processing = false; render();
}

async function clearAll() {
  try { await fetch(API+'/api/clear', {method:'POST'}); } catch(e) {}
  state.engineers = []; state.projects = []; state.matches = [];
  state.graphStats = null; render();
}

function loadDemo() {
  const demoEng = [
    {fileName:'佐藤健一.txt', text:'氏名：佐藤 健一\nスキル：Java, Spring Boot, AWS, Docker, PostgreSQL\n経験：8年\n希望単価：80万円\n勤務地希望：東京、リモート可\n参画可能：2026年5月'},
    {fileName:'鈴木麻衣.txt', text:'氏名：鈴木 麻衣\nスキル：React, TypeScript, Next.js, Node.js, JavaScript\n経験：5年\n希望単価：78万円\n勤務地希望：フルリモート\n参画可能：即日'},
    {fileName:'田中亮.txt', text:'氏名：田中 亮\nスキル：Python, Azure, ETL, SQL, Docker, 機械学習\n経験：7年\n希望単価：90万円\n勤務地希望：横浜、リモート\n参画可能：2026年6月'},
    {fileName:'高橋さくら.txt', text:'氏名：高橋 さくら\nスキル：Python, 機械学習, PyTorch, AWS, Docker, Kubernetes\n経験：6年\n希望単価：85万円\n勤務地希望：東京\n参画可能：2026年5月'},
  ];
  const demoProj = [
    {fileName:'EC基盤刷新.txt', text:'案件名：大手小売向けEC基盤刷新\n必須スキル：Java, Spring Boot, AWS\n歓迎：Docker, PostgreSQL\n単価：75〜85万円\n勤務地：東京・リモート併用\n開始：2026年5月'},
    {fileName:'Webアプリ開発.txt', text:'案件名：通信会社向けWebアプリ開発\n必須スキル：React, TypeScript, Node.js\n歓迎：Next.js, AWS\n単価：70〜85万円\n勤務地：新宿・フルリモート可\n開始：2026年5月'},
    {fileName:'データ分析基盤.txt', text:'案件名：製造業向けデータ分析基盤構築\n必須スキル：Python, ETL, Azure\n歓迎：機械学習, Docker\n単価：80〜95万円\n勤務地：横浜・リモート併用\n開始：2026年6月'},
    {fileName:'医療AI.txt', text:'案件名：医療データ分析PF\n必須スキル：Python, 機械学習, AWS\n歓迎：PyTorch, Docker, Kubernetes\n単価：85〜100万円\n勤務地：東京・リモート併用\n開始：2026年7月'},
  ];
  processTexts(demoEng, 'engineer');
  processTexts(demoProj, 'project');
}

// ─── File handling ───
function handleFileInput(type, input) {
  const files = [...input.files].map(f => ({fileName: f.name, textPromise: f.text()}));
  Promise.all(files.map(async f => ({fileName: f.fileName, text: await f.textPromise})))
    .then(results => processTexts(results, type));
  input.value = '';
}

function setupDropListeners() {
  ['engineer','project'].forEach(type => {
    const el = document.getElementById('drop-'+type);
    if (!el) return;
    el.onclick = () => document.getElementById('file-'+type)?.click();
    el.ondragover = (e) => { e.preventDefault(); el.classList.add('drag'); };
    el.ondragleave = () => el.classList.remove('drag');
    el.ondrop = (e) => {
      e.preventDefault(); el.classList.remove('drag');
      const files = [...e.dataTransfer.files].filter(f => /\.(txt|csv|md|text|json|tsv)$/i.test(f.name) || f.type.startsWith('text/'));
      if (files.length === 0) {
        const t = e.dataTransfer.getData('text/plain');
        if (t) processTexts([{fileName:'pasted.txt',text:t}], type);
        return;
      }
      Promise.all(files.map(async f => ({fileName:f.name, text: await f.text()}))).then(r => processTexts(r, type));
    };
  });
}

// ─── Init ───
fetch(API+'/api/health').then(r=>r.json()).then(h=>{state.health=h;render();}).catch(()=>{state.health={status:'error',neo4j:'disconnected',qwen_api:'unknown'};render();});
render();
</script>
</body>
</html>
"""


# ──────────────────────────────────────────────
# 起動
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Skill Matching Backend")
    print("  Flask + Qwen API + Neo4j + GraphRAG")
    print("=" * 60)
    print()

    # Neo4j 接続テスト
    try:
        get_driver()
    except Exception as e:
        print(f"  Neo4j 接続エラー: {e}")
        print(f"  Neo4j を起動してから再実行してください")
        exit(1)

    # GraphRAG エンドポイント登録
    try:
        from graphrag_matching import register_graphrag_routes
        register_graphrag_routes(app)
    except ImportError:
        print("  GraphRAG: スキップ")
    except Exception as e:
        print(f"  GraphRAG エラー: {e}")

    # LightRAG エンドポイント登録
    try:
        from lightrag_matching import register_lightrag_routes
        register_lightrag_routes(app)
    except ImportError:
        print("  LightRAG: lightrag_matching.py が見つかりません")
    except Exception as e:
        print(f"  LightRAG エラー: {e}")

    # LLM チェック（Ollama → Qwen の優先順位）
    if is_ollama_available():
        print(f"  LLM: Ollama 検出 (model: {OLLAMA_MODEL})")
    elif QWEN_API_KEY != "YOUR_API_KEY_HERE":
        print(f"  LLM: Qwen API (model: {QWEN_MODEL})")
    else:
        print("  LLM: なし (ルールベース抽出を使用)")
        print("  Ollama 設定: ollama run qwen2.5:7b")
        print("  Qwen 設定:   set DASHSCOPE_API_KEY=sk-xxxxx")

    print()
    print("  フロントエンド: http://localhost:5000")
    print("  ルールベース:   /api/matching")
    print("  GraphRAG:      /api/graphrag/matching")
    print("  LightRAG:      /api/lightrag/matching")
    print()

    app.run(debug=True, port=5000, host="0.0.0.0")
