"""
GraphRAG スキルマッチング拡張
════════════════════════════════
既存の Neo4j グラフマイニングに GraphRAG レイヤーを追加。

LightRAG 的アプローチ:
  1. LLM でテキストからエンティティ＆リレーションを自動抽出
  2. Neo4j グラフに投入（動的スキーマ）
  3. クエリ時にサブグラフを検索・抽出
  4. サブグラフ + クエリを LLM に渡して推論
  5. 理由付きマッチングスコアを生成

既存のルールベースとの違い:
  - スキル関連性が手動定義ではなく、文脈から自動推論
  - 「マイクロサービス経験3年」→ スキルノードだけでなく経験ノードも生成
  - マッチング理由を自然言語で説明

使い方:
  既存の app_qwen.py に追加エンドポイントとして組み込む

依存: neo4j, requests, numpy
"""

import os
import json
import re
import hashlib
import numpy as np
from collections import defaultdict
from neo4j import GraphDatabase
import requests

# ──────────────────────────────────────────────
# 設定
# ──────────────────────────────────────────────

QWEN_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "YOUR_API_KEY_HERE")
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
QWEN_MODEL = "qwen-plus"

NEO4J_URI = os.environ.get("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_AUTH = (
    os.environ.get("NEO4J_USER", "neo4j"),
    os.environ.get("NEO4J_PASS", "unko1234"),
)

driver = None

def get_driver():
    global driver
    if driver is None:
        driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        driver.verify_connectivity()
    return driver


def call_qwen(system_prompt, user_prompt, temperature=0.1):
    """Qwen API 呼び出し（共通関数）"""
    if QWEN_API_KEY == "YOUR_API_KEY_HERE":
        return None

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {QWEN_API_KEY}",
    }
    payload = {
        "model": QWEN_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": 3000,
    }
    try:
        resp = requests.post(QWEN_BASE_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return content
    except Exception as e:
        print(f"  Qwen API error: {e}")
        return None


# ──────────────────────────────────────────────
# Phase 1: LLM ベースのエンティティ＆リレーション抽出
# ──────────────────────────────────────────────

GRAPH_EXTRACTION_PROMPT = """あなたはIT人材のスキルシートや求人票からナレッジグラフを構築するエキスパートです。

テキストからエンティティ（ノード）とリレーション（エッジ）を抽出してください。
固定スキーマに縛られず、テキストに含まれる情報を最大限構造化してください。

出力はJSON**のみ**（マークダウン不要）:
{
  "entities": [
    {
      "id": "一意のID（例: skill_java, exp_microservice）",
      "type": "Person|Skill|Experience|Project|Industry|Certification|Location|Tool|Methodology",
      "name": "表示名",
      "properties": {"key": "value"}
    }
  ],
  "relations": [
    {
      "source": "エンティティID",
      "target": "エンティティID",
      "type": "HAS_SKILL|HAS_EXPERIENCE|USED_IN|RELATED_TO|WORKS_IN|CERTIFIED_IN|LOCATED_IN|REQUIRES|PREFERS",
      "properties": {"key": "value"}
    }
  ]
}

抽出のポイント:
- スキル名だけでなく「何に使ったか」の文脈も Experience ノードとして抽出
  例: "Spring Boot でマイクロサービスを構築" →
    Skill:Spring Boot, Experience:マイクロサービス構築, (Spring Boot)-[USED_IN]->(マイクロサービス構築)
- 業界経験も抽出: "金融系システム開発5年" → Industry:金融, Experience:金融系システム開発
- 関連スキルの推論: Java → Spring Boot は RELATED_TO
- 経験年数は properties に含める
- 単価、勤務地、参画時期なども properties に"""


def extract_graph_entities(text: str, doc_type: str) -> dict:
    """LLM でテキストからグラフ構造を抽出"""
    type_label = "スキルシート" if doc_type == "engineer" else "求人票"
    
    content = call_qwen(
        GRAPH_EXTRACTION_PROMPT,
        f"以下の{type_label}からエンティティとリレーションを抽出してください:\n\n{text[:4000]}"
    )
    
    if content is None:
        return _fallback_extract(text, doc_type)
    
    try:
        content = re.sub(r"```json\s*|```\s*", "", content).strip()
        result = json.loads(content)
        return result
    except (json.JSONDecodeError, KeyError) as e:
        print(f"  JSON parse error: {e}")
        return _fallback_extract(text, doc_type)


def _fallback_extract(text: str, doc_type: str) -> dict:
    """LLM なしのフォールバック抽出"""
    from app_qwen import extract_with_rules
    rules_result = extract_with_rules(text, doc_type)
    
    entities = []
    relations = []
    
    # Person / Project ノード
    main_id = f"{'person' if doc_type == 'engineer' else 'project'}_{hashlib.md5(text[:100].encode()).hexdigest()[:8]}"
    main_type = "Person" if doc_type == "engineer" else "Project"
    entities.append({
        "id": main_id,
        "type": main_type,
        "name": rules_result.get("name") or f"{'要員' if doc_type == 'engineer' else '案件'}",
        "properties": {
            "price_min": rules_result.get("price_min"),
            "price_max": rules_result.get("price_max"),
            "available": rules_result.get("available"),
            "remote": rules_result.get("remote"),
        },
    })
    
    # Skill ノード
    for skill in rules_result.get("skills", []):
        s_id = f"skill_{skill['name'].lower().replace(' ', '_')}"
        entities.append({
            "id": s_id,
            "type": "Skill",
            "name": skill["name"],
            "properties": {"category": skill.get("category")},
        })
        rel_type = "HAS_SKILL" if doc_type == "engineer" else "REQUIRES"
        relations.append({
            "source": main_id,
            "target": s_id,
            "type": rel_type,
            "properties": {"years": skill.get("years")},
        })
    
    # Location ノード
    for loc in rules_result.get("locations", []):
        l_id = f"loc_{loc}"
        entities.append({
            "id": l_id, "type": "Location", "name": loc, "properties": {},
        })
        rel_type = "PREFERS" if doc_type == "engineer" else "LOCATED_IN"
        relations.append({
            "source": main_id, "target": l_id, "type": rel_type, "properties": {},
        })
    
    return {"entities": entities, "relations": relations}


# ──────────────────────────────────────────────
# Phase 2: Neo4j グラフへの動的登録
# ──────────────────────────────────────────────

def register_graph_entities(extracted: dict, doc_type: str, doc_id: str) -> dict:
    """抽出したエンティティ＆リレーションを Neo4j に登録
    
    固定スキーマではなく、抽出されたタイプをそのままラベルとして使う。
    全ノードに :GraphRAG ラベルも付与して検索を容易にする。
    """
    stats = {"nodes_created": 0, "relations_created": 0}
    
    with get_driver().session() as sess:
        # エンティティ登録
        for entity in extracted.get("entities", []):
            eid = entity["id"]
            etype = entity["type"]
            name = entity["name"]
            props = entity.get("properties", {})
            
            # None を除外
            clean_props = {k: v for k, v in props.items() if v is not None}
            
            # MERGE で重複防止
            query = f"""
                MERGE (n:{etype}:GraphRAG {{entity_id: $eid}})
                SET n.name = $name, n.doc_id = $doc_id, n.doc_type = $doc_type
            """
            # properties を動的に SET
            if clean_props:
                for k, v in clean_props.items():
                    query += f", n.{k} = ${k}"
            
            params = {"eid": eid, "name": name, "doc_id": doc_id, "doc_type": doc_type, **clean_props}
            
            try:
                sess.run(query, **params)
                stats["nodes_created"] += 1
            except Exception as e:
                print(f"  Node error ({eid}): {e}")
        
        # リレーション登録
        for rel in extracted.get("relations", []):
            src = rel["source"]
            tgt = rel["target"]
            rtype = rel["type"]
            props = rel.get("properties", {})
            clean_props = {k: v for k, v in props.items() if v is not None}
            
            query = f"""
                MATCH (a:GraphRAG {{entity_id: $src}})
                MATCH (b:GraphRAG {{entity_id: $tgt}})
                MERGE (a)-[r:{rtype}]->(b)
            """
            if clean_props:
                for k, v in clean_props.items():
                    query += f" SET r.{k} = ${k}"
            
            params = {"src": src, "tgt": tgt, **clean_props}
            
            try:
                sess.run(query, **params)
                stats["relations_created"] += 1
            except Exception as e:
                print(f"  Relation error ({src}->{tgt}): {e}")
    
    return stats


# ──────────────────────────────────────────────
# Phase 3: サブグラフ検索（Retrieval）
# ──────────────────────────────────────────────

def retrieve_subgraph_for_matching(project_id: str, max_hops: int = 3) -> dict:
    """案件ノード周辺のサブグラフを検索して取得
    
    LightRAG 的な dual-level retrieval:
      低レベル: 案件が REQUIRES するスキルに直接接続する要員
      高レベル: RELATED_TO / USED_IN 経由で間接的に関連する要員・経験
    """
    with get_driver().session() as sess:
        # 案件の周辺サブグラフ（max_hops ホップ以内）
        result = sess.run("""
            MATCH (p:GraphRAG {entity_id: $pid})
            CALL apoc.path.subgraphAll(p, {
                maxLevel: $hops,
                relationshipFilter: "REQUIRES|HAS_SKILL|RELATED_TO|USED_IN|HAS_EXPERIENCE|LOCATED_IN|PREFERS|WORKS_IN"
            })
            YIELD nodes, relationships
            RETURN nodes, relationships
        """, pid=project_id, hops=max_hops).data()
        
        if not result:
            # APOC がない場合のフォールバック
            return _retrieve_subgraph_basic(sess, project_id)
        
        nodes = []
        edges = []
        for record in result:
            for node in record.get("nodes", []):
                labels = list(node.labels)
                props = dict(node)
                nodes.append({"labels": labels, "properties": props})
            for rel in record.get("relationships", []):
                edges.append({
                    "type": rel.type,
                    "source": dict(rel.start_node).get("entity_id", ""),
                    "target": dict(rel.end_node).get("entity_id", ""),
                    "properties": dict(rel),
                })
        
        return {"nodes": nodes, "edges": edges}


def _retrieve_subgraph_basic(sess, project_id: str) -> dict:
    """APOC なしのフォールバック: Cypher で2ホップ探索"""
    # 案件→スキル
    result1 = sess.run("""
        MATCH (p:GraphRAG {entity_id: $pid})
        OPTIONAL MATCH (p)-[r1]->(n1:GraphRAG)
        OPTIONAL MATCH (n1)-[r2]->(n2:GraphRAG)
        OPTIONAL MATCH (n2)-[r3]->(n3:GraphRAG)
        WITH collect(DISTINCT p) + collect(DISTINCT n1) + collect(DISTINCT n2) + collect(DISTINCT n3) AS all_nodes,
             collect(DISTINCT r1) + collect(DISTINCT r2) + collect(DISTINCT r3) AS all_rels
        UNWIND all_nodes AS n
        WITH collect(DISTINCT n) AS nodes, all_rels
        UNWIND all_rels AS r
        RETURN nodes, collect(DISTINCT r) AS rels
    """, pid=project_id).data()
    
    # 逆方向: 要員→スキル→案件
    result2 = sess.run("""
        MATCH (p:GraphRAG {entity_id: $pid})-[:REQUIRES]->(s:Skill)
        OPTIONAL MATCH (e:Person)-[:HAS_SKILL]->(s)
        OPTIONAL MATCH (e)-[r]->(related:GraphRAG)
        RETURN collect(DISTINCT e) AS engineers,
               collect(DISTINCT s) AS skills,
               collect(DISTINCT related) AS related_nodes,
               collect(DISTINCT r) AS rels
    """, pid=project_id).data()
    
    nodes = []
    edges = []
    seen_ids = set()
    
    def add_node(node):
        if node is None:
            return
        props = dict(node)
        eid = props.get("entity_id", "")
        if eid and eid not in seen_ids:
            seen_ids.add(eid)
            nodes.append({
                "labels": list(node.labels),
                "properties": props,
            })
    
    for r in result1:
        for n in (r.get("nodes") or []):
            add_node(n)
    
    for r in result2:
        for key in ["engineers", "skills", "related_nodes"]:
            for n in (r.get(key) or []):
                add_node(n)
    
    return {"nodes": nodes, "edges": edges}


def subgraph_to_text(subgraph: dict) -> str:
    """サブグラフをテキスト表現に変換（LLM に渡す用）"""
    lines = []
    
    # ノード
    persons = []
    projects = []
    skills = []
    others = []
    
    for node in subgraph.get("nodes", []):
        labels = node.get("labels", [])
        props = node.get("properties", {})
        name = props.get("name", "?")
        
        if "Person" in labels:
            details = []
            if props.get("price_min"):
                details.append(f"単価:{props['price_min']}万円")
            if props.get("available"):
                details.append(f"参画可能:{props['available']}")
            if props.get("remote"):
                details.append(f"{props['remote']}")
            persons.append(f"  要員: {name}" + (f" ({', '.join(details)})" if details else ""))
        elif "Project" in labels:
            details = []
            if props.get("price_min") and props.get("price_max"):
                details.append(f"単価:{props['price_min']}〜{props['price_max']}万円")
            if props.get("available"):
                details.append(f"開始:{props['available']}")
            projects.append(f"  案件: {name}" + (f" ({', '.join(details)})" if details else ""))
        elif "Skill" in labels:
            cat = props.get("category", "")
            skills.append(f"  スキル: {name}" + (f" [{cat}]" if cat else ""))
        else:
            label_str = "/".join(l for l in labels if l != "GraphRAG")
            others.append(f"  {label_str}: {name}")
    
    if projects:
        lines.append("【案件】")
        lines.extend(projects)
    if persons:
        lines.append("\n【要員候補】")
        lines.extend(persons)
    if skills:
        lines.append("\n【スキルノード】")
        lines.extend(skills)
    if others:
        lines.append("\n【その他のエンティティ】")
        lines.extend(others)
    
    # エッジ
    edges = subgraph.get("edges", [])
    if edges:
        lines.append("\n【グラフのリレーション】")
        for edge in edges[:30]:
            props_str = ""
            eprops = edge.get("properties", {})
            if eprops.get("years"):
                props_str = f" ({eprops['years']}年)"
            lines.append(f"  ({edge['source']}) -[{edge['type']}]-> ({edge['target']}){props_str}")
    
    return "\n".join(lines)


# ──────────────────────────────────────────────
# Phase 4: LLM ベースの推論マッチング
# ──────────────────────────────────────────────

MATCHING_PROMPT = """あなたはIT人材と案件のマッチングエキスパートです。

以下のナレッジグラフ情報（案件と候補要員のサブグラフ）を分析し、
各要員と案件の適合度を評価してください。

出力はJSON**のみ**:
{
  "matches": [
    {
      "engineer_name": "要員名",
      "score": 0-100の整数,
      "rank": "最有力|有力|候補|要検討",
      "reasoning": "マッチング理由（2-3文の日本語）",
      "strengths": ["強み1", "強み2"],
      "gaps": ["不足点1"],
      "recommendation": "提案方針（1文）"
    }
  ]
}

評価基準:
1. スキル一致（直接一致 + グラフ上の関連スキル）: 50%
2. 経験の質（同じスキルでも使い方の文脈が合うか）: 20%
3. 単価適合: 15%
4. 勤務地・時期: 15%

重要: グラフの RELATED_TO リレーションを考慮し、
直接マッチしないが関連するスキルも評価に含めてください。
例: React経験者は Next.js 案件に対して RELATED_TO(0.85) で間接マッチ。"""


def graphrag_matching(project_id: str) -> list:
    """GraphRAG ベースのマッチング
    
    1. 案件周辺のサブグラフを取得
    2. サブグラフをテキスト化
    3. LLM に推論させる
    4. 結果をパース
    """
    # サブグラフ取得
    subgraph = _retrieve_subgraph_basic(get_driver().session(), project_id)
    
    if not subgraph["nodes"]:
        return []
    
    # テキスト化
    graph_text = subgraph_to_text(subgraph)
    
    if not graph_text.strip():
        return []
    
    # LLM 推論
    content = call_qwen(
        MATCHING_PROMPT,
        f"以下のナレッジグラフ情報からマッチング評価を行ってください:\n\n{graph_text}"
    )
    
    if content is None:
        # LLM なしの場合はルールベースにフォールバック
        return _fallback_matching(subgraph)
    
    try:
        content = re.sub(r"```json\s*|```\s*", "", content).strip()
        result = json.loads(content)
        matches = result.get("matches", [])
        
        # project 情報を追加
        for m in matches:
            m["project_id"] = project_id
            m["method"] = "graphrag"
        
        return matches
    except (json.JSONDecodeError, KeyError) as e:
        print(f"  Matching parse error: {e}")
        return _fallback_matching(subgraph)


def _fallback_matching(subgraph: dict) -> list:
    """LLM なしのフォールバック: サブグラフからルールベースでスコア計算"""
    persons = []
    projects = []
    skills_by_entity = defaultdict(set)
    
    for node in subgraph.get("nodes", []):
        labels = node.get("labels", [])
        props = node.get("properties", {})
        if "Person" in labels:
            persons.append(props)
        elif "Project" in labels:
            projects.append(props)
    
    # エッジからスキル関連を構築
    for edge in subgraph.get("edges", []):
        if edge["type"] in ("HAS_SKILL", "REQUIRES"):
            skills_by_entity[edge["source"]].add(edge["target"])
    
    matches = []
    for proj in projects:
        pid = proj.get("entity_id", "")
        proj_skills = skills_by_entity.get(pid, set())
        
        for person in persons:
            eid = person.get("entity_id", "")
            eng_skills = skills_by_entity.get(eid, set())
            
            overlap = proj_skills & eng_skills
            score = int(len(overlap) / max(len(proj_skills), 1) * 100)
            
            rank = "最有力" if score >= 85 else "有力" if score >= 70 else "候補" if score >= 50 else "要検討"
            
            matches.append({
                "engineer_name": person.get("name", "?"),
                "project_id": pid,
                "score": score,
                "rank": rank,
                "reasoning": f"スキル一致: {len(overlap)}/{len(proj_skills)}",
                "strengths": list(overlap)[:5],
                "gaps": list(proj_skills - eng_skills)[:5],
                "recommendation": "ルールベース評価",
                "method": "fallback",
            })
    
    matches.sort(key=lambda x: x["score"], reverse=True)
    return matches


# ──────────────────────────────────────────────
# Phase 5: 全案件マッチング
# ──────────────────────────────────────────────

def graphrag_matching_all() -> list:
    """全案件に対して GraphRAG マッチングを実行"""
    with get_driver().session() as sess:
        projects = sess.run("""
            MATCH (p:Project:GraphRAG)
            RETURN p.entity_id AS pid, p.name AS name
        """).data()
    
    all_matches = []
    for proj in projects:
        pid = proj["pid"]
        matches = graphrag_matching(pid)
        for m in matches:
            m["project_name"] = proj["name"]
        all_matches.extend(matches)
    
    all_matches.sort(key=lambda x: x.get("score", 0), reverse=True)
    return all_matches


# ──────────────────────────────────────────────
# Flask 統合用のエンドポイント関数
# ──────────────────────────────────────────────

def register_graphrag_routes(app):
    """Flask app に GraphRAG エンドポイントを追加"""
    from flask import jsonify, request
    
    @app.route("/api/graphrag/extract", methods=["POST"])
    def graphrag_extract():
        """GraphRAG: テキストからグラフ構造を抽出"""
        data = request.json
        text = data.get("text", "")
        doc_type = data.get("type", "engineer")
        
        if not text.strip():
            return jsonify({"error": "テキストが空です"}), 400
        
        extracted = extract_graph_entities(text, doc_type)
        
        # Neo4j に登録
        doc_id = hashlib.md5(text[:200].encode()).hexdigest()[:12]
        stats = register_graph_entities(extracted, doc_type, doc_id)
        
        return jsonify({
            "extracted": extracted,
            "registered": stats,
            "doc_id": doc_id,
        })
    
    @app.route("/api/graphrag/matching", methods=["GET"])
    def graphrag_match():
        """GraphRAG: LLM 推論ベースのマッチング"""
        project_id = request.args.get("project_id", "")
        
        if project_id:
            matches = graphrag_matching(project_id)
        else:
            matches = graphrag_matching_all()
        
        return jsonify(matches)
    
    @app.route("/api/graphrag/subgraph/<entity_id>", methods=["GET"])
    def graphrag_subgraph(entity_id):
        """GraphRAG: 特定ノード周辺のサブグラフを取得"""
        with get_driver().session() as sess:
            subgraph = _retrieve_subgraph_basic(sess, entity_id)
        
        return jsonify({
            "subgraph": subgraph,
            "text_representation": subgraph_to_text(subgraph),
        })
    
    @app.route("/api/graphrag/stats", methods=["GET"])
    def graphrag_stats():
        """GraphRAG: グラフ統計"""
        with get_driver().session() as sess:
            stats = sess.run("""
                MATCH (n:GraphRAG)
                WITH labels(n) AS lbls, count(n) AS cnt
                UNWIND lbls AS lbl
                WITH lbl, sum(cnt) AS total
                WHERE lbl <> 'GraphRAG'
                RETURN lbl AS label, total AS count
                ORDER BY total DESC
            """).data()
            
            rel_stats = sess.run("""
                MATCH (:GraphRAG)-[r]->(:GraphRAG)
                RETURN type(r) AS type, count(r) AS count
                ORDER BY count DESC
            """).data()
        
        return jsonify({
            "node_types": stats,
            "relation_types": rel_stats,
        })
    
    @app.route("/api/graphrag/clear", methods=["POST"])
    def graphrag_clear():
        """GraphRAG: GraphRAG ラベル付きノードのみクリア"""
        with get_driver().session() as sess:
            result = sess.run("""
                MATCH (n:GraphRAG)
                WHERE NOT n:Skill OR n.doc_id IS NOT NULL
                DETACH DELETE n
                RETURN count(n) AS deleted
            """).single()
        
        return jsonify({"deleted": result["deleted"]})
    
    print("  GraphRAG エンドポイント登録完了:")
    print("    POST /api/graphrag/extract")
    print("    GET  /api/graphrag/matching")
    print("    GET  /api/graphrag/subgraph/<id>")
    print("    GET  /api/graphrag/stats")
    print("    POST /api/graphrag/clear")


# ──────────────────────────────────────────────
# テスト実行
# ──────────────────────────────────────────────

def test_graphrag():
    """GraphRAG パイプラインのテスト"""
    print("=" * 60)
    print("  GraphRAG スキルマッチング テスト")
    print("=" * 60)
    
    # テストデータ
    engineer_text = """
氏名：佐藤 健一
経歴概要：
Java/Spring Boot を中心に8年のバックエンド開発経験。
直近3年間はAWS上でマイクロサービスアーキテクチャの設計・構築に従事。
EC系サービスのAPI設計、決済連携、在庫管理システムの開発実績あり。
Docker/Kubernetes での CI/CD 構築も経験。
PostgreSQL でのデータモデリング、パフォーマンスチューニングが得意。

技術スキル：
- Java (8年), Spring Boot (6年), AWS (5年)
- Docker (3年), Kubernetes (2年)
- PostgreSQL (4年), Redis (2年)
- マイクロサービス設計, REST API, GraphQL

希望条件：
- 単価：80万円
- 勤務地：東京、リモート可
- 参画可能：2026年5月
"""

    project_text = """
案件名：大手小売向けEC基盤刷新
顧客：大手小売企業（売上高5000億円規模）

プロジェクト概要：
既存モノリシックECプラットフォームのマイクロサービス化。
Spring Boot ベースの API Gateway + 各種ドメインサービスの設計・開発。
AWS ECS/EKS 上での運用を前提としたクラウドネイティブ設計。

必須スキル：Java, Spring Boot, AWS
歓迎スキル：Docker, Kubernetes, マイクロサービス設計経験
単価：75〜85万円
勤務地：東京・リモート併用
開始：2026年5月
"""
    
    # Phase 1: 抽出
    print("\n[Phase 1] エンティティ抽出")
    eng_graph = extract_graph_entities(engineer_text, "engineer")
    proj_graph = extract_graph_entities(project_text, "project")
    
    print(f"  要員: {len(eng_graph.get('entities', []))} entities, "
          f"{len(eng_graph.get('relations', []))} relations")
    print(f"  案件: {len(proj_graph.get('entities', []))} entities, "
          f"{len(proj_graph.get('relations', []))} relations")
    
    # Phase 2: グラフ登録
    print("\n[Phase 2] Neo4j 登録")
    eng_stats = register_graph_entities(eng_graph, "engineer", "test_eng_001")
    proj_stats = register_graph_entities(proj_graph, "project", "test_proj_001")
    print(f"  要員: {eng_stats}")
    print(f"  案件: {proj_stats}")
    
    # Phase 3-4: マッチング
    print("\n[Phase 3-4] GraphRAG マッチング")
    # 案件のIDを見つける
    proj_entities = proj_graph.get("entities", [])
    proj_id = None
    for e in proj_entities:
        if e["type"] == "Project":
            proj_id = e["id"]
            break
    
    if proj_id:
        matches = graphrag_matching(proj_id)
        print(f"  結果: {len(matches)} マッチ")
        for m in matches:
            print(f"    {m.get('engineer_name', '?')}: {m.get('score', 0)}% "
                  f"({m.get('rank', '?')}) - {m.get('reasoning', '')[:60]}")
    else:
        print("  案件IDが見つかりません")
    
    print("\n  テスト完了！")


if __name__ == "__main__":
    test_graphrag()