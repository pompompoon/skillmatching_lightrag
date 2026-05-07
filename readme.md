# LightRAG スキルマッチングシステム

IT人材のスキルシートと求人票をドロップするだけで、LightRAGアーキテクチャによるインテリジェントなマッチングを実行するシステム。

## 概要

本システムは、香港大学（HKUDS）が提案した [LightRAG](https://github.com/HKUDS/LightRAG) のアーキテクチャに準拠し、IT人材マッチングドメインに特化して実装したものです。テキスト形式のスキルシート・求人票から自動的にナレッジグラフを構築し、ベクトル埋め込みとグラフ構造の両方を活用して、文字列一致では発見できない「意味的に近いスキル」の間接マッチングを実現します。

従来のルールベースマッチング（スキル名の完全一致 + 手動定義の関連性）に対して、LightRAGは以下の優位性を持ちます。

- 「マイクロサービス設計」と「分散システム構築」のような、文字列は異なるが意味的に近いスキルを自動検出
- スキルシートの自由記述テキストから、経験の文脈（何に使ったか、どの業界か）も構造化
- エンティティの表記揺れ（Spring Boot / SpringBoot / スプリングブート）を自動統合
- マッチング理由を自然言語で説明

  <img width="1778" height="1125" alt="image" src="https://github.com/user-attachments/assets/97860e05-54db-49a1-8ebc-b44352646567" />
スキルシート、求人データ入力画面
  <img width="1652" height="1125" alt="image" src="https://github.com/user-attachments/assets/bce85282-5f0a-4f1c-a202-066254041aae" />
  
■マッチング結果
  <img width="1800" height="1047" alt="image" src="https://github.com/user-attachments/assets/46ba78cb-a70f-43a1-9e86-7d3658a7c1bf" />

■スキルグラフ
  <img width="2000" height="745" alt="image" src="https://github.com/user-attachments/assets/cf616695-59d2-4393-888c-2455c3ecdffb" />
<img width="1650" height="644" alt="image" src="https://github.com/user-attachments/assets/6dc7642a-a76c-4fe3-baf2-a5a5be550ee1" />




## アーキテクチャ

```
┌─────────────────────────────────────────────────────────┐
│                    フロントエンド                          │
│              http://localhost:5000                        │
│    ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│    │Rule-based│  │ GraphRAG │  │ LightRAG │ ← 3モード   │
│    └──────────┘  └──────────┘  └──────────┘             │
└──────────────────────┬──────────────────────────────────┘
                       │ REST API
┌──────────────────────┴──────────────────────────────────┐
│                  Flask Backend                           │
│                  app_qwen.py                             │
│                                                          │
│  ┌─────────────┐ ┌──────────────┐ ┌──────────────────┐  │
│  │ Rule-based  │ │ GraphRAG     │ │ LightRAG         │  │
│  │ (Cypher)    │ │ (SubGraph    │ │ (Vector+Graph    │  │
│  │             │ │  +LLM推論)   │ │  +Dual Retrieval)│  │
│  └─────────────┘ └──────────────┘ └──────────────────┘  │
└────────┬─────────────────┬──────────────────┬───────────┘
         │                 │                  │
    ┌────┴────┐     ┌──────┴──────┐   ┌──────┴──────┐
    │ Neo4j   │     │ LLM        │   │ sentence-   │
    │ Graph   │     │ (自動検出)  │   │ transformers│
    │ Database│     │ ①Ollama    │   │ + hnswlib   │
    └─────────┘     │ ②Qwen API │   └─────────────┘
                    │ ③なし(規則)│
                    └─────────────┘
```

## LightRAG パイプライン詳細

### Phase 1: Insert（データ投入）

```
テキスト入力
    │
    ▼
┌──────────────┐
│ チャンク分割   │  500文字単位、50文字オーバーラップ
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ LLM          │  Qwen API でエンティティ＆リレーション抽出
│ Entity       │  ノード型: Person, Skill, Experience,
│ Extraction   │           Project, Industry, Location
└──────┬───────┘  エッジ型: HAS_SKILL, REQUIRES,
       │                   RELATED_TO, USED_IN, ...
       ▼
┌──────────────┐
│ Entity       │  LLM で表記揺れを統合
│ Deduplication│  "Spring Boot" = "SpringBoot" = "スプリングブート"
└──────┬───────┘
       │
       ├──────────────────────────────┐
       ▼                              ▼
┌──────────────┐              ┌──────────────┐
│ Neo4j        │              │ Vector Store │
│ グラフ格納    │              │ (hnswlib)    │
│ :LR_Entity   │              │ entity_store │
│ ノード+エッジ │              │ relation_store│
└──────────────┘              │ chunk_store  │
                              └──────────────┘
```

### Phase 2: Query（検索＋マッチング）

```
クエリ: 「EC基盤刷新に合う人は？」
    │
    ▼
┌──────────────┐
│ LLM          │  low_level:  ["Java", "Spring Boot", "AWS"]
│ Keyword      │  high_level: ["EC系開発", "マイクロサービス"]
│ Extraction   │
└──────┬───────┘
       │
       ├─── low-level ───┐         ├─── high-level ──┐
       ▼                 │         ▼                  │
┌──────────────┐         │  ┌──────────────┐          │
│ Entity       │         │  │ Relation     │          │
│ Vector       │         │  │ Vector       │          │
│ Search       │         │  │ Search       │          │
└──────┬───────┘         │  └──────┬───────┘          │
       │                 │         │                   │
       ▼                 │         ▼                   │
┌──────────────┐         │  ┌──────────────┐          │
│ Graph        │         │  │ Chunk        │          │
│ Neighbor     │◄────────┘  │ Vector       │◄─────────┘
│ Expansion    │            │ Search       │
└──────┬───────┘            └──────┬───────┘
       │                           │
       └─────────┬─────────────────┘
                 ▼
┌────────────────────────┐
│ Context Assembly       │  検索結果をテキスト化
│ + LLM Reasoning        │  → LLM で推論・スコアリング
└────────────────────────┘
```

### スキルマッチング固有の処理

通常の LightRAG の query パイプラインに加えて、スキルマッチング固有の `match_project` メソッドを実装しています。

```
案件の必須スキル ──── 直接マッチ ────── 要員のスキル
  {Java,                 │              {Java,
   Spring Boot,          │               Spring Boot,
   AWS}                  │               PostgreSQL,
                         │               Docker}
                         │
                         │  ✓ Java: 完全一致
                         │  ✓ Spring Boot: 完全一致
                         │  ✗ AWS: 不一致
                         │
                         ▼
             ベクトル間接マッチ
             (similarity ≥ 0.75)
                         │
             AWS の埋め込み ←→ Docker の埋め込み
             cosine_sim = 0.62 → ✗ 不採用
                         │
             AWS の埋め込み ←→ 他のスキル...
                         │
          ※ 1:1 貪欲マッチング（最良ペアのみ採用）
```

スコア計算:

```
直接マッチ: 1.0 点 / スキル
間接マッチ: similarity × 0.6 点 / スキル
スキルスコア = 合計点 / 必須スキル数 (上限 1.0)

総合スコア = スキル × 60% + 単価 × 20% + 勤務地 × 20%
```

## ファイル構成

```
skillmatching/
├── app_qwen.py              # メインサーバー (Flask + フロントエンドHTML)
│                             # ルールベースマッチング + 全モジュール統合
├── lightrag_matching.py      # LightRAG 本体
│                             # Vector Store, Embedding, Dual-level Retrieval
├── graphrag_matching.py      # GraphRAG モジュール (比較用)
├── setup_neo4j.py            # Neo4j 初期スキーマ + デモデータ投入
├── gnn_matching.py           # GATv2Conv GNN (オプション)
└── requirements.txt          # Python 依存パッケージ
```

## セットアップ

### 1. 前提条件

- Python 3.10+
- Neo4j Desktop 2.x（ローカルインスタンス起動済み）
- Ollama（推奨、ローカル LLM 用）または Qwen API キー（オプション）

### 2. パッケージインストール

```bash
pip install flask flask-cors neo4j requests numpy sentence-transformers hnswlib
```

`sentence-transformers` がインストールできない場合は TF-IDF フォールバックで動作します。`hnswlib` がインストールできない場合は numpy brute-force で動作します。最低限必要なのは `flask flask-cors neo4j requests numpy` のみです。

### 3. Neo4j 設定

Neo4j Desktop でインスタンスを作成し、RUNNING 状態にしてください。

```
Connection URI: neo4j://127.0.0.1:7687
Username: neo4j
Password: （インスタンス作成時に設定したもの）
```

パスワードが異なる場合は、`app_qwen.py` と `lightrag_matching.py` の `NEO4J_AUTH` を修正するか、環境変数で指定してください。

```bash
# Windows
set NEO4J_PASS=your_password

# Mac/Linux
export NEO4J_PASS=your_password
```

### 4. LLM 設定

LLM はエンティティ抽出、重複排除、キーワード抽出、マッチング理由説明に使用します。システムは起動時に以下の優先順位で自動検出します。

```
①  Ollama (localhost:11434)      ← ローカル、無料、推奨
②  Qwen API (DASHSCOPE_API_KEY)  ← クラウド、無料枠あり
③  なし                          ← ルールベース抽出のみ（ベクトル検索は動作）
```

LLM がなくてもベクトル検索による間接マッチは動作します。LLM があるとエンティティ抽出の精度向上、表記揺れ統合、理由説明が追加されます。

#### 方法A: Ollama（推奨）

ローカルで動作し、API キー不要です。

```bash
# 1. Ollama インストール
#    https://ollama.com からダウンロード

# 2. モデルをダウンロード
ollama pull qwen3:4b

# 3. 環境変数でモデル指定（省略時は qwen2.5:7b）
set OLLAMA_MODEL=qwen2.5:7b 
set OLLAMA_MODEL=qwen3:4b        # Windows
export OLLAMA_MODEL=qwen3:4b     # Mac/Linux

# 別のターミナルで先にモデルをウォームアップ
ollama run qwen3:4b "hello"
ollama run qwen2.5:7b "hello"

# 4. 起動（Ollama が localhost:11434 で動いていれば自動検出）
python app_qwen.py
```

#### モデル選択ガイド（16GB PC 向け）

| モデル | サイズ | JSON安定性 | 推奨度 | 備考 |
|--------|--------|----------|--------|------|
| `qwen3:4b` | 2.6GB | 安定 | ◎ 最推奨 | テキスト専用、JSON出力が素直 |
| `qwen2.5:7b` | 4.7GB | 非常に安定 | ◎ | 実績多い、最も無難 |
| `qwen3.5:9b` | 6.6GB | 安定 | ○ | マルチモーダル、16GBで動作可能 |
| `qwen3.5:4b` | 3.4GB | やや不安定 | △ | OllamaでJSON崩れ報告あり |
| `gemma3:4b` | 3.3GB | 安定 | ○ | Google製、日本語もそこそこ |

本システムのタスクは「テキストから JSON 構造を抽出する」ことなので、JSON 出力の安定性が最重要です。モデルサイズが大きいほど安定しますが、16GB PC では 7B〜9B が上限の目安です。JSON パースエラーが発生した場合は自動的にルールベース（200+スキルの正規表現辞書）にフォールバックするため、マッチング自体は止まりません。

#### 方法B: Qwen API（クラウド）

```bash
# 1. DashScope でアカウント作成（無料）
#    https://dashscope.console.aliyun.com/

# 2. API-KEY管理 → キー作成

# 3. 環境変数に設定
set DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxx    # Windows
export DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxx # Mac/Linux

# 4. 起動
python app_qwen.py
```

#### 方法C: LLM なし

API キーも Ollama も不要です。ルールベース抽出（正規表現辞書）+ ベクトル検索で動作します。LLM による理由説明やエンティティ重複排除は使用できませんが、スキルマッチング自体は動作します。

```bash
python app_qwen.py
# 起動ログに "LLM: なし" と表示される
```

### 5. 起動

```bash
python app_qwen.py
```

起動ログで LLM バックエンドの検出結果が表示されます。

```
============================================================
  Skill Matching Backend
  Flask + Qwen API + Neo4j + GraphRAG
============================================================
  Neo4j 接続成功: neo4j://127.0.0.1:7687
  LLM: Ollama 検出 (model: qwen3:4b)     ← または "Qwen API" / "なし"
  LightRAG エンドポイント登録:
    POST /api/lightrag/insert
    POST /api/lightrag/query
    GET  /api/lightrag/matching

  フロントエンド: http://localhost:5000
```

ブラウザで http://localhost:5000 にアクセスしてください。

## 使い方

### データ入力

3つの方法でデータを投入できます。

1. **ファイルドロップ**: `.txt` / `.csv` / `.md` ファイルを左右のドロップゾーンにドラッグ＆ドロップ
2. **手入力**: 「+スキルシート手入力」「+求人データ手入力」ボタンからテキスト入力
3. **デモデータ**: 「サンプルデータ読込」ボタンで4名×4案件のデモデータを投入

### マッチング実行

画面下部の3つのボタンから実行方法を選択します。

| ボタン | 方式 | 特徴 |
|--------|------|------|
| **Rule-based** | Cypher 固定クエリ | 高速、決定的、文字列完全一致のみ |
| **GraphRAG** | サブグラフ + LLM 推論 | 理由説明あり、LLM 必要 |
| **LightRAG** | ベクトル検索 + グラフ + LLM | 間接マッチ検出、LLM なしでも動作 |

### マッチング結果の見方

各マッチング結果カードには以下の情報が含まれます。

- **スコア (0-100%)**: 総合適合度
- **ランク**: 最有力 (≥80) / 有力 (≥60) / 候補 (≥40) / 要検討 (<40)
- **青タグ**: 直接マッチしたスキル（名前が完全一致）
- **黄タグ**: 間接マッチしたスキル（ベクトル類似度 ≥ 0.75）
- **赤取消線タグ**: 不足スキル（直接にも間接にもマッチしなかった）
- **スコア内訳**: スキル / 単価 / 勤務地 の各サブスコア

カードをクリックすると詳細が展開され、間接マッチの `has →[similarity%]→ required` や LLM による理由説明が表示されます。

## API リファレンス

### LightRAG エンドポイント

```
POST /api/lightrag/insert
  Body: {"text": "スキルシートテキスト", "type": "engineer|project"}
  Response: {"doc_id": "...", "stats": {"chunks": 1, "entities": 5, "relations": 4}}

POST /api/lightrag/query
  Body: {"query": "Java経験のある要員は？", "mode": "hybrid|low|high"}
  Response: {"answer": "...", "context": "...", "entities": [...], "keywords": {...}}

GET /api/lightrag/matching
  Query: ?project_id=xxx (省略時は全案件)
  Response: [{score, rank, direct_matches, indirect_matches, missing_skills, ...}]

GET /api/lightrag/stats
  Response: {entities_vectorized, relations_vectorized, neo4j_nodes, neo4j_edges}

POST /api/lightrag/clear
  Response: {"status": "cleared"}
```

### ルールベースエンドポイント

```
POST /api/extract       テキストからエンティティ抽出
POST /api/register      抽出結果をグラフ登録
GET  /api/matching      ルールベースマッチング
GET  /api/graph/stats   グラフ統計
GET  /api/health        ヘルスチェック
POST /api/clear         データクリア
```

## 技術詳細

### Embedding モデル

デフォルトで `paraphrase-multilingual-MiniLM-L12-v2`（384次元）を使用します。日本語と英語の両方に対応した多言語モデルで、IT スキル名のような混合テキストの埋め込みに適しています。

`sentence-transformers` が未インストールの場合は、簡易 TF-IDF ベースのフォールバックが動作しますが、間接マッチの精度は低下します。

### Vector Store

`hnswlib` による近似最近傍検索を使用します（HNSW アルゴリズム、cosine 距離）。未インストール時は numpy による brute-force 検索にフォールバックします。データ量が数千件以下であれば brute-force でも十分な速度が出ます。

### Dual-level Retrieval

LightRAG の核心的な検索機構です。

- **Low-level**: 具体的なエンティティ名（`Java`, `Spring Boot`, `佐藤健一`）でエンティティベクトルストアを検索し、ヒットしたノードのグラフ上の隣接ノードも取得
- **High-level**: 抽象的なトピック（`バックエンド開発`, `クラウドネイティブ`）でリレーションベクトルストア＋チャンクベクトルストアを検索

### コンポーネント実装の選択

本実装は LightRAG のアーキテクチャ（LLMエンティティ抽出 → Entity dedup → ベクトル埋め込み → Dual-level retrieval → グラフ隣接拡張 → LLM回答生成）に準拠しています。以下の表は、本家リポジトリとのコンポーネントレベルでの実装選択の違いです。アーキテクチャ上の差異ではありません。

| コンポーネント | 本家リポジトリ | 本実装 |
|---------------|-------------|--------|
| LLM | GPT-4o / GPT-4o-mini | Ollama (qwen3:4b等) / Qwen API / ルールフォールバック |
| グラフストレージ | NetworkX / Neo4j | Neo4j |
| ベクトルストア | nano-vectordb | hnswlib + numpy フォールバック |
| 埋め込みモデル | text-embedding-3-large | paraphrase-multilingual-MiniLM-L12-v2 |

LLM バックエンドは起動時に自動検出されます（Ollama → Qwen API → なし）。本家が GPT-4o を前提とする部分を、ローカル LLM でも動作するようにフォールバック機構を組み込んでいます。

加えて、本家 LightRAG は汎用QAシステムですが、本実装ではスキルマッチングドメインに特化した `match_project` メソッド（ベクトル類似度による1:1貪欲スキルマッチング、単価・勤務地スコアリング）を追加しています。

## 3つのマッチング方式の比較

```
                文字列一致    意味的類似   グラフ構造   LLM推論   LLMコスト   速度
Rule-based      ✓            ✗           ✓(手動定義)  ✗        なし        高速
GraphRAG        ✓            ✗           ✓(LLM構築)   ✓        必須        遅い
LightRAG        ✓            ✓(vector)   ✓(LLM構築)   △(opt)   任意(Ollama) 中速
```

Rule-based は「Java = Java」の完全一致のみ。GraphRAG は LLM にサブグラフを渡して推論させるが、ベクトル検索がないため検索が粗い。LightRAG はベクトル埋め込みによる意味的類似度で間接マッチを検出しつつ、グラフ構造で関連ノードを拡張する。LLM がなくてもベクトル検索 + グラフは動作する。

## トラブルシューティング

### Neo4j に接続できない

Neo4j Desktop でインスタンスが RUNNING になっているか確認してください。Connection URI が `neo4j://127.0.0.1:7687` であること、パスワードが一致していることを確認してください。

### LightRAG マッチングが空を返す

「サンプルデータ読込」→「LightRAG」ボタンの順で実行してください。LightRAG は insert → matching の順序で動作し、データ投入前にマッチングを実行しても結果は空になります。

### 間接マッチが検出されない

`sentence-transformers` がインストールされていない場合、TF-IDF フォールバックではスキル間の意味的類似度が正確に計算されません。`pip install sentence-transformers` でインストールしてください。

### Qwen API エラー

API キーが未設定の場合はルールベース抽出にフォールバックします。LLM による理由説明やエンティティ重複排除は使用できませんが、ベクトル検索ベースのマッチングは動作します。

### Ollama が検出されない

Ollama が起動しているか確認してください。ブラウザで `http://localhost:11434` にアクセスして「Ollama is running」と表示されればOKです。表示されない場合は Ollama アプリケーションを起動してください。

```bash
# モデルが pull 済みか確認
ollama list

# まだなら pull
ollama pull qwen3:4b
```

### Ollama で JSON パースエラーが多発する

モデルサイズが小さいと JSON 出力が崩れることがあります。`qwen3:4b` や `qwen2.5:7b` が安定しています。`qwen3.5:4b` は Ollama との互換性に問題が報告されているため、エラーが多い場合は `qwen3:4b` に変更してください。

```bash
set OLLAMA_MODEL=qwen3:4b
python app_qwen.py
```

## ライセンス

本プロジェクトのコードは自由に利用・改変できます。LightRAG のアーキテクチャは [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG) (MIT License) に基づいています。
