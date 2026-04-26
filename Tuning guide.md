# LightRAG スキルマッチング チューニングガイド

## チューニングの優先順位

| 優先度 | 項目 | 効果 | コスト | 概要 |
|--------|------|------|--------|------|
| 1 | sentence-transformers 導入 | 最大 | 低 | ベクトル間接マッチの前提条件 |
| 2 | スキル名正規化 | 大 | 低 | 表記揺れで直接マッチが失敗する問題を解消 |
| 3 | 類似度閾値の調整 | 大 | 低 | tuning.py で最適値を自動計算 |
| 4 | Embedding モデル変更 | 大 | 中 | ITスキル間の意味的距離がより正確に |
| 5 | スコア重みの調整 | 中 | 低 | スキル/単価/勤務地の配分を最適化 |
| 6 | 間接マッチの減衰係数 | 中 | 低 | 間接マッチの評価比重を調整 |

## 1. sentence-transformers を入れる（最優先）

これがないと TF-IDF フォールバックになり、間接マッチの精度が大幅に低下する。入れるだけで劇的に改善する。

```bash
pip install sentence-transformers
```

初回実行時にモデル（約500MB）がダウンロードされる。

## 2. スキル名の正規化（効果大・コスト低）

「JS」と「JavaScript」が別ノードになると直接マッチすら失敗する。`tuning.py` 内の `SKILL_SYNONYMS` 辞書と `normalize_skill_name()` 関数を `lightrag_matching.py` の `_rule_extract` に組み込むことで、表記揺れを解消できる。

```python
SKILL_SYNONYMS = {
    "JavaScript": ["JS", "javascript", "Java Script"],
    "TypeScript": ["TS", "typescript"],
    "Kubernetes": ["k8s", "K8s", "kubernetes"],
    "PostgreSQL": ["Postgres", "postgres", "PG"],
    "機械学習": ["ML", "Machine Learning"],
    "深層学習": ["DL", "Deep Learning", "ディープラーニング"],
    "自然言語処理": ["NLP", "Natural Language Processing"],
    "Spring Boot": ["SpringBoot", "spring boot"],
    "Next.js": ["NextJS", "Nextjs", "next.js"],
    "Vue.js": ["VueJS", "Vuejs", "vue.js", "Vue"],
    "Node.js": ["NodeJS", "Nodejs", "node.js"],
    ".NET": ["dotnet", "DotNet", ".net"],
    "scikit-learn": ["sklearn", "Scikit-learn"],
}
```

## 3. 類似度閾値の調整（効果大）

`python tuning.py` を実行すると、正例（マッチすべきペア）と負例（マッチすべきでないペア）の類似度分布が表示され、最適閾値が自動計算される。

```bash
python tuning.py
```

出力例:

```
  閾値    正例Hit  負例排除    F1
  0.50    22/22    3/10    0.815
  0.55    20/22    5/10    0.851
  0.60    18/22    7/10    0.878  ◀ best
  0.65    15/22    8/10    0.857
  0.70    12/22    9/10    0.800
  0.75    10/22   10/10    0.769

  推奨閾値: 0.60 (F1: 0.878)
```

算出された閾値を `lightrag_matching.py` の `match_project` 内に反映する:

```python
# 変更前
if sim < 0.75:
    break

# 変更後（tuning.py の推奨値に合わせる）
if sim < 0.60:
    break
```

モデルを変更したら `tuning.py` を再実行して閾値を再計算すること。

## 4. Embedding モデルの変更（効果大・メモリ増）

`lightrag_matching.py` の設定を変更する。

```python
# 現在（汎用、軽量）
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # 384次元, 500MB
EMBED_DIM = 384

# 推奨: 日本語特化（精度↑↑、16GB PC で動作可能）
EMBED_MODEL = "intfloat/multilingual-e5-base"  # 768次元, 1.1GB
EMBED_DIM = 768

# 最高精度（メモリに余裕がある場合）
EMBED_MODEL = "intfloat/multilingual-e5-large"  # 1024次元, 2.2GB
EMBED_DIM = 1024
```

16GB PC で Ollama + Neo4j と同時運用する場合は `e5-base`（1.1GB）が現実的な上限。`e5-large` は Ollama を停止すれば使える。

## 5. スコア重みの調整（効果中）

`tuning.py` の `simulate_scoring()` で、異なる重み配分でのスコアを比較できる。

```python
# lightrag_matching.py の match_project 内

# 現在
total = int(skill_score * 60 + price_score * 20 + 0.5 * 20)

# スキル重視（スキル適合を最重要視する場合）
total = int(skill_score * 70 + price_score * 15 + 0.5 * 15)

# バランス型（単価・勤務地も重視する場合）
total = int(skill_score * 50 + price_score * 25 + 0.5 * 25)
```

どの重みが最適かは、実際のマッチング結果と営業判断の一致度で判断する。

## 6. 間接マッチの減衰係数（効果中）

間接マッチが直接マッチに対してどの程度の価値を持つかを制御する。

```python
# lightrag_matching.py の match_project 内

# 現在: 間接マッチは直接マッチの最大60%の価値
skill_points += ind["similarity"] * 0.6

# 積極評価: 間接マッチを高く評価（関連スキル保有者を拾いやすい）
skill_points += ind["similarity"] * 0.8

# 保守評価: 間接マッチを低く評価（直接一致を重視）
skill_points += ind["similarity"] * 0.4
```

シミュレーション（案件: Java/Spring Boot/AWS、要員: Java, Spring Boot, Docker(sim=0.62)）:

```
decay=0.3  skill_score=0.729  final=64%
decay=0.4  skill_score=0.749  final=65%
decay=0.5  skill_score=0.770  final=66%
decay=0.6  skill_score=0.791  final=67%  ◀ 現在
decay=0.7  skill_score=0.811  final=69%
decay=0.8  skill_score=0.832  final=70%
```

## チューニングの実行手順

```bash
# 1. sentence-transformers インストール
pip install sentence-transformers

# 2. チューニングツール実行
python tuning.py

# 3. 出力された推奨閾値を lightrag_matching.py に反映

# 4. Flask 再起動
python app_qwen.py

# 5. デモデータでマッチング結果を確認

# 6. 結果が期待と異なれば、重み・減衰係数を調整して再実行
```
