"""
LightRAG チューニングツール
═══════════════════════════
1. スキルペアの類似度を確認して閾値を決定
2. スキル同義語辞書の効果を検証
3. 重みパラメータの最適化

使い方:
  python tuning.py
"""

import numpy as np
import sys

# ──────────────────────────────────────
# 1. Embedding モデルのロード
# ──────────────────────────────────────

def load_embedder():
    try:
        from sentence_transformers import SentenceTransformer
        model_name = "paraphrase-multilingual-MiniLM-L12-v2"
        print(f"Loading: {model_name}")
        model = SentenceTransformer(model_name)
        return model
    except ImportError:
        print("sentence-transformers 未インストール")
        print("pip install sentence-transformers")
        sys.exit(1)


# ──────────────────────────────────────
# 2. スキルペア類似度チェック
# ──────────────────────────────────────

def check_skill_similarities(model):
    """スキルペアの類似度を確認して閾値決定の参考にする"""

    # マッチすべきペア（正例）
    positive_pairs = [
        # 同義語・表記揺れ
        ("JavaScript", "JS"),
        ("TypeScript", "TS"),
        ("Kubernetes", "k8s"),
        ("機械学習", "Machine Learning"),
        ("深層学習", "Deep Learning"),
        ("自然言語処理", "NLP"),
        # 強い関連
        ("Java", "Spring Boot"),
        ("React", "Next.js"),
        ("TypeScript", "React"),
        ("Python", "Django"),
        ("Python", "機械学習"),
        ("C#", ".NET"),
        ("Vue.js", "Nuxt"),
        ("AWS", "Lambda"),
        ("Docker", "Kubernetes"),
        # 中程度の関連
        ("React", "Vue.js"),
        ("AWS", "GCP"),
        ("PostgreSQL", "MySQL"),
        ("Flask", "FastAPI"),
        ("マイクロサービス", "Docker"),
        ("バックエンド開発", "API設計"),
        ("フロントエンド開発", "UI設計"),
    ]

    # マッチすべきでないペア（負例）
    negative_pairs = [
        ("Java", "PostgreSQL"),
        ("React", "SQL Server"),
        ("Python", "Nginx"),
        ("TypeScript", "Oracle"),
        ("AWS", "Ruby"),
        ("Docker", "jQuery"),
        ("機械学習", "JIRA"),
        ("Spring Boot", "Vue.js"),
        ("フロントエンド開発", "データベース設計"),
        ("iOS開発", "ETL"),
    ]

    print("\n" + "=" * 70)
    print("  スキルペア類似度チェック")
    print("=" * 70)

    # 正例
    print("\n  ✓ マッチすべきペア（正例）:")
    print(f"  {'スキルA':<20s} {'スキルB':<20s} {'類似度':>8s}  判定")
    print("  " + "-" * 62)

    pos_scores = []
    for a, b in positive_pairs:
        vecs = model.encode([a, b], normalize_embeddings=True)
        sim = float(np.dot(vecs[0], vecs[1]))
        pos_scores.append(sim)
        marker = "✓" if sim >= 0.75 else "△" if sim >= 0.5 else "✗"
        print(f"  {a:<20s} {b:<20s} {sim:>8.3f}  {marker}")

    # 負例
    print(f"\n  ✗ マッチすべきでないペア（負例）:")
    print(f"  {'スキルA':<20s} {'スキルB':<20s} {'類似度':>8s}  判定")
    print("  " + "-" * 62)

    neg_scores = []
    for a, b in negative_pairs:
        vecs = model.encode([a, b], normalize_embeddings=True)
        sim = float(np.dot(vecs[0], vecs[1]))
        neg_scores.append(sim)
        marker = "✓" if sim < 0.75 else "✗ (誤マッチ)"
        print(f"  {a:<20s} {b:<20s} {sim:>8.3f}  {marker}")

    # 最適閾値の推定
    print("\n" + "=" * 70)
    print("  閾値分析")
    print("=" * 70)
    print(f"  正例 平均: {np.mean(pos_scores):.3f}  "
          f"最小: {np.min(pos_scores):.3f}  最大: {np.max(pos_scores):.3f}")
    print(f"  負例 平均: {np.mean(neg_scores):.3f}  "
          f"最小: {np.min(neg_scores):.3f}  最大: {np.max(neg_scores):.3f}")

    # 各閾値でのF1スコア
    print(f"\n  {'閾値':>6s}  {'正例Hit':>7s}  {'負例排除':>8s}  {'F1':>6s}")
    print("  " + "-" * 40)
    best_f1 = 0
    best_threshold = 0.5

    for t in np.arange(0.3, 0.95, 0.05):
        tp = sum(1 for s in pos_scores if s >= t)
        tn = sum(1 for s in neg_scores if s < t)
        fp = sum(1 for s in neg_scores if s >= t)
        fn = sum(1 for s in pos_scores if s < t)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        marker = " ◀ best" if f1 > best_f1 else ""
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

        print(f"  {t:>6.2f}  {tp:>3d}/{len(pos_scores):<3d}  "
              f"{tn:>3d}/{len(neg_scores):<3d}   {f1:>5.3f}{marker}")

    print(f"\n  推奨閾値: {best_threshold:.2f} (F1: {best_f1:.3f})")
    print(f"  → lightrag_matching.py の match_project 内:")
    print(f"    if sim < {best_threshold:.2f}:  # この値に変更")

    return best_threshold


# ──────────────────────────────────────
# 3. スキル同義語辞書
# ──────────────────────────────────────

SKILL_SYNONYMS = {
    # 正規名: [同義語リスト]
    "JavaScript": ["JS", "javascript", "Java Script"],
    "TypeScript": ["TS", "typescript"],
    "Kubernetes": ["k8s", "K8s", "kubernetes"],
    "PostgreSQL": ["Postgres", "postgres", "PG"],
    "MongoDB": ["Mongo", "mongo"],
    "機械学習": ["ML", "Machine Learning", "machine learning"],
    "深層学習": ["DL", "Deep Learning", "deep learning", "ディープラーニング"],
    "自然言語処理": ["NLP", "nlp", "Natural Language Processing"],
    "画像認識": ["CV", "Computer Vision", "コンピュータビジョン"],
    "Spring Boot": ["SpringBoot", "spring boot", "Spring boot"],
    "Next.js": ["NextJS", "Nextjs", "next.js"],
    "Vue.js": ["VueJS", "Vuejs", "vue.js", "Vue"],
    "Node.js": ["NodeJS", "Nodejs", "node.js"],
    "React": ["ReactJS", "React.js", "react"],
    ".NET": ["dotnet", "DotNet", ".net"],
    "scikit-learn": ["sklearn", "Scikit-learn"],
    "GitHub Actions": ["GHA", "Github Actions"],
    "CI/CD": ["CICD", "CI CD"],
}


def normalize_skill_name(name: str) -> str:
    """スキル名を正規化"""
    # 完全一致チェック
    for canonical, synonyms in SKILL_SYNONYMS.items():
        if name == canonical:
            return canonical
        if name in synonyms:
            return canonical

    # 大文字小文字無視
    name_lower = name.lower().strip()
    for canonical, synonyms in SKILL_SYNONYMS.items():
        if name_lower == canonical.lower():
            return canonical
        if name_lower in [s.lower() for s in synonyms]:
            return canonical

    return name


def test_normalization():
    """正規化テスト"""
    print("\n" + "=" * 70)
    print("  スキル名正規化テスト")
    print("=" * 70)

    test_cases = [
        ("JS", "JavaScript"),
        ("k8s", "Kubernetes"),
        ("TS", "TypeScript"),
        ("ML", "機械学習"),
        ("sklearn", "scikit-learn"),
        ("SpringBoot", "Spring Boot"),
        ("NextJS", "Next.js"),
        ("Postgres", "PostgreSQL"),
        ("DL", "深層学習"),
        ("NLP", "自然言語処理"),
        ("VueJS", "Vue.js"),
        ("NodeJS", "Node.js"),
        ("react", "React"),
    ]

    print(f"  {'入力':<20s} {'正規化結果':<20s} {'期待値':<20s} 判定")
    print("  " + "-" * 65)

    correct = 0
    for input_name, expected in test_cases:
        result = normalize_skill_name(input_name)
        ok = result == expected
        if ok:
            correct += 1
        print(f"  {input_name:<20s} {result:<20s} {expected:<20s} {'✓' if ok else '✗'}")

    print(f"\n  正解率: {correct}/{len(test_cases)} ({correct/len(test_cases)*100:.0f}%)")


# ──────────────────────────────────────
# 4. スコア重み最適化
# ──────────────────────────────────────

def simulate_scoring():
    """異なる重みでのスコアシミュレーション"""
    print("\n" + "=" * 70)
    print("  スコア重み シミュレーション")
    print("=" * 70)

    # テストケース: (skill_score, price_score, location_score, expected_rank)
    test_cases = [
        # 完全マッチ
        {"name": "佐藤(Java,SB,AWS) → EC基盤(Java,SB,AWS)",
         "skill": 1.0, "price": 1.0, "loc": 1.0, "expected": "最有力"},
        # スキル完全、単価外
        {"name": "佐藤(80万) → EC基盤(75-85万)",
         "skill": 1.0, "price": 0.5, "loc": 1.0, "expected": "有力"},
        # スキル2/3マッチ
        {"name": "鈴木(React,TS) → EC基盤(Java,SB,AWS)",
         "skill": 0.0, "price": 0.8, "loc": 0.9, "expected": "要検討"},
        # 間接マッチあり
        {"name": "田中(Python,ETL) → データ分析(Python,ETL,Azure)",
         "skill": 0.8, "price": 0.8, "loc": 1.0, "expected": "最有力"},
        # スキル1/3 + 間接1
        {"name": "渡辺(React,Vue) → Web開発(React,TS,Node)",
         "skill": 0.55, "price": 0.9, "loc": 1.0, "expected": "有力"},
    ]

    weight_sets = [
        {"name": "現在 (60/20/20)", "skill": 60, "price": 20, "loc": 20},
        {"name": "スキル重視 (70/15/15)", "skill": 70, "price": 15, "loc": 15},
        {"name": "バランス (50/25/25)", "skill": 50, "price": 25, "loc": 25},
        {"name": "スキル最重視 (80/10/10)", "skill": 80, "price": 10, "loc": 10},
    ]

    for ws in weight_sets:
        print(f"\n  重み: {ws['name']}")
        print(f"  {'ケース':<45s} {'スコア':>5s} {'ランク':<8s} {'期待':<8s} 判定")
        print("  " + "-" * 75)

        correct = 0
        for tc in test_cases:
            total = int(
                tc["skill"] * ws["skill"] +
                tc["price"] * ws["price"] +
                tc["loc"] * ws["loc"]
            )
            total = min(total, 100)
            rank = ("最有力" if total >= 80 else "有力" if total >= 60
                    else "候補" if total >= 40 else "要検討")
            ok = rank == tc["expected"]
            if ok:
                correct += 1
            print(f"  {tc['name']:<45s} {total:>4d}% {rank:<8s} {tc['expected']:<8s} {'✓' if ok else '✗'}")

        print(f"  正解率: {correct}/{len(test_cases)}")


# ──────────────────────────────────────
# 5. 間接マッチスコアの減衰係数
# ──────────────────────────────────────

def simulate_indirect_decay():
    """間接マッチの減衰係数による影響"""
    print("\n" + "=" * 70)
    print("  間接マッチ 減衰係数シミュレーション")
    print("=" * 70)
    print()
    print("  案件: 必須スキル3つ (Java, Spring Boot, AWS)")
    print("  要員: Java(直接), Spring Boot(直接), Docker(間接 sim=0.62)")
    print()

    for decay in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        direct_points = 2  # Java + Spring Boot
        indirect_points = 0.62 * decay  # Docker → AWS
        total_skills = 3
        skill_score = (direct_points + indirect_points) / total_skills
        final = int(skill_score * 60 + 0.8 * 20 + 1.0 * 20)  # price=0.8, loc=1.0

        print(f"  decay={decay:.1f}  "
              f"skill_score={skill_score:.3f}  "
              f"final={final}%  "
              f"{'◀ 現在' if decay == 0.6 else ''}")

    print()
    print("  → lightrag_matching.py の match_project 内:")
    print("    skill_points += ind['similarity'] * 0.6  # この 0.6 を調整")


# ──────────────────────────────────────
# メイン
# ──────────────────────────────────────

def main():
    print("=" * 70)
    print("  LightRAG スキルマッチング チューニングツール")
    print("=" * 70)

    # スキル名正規化テスト
    test_normalization()

    # Embedding 類似度チェック
    model = load_embedder()
    threshold = check_skill_similarities(model)

    # スコアリングシミュレーション
    simulate_scoring()

    # 間接マッチ減衰シミュレーション
    simulate_indirect_decay()

    print("\n" + "=" * 70)
    print("  チューニング推奨サマリー")
    print("=" * 70)
    print(f"  1. 類似度閾値:      {threshold:.2f}")
    print(f"  2. 正規化辞書:      {len(SKILL_SYNONYMS)} エントリ")
    print(f"  3. スコア重み:      simulate_scoring() の結果を参考に")
    print(f"  4. 間接マッチ減衰:  simulate_indirect_decay() の結果を参考に")
    print(f"  5. Embedding モデル: multilingual-e5-base を推奨")
    print()


if __name__ == "__main__":
    main()