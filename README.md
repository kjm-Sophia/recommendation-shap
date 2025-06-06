# 推薦システム & SHAP解析プラットフォーム

## 概要

このプロジェクトは、推薦システムの評価とSHAP（SHapley Additive exPlanations）解析を統合したStreamlitアプリケーションです。機械学習モデルの解釈可能性を向上させ、推薦システムの意思決定プロセスを可視化します。

## 主な機能

### 1. Model Training & Evaluation
- 複数の推薦アルゴリズム（SVD、NMFなど）の訓練と評価
- 性能指標の比較とベンチマーク
- モデルパラメータの調整

### 2. SHAP Analysis
- 推薦モデルの予測に対する特徴量の影響度分析
- SHAP値による解釈可能な機械学習
- インタラクティブな可視化

### 3. Content-Based Filtering SHAP (CF for SHAP)
- TF-IDFベースのコンテンツフィルタリング
- ジャンル・単語レベルでの影響度分析
- 評価予測の要因分解

### 4. Recommendation Execution
- 訓練済みモデルによる推薦実行
- ユーザー別・アイテム別推薦結果の生成
- 推薦品質の評価

### 5. Metrics Calculation
- 精度、再現率、F1スコアなどの評価指標計算
- ランキング品質の評価
- A/Bテスト用の統計分析

### 6. LLM for SHAP
- 大規模言語モデルを活用したSHAP結果の自然言語解釈
- 推薦理由の自動生成
- ビジネス向けレポート作成

## インストール方法

### 1. リポジトリのクローン
```bash
git clone https://github.com/kjm-Sophia/recommendation-shap.git
cd recommendation-shap
```

### 2. 仮想環境の作成
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 3. 依存関係のインストール
```bash
pip install -r requirements.txt
```

## 使用方法

### 1. アプリケーションの起動
```bash
streamlit run enhanced_rec_app.py
```

### 2. ブラウザでアクセス
アプリケーション起動後、ブラウザで `http://localhost:8501` にアクセス

### 3. データの準備

#### 訓練・検証データ形式
```csv
user_id,item_id,rating
505,224,3.0
86,150,4.0
534,2125,3.5
```

#### メタデータ形式（CF for SHAP用）
```csv
itemId,title,category,description,option1,option2,option3
949,Heat,"[{'id': 28, 'name': 'Action'}]","Obsessive master thief...",en,1995-12-15,Released
710,GoldenEye,"[{'id': 12, 'name': 'Adventure'}]","James Bond must unmask...",en,1995-11-16,Released
```

## CF for SHAP Analysis 詳細

Content-Based Filtering with SHAP Analysis の主要機能：

### 特徴量抽出
- **TF-IDF特徴量**: アイテムの説明文から300次元の特徴量を抽出
- **ジャンル特徴量**: カテゴリ情報から最大20ジャンルをワンホットエンコーディング
- **基本特徴量**: タイトル長、説明文長、公開年、ユーザー平均評価

### SHAP解析結果
- **重要特徴量ランキング**: 予測に最も影響する特徴量の特定
- **ジャンル影響分析**: 各ジャンルが評価予測に与える影響
- **単語影響分析**: 評価を上げる/下げる単語の特定
- **可視化**: Summary Plot、Feature Importance、相互作用分析

### 分析結果例
```
Top 5 重要ジャンル:
  Comedy: 0.0016
  Adventure: 0.0010
  Family: 0.0005
  Drama: 0.0005
  Action: 0.0004

最も評価を上げる単語:
  'later': +0.0026
  'big': +0.0023
  'gives': +0.0023

最も評価を下げる単語:
  'german': -0.0019
  'order': -0.0016
  'law': -0.0011

Top 10 重要特徴量:
  user_avg_rating: 0.4639
  word_gets: 0.0064
  title_length: 0.0060
  desc_length: 0.0058
  word_later: 0.0031
  year: 0.0030
  word_big: 0.0029
  word_order: 0.0025
  word_german: 0.0023
  word_gives: 0.0023
```

## 技術仕様

### 主要な依存関係
- `streamlit` - Webアプリケーションフレームワーク
- `scikit-surprise` - 推薦システムライブラリ
- `shap` - SHAP解析ライブラリ
- `pandas`, `numpy` - データ処理
- `scikit-learn` - 機械学習
- `matplotlib`, `seaborn` - データ可視化
- `plotly` - インタラクティブ可視化
