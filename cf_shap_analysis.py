import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import shap
import io

def run_content_filtering_shap_analysis(train_file, validation_file, metadata_file):
    """
    Content-Based Filtering SHAP解析を実行
    
    Args:
        train_file: 訓練データファイル
        validation_file: 検証データファイル  
        metadata_file: メタデータファイル
    """
    
    st.header("CF for SHAP Analysis")
    st.write("Content-Based Filtering with SHAP Analysis")
    
    try:
        # データ読み込み
        with st.spinner("Loading data..."):
            train_data, validation_data, metadata = load_cf_data(train_file, validation_file, metadata_file)
            st.success(f"Training data: {len(train_data)} rows")
            st.success(f"Validation data: {len(validation_data)} rows")
        
        # 特徴量作成
        with st.spinner("Creating features..."):
            X_train, X_val, feature_names = create_cf_features(train_data, validation_data, metadata)
            y_train = train_data['rating'].values
            y_val = validation_data['rating'].values
            st.success(f"Total features: {len(feature_names)}")
        
        # モデル訓練
        with st.spinner("Training model..."):
            model, train_rmse, val_rmse = train_cf_model(X_train, y_train, X_val, y_val)
            st.success(f"Training RMSE: {train_rmse:.4f}")
            st.success(f"Validation RMSE: {val_rmse:.4f}")
        
        # SHAP解析
        with st.spinner("Performing SHAP analysis..."):
            explainer, shap_values, X_sample = perform_cf_shap_analysis(model, X_train, X_val, feature_names)
            st.success(f"SHAP calculation completed: {shap_values.shape}")
        
        # 結果表示
        st.subheader("Analysis Results")
        display_cf_results(shap_values, X_sample, feature_names, explainer)
        
        # 結果保存
        results_data = save_cf_results(validation_data, model.predict(X_val), shap_values, feature_names)
        
        # ダウンロードボタン
        st.subheader("Download Results")
        col1, col2 = st.columns(2)
        
        with col1:
            predictions_csv = results_data['predictions'].to_csv(index=False)
            st.download_button(
                label="Download Predictions",
                data=predictions_csv,
                file_name="cf_predictions.csv",
                mime="text/csv"
            )
        
        with col2:
            importance_csv = results_data['importance'].to_csv(index=False)
            st.download_button(
                label="Download Feature Importance",
                data=importance_csv,
                file_name="cf_feature_importance.csv",
                mime="text/csv"
            )
        
    except Exception as e:
        st.error(f"Error occurred: {str(e)}")
        st.exception(e)

def load_cf_data(train_file, validation_file, metadata_file):
    """データ読み込みと前処理"""
    
    # データ読み込み
    train = pd.read_csv(train_file)
    validation = pd.read_csv(validation_file)
    metadata = pd.read_csv(metadata_file)
    
    # カラム名統一
    train = train.rename(columns={'user_id': 'userId', 'item_id': 'itemId'})
    validation = validation.rename(columns={'user_id': 'userId', 'item_id': 'itemId'})
    
    # データ型変換
    for df in [train, validation]:
        df['userId'] = df['userId'].astype(str)
        df['itemId'] = df['itemId'].astype(str)
    metadata['itemId'] = metadata['itemId'].astype(str)
    
    # メタデータ結合
    train = pd.merge(train, metadata, on='itemId', how='left')
    validation = pd.merge(validation, metadata, on='itemId', how='left')
    
    # 欠損値処理
    for df in [train, validation]:
        df['description'] = df['description'].fillna(df['title']).fillna("")
        df['category'] = df['category'].fillna("")
        df['title'] = df['title'].fillna("")
    
    return train, validation, metadata

def create_cf_features(train_data, validation_data, metadata):
    """特徴量作成"""
    
    # TF-IDF特徴量
    st.write("- Creating TF-IDF features...")
    tfidf = TfidfVectorizer(max_features=300, stop_words='english', ngram_range=(1,2), min_df=2)
    
    tfidf_train = tfidf.fit_transform(train_data['description']).toarray()
    tfidf_val = tfidf.transform(validation_data['description']).toarray()
    
    tfidf_names = [f"word_{word}" for word in tfidf.get_feature_names_out()]
    
    # ジャンル特徴量
    st.write("- Creating genre features...")
    
    # ジャンル抽出
    all_genres = set()
    for data in [train_data, validation_data]:
        for cat_str in data['category']:
            genres = extract_cf_genres(cat_str)
            all_genres.update(genres)
    
    all_genres = sorted(list(all_genres))[:20]  # 上位20ジャンル
    st.write(f"  Detected genres: {len(all_genres)}")
    
    # ジャンルエンコーディング
    genre_train = create_cf_genre_features(train_data, all_genres)
    genre_val = create_cf_genre_features(validation_data, all_genres)
    
    genre_names = [f"genre_{genre}" for genre in all_genres]
    
    # 基本特徴量
    st.write("- Creating basic features...")
    
    basic_train = create_cf_basic_features(train_data)
    basic_val = create_cf_basic_features(validation_data)
    
    basic_names = ['title_length', 'desc_length', 'year', 'user_avg_rating']
    
    # 特徴量結合
    X_train = np.hstack([tfidf_train, genre_train, basic_train])
    X_val = np.hstack([tfidf_val, genre_val, basic_val])
    
    feature_names = tfidf_names + genre_names + basic_names
    
    return X_train, X_val, feature_names

def extract_cf_genres(category_str):
    """カテゴリ文字列からジャンル抽出"""
    if pd.isna(category_str) or category_str == "":
        return []
    
    import re
    genres = re.findall(r"'name': '([^']+)'", str(category_str))
    return genres

def create_cf_genre_features(data, all_genres):
    """ジャンル特徴量作成"""
    genre_matrix = np.zeros((len(data), len(all_genres)))
    
    for i, cat_str in enumerate(data['category']):
        item_genres = extract_cf_genres(cat_str)
        for genre in item_genres:
            if genre in all_genres:
                genre_idx = all_genres.index(genre)
                genre_matrix[i, genre_idx] = 1
    
    return genre_matrix

def create_cf_basic_features(data):
    """基本特徴量作成"""
    features = []
    
    # タイトル長
    title_lens = data['title'].fillna("").apply(len).values.reshape(-1, 1)
    features.append(title_lens)
    
    # 説明文長
    desc_lens = data['description'].fillna("").apply(len).values.reshape(-1, 1)
    features.append(desc_lens)
    
    # 年度
    years = []
    for date_str in data['option2'].fillna(""):
        try:
            year = int(str(date_str).split('-')[0]) if '-' in str(date_str) else 2000
        except:
            year = 2000
        years.append(year)
    features.append(np.array(years).reshape(-1, 1))
    
    # ユーザー平均評価
    user_avg = data.groupby('userId')['rating'].transform('mean').values.reshape(-1, 1)
    features.append(user_avg)
    
    return np.hstack(features)

def train_cf_model(X_train, y_train, X_val, y_val):
    """モデル訓練"""
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # 評価
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    
    return model, train_rmse, val_rmse

def perform_cf_shap_analysis(model, X_train, X_val, feature_names, sample_size=100):
    """SHAP解析実行"""
    
    # サンプリング
    if len(X_train) > 500:
        bg_indices = np.random.choice(len(X_train), 500, replace=False)
        X_background = X_train[bg_indices]
    else:
        X_background = X_train
    
    if len(X_val) > sample_size:
        sample_indices = np.random.choice(len(X_val), sample_size, replace=False)
        X_sample = X_val[sample_indices]
    else:
        X_sample = X_val
    
    # SHAP計算
    explainer = shap.TreeExplainer(model, X_background)
    shap_values = explainer.shap_values(X_sample)
    
    return explainer, shap_values, X_sample

def display_cf_results(shap_values, X_sample, feature_names, explainer):
    """結果表示"""
    
    # 特徴量重要度計算
    feature_importance = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    # タブで結果表示
    tab1, tab2, tab3, tab4 = st.tabs(["SHAP Summary", "Feature Importance", "Genre Analysis", "Word Analysis"])
    
    with tab1:
        st.subheader("SHAP Summary Plot")
        fig_summary = create_cf_shap_summary_plot(shap_values, X_sample, feature_names)
        st.pyplot(fig_summary)
    
    with tab2:
        st.subheader("Top 15 Feature Importance")
        fig_importance = create_cf_importance_plot(importance_df)
        st.pyplot(fig_importance)
        
        # 重要特徴量テーブル
        st.subheader("Top 10 Important Features")
        top_features = importance_df.head(10)
        for i, row in top_features.iterrows():
            st.write(f"  {row['feature']}: {row['importance']:.4f}")
    
    with tab3:
        st.subheader("Genre Analysis")
        fig_genre = analyze_cf_genres(shap_values, feature_names)
        if fig_genre:
            st.pyplot(fig_genre)
    
    with tab4:
        st.subheader("Word Analysis")
        fig_word = analyze_cf_words(shap_values, feature_names)
        if fig_word:
            st.pyplot(fig_word)

def create_cf_shap_summary_plot(shap_values, X_sample, feature_names):
    """SHAP要約プロット作成（バージョン互換性対応）"""
    try:
        # Method 1: 標準的なSHAP summary plot
        fig = plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                         max_display=20, show=False)
        plt.title('SHAP Summary Plot - Content-Based Filtering')
        plt.tight_layout()
        return fig
        
    except Exception as e:
        # Method 2: 代替案 - 手動でプロット作成
        st.warning(f"SHAP summary plot error: {e}. Using alternative visualization.")
        
        # 特徴量重要度計算
        feature_importance = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False).head(20)
        
        # 代替プロット
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(range(len(importance_df)), importance_df['importance'])
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df['feature'])
        ax.set_xlabel('Mean |SHAP Value|')
        ax.set_title('Feature Importance (Alternative to SHAP Summary)')
        ax.invert_yaxis()
        
        # 値表示
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + max(importance_df['importance']) * 0.01, 
                   bar.get_y() + bar.get_height()/2, 
                   f'{width:.4f}', ha='left', va='center', fontsize=8)
        
        plt.tight_layout()
        return fig

def create_cf_importance_plot(importance_df):
    """特徴量重要度プロット作成"""
    fig, ax = plt.subplots(figsize=(10, 8))
    top_features = importance_df.head(15)
    bars = ax.barh(range(len(top_features)), top_features['importance'])
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Mean |SHAP Value|')
    ax.set_title('Top 15 Feature Importance')
    ax.invert_yaxis()
    
    # 値表示
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + max(top_features['importance']) * 0.01, 
               bar.get_y() + bar.get_height()/2, 
               f'{width:.4f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    return fig

def analyze_cf_genres(shap_values, feature_names):
    """ジャンル分析"""
    genre_indices = [i for i, name in enumerate(feature_names) if name.startswith('genre_')]
    
    if len(genre_indices) == 0:
        st.write("No genre features found")
        return None
    
    genre_shap = shap_values[:, genre_indices]
    genre_names = [feature_names[i].replace('genre_', '') for i in genre_indices]
    genre_importance = np.abs(genre_shap).mean(axis=0)
    
    # 上位ジャンル表示
    top_n = min(5, len(genre_names))
    top_indices = np.argsort(genre_importance)[-top_n:]
    
    st.write("Top 5 Important Genres:")
    for i in reversed(top_indices):
        st.write(f"  {genre_names[i]}: {genre_importance[i]:.4f}")
    
    # 可視化
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(top_n), genre_importance[top_indices], color='steelblue', alpha=0.7)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([genre_names[i] for i in top_indices])
    ax.set_xlabel('Average |SHAP Value|')
    ax.set_title('Genre Impact on Ratings')
    
    # 値表示
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + max(genre_importance[top_indices]) * 0.05, 
               bar.get_y() + bar.get_height()/2, 
               f'{width:.4f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    return fig

def analyze_cf_words(shap_values, feature_names):
    """単語分析"""
    word_indices = [i for i, name in enumerate(feature_names) if name.startswith('word_')]
    
    if len(word_indices) == 0:
        st.write("No word features found")
        return None
    
    word_shap = shap_values[:, word_indices]
    word_names = [feature_names[i].replace('word_', '') for i in word_indices]
    
    # 正負の影響分析
    word_impact = word_shap.mean(axis=0)
    word_abs_impact = np.abs(word_shap).mean(axis=0)
    
    # 最も正の影響
    top_positive = np.argsort(word_impact)[-5:]
    st.write("Words that increase ratings:")
    for i in reversed(top_positive):
        st.write(f"  '{word_names[i]}': +{word_impact[i]:.4f}")
    
    # 最も負の影響
    top_negative = np.argsort(word_impact)[:5]
    st.write("Words that decrease ratings:")
    for i in top_negative:
        st.write(f"  '{word_names[i]}': {word_impact[i]:.4f}")
    
    # 可視化
    top_word_indices = np.argsort(word_abs_impact)[-15:]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['green' if word_impact[i] >= 0 else 'red' for i in top_word_indices]
    bars = ax.barh(range(15), word_abs_impact[top_word_indices], color=colors, alpha=0.7)
    ax.set_yticks(range(15))
    ax.set_yticklabels([word_names[i] for i in top_word_indices])
    ax.set_xlabel('Average |SHAP Value|')
    ax.set_title('Word Impact on Ratings (Green=Positive, Red=Negative)')
    
    # 値表示
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + max(word_abs_impact[top_word_indices]) * 0.02, 
               bar.get_y() + bar.get_height()/2, 
               f'{width:.4f}', ha='left', va='center', fontsize=8)
    
    plt.tight_layout()
    return fig

def save_cf_results(validation_data, predictions, shap_values, feature_names):
    """結果保存"""
    
    # 予測結果
    predictions_df = pd.DataFrame({
        'userId_itemId': validation_data['userId'].astype(str) + '_' + validation_data['itemId'].astype(str),
        'true_rating': validation_data['rating'],
        'predicted_rating': predictions
    })
    
    # 特徴量重要度
    feature_importance = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    return {
        'predictions': predictions_df,
        'importance': importance_df
    }