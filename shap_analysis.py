import streamlit as st
import pandas as pd
import numpy as np
from surprise import dump, Reader, Dataset

try:
    import shap
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split as sklearn_train_test_split
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

class SVDSHAPAnalyzer:
    def __init__(self):
        self.svd_model = None
        self.explanation_model = None
        self.user_factors = None
        self.item_factors = None
        self.user_bias = None
        self.item_bias = None
        self.global_mean = None
        
    def load_model(self, model_path):
        """学習済みSVDモデルを読み込み"""
        try:
            predictions, algo = dump.load(model_path)
            self.svd_model = algo
            
            # 潜在因子とバイアスを抽出
            self.user_factors = self.svd_model.pu
            self.item_factors = self.svd_model.qi
            self.user_bias = self.svd_model.bu
            self.item_bias = self.svd_model.bi
            self.global_mean = self.svd_model.trainset.global_mean
            
            return True
        except Exception as e:
            st.error(f"Model loading error: {str(e)}")
            return False
    
    def create_features_for_prediction(self, user_ids, item_ids):
        """ユーザーとアイテムのペアから潜在因子特徴量を作成"""
        features = []
        valid_indices = []
        
        for i, (user_id, item_id) in enumerate(zip(user_ids, item_ids)):
            try:
                user_inner_id = self.svd_model.trainset.to_inner_uid(user_id)
                item_inner_id = self.svd_model.trainset.to_inner_iid(item_id)
                
                user_factor = self.user_factors[user_inner_id]
                item_factor = self.item_factors[item_inner_id]
                
                feature_vector = []
                
                # 1. ユーザー因子とアイテム因子の要素積 (最重要)
                elementwise_product = user_factor * item_factor
                feature_vector.extend(elementwise_product)
                
                # 2. ユーザー因子の値
                feature_vector.extend(user_factor)
                
                # 3. アイテム因子の値  
                feature_vector.extend(item_factor)
                
                # 4. バイアス項
                user_bias_val = self.user_bias[user_inner_id]
                item_bias_val = self.item_bias[item_inner_id]
                feature_vector.extend([user_bias_val, item_bias_val, self.global_mean])
                
                features.append(feature_vector)
                valid_indices.append(i)
                
            except ValueError:
                continue
                
        return np.array(features), valid_indices
    
    def create_feature_names(self):
        """特徴量名を作成"""
        n_factors = self.user_factors.shape[1]
        feature_names = []
        
        # ユーザー×アイテム因子の積
        for i in range(n_factors):
            feature_names.append(f'UserItem_Factor_{i+1}')
        
        # ユーザー因子
        for i in range(n_factors):
            feature_names.append(f'User_Factor_{i+1}')
        
        # アイテム因子
        for i in range(n_factors):
            feature_names.append(f'Item_Factor_{i+1}')
        
        # バイアス項
        feature_names.extend(['User_Bias', 'Item_Bias', 'Global_Mean'])
        
        return feature_names
    
    def train_explanation_model(self, validation_data):
        """説明用のRandom Forestモデルを訓練"""
        features, valid_indices = self.create_features_for_prediction(
            validation_data['user_id'].values, 
            validation_data['item_id'].values
        )
        
        if len(features) == 0:
            raise ValueError("No valid user-item pairs found in validation data")
        
        valid_ratings = validation_data.iloc[valid_indices]['rating'].values
        
        st.info(f"SHAP analysis available pairs: {len(features)}/{len(validation_data)} ({len(features)/len(validation_data)*100:.1f}%)")
        
        self.explanation_model = RandomForestRegressor(
            n_estimators=100, 
            random_state=42, 
            max_depth=10,
            n_jobs=1
        )
        
        # 説明モデルを全データで訓練
        self.explanation_model.fit(features, valid_ratings)
        
        # 性能評価のために一部をテスト用に分ける
        if len(features) > 100:
            X_train, X_test, y_train, y_test = sklearn_train_test_split(
                features, valid_ratings, test_size=0.2, random_state=42
            )
            
            y_pred = self.explanation_model.predict(X_test)
            mae = np.mean(np.abs(y_test - y_pred))
            rmse = np.sqrt(np.mean((y_test - y_pred)**2))
            
            st.success("Explanation model trained successfully")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Explanation Model MAE", f"{mae:.4f}")
            with col2:
                st.metric("Explanation Model RMSE", f"{rmse:.4f}")
        
        # valid_indicesも返すように修正
        return features, valid_ratings, valid_indices
    
    def create_feature_scatter_plot(self, shap_values, features, feature_names, n_samples):
        """特徴量選択可能な散布図を作成（matplotlib版）"""
        st.subheader("Feature vs SHAP Value Scatter Plot")
        
        if not MATPLOTLIB_AVAILABLE:
            st.error("matplotlib not available. Please install: pip install matplotlib seaborn")
            return
        
        # 重要度でソートした特徴量リストを作成
        mean_shap = np.abs(shap_values).mean(0)
        non_zero_indices = mean_shap > 1e-6
        
        if not np.any(non_zero_indices):
            st.warning("No significant features found for scatter plot.")
            return
        
        filtered_features = [(i, feature_names[i]) for i in range(len(feature_names)) if non_zero_indices[i]]
        filtered_features.sort(key=lambda x: mean_shap[x[0]], reverse=True)
        
        # 特徴量選択ドロップダウン
        feature_options = [f"{name} (Importance: {mean_shap[idx]:.4f})" for idx, name in filtered_features]
        
        selected_feature_display = st.selectbox(
            "Select feature to plot:",
            options=feature_options,
            index=0,
            key="scatter_feature_select"
        )
        
        # 選択された特徴量のインデックスを取得
        selected_idx = None
        for idx, name in filtered_features:
            if f"{name} (Importance: {mean_shap[idx]:.4f})" == selected_feature_display:
                selected_idx = idx
                break
        
        if selected_idx is not None:
            selected_feature_name = feature_names[selected_idx]
            
            # データ準備
            x_values = features[:n_samples, selected_idx]
            y_values = shap_values[:, selected_idx]
            
            # matplotlibで散布図作成
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # 特徴量値で色分け（RdYlBu_r カラーマップ）
            scatter = ax.scatter(
                x_values, 
                y_values, 
                c=x_values, 
                cmap='RdYlBu_r', 
                alpha=0.7, 
                s=50,
                edgecolors='white',
                linewidth=0.5
            )
            
            # カラーバー
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Feature Value', rotation=270, labelpad=15)
            
            # 軸とタイトル
            ax.set_xlabel(f'Feature Value ({selected_feature_name})')
            ax.set_ylabel('SHAP Value')
            ax.set_title(f'Feature Value vs SHAP Value: {selected_feature_name}')
            
            # 0線を追加
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # 統計情報表示
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean Feature Value", f"{np.mean(x_values):.4f}")
            with col2:
                st.metric("Std Feature Value", f"{np.std(x_values):.4f}")
            with col3:
                st.metric("Mean SHAP Value", f"{np.mean(y_values):.4f}")
            with col4:
                st.metric("Std SHAP Value", f"{np.std(y_values):.4f}")
            
            # 相関係数
            correlation = np.corrcoef(x_values, y_values)[0, 1]
            st.metric("Correlation (Feature vs SHAP)", f"{correlation:.4f}")
            
            # 解釈のヘルプ
            with st.expander("How to interpret this plot"):
                st.write("""
                **Reading the scatter plot:**
                - **X-axis**: Values of the selected feature
                - **Y-axis**: SHAP values (contribution to prediction)
                - **Color**: Feature value (red=high, blue=low)
                - **Points above 0**: Feature contributes positively to the prediction
                - **Points below 0**: Feature contributes negatively to the prediction
                
                **Interpretation patterns:**
                - **Positive correlation**: Higher feature values → higher SHAP values
                - **Negative correlation**: Higher feature values → lower SHAP values  
                - **No correlation**: Feature value doesn't directly relate to SHAP contribution
                - **Non-linear patterns**: Complex relationships between feature and prediction
                """)

    def create_feature_importance_plot(self, mean_shap, feature_names):
        """特徴量重要度のバープロットを作成（matplotlib版）"""
        if not MATPLOTLIB_AVAILABLE:
            st.error("matplotlib not available. Please install: pip install matplotlib seaborn")
            return
        
        # 0でない特徴量のみフィルタリング
        non_zero_indices = mean_shap > 1e-6
        filtered_features = [feature_names[i] for i in range(len(feature_names)) if non_zero_indices[i]]
        filtered_importance = mean_shap[non_zero_indices]
        
        if len(filtered_features) == 0:
            st.warning("No significant features found.")
            return
        
        # 重要度順にソート
        sorted_indices = np.argsort(filtered_importance)
        top_15_indices = sorted_indices[-15:] if len(sorted_indices) >= 15 else sorted_indices
        
        top_features = [filtered_features[i] for i in top_15_indices]
        top_importance = filtered_importance[top_15_indices]
        
        # matplotlib版バープロット
        fig, ax = plt.subplots(figsize=(10, 8))
        
        bars = ax.barh(range(len(top_features)), top_importance, color='steelblue', alpha=0.7)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels([feat[:30] + "..." if len(feat) > 30 else feat for feat in top_features])
        ax.set_xlabel('Mean |SHAP Value|')
        ax.set_title('Top 15 Feature Importance')
        ax.grid(True, alpha=0.3)
        
        # 値をバーに表示
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + max(top_importance) * 0.01, 
                   bar.get_y() + bar.get_height()/2, 
                   f'{width:.4f}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    def analyze_with_shap(self, features, ratings, valid_indices, validation_data, max_samples=1000):
        """SHAPを使用した分析"""
        st.info("Running SHAP analysis...")
        
        n_samples = min(max_samples, len(features))
        
        # SHAP分析実行
        explainer = shap.TreeExplainer(self.explanation_model)
        shap_values = explainer.shap_values(features[:n_samples])
        
        feature_names = self.create_feature_names()
        
        st.success(f"SHAP analysis completed for {n_samples} samples")
        
        # 1. Feature Importance
        st.subheader("Feature Importance Analysis")
        mean_shap = np.abs(shap_values).mean(0)
        
        # matplotlib版特徴量重要度プロット
        self.create_feature_importance_plot(mean_shap, feature_names)
        
        # 2. SHAP Summary Plot (matplotlib版)
        st.subheader("SHAP Summary Plot (Beeswarm)")
        
        if MATPLOTLIB_AVAILABLE:
            # 0でない特徴量のみフィルタリング
            non_zero_indices = mean_shap > 1e-6
            filtered_features = [feature_names[i] for i in range(len(feature_names)) if non_zero_indices[i]]
            filtered_importance = mean_shap[non_zero_indices]
            
            if len(filtered_features) > 0:
                # 重要度順にソート
                sorted_indices = np.argsort(filtered_importance)
                top_10_indices = sorted_indices[-10:] if len(sorted_indices) >= 10 else sorted_indices
                
                # 実際の特徴量インデックスを取得
                original_indices = [i for i in range(len(feature_names)) if non_zero_indices[i]]
                top_10_features = [filtered_features[i] for i in top_10_indices]
                top_10_original_indices = [original_indices[i] for i in top_10_indices]
                
                # matplotlib/seabornでbeeswarmプロット作成
                fig, ax = plt.subplots(figsize=(12, 8))
                
                plot_data = []
                for i, feat_idx in enumerate(top_10_original_indices):
                    feature_name = feature_names[feat_idx]
                    shap_vals = shap_values[:, feat_idx]
                    feature_vals = features[:n_samples, feat_idx]
                    
                    for j in range(len(shap_vals)):
                        plot_data.append({
                            'Feature': feature_name,
                            'Feature_Index': i,
                            'SHAP_Value': float(shap_vals[j]),
                            'Feature_Value': float(feature_vals[j])
                        })
                
                plot_df = pd.DataFrame(plot_data)
                
                # 各特徴量について散布図を作成
                for i, feature in enumerate(top_10_features):
                    feature_data = plot_df[plot_df['Feature'] == feature]
                    
                    if len(feature_data) > 0:
                        # Y軸にランダムなjitterを追加
                        np.random.seed(42 + i)
                        y_jitter = np.random.normal(i, 0.1, len(feature_data))
                        
                        # 特徴量値で色分け
                        scatter = ax.scatter(
                            feature_data['SHAP_Value'].values,
                            y_jitter,
                            c=feature_data['Feature_Value'].values,
                            cmap='RdYlBu_r',
                            alpha=0.7,
                            s=30
                        )
                
                # 軸設定
                ax.set_yticks(range(len(top_10_features)))
                ax.set_yticklabels([feat[:30] + "..." if len(feat) > 30 else feat for feat in top_10_features])
                ax.set_xlabel('SHAP Value')
                ax.set_ylabel('Features')
                ax.set_title('SHAP Summary Plot (Beeswarm Style)')
                ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
                ax.grid(True, alpha=0.3)
                
                # カラーバー
                plt.colorbar(scatter, ax=ax, label='Feature Value')
                plt.tight_layout()
                
                st.pyplot(fig)
                plt.close()
                
                st.write("""
                **How to read:** 
                - Each dot represents a sample
                - Color indicates feature value (red=high, blue=low)
                - X-position shows SHAP impact (positive pushes prediction up, negative pushes down)
                - Vertical spread shows distribution of SHAP values for each feature
                """)
            else:
                st.warning("No significant features found for beeswarm plot.")
        else:
            st.error("matplotlib/seaborn not available. Please install: pip install matplotlib seaborn")

        # 3. 新しい散布図を追加
        self.create_feature_scatter_plot(shap_values, features, feature_names, n_samples)
                
        # 4. Individual Prediction Explanations
        st.subheader("Individual Prediction Explanations")

        sample_idx = 0
        st.info(f"Showing explanation for Sample {sample_idx + 1} (first sample)")

        if sample_idx < len(shap_values):
            try:
                sample_shap = shap_values[sample_idx]
                sample_features = features[sample_idx]
                
                # numpy配列を明示的にfloat配列に変換
                sample_shap_float = [float(val) for val in sample_shap]
                sample_features_float = [float(val) for val in sample_features]
                
                # 予測値と基準値
                prediction = float(self.explanation_model.predict([sample_features])[0])
                base_value = float(explainer.expected_value)
                actual = float(ratings[sample_idx]) if sample_idx < len(ratings) else None
                
                # 情報表示
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Predicted Rating", f"{prediction:.3f}")
                with col2:
                    if actual is not None:
                        st.metric("Actual Rating", f"{actual:.3f}")
                    else:
                        st.metric("Actual Rating", "N/A")
                with col3:
                    st.metric("Base Value", f"{base_value:.3f}")
                
                # 上位貢献因子
                feature_contributions = []
                for i, (feat_name, shap_val, feat_val) in enumerate(zip(feature_names, sample_shap_float, sample_features_float)):
                    if abs(shap_val) > 1e-6:
                        feature_contributions.append((feat_name, shap_val, feat_val))
                
                feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
                top_contributions = feature_contributions[:10]
                
                if top_contributions and MATPLOTLIB_AVAILABLE:
                    contrib_names = [feat[:25] + "..." if len(feat) > 25 else feat for feat, _, _ in top_contributions]
                    contrib_values = [shap_val for _, shap_val, _ in top_contributions]
                    colors = ['green' if val > 0 else 'red' for val in contrib_values]
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.barh(contrib_names, contrib_values, color=colors, alpha=0.7)
                    
                    # 値をバーに表示
                    for bar, val in zip(bars, contrib_values):
                        width = bar.get_width()
                        ax.text(width + (0.01 if width > 0 else -0.01), bar.get_y() + bar.get_height()/2, 
                            f'{val:.3f}', ha='left' if width > 0 else 'right', va='center')
                    
                    ax.set_xlabel('SHAP Value')
                    ax.set_title(f'SHAP Contributions for Sample {sample_idx + 1}')
                    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    plt.close()
                    
                    # 詳細テーブル
                    st.write("**Detailed Breakdown:**")
                    detail_df = pd.DataFrame([{
                        'Feature': feat,
                        'Feature_Value': f"{feat_val:.4f}",
                        'SHAP_Contribution': f"{shap_val:.4f}",
                        'Impact': 'Positive' if shap_val > 0 else 'Negative'
                    } for feat, shap_val, feat_val in top_contributions])
                    
                    st.dataframe(detail_df, use_container_width=True)
                    
                    # 追加情報
                    st.write("**Interpretation:**")
                    positive_contrib = [f for f, v, _ in top_contributions if v > 0]
                    negative_contrib = [f for f, v, _ in top_contributions if v < 0]
                    
                    if positive_contrib:
                        st.write(f"**Factors increasing the rating:** {len(positive_contrib)} features")
                        st.write(f"Top positive: {positive_contrib[0][:30]}...")
                    
                    if negative_contrib:
                        st.write(f"**Factors decreasing the rating:** {len(negative_contrib)} features")
                        st.write(f"Top negative: {negative_contrib[0][:30]}...")
                        
                else:
                    st.warning("No significant contributions found for this sample or matplotlib not available.")
                    
            except Exception as e:
                st.error(f"Error in individual explanation: {str(e)}")

        else:
            st.warning("No samples available for individual explanation.")        
        
        # 5. SHAP値統計
        st.subheader("SHAP Value Statistics")
        try:
            # numpy配列を明示的にfloat変換
            mean_shap_vals = [float(val) for val in np.mean(shap_values, axis=0)]
            mean_abs_shap_vals = [float(val) for val in np.abs(shap_values).mean(0)]
            std_shap_vals = [float(val) for val in np.std(shap_values, axis=0)]
            
            shap_stats = pd.DataFrame({
                'Feature': feature_names,
                'Mean_SHAP': mean_shap_vals,
                'Mean_Abs_SHAP': mean_abs_shap_vals,
                'Std_SHAP': std_shap_vals
            })
            
            # 0でない特徴量のみ表示
            shap_stats_filtered = shap_stats[shap_stats['Mean_Abs_SHAP'] > 1e-6].sort_values('Mean_Abs_SHAP', ascending=False)
            st.dataframe(shap_stats_filtered.head(10), use_container_width=True)
        except Exception as e:
            st.error(f"Error in statistics: {str(e)}")
        
        # 6. 因子別分析
        st.subheader("Latent Factor Analysis")
        
        try:
            actual_n_factors = self.user_factors.shape[1]
            st.write(f"Actual number of factors in SVD model: {actual_n_factors}")
            
            # UserItem因子の重要度を計算
            useritem_importance = []
            for i in range(actual_n_factors):
                factor_name = f'UserItem_Factor_{i+1}'
                if factor_name in feature_names:
                    idx = feature_names.index(factor_name)
                    if idx < len(mean_shap):
                        importance = float(mean_shap[idx])
                        if importance > 1e-6:
                            useritem_importance.append({
                                'Factor': i+1, 
                                'Importance': importance,
                                'Feature_Name': factor_name
                            })
            
            if useritem_importance and MATPLOTLIB_AVAILABLE:
                factor_df = pd.DataFrame(useritem_importance)
                
                factors = factor_df['Factor'].values
                importances = factor_df['Importance'].values
                
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(factors, importances, color='steelblue', alpha=0.7)
                
                # 値をバーに表示
                for bar, val in zip(bars, importances):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{val:.4f}', ha='center', va='bottom')
                
                ax.set_xlabel('Factor Number')
                ax.set_ylabel('Importance (Mean |SHAP Value|)')
                ax.set_title('User×Item Factor Importance')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                
                st.pyplot(fig)
                plt.close()
                
                st.write("**Factor Details:**")
                st.dataframe(factor_df, use_container_width=True)
            else:
                if not MATPLOTLIB_AVAILABLE:
                    st.error("matplotlib not available for factor analysis.")
                else:
                    st.warning("No significant User×Item factors found.")
        except Exception as e:
            st.error(f"Error in factor analysis: {str(e)}")
        
        return shap_values, feature_names, valid_indices, validation_data

def shap_analysis():
    st.subheader("SHAP Analysis for SVD Model")
    
    if not SHAP_AVAILABLE:
        st.error("SHAP library is not available. Please install it with: pip install shap")
        st.code("pip install shap scikit-learn")
        return
    
    if not MATPLOTLIB_AVAILABLE:
        st.error("matplotlib library is not available. Please install it with: pip install matplotlib seaborn")
        st.code("pip install matplotlib seaborn")
        return
    
    st.write("""
    This analysis uses the latent factors from a trained SVD model to provide interpretable explanations.
    The features include user factors, item factors, their element-wise products, and bias terms.
    """)
    
    # セッション状態の初期化
    if 'shap_analysis_complete' not in st.session_state:
        st.session_state.shap_analysis_complete = False
    if 'shap_results' not in st.session_state:
        st.session_state.shap_results = None
    
    # モデル読み込み
    st.subheader("Load Trained SVD Model")
    uploaded_model = st.file_uploader(
        "Upload trained SVD model (.pkl)", 
        type=["pkl"],
        key="shap_model"
    )
    
    # 検証データ読み込み
    st.subheader("Load Validation Data")
    uploaded_data = st.file_uploader(
        "Upload validation data (CSV)", 
        type="csv",
        help="CSV with user_id, item_id, rating columns",
        key="shap_validation_data"
    )
    
    if uploaded_model and uploaded_data:
        try:
            # モデル読み込み
            with open("temp_shap_model.pkl", "wb") as f:
                f.write(uploaded_model.read())
            
            analyzer = SVDSHAPAnalyzer()
            if not analyzer.load_model("temp_shap_model.pkl"):
                return
            
            # データ読み込み
            validation_df = pd.read_csv(uploaded_data)
            st.write("Validation Data Preview:")
            st.dataframe(validation_df.head())
            
            # データ検証
            required_cols = ['user_id', 'item_id', 'rating']
            missing_cols = [col for col in required_cols if col not in validation_df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
                return
            
            # 分析設定
            st.subheader("Analysis Configuration")
            col1, col2 = st.columns(2)
            
            with col1:
                max_samples = st.slider(
                    "Maximum samples for SHAP analysis", 
                    min_value=100, 
                    max_value=min(2000, len(validation_df)), 
                    value=min(1000, len(validation_df))
                )
            
            with col2:
                n_estimators = st.slider("Random Forest estimators", min_value=50, max_value=200, value=100)
            
            # SHAP分析実行
            if st.button("Run SHAP Analysis"):
                with st.spinner("Running SHAP analysis..."):
                    try:
                        # 説明用モデルの訓練
                        features, ratings, valid_indices = analyzer.train_explanation_model(validation_df)
                        
                        # SHAP分析の実行
                        shap_values, feature_names, valid_indices, validation_data = analyzer.analyze_with_shap(
                            features, ratings, valid_indices, validation_df, max_samples
                        )
                        
                        # セッション状態に結果を保存
                        st.session_state.shap_results = {
                            'shap_values': shap_values,
                            'features': features,
                            'feature_names': feature_names,
                            'valid_indices': valid_indices,
                            'validation_data': validation_df,
                            'ratings': ratings,
                            'max_samples': max_samples,
                            'analyzer': analyzer
                        }
                        st.session_state.shap_analysis_complete = True
                        
                    except Exception as e:
                        st.error(f"SHAP analysis error: {str(e)}")
                        st.write("Please ensure your model and data are compatible.")
            
            # SHAP分析完了後、結果を表示（セッション状態から）
            if st.session_state.shap_analysis_complete and st.session_state.shap_results:
                results = st.session_state.shap_results
                
                # 散布図セクション（セッション状態から）
                st.markdown("---")
                results['analyzer'].create_feature_scatter_plot(
                    results['shap_values'],
                    results['features'],
                    results['feature_names'],
                    results['max_samples']
                )
                
                # 結果の保存オプション
                st.subheader("Export Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # ユーザーIDとアイテムIDの両方を含めてSHAP値を保存
                    valid_user_ids = results['validation_data'].iloc[results['valid_indices']]['user_id'].values
                    valid_item_ids = results['validation_data'].iloc[results['valid_indices']]['item_id'].values
                    
                    # SHAP値の実際のサンプル数に合わせる
                    actual_samples = min(len(results['shap_values']), len(results['valid_indices']))
                    
                    shap_df = pd.DataFrame(
                        results['shap_values'][:actual_samples], 
                        columns=results['feature_names']
                    )
                    shap_df.insert(0, 'user_id', valid_user_ids[:actual_samples])
                    shap_df.insert(1, 'item_id', valid_item_ids[:actual_samples])
                    
                    csv_shap = shap_df.to_csv(index=False)
                    st.download_button(
                        label="Download SHAP Values CSV",
                        data=csv_shap,
                        file_name="shap_values.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # 特徴量重要度をCSVで保存
                    importance_df = pd.DataFrame({
                        'Feature': results['feature_names'],
                        'Mean_Abs_SHAP': np.abs(results['shap_values']).mean(0)
                    }).sort_values('Mean_Abs_SHAP', ascending=False)
                    
                    csv_importance = importance_df.to_csv(index=False)
                    st.download_button(
                        label="Download Feature Importance CSV",
                        data=csv_importance,
                        file_name="feature_importance.csv",
                        mime="text/csv"
                    )
                
                # 分析をリセットするボタン
                if st.button("Reset Analysis"):
                    st.session_state.shap_analysis_complete = False
                    st.session_state.shap_results = None
                    st.rerun()
                        
        except Exception as e:
            st.error(f"Data loading error: {str(e)}")
            st.write("Please ensure your files are properly formatted.")
    
    else:
        st.info("Please upload both a trained SVD model and validation data to begin SHAP analysis.")
        
        # 使用例の表示
        with st.expander("Data Format Examples"):
            st.write("**Validation Data Format:**")
            st.code("""user_id,item_id,rating
1,101,5
1,102,3
2,103,4
2,104,2""")
            
            st.write("**Expected Features from SVD Model:**")
            st.write("- User×Item factor products (most important)")
            st.write("- Individual user latent factors")
            st.write("- Individual item latent factors") 
            st.write("- User and item bias terms")
            st.write("- Global rating mean")