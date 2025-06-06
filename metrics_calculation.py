import streamlit as st
import pandas as pd
import numpy as np
from metrics import Metrics

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

def calculate_rating_metrics(true_df, pred_df):
    st.write("### Rating Prediction Metrics")
    
    try:
        merged_df = pd.merge(
            true_df, pred_df, 
            on=['user_id', 'item_id'], 
            suffixes=('_true', '_pred')
        )
        
        if len(merged_df) == 0:
            st.error("Data merge failed. Please check column names.")
            return
        
        st.success(f"Matched data points: {len(merged_df)}")
        
        metrics = Metrics()
        true_ratings = merged_df['rating_true' if 'rating_true' in merged_df.columns else merged_df.columns[2]].tolist()
        pred_ratings = merged_df['rating_pred' if 'rating_pred' in merged_df.columns else merged_df.columns[5]].tolist()
        
        rmse = metrics.rmse(true_ratings, pred_ratings)
        mae = metrics.mae(true_ratings, pred_ratings)
        mse = metrics.mse(true_ratings, pred_ratings)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("RMSE", f"{rmse:.4f}")
        with col2:
            st.metric("MAE", f"{mae:.4f}")
        with col3:
            st.metric("MSE", f"{mse:.4f}")
        
        # 散布図と誤差分布をmatplotlibで作成
        if MATPLOTLIB_AVAILABLE:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 散布図
            ax1.scatter(true_ratings, pred_ratings, alpha=0.6, s=20)
            
            # 完全予測線
            min_val = min(min(true_ratings), min(pred_ratings))
            max_val = max(max(true_ratings), max(pred_ratings))
            ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
            
            # R²値を追加
            correlation = np.corrcoef(true_ratings, pred_ratings)[0, 1]
            r_squared = correlation ** 2
            ax1.text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=ax1.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            ax1.set_xlabel('Actual Rating')
            ax1.set_ylabel('Predicted Rating')
            ax1.set_title('Prediction Accuracy Visualization')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # エラー分布
            errors = np.array(pred_ratings) - np.array(true_ratings)
            ax2.hist(errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            
            # 統計情報を追加
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            ax2.axvline(mean_error, color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {mean_error:.3f}')
            ax2.text(0.7, 0.9, f'Std: {std_error:.3f}', transform=ax2.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            ax2.set_xlabel('Error (Predicted - Actual)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Prediction Error Distribution')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.warning("Matplotlib not available. Install with: pip install matplotlib")
        
        # 詳細統計
        st.subheader("Detailed Statistics")
        stats_df = pd.DataFrame({
            'Metric': ['Mean Error', 'Std Error', 'Min Error', 'Max Error', 'Correlation'],
            'Value': [mean_error, std_error, np.min(errors), np.max(errors), correlation]
        })
        st.dataframe(stats_df)
        
    except Exception as e:
        st.error(f"Calculation error: {str(e)}")

def calculate_ranking_metrics(true_df, pred_df):
    st.write("### Ranking Metrics")
    
    k_values = st.multiselect(
        "Select K values (multiple selection allowed)",
        [1, 3, 5, 10, 20, 50],
        default=[5, 10]
    )
    
    relevance_threshold = st.slider(
        "Relevance threshold (ratings above this value are considered relevant)",
        min_value=1.0, max_value=5.0, value=4.0, step=0.5
    )
    
    if st.button("Calculate Ranking Metrics"):
        try:
            metrics = Metrics()
            results = []
            
            users = set(true_df['user_id'].unique()) & set(pred_df['user_id'].unique())
            
            for user in users:
                user_true = true_df[true_df['user_id'] == user]
                user_pred = pred_df[pred_df['user_id'] == user]
                
                relevant_items = user_true[user_true['rating'] >= relevance_threshold]['item_id'].tolist()
                
                user_pred_sorted = user_pred.sort_values('rating', ascending=False)
                predicted_items = user_pred_sorted['item_id'].tolist()
                
                relevances = [1 if item in relevant_items else 0 for item in predicted_items]
                
                user_results = {'user_id': user}
                
                for k in k_values:
                    if k <= len(predicted_items):
                        prec_k = metrics.precision_at_k(relevant_items, predicted_items, k)
                        recall_k = metrics.recall_at_k(relevant_items, predicted_items, k)
                        f1_k = metrics.f1_at_k(relevant_items, predicted_items, k)
                        ndcg_k = metrics.ndcg_at_k(relevances, k)
                        
                        user_results.update({
                            f'precision@{k}': prec_k,
                            f'recall@{k}': recall_k,
                            f'f1@{k}': f1_k,
                            f'ndcg@{k}': ndcg_k
                        })
                
                results.append(user_results)
            
            results_df = pd.DataFrame(results)
            
            avg_results = {}
            for col in results_df.columns:
                if col != 'user_id':
                    avg_results[col] = results_df[col].mean()
            
            st.write("**User-wise Results:**")
            st.dataframe(results_df, use_container_width=True)
            
            st.write("**Average Metrics:**")
            metrics_display = []
            for k in k_values:
                metrics_display.append({
                    'K': k,
                    'Precision@K': f"{avg_results.get(f'precision@{k}', 0):.4f}",
                    'Recall@K': f"{avg_results.get(f'recall@{k}', 0):.4f}",
                    'F1@K': f"{avg_results.get(f'f1@{k}', 0):.4f}",
                    'NDCG@K': f"{avg_results.get(f'ndcg@{k}', 0):.4f}"
                })
            
            st.dataframe(pd.DataFrame(metrics_display), use_container_width=True)
            
            # 可視化をmatplotlibで作成
            if MATPLOTLIB_AVAILABLE:
                fig, ax = plt.subplots(figsize=(12, 8))
                
                colors = ['blue', 'green', 'red', 'orange']
                for i, metric in enumerate(['precision', 'recall', 'f1', 'ndcg']):
                    y_values = [avg_results.get(f'{metric}@{k}', 0) for k in k_values]
                    ax.plot(k_values, y_values, marker='o', linewidth=3, markersize=8, 
                           color=colors[i], label=metric.upper())
                
                ax.set_xlabel('K Value')
                ax.set_ylabel('Score')
                ax.set_title('Ranking Metrics Comparison')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.warning("Matplotlib not available for visualization")
            
            # CSV エクスポート
            if st.button("Download Results as CSV"):
                csv_data = pd.DataFrame(metrics_display).to_csv(index=False)
                st.download_button(
                    label="Download Ranking Metrics CSV",
                    data=csv_data,
                    file_name="ranking_metrics.csv",
                    mime="text/csv"
                )
            
        except Exception as e:
            st.error(f"Calculation error: {str(e)}")
            st.write("Please check data format. Required columns: user_id, item_id, rating")

def metrics_calculation():
    st.subheader("Metrics Calculation")
    st.write("""
    This section evaluates trained model performance using validation data.
    
    **Recommended Workflow:**
    1. Use "Model Training & Evaluation" to train model with training data
    2. Use "Recommendation Execution" → "Validation Predictions" to generate predictions for validation data
    3. Calculate detailed evaluation metrics here
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Actual Ratings (Ground Truth)**")
        st.caption("True labels for validation data")
        true_file = st.file_uploader("Actual ratings CSV", type="csv", key="true")
        
    with col2:
        st.write("**Predicted Ratings**")
        st.caption("Model predictions")
        pred_file = st.file_uploader("Predicted ratings CSV", type="csv", key="pred")
    
    with st.expander("Data Format Examples"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Actual ratings example:**")
            st.code("""user_id,item_id,rating
1,101,5
1,102,3
2,103,4""")
        with col2:
            st.write("**Predicted ratings example:**")
            st.code("""user_id,item_id,rating
1,101,4.8
1,102,3.2
2,103,3.9""")
    
    if true_file and pred_file:
        true_df = pd.read_csv(true_file)
        pred_df = pd.read_csv(pred_file)
        
        st.write("Data Preview:")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Actual ratings:")
            st.dataframe(true_df.head())
        with col2:
            st.write("Predicted ratings:")
            st.dataframe(pred_df.head())
        
        st.subheader("Calculation Settings")
        
        metric_type = st.selectbox(
            "Select metric type",
            ["Rating Prediction Metrics (RMSE, MAE, etc.)", "Ranking Metrics (Precision@K, NDCG@K, etc.)"]
        )
        
        if metric_type == "Rating Prediction Metrics (RMSE, MAE, etc.)":
            calculate_rating_metrics(true_df, pred_df)
        else:
            calculate_ranking_metrics(true_df, pred_df)