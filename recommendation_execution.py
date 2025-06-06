import streamlit as st
import pandas as pd
import numpy as np
from surprise import dump
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

def load_trained_model():
    uploaded_file = st.file_uploader(
        "Upload trained model", 
        type=["pkl"]
    )
    
    if uploaded_file:
        with st.spinner("Loading model..."):
            with open("temp_model.pkl", "wb") as f:
                f.write(uploaded_file.read())
            predictions, loaded_algo = dump.load("temp_model.pkl")
            st.success("Model loaded successfully")
            return predictions, loaded_algo
    
    if 'last_predictions' in st.session_state and 'last_algo' in st.session_state:
        st.info("Using model from previous training session")
        return st.session_state['last_predictions'], st.session_state['last_algo']
    
    return None, None

def generate_validation_predictions(loaded_algo):
    st.subheader("Validation Predictions Generation")
    st.write("Generate model predictions for validation data to be used in metrics calculation.")
    
    uploaded_file = st.file_uploader(
        "Upload validation data (actual ratings)", 
        type="csv",
        help="Columns required: user_id, item_id, rating"
    )
    
    if uploaded_file:
        try:
            validation_df = pd.read_csv(uploaded_file)
            st.write("Validation Data Preview:")
            st.dataframe(validation_df.head())
            
            required_cols = ['user_id', 'item_id', 'rating']
            missing_cols = [col for col in required_cols if col not in validation_df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
                return
            
            if st.button("Generate Predictions"):
                with st.spinner("Generating predictions..."):
                    predictions_list = []
                    
                    for _, row in validation_df.iterrows():
                        user_id = row['user_id']
                        item_id = row['item_id']
                        true_rating = row['rating']
                        
                        pred = loaded_algo.predict(user_id, item_id)
                        
                        predictions_list.append({
                            'user_id': user_id,
                            'item_id': item_id,
                            'true_rating': true_rating,
                            'predicted_rating': pred.est
                        })
                    
                    predictions_df = pd.DataFrame(predictions_list)
                    
                    st.success("Predictions completed")
                    st.write("Prediction Results:")
                    st.dataframe(predictions_df.head(10))
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Predictions", len(predictions_df))
                    with col2:
                        mae = np.mean(np.abs(predictions_df['predicted_rating'] - predictions_df['true_rating']))
                        st.metric("MAE (Reference)", f"{mae:.4f}")
                    with col3:
                        rmse = np.sqrt(np.mean((predictions_df['predicted_rating'] - predictions_df['true_rating'])**2))
                        st.metric("RMSE (Reference)", f"{rmse:.4f}")
                    
                    # 散布図をmatplotlibで作成
                    if MATPLOTLIB_AVAILABLE:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.scatter(predictions_df['true_rating'], predictions_df['predicted_rating'], alpha=0.6, s=20)
                        
                        # 完全予測線
                        min_val = predictions_df['true_rating'].min()
                        max_val = predictions_df['true_rating'].max()
                        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
                        
                        # R²値計算
                        correlation = np.corrcoef(predictions_df['true_rating'], predictions_df['predicted_rating'])[0, 1]
                        r_squared = correlation ** 2
                        ax.text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=ax.transAxes, 
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                        
                        ax.set_xlabel('Actual Rating')
                        ax.set_ylabel('Predicted Rating')
                        ax.set_title('Prediction vs Actual (Validation Data)')
                        ax.grid(True, alpha=0.3)
                        ax.legend()
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                    else:
                        st.warning("Matplotlib not available. Install with: pip install matplotlib")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Actual Ratings (Ground Truth)**")
                        true_data = predictions_df[['user_id', 'item_id', 'true_rating']].rename(
                            columns={'true_rating': 'rating'}
                        )
                        st.download_button(
                            label="Download Actual Ratings CSV",
                            data=true_data.to_csv(index=False),
                            file_name="validation_true.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        st.write("**Predicted Ratings**")
                        pred_data = predictions_df[['user_id', 'item_id', 'predicted_rating']].rename(
                            columns={'predicted_rating': 'rating'}
                        )
                        st.download_button(
                            label="Download Predicted Ratings CSV", 
                            data=pred_data.to_csv(index=False),
                            file_name="validation_pred.csv",
                            mime="text/csv"
                        )
                    
                    st.info("Use these CSV files in 'Metrics Calculation' for detailed evaluation.")
                    
        except Exception as e:
            st.error(f"Error: {str(e)}")

def recommend_existing_users(predictions, loaded_algo):
    st.subheader("Existing User Recommendations")
    
    col1, col2 = st.columns(2)
    with col1:
        top_n = st.slider("Number of recommendations", min_value=5, max_value=50, value=10)
    with col2:
        threshold = st.slider("Recommendation threshold", min_value=3.0, max_value=5.0, value=4.0, step=0.1)
    
    if st.button("Generate Recommendations"):
        with st.spinner("Generating recommendations..."):
            top_n_df = get_top_n(predictions, n=top_n, threshold=threshold)
            
            st.write(f"**Recommendation Results (Threshold: {threshold}+)**")
            st.dataframe(top_n_df, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Recommendations", len(top_n_df))
            with col2:
                st.metric("Target Users", top_n_df['User'].nunique())
            with col3:
                st.metric("Average Predicted Rating", f"{top_n_df['Estimate'].mean():.2f}")
            
            # 推薦分布の可視化をmatplotlibで作成
            if MATPLOTLIB_AVAILABLE:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(top_n_df['Estimate'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                ax.set_xlabel('Predicted Rating')
                ax.set_ylabel('Frequency')
                ax.set_title('Predicted Rating Distribution')
                ax.grid(True, alpha=0.3)
                
                # 統計情報を追加
                mean_rating = top_n_df['Estimate'].mean()
                ax.axvline(mean_rating, color='red', linestyle='--', linewidth=2, 
                          label=f'Mean: {mean_rating:.2f}')
                ax.legend()
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.warning("Matplotlib not available for visualization")
            
            save_filename = st.text_input("Save filename", "existing_user_recommendations.csv")
            if st.button("Save CSV"):
                top_n_df.to_csv(save_filename, index=False)
                st.success(f"Results saved as {save_filename}")

def recommend_new_users(loaded_algo):
    st.subheader("New User Recommendations")
    
    uploaded_file = st.file_uploader(
        "New user rating data CSV", 
        type="csv"
    )
    
    if uploaded_file:
        new_user_ratings = pd.read_csv(uploaded_file)
        st.write("Uploaded data:")
        st.dataframe(new_user_ratings.head())
        
        top_n = st.slider("Number of recommendations", min_value=5, max_value=50, value=10)
        
        if st.button("Generate Recommendations"):
            with st.spinner("Generating recommendations..."):
                recommendations = generate_new_user_recommendations(
                    new_user_ratings, loaded_algo, top_n
                )
                
                st.write("**Recommendation Results:**")
                st.dataframe(recommendations, use_container_width=True)
                
                save_filename = st.text_input("Save filename", "new_user_recommendations.csv")
                if st.button("Save CSV"):
                    recommendations.to_csv(save_filename, index=False)
                    st.success(f"Results saved as {save_filename}")

def generate_new_user_recommendations(new_user_ratings, loaded_algo, top_n):
    trainset = loaded_algo.trainset
    unique_users = new_user_ratings['user_id'].unique()
    recommendations = []
    
    for user in unique_users:
        user_ratings = new_user_ratings[new_user_ratings['user_id'] == user]
        rated_items = user_ratings['item_id'].values.tolist()
        
        preds = []
        for iid in trainset.all_items():
            raw_iid = trainset.to_raw_iid(iid)
            if raw_iid not in rated_items:
                pred = loaded_algo.predict(user, raw_iid)
                preds.append((user, raw_iid, pred.est))
        
        preds.sort(key=lambda x: x[2], reverse=True)
        for user_id, item_id, rating in preds[:top_n]:
            recommendations.append({
                'user_id': user_id,
                'item_id': item_id,
                'predicted_rating': rating
            })
    
    return pd.DataFrame(recommendations)

def get_top_n(predictions, n=10, threshold=4.0):
    rows = []
    user_ratings = defaultdict(list)
    
    for uid, iid, true_r, est, _ in predictions:
        if est >= threshold:
            user_ratings[uid].append((iid, est))
    
    for uid, ratings in user_ratings.items():
        ratings.sort(key=lambda x: x[1], reverse=True)
        top_ratings = ratings[:n]
        for iid, est in top_ratings:
            rows.append([uid, iid, est])
    
    return pd.DataFrame(rows, columns=['User', 'Item', 'Estimate'])

def recommend_execution():
    st.subheader("Recommendation Execution")
    
    predictions, loaded_algo = load_trained_model()
    
    if loaded_algo:
        rec_type = st.radio(
            "Select recommendation type",
            ["Existing User Recommendations", "New User Recommendations", "Validation Predictions"]
        )
        
        if rec_type == "Existing User Recommendations":
            recommend_existing_users(predictions, loaded_algo)
        elif rec_type == "New User Recommendations":
            recommend_new_users(loaded_algo)
        else:
            generate_validation_predictions(loaded_algo)