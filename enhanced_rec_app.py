import streamlit as st
import sys
import os

# メニューごとのモジュールをインポート
from model_training import model_training_and_evaluation
from shap_analysis import shap_analysis
from recommendation_execution import recommend_execution
from metrics_calculation import metrics_calculation
from llm_shap import llm_for_shap
from cf_shap_analysis import run_content_filtering_shap_analysis

def main():
    st.set_page_config(page_title="Recommendation System Evaluation", layout="wide")
    st.title("Recommendation System & Evaluation Metrics")
    
    menu = [
        "Model Training & Evaluation", 
        "SHAP Analysis",
        "Recommendation Execution", 
        "Metrics Calculation",
        "LLM for SHAP",
        "CF for SHAP"
    ]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Model Training & Evaluation":
        model_training_and_evaluation()
    elif choice == "SHAP Analysis":
        shap_analysis()
    elif choice == "Recommendation Execution":
        recommend_execution()
    elif choice == "Metrics Calculation":
        metrics_calculation()
    elif choice == "LLM for SHAP":
        llm_for_shap()
    elif choice == "CF for SHAP":
        cf_for_shap_interface()

def cf_for_shap_interface():
    """CF for SHAP メニューのインターフェース"""
    st.header("Content-Based Filtering SHAP Analysis")
    st.write("Upload training data, validation data, and metadata to perform content-based filtering with SHAP analysis.")
    
    # ファイルアップロード用のカラム
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Training Data")
        train_file = st.file_uploader(
            "Upload training CSV file", 
            type=['csv'], 
            key="cf_train_file",
            help="CSV file with columns: user_id, item_id, rating"
        )
        if train_file:
            st.success(f"Training file uploaded: {train_file.name}")
    
    with col2:
        st.subheader("Validation Data")
        validation_file = st.file_uploader(
            "Upload validation CSV file", 
            type=['csv'], 
            key="cf_validation_file",
            help="CSV file with columns: user_id, item_id, rating"
        )
        if validation_file:
            st.success(f"Validation file uploaded: {validation_file.name}")
    
    with col3:
        st.subheader("Metadata")
        metadata_file = st.file_uploader(
            "Upload metadata CSV file", 
            type=['csv'], 
            key="cf_metadata_file",
            help="CSV file with item information: itemId, title, category, description, etc."
        )
        if metadata_file:
            st.success(f"Metadata file uploaded: {metadata_file.name}")
    
    # 分析実行ボタン
    if st.button("Run CF SHAP Analysis", type="primary", use_container_width=True):
        if train_file and validation_file and metadata_file:
            try:
                # 分析実行
                run_content_filtering_shap_analysis(train_file, validation_file, metadata_file)
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                with st.expander("Error Details"):
                    st.exception(e)
        else:
            st.warning("Please upload all required files (training data, validation data, and metadata)")
    
    # データ形式説明
    with st.expander("Required Data Format"):
        st.write("**Training & Validation Data Format:**")
        st.code("""
user_id,item_id,rating
505,224,3.0
86,150,4.0
534,2125,3.5
        """)
        
        st.write("**Metadata Format:**")
        st.code("""
itemId,title,category,description,option1,option2,option3
949,Heat,"[{'id': 28, 'name': 'Action'}]","Obsessive master thief...",en,1995-12-15,Released
710,GoldenEye,"[{'id': 12, 'name': 'Adventure'}]","James Bond must unmask...",en,1995-11-16,Released
        """)
        
        st.write("**Key Requirements:**")
        st.write("- Training/Validation: user_id, item_id, rating columns")
        st.write("- Metadata: itemId, title, category, description columns")
        st.write("- Category should contain genre information in JSON-like format")
        st.write("- Description contains text content for TF-IDF analysis")

if __name__ == "__main__":
    main()