import streamlit as st
import pandas as pd
import numpy as np
from surprise import SVD, NMF, KNNBasic, KNNWithMeans, SlopeOne, CoClustering, Dataset, dump, Reader
from surprise.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from metrics import Metrics

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

def load_data():
    st.subheader("Data Loading")
    data_source = st.radio(
        "Select data source", 
        ["CSV File", "Sample Data (ml-100k)"]
    )

    if data_source == "Sample Data (ml-100k)":
        try:
            return Dataset.load_builtin('ml-100k')
        except Exception as e:
            st.error(f"Sample data loading error: {str(e)}")
            return None
    
    uploaded_file = st.file_uploader(
        "Select CSV file", 
        type="csv",
        help="Columns required: user_id, item_id, rating"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Data Preview:")
            st.dataframe(df.head())
            
            st.write("Column Names:", list(df.columns))
            
            required_cols = ['user_id', 'item_id', 'rating']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
                st.write("CSV file must contain the following columns:")
                st.write("- user_id: User identifier")
                st.write("- item_id: Item identifier") 
                st.write("- rating: Rating value")
                return None
            
            st.write("Data Statistics:")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Users", df['user_id'].nunique())
            with col2:
                st.metric("Items", df['item_id'].nunique())
            with col3:
                st.metric("Ratings", len(df))
            
            rating_min, rating_max = df['rating'].min(), df['rating'].max()
            st.write(f"Rating Range: {rating_min} ~ {rating_max}")
            
            reader = Reader(rating_scale=(rating_min, rating_max))
            dataset = Dataset.load_from_df(df[["user_id", "item_id", "rating"]], reader)
            
            st.success("Data loading completed")
            return dataset
            
        except Exception as e:
            st.error(f"File loading error: {str(e)}")
            st.write("Please check your CSV file format")
            return None
    
    return None

def get_algorithm_parameters(algorithm):
    """アルゴリズム別のパラメーター設定UI"""
    params = {}
    
    if algorithm == "SVD":
        st.write("**SVD Parameters:**")
        col1, col2 = st.columns(2)
        with col1:
            params['n_factors'] = st.slider("Number of factors", 10, 200, 50)
            params['n_epochs'] = st.slider("Number of epochs", 5, 50, 20)
        with col2:
            params['lr_all'] = st.slider("Learning rate", 0.001, 0.02, 0.005, 0.001)
            params['reg_all'] = st.slider("Regularization", 0.01, 0.5, 0.02, 0.01)
    
    elif algorithm == "NMF":
        st.write("**NMF Parameters:**")
        col1, col2 = st.columns(2)
        with col1:
            params['n_factors'] = st.slider("Number of factors", 10, 200, 15)
            params['n_epochs'] = st.slider("Number of epochs", 5, 50, 50)
        with col2:
            params['reg_pu'] = st.slider("User regularization", 0.01, 0.5, 0.06, 0.01)
            params['reg_qi'] = st.slider("Item regularization", 0.01, 0.5, 0.06, 0.01)
    
    elif algorithm in ["KNNBasic", "KNNWithMeans"]:
        st.write(f"**{algorithm} Parameters:**")
        col1, col2 = st.columns(2)
        with col1:
            params['k'] = st.slider("Number of neighbors (k)", 10, 100, 40)
            params['min_k'] = st.slider("Minimum k", 1, 10, 1)
        with col2:
            similarity = st.selectbox("Similarity measure", ["cosine", "pearson", "msd"])
            user_based = st.checkbox("User-based (unchecked = Item-based)", value=True)
            params['sim_options'] = {
                'name': similarity,
                'user_based': user_based
            }
    
    elif algorithm == "SlopeOne":
        st.write("**SlopeOne Parameters:**")
        st.info("SlopeOne has no tunable parameters")
    
    elif algorithm == "CoClustering":
        st.write("**CoClustering Parameters:**")
        col1, col2 = st.columns(2)
        with col1:
            params['n_cltr_u'] = st.slider("User clusters", 2, 10, 3)
            params['n_cltr_i'] = st.slider("Item clusters", 2, 10, 3)
        with col2:
            params['n_epochs'] = st.slider("Number of epochs", 5, 50, 20)
    
    return params

def hyperparameter_tuning(data, cv_value, algorithm):
    try:
        st.info("Hyperparameter tuning in progress...")
        
        param_grids = {
            'SVD': {
                'n_epochs': [5, 10, 20], 
                'lr_all': [0.002, 0.005, 0.01],
                'reg_all': [0.02, 0.1, 0.2],
                'n_factors': [15, 50, 100, 150, 200]
            },
            'NMF': {
                'n_epochs': [5, 10, 20],
                'n_factors': [15, 50, 100,150],
                'reg_pu': [0.06, 0.12, 0.18],
                'reg_qi': [0.06, 0.12, 0.18]
            },
            'KNNBasic': {
                'k': [20, 40, 50],
                'sim_options': [
                    {'name': 'cosine', 'user_based': True},
                    {'name': 'cosine', 'user_based': False},
                    {'name': 'pearson', 'user_based': True}
                ]
            },
            'KNNWithMeans': {
                'k': [20, 40, 50],
                'sim_options': [
                    {'name': 'cosine', 'user_based': True},
                    {'name': 'cosine', 'user_based': False},
                    {'name': 'pearson', 'user_based': True}
                ]
            },
            'SlopeOne': {},
            'CoClustering': {
                'n_cltr_u': [3, 5, 7],
                'n_cltr_i': [3, 5, 7],
                'n_epochs': [10, 20]
            }
        }
        
        algorithm_classes = {
            'SVD': SVD,
            'NMF': NMF,
            'KNNBasic': KNNBasic,
            'KNNWithMeans': KNNWithMeans,
            'SlopeOne': SlopeOne,
            'CoClustering': CoClustering
        }
        
        param_grid = param_grids.get(algorithm, {})
        algo_class = algorithm_classes[algorithm]
        
        if param_grid:
            gs = GridSearchCV(algo_class, param_grid, measures=['rmse', 'mae'], cv=cv_value, n_jobs=1)
            gs.fit(data)
            
            st.success("Tuning completed")
            st.write("**Best Parameters (RMSE-based):**")
            for param, value in gs.best_params['rmse'].items():
                st.write(f"- {param}: {value}")
            
            st.write("**Best Scores:**")
            st.write(f"- RMSE: {gs.best_score['rmse']:.4f}")
            st.write(f"- MAE: {gs.best_score['mae']:.4f}")
            
            return gs.best_params
        else:
            st.info(f"{algorithm} has no tunable parameters")
            return None
        
    except Exception as e:
        st.error(f"Tuning error: {str(e)}")
        return None

def create_prediction_plots(true_ratings, pred_ratings, algorithm):
    """予測精度の可視化を作成（matplotlib版）"""
    if len(true_ratings) == 0 or not MATPLOTLIB_AVAILABLE:
        return
    
    try:
        # 散布図
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 予測精度散布図
        ax1.scatter(true_ratings, pred_ratings, alpha=0.6, s=20)
        
        # 完全予測線を追加
        min_val = min(min(true_ratings), min(pred_ratings))
        max_val = max(max(true_ratings), max(pred_ratings))
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # R²値を計算して表示
        correlation = np.corrcoef(true_ratings, pred_ratings)[0, 1]
        r_squared = correlation ** 2
        ax1.text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=ax1.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        ax1.set_xlabel('Actual Rating')
        ax1.set_ylabel('Predicted Rating')
        ax1.set_title(f'Prediction Accuracy - {algorithm}')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # エラー分布
        errors = np.array(pred_ratings) - np.array(true_ratings)
        ax2.hist(errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        
        # 統計情報を追加
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        ax2.axvline(mean_error, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_error:.3f}')
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
        
    except Exception as e:
        st.error(f"Error creating plots: {str(e)}")

def model_training_and_evaluation():
    st.subheader("Model Training & Evaluation")
    
    data = load_data()
    if data is None:
        st.warning("Please load data")
        return
    
    st.subheader("Training Configuration")
    
    # アルゴリズム選択
    algorithm = st.selectbox(
        "Select Algorithm",
        ["SVD", "NMF", "KNNBasic", "KNNWithMeans", "SlopeOne", "CoClustering"]
    )
    
    algorithm_descriptions = {
        "SVD": "Singular Value Decomposition - Matrix factorization approach",
        "NMF": "Non-negative Matrix Factorization - Non-negative latent factors",
        "KNNBasic": "K-Nearest Neighbors - Basic collaborative filtering",
        "KNNWithMeans": "K-Nearest Neighbors with user/item means normalization",
        "SlopeOne": "Slope One - Simple and efficient collaborative filtering",
        "CoClustering": "Co-clustering approach for simultaneous user-item clustering"
    }
    st.info(algorithm_descriptions[algorithm])
    
    # パラメーター設定方法の選択
    param_method = st.radio(
        "Parameter Setting Method",
        ["Manual Setting", "Hyperparameter Tuning", "Default Parameters"]
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if param_method == "Manual Setting":
            st.subheader("Manual Parameter Setting")
            user_params = get_algorithm_parameters(algorithm)
        elif param_method == "Hyperparameter Tuning":
            st.subheader("Hyperparameter Tuning Settings")
            cv_value = st.slider("CV Folds", min_value=2, max_value=5, value=3, step=1)
            user_params = None
        else:
            st.subheader("Default Parameters")
            st.info("Using algorithm default parameters")
            user_params = None
    
    with col2:
        st.subheader("Training Settings")
        test_size = st.slider("Test Data Ratio", min_value=0.1, max_value=0.5, value=0.25, step=0.05)
        save_path = st.text_input("Save File Name", f"{algorithm.lower()}_model.pkl")
        
        # 可視化オプション
        show_plots = st.checkbox("Show prediction plots", value=True)
        if not MATPLOTLIB_AVAILABLE:
            st.warning("Matplotlib not available. Plots will be disabled.")
            show_plots = False
    
    # 学習・評価実行
    if st.button("Start Training & Evaluation", type="primary"):
        try:
            with st.spinner("Training & evaluation in progress..."):
                trainset, testset = train_test_split(data, test_size=test_size, random_state=42)
                
                algorithm_classes = {
                    'SVD': SVD,
                    'NMF': NMF,
                    'KNNBasic': KNNBasic,
                    'KNNWithMeans': KNNWithMeans,
                    'SlopeOne': SlopeOne,
                    'CoClustering': CoClustering
                }
                
                algo_class = algorithm_classes[algorithm]
                
                # パラメーター設定に基づいてモデル作成
                if param_method == "Manual Setting" and user_params:
                    st.info("Using manually set parameters")
                    algo = algo_class(**user_params)
                elif param_method == "Hyperparameter Tuning":
                    best_params = hyperparameter_tuning(data, cv_value, algorithm)
                    if best_params and best_params.get('rmse'):
                        st.info("Using optimized parameters")
                        algo = algo_class(**best_params['rmse'])
                    else:
                        st.warning("Tuning failed or no parameters. Using default parameters.")
                        algo = algo_class()
                else:
                    st.info("Using default parameters")
                    algo = algo_class()
                
                # モデル学習
                algo.fit(trainset)
                predictions = algo.test(testset)
                
                # メトリクス計算
                metrics = Metrics()
                true_ratings = [pred.r_ui for pred in predictions]
                pred_ratings = [pred.est for pred in predictions]
                
                rmse = metrics.rmse(true_ratings, pred_ratings)
                mae = metrics.mae(true_ratings, pred_ratings)
                mse = metrics.mse(true_ratings, pred_ratings)
                
                st.success("Training & evaluation completed")
                
                # 結果表示
                st.subheader("Performance Metrics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("RMSE", f"{rmse:.4f}")
                with col2:
                    st.metric("MAE", f"{mae:.4f}")
                with col3:
                    st.metric("MSE", f"{mse:.4f}")
                
                # 追加統計
                correlation = np.corrcoef(true_ratings, pred_ratings)[0, 1]
                r_squared = correlation ** 2
                st.write(f"**Correlation**: {correlation:.4f}")
                st.write(f"**R² Score**: {r_squared:.4f}")
                
                # 可視化
                if show_plots and len(true_ratings) > 0:
                    st.subheader("Prediction Analysis")
                    create_prediction_plots(true_ratings, pred_ratings, algorithm)
                
                # 使用パラメーター表示
                st.subheader("Model Configuration")
                if hasattr(algo, 'n_factors'):
                    st.write(f"**Number of factors**: {algo.n_factors}")
                if hasattr(algo, 'n_epochs'):
                    st.write(f"**Number of epochs**: {algo.n_epochs}")
                if hasattr(algo, 'lr_all'):
                    st.write(f"**Learning rate**: {algo.lr_all}")
                if hasattr(algo, 'reg_all'):
                    st.write(f"**Regularization**: {algo.reg_all}")
                
                # モデル保存
                try:
                    full_trainset = data.build_full_trainset()
                    algo.fit(full_trainset)
                    anti_testset = full_trainset.build_anti_testset()
                    full_predictions = algo.test(anti_testset)
                    
                    dump.dump(save_path, predictions=full_predictions, algo=algo)
                    st.success(f"Model saved as {save_path}")
                    
                    # ダウンロード機能
                    with open(save_path, "rb") as file:
                        st.download_button(
                            label="Download Trained Model",
                            data=file.read(),
                            file_name=save_path,
                            mime="application/octet-stream"
                        )
                    
                    # セッション状態に保存
                    st.session_state['last_predictions'] = full_predictions
                    st.session_state['last_algo'] = algo
                    st.session_state['last_algorithm'] = algorithm
                    
                except Exception as e:
                    st.error(f"Model saving error: {str(e)}")
                    
        except Exception as e:
            st.error(f"Training error: {str(e)}")
            st.write("Please check data format and parameters")