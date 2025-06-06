import streamlit as st
import pandas as pd
import numpy as np
import json
import requests
from datetime import datetime
import ast

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# LLM API Configuration
def setup_llm_api():
    """LLM API Configuration"""
    st.sidebar.subheader("LLM API Settings")
    
    api_provider = st.sidebar.selectbox(
        "Select LLM Provider:",
        ["Gemini", "OpenAI"]
    )
    
    api_key = st.sidebar.text_input(
        "API Key:",
        type="password",
        help="Enter your API key"
    )
    
    return api_provider, api_key

def call_llm_api(prompt, provider, api_key):
    """Call LLM API"""
    if not api_key:
        return "API key is not set. Please enter API key in the sidebar."
    
    try:
        if provider == "Gemini":
            return call_gemini_api(prompt, api_key)
        elif provider == "OpenAI":
            return call_openai_api(prompt, api_key)
        else:
            return "Selected provider is not supported."
    except Exception as e:
        return f"API call error: {str(e)}"

def call_gemini_api(prompt, api_key):
    """Call Gemini API"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    data = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }],
        "generationConfig": {
            "temperature": 0.1,  # 0.7から0.1に下げて一貫性を向上
            "topP": 0.8,         # 追加: より決定論的な出力
            "topK": 20,          # 追加: 候補を制限
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)  # タイムアウトを60秒に延長
        
        if response.status_code == 200:
            result = response.json()
            
            if 'candidates' in result and len(result['candidates']) > 0:
                candidate = result['candidates'][0]
                
                finish_reason = candidate.get('finishReason', 'UNKNOWN')
                
                if finish_reason == 'MAX_TOKENS':
                    st.warning("Response was truncated due to token limit")
                
                if 'content' in candidate:
                    content = candidate['content']
                    
                    if content.get('role') == 'model' and 'parts' not in content:
                        if 'text' in content:
                            return content['text']
                        else:
                            return "Error: Text content not found"
                    
                    elif 'parts' in content and len(content['parts']) > 0:
                        text_content = content['parts'][0].get('text', '')
                        
                        if finish_reason == 'MAX_TOKENS':
                            text_content += "\n\n[Note: Response was cut off due to limit]"
                        
                        return text_content
                    else:
                        return f"Error: Invalid content structure"
                else:
                    return f"Error: Content not found"
            else:
                return f"Error: No candidates found"
        else:
            return f"API Error: {response.status_code} - {response.text}"
            
    except requests.exceptions.Timeout:
        return "Error: API request timed out (60 seconds). The prompt may be too long."
    except Exception as e:
        return f"Error: {str(e)}"

def call_openai_api(prompt, api_key):
    """Call OpenAI API"""
    url = "https://api.openai.com/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 2048,
        "temperature": 0.1,      # 0.7から0.1に下げて一貫性を向上
        "top_p": 0.8,            # 追加: より決定論的な出力
        "frequency_penalty": 0.1, # 追加: 繰り返しを少し抑制
        "presence_penalty": 0.1   # 追加: 多様性を少し抑制
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)  # タイムアウト延長
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return f"API Error: {response.status_code} - {response.text}"
            
    except requests.exceptions.Timeout:
        return "Error: API request timed out (60 seconds). The prompt may be too long."
    except Exception as e:
        return f"Error: {str(e)}"

# Data Processing Functions
def extract_genre_names(genre_string):
    """Extract genre names list from genre string"""
    if pd.isna(genre_string) or genre_string == '':
        return []
    
    try:
        if isinstance(genre_string, str):
            if genre_string.startswith('['):
                genres_list = ast.literal_eval(genre_string)
            else:
                return [genre_string.strip()]
        else:
            genres_list = genre_string
            
        genre_names = []
        for genre in genres_list:
            if isinstance(genre, dict) and 'name' in genre:
                genre_names.append(genre['name'])
            elif isinstance(genre, str):
                genre_names.append(genre)
        
        return genre_names
    except:
        return []

def preprocess_metadata(metadata_df):
    """Preprocess metadata for analysis"""
    processed_df = metadata_df.copy()
    
    # カテゴリー（ジャンル）の処理 - 実際の列名に対応
    genre_column = None
    for col in ['category', 'category2', 'genres']:
        if col in processed_df.columns:
            genre_column = col
            break
    
    if genre_column:
        try:
            processed_df['genres'] = processed_df[genre_column].apply(extract_genre_names)
            processed_df['primary_genre'] = processed_df['genres'].apply(lambda x: x[0] if x else 'Unknown')
        except Exception as e:
            st.warning(f"Genre processing error: {str(e)}")
            processed_df['primary_genre'] = 'Unknown'
    else:
        processed_df['primary_genre'] = 'Unknown'
    
    # 日付から年を抽出 - 実際の列名に対応
    date_column = None
    for col in ['option2', 'release_date', 'date', 'year']:
        if col in processed_df.columns:
            date_column = col
            break
    
    if date_column:
        try:
            processed_df['release_year'] = pd.to_datetime(processed_df[date_column], errors='coerce').dt.year
        except Exception as e:
            st.warning(f"Year processing error: {str(e)}")
            processed_df['release_year'] = None
    else:
        processed_df['release_year'] = None
    
    return processed_df

def analyze_feature_importance(feature_importance_df, top_n=10):
    """Analyze feature importance"""
    st.subheader("Top Important Features")
    
    top_features = feature_importance_df.nlargest(top_n, 'Mean_Abs_SHAP')
    
    if MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(range(len(top_features)), top_features['Mean_Abs_SHAP'].values)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels([feat[:25] + "..." if len(feat) > 25 else feat for feat in top_features['Feature']])
        ax.set_xlabel('Mean |SHAP Value|')
        ax.set_title(f'Top {top_n} Most Important Features')
        
        for i, (bar, val) in enumerate(zip(bars, top_features['Mean_Abs_SHAP'].values)):
            ax.text(val + val*0.01, bar.get_y() + bar.get_height()/2, 
                   f'{val:.4f}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    st.dataframe(top_features.reset_index(drop=True))
    return top_features

def get_representative_items(factor_name, shap_values_df, metadata_df, top=True, n=10):
    """Get representative items (updated version: default 10 items)"""
    if factor_name not in shap_values_df.columns:
        return pd.DataFrame()
    
    # 多めに候補を取得
    candidate_size = min(50, len(shap_values_df))
    
    if top:
        top_indices = shap_values_df[factor_name].nlargest(candidate_size)
    else:
        top_indices = shap_values_df[factor_name].nsmallest(candidate_size)
    
    # データマッチング分析
    shap_item_ids = set(shap_values_df.index)
    metadata_item_ids = set(metadata_df['itemId'])
    common_ids = shap_item_ids.intersection(metadata_item_ids)
    
    # 選択されたアイテムIDのうち、メタデータに存在するもののみ
    selected_ids = list(top_indices.index)
    matching_ids = [item_id for item_id in selected_ids if item_id in metadata_item_ids]
    
    # 目標数に達していない場合は共通IDから追加選択
    if len(matching_ids) < n and common_ids:
        common_shap_data = shap_values_df[shap_values_df.index.isin(common_ids)]
        
        if factor_name in common_shap_data.columns:
            if top:
                all_common_sorted = common_shap_data[factor_name].nlargest(len(common_shap_data))
            else:
                all_common_sorted = common_shap_data[factor_name].nsmallest(len(common_shap_data))
            
            additional_ids = [item_id for item_id in all_common_sorted.index 
                            if item_id not in matching_ids][:n]
            
            matching_ids = matching_ids + additional_ids
    
    # 最終的に目標数まで絞り込み
    final_matching_ids = matching_ids[:n]
    
    # メタデータから該当する映画を取得
    representative_items = metadata_df[metadata_df['itemId'].isin(final_matching_ids)]
    
    if len(representative_items) == 0:
        return pd.DataFrame()
    
    # 因子値の辞書を作成（重複対応）
    factor_value_dict = {}
    for item_id in final_matching_ids:
        try:
            if item_id in shap_values_df.index:
                factor_series = shap_values_df.loc[item_id, factor_name]
                
                if isinstance(factor_series, pd.Series):
                    factor_value_dict[item_id] = float(factor_series.iloc[0])
                else:
                    factor_value_dict[item_id] = float(factor_series)
        except Exception:
            factor_value_dict[item_id] = None
    
    # 新しいDataFrame作成
    result_data = []
    for _, row in representative_items.iterrows():
        row_data = {}
        for col_name in representative_items.columns:
            row_data[col_name] = row[col_name]
        
        item_id = row['itemId']
        if item_id in factor_value_dict and factor_value_dict[item_id] is not None:
            row_data['factor_value'] = factor_value_dict[item_id]
        else:
            row_data['factor_value'] = None
        
        result_data.append(row_data)
    
    if not result_data:
        return pd.DataFrame()
    
    result_df = pd.DataFrame(result_data)
    result_df = result_df.dropna(subset=['factor_value'])
    
    # ソート
    if len(result_df) > 0:
        try:
            if 'factor_value' in result_df.columns:
                result_df['factor_value'] = pd.to_numeric(result_df['factor_value'], errors='coerce')
                result_df = result_df.dropna(subset=['factor_value'])
                
                if len(result_df) > 0:
                    result_df = result_df.sort_values('factor_value', ascending=not top)
        except Exception:
            pass
    
    return result_df

def format_movie_samples_compact(movies_df, max_movies=5):
    """Format movie samples compactly (for factor naming)"""
    if movies_df is None or not isinstance(movies_df, pd.DataFrame) or len(movies_df) == 0:
        return "No data"
    
    # 最大表示数を制限
    display_movies = movies_df.head(max_movies)
    formatted_samples = []
    
    try:
        for idx in range(len(display_movies)):
            try:
                row = display_movies.iloc[idx]
                movie = row.to_dict()
            except Exception:
                continue
            
            title = movie.get('title', 'N/A')
            genre = movie.get('primary_genre', 'Unknown')
            year = movie.get('release_year')
            
            sample_text = f"「{title}」({genre}"
            if pd.notna(year):
                sample_text += f", {int(year)}年"
            sample_text += ")"
            
            formatted_samples.append(sample_text)
            
    except Exception:
        return "Format error"
    
    if not formatted_samples:
        return "No data"
    
    result = ", ".join(formatted_samples)
    if len(movies_df) > max_movies:
        result += f" など({len(movies_df)}件)"
    
    return result

def format_movie_samples(movies_df):
    """Format movie samples (for detailed analysis)"""
    if movies_df is None or not isinstance(movies_df, pd.DataFrame) or len(movies_df) == 0:
        return "No data"
    
    formatted_samples = []
    
    try:
        for idx in range(len(movies_df)):
            try:
                row = movies_df.iloc[idx]
                movie = row.to_dict()
            except Exception:
                continue
            
            title = movie.get('title', 'N/A')
            item_id = movie.get('itemId', 'N/A')
            sample_text = f"【映画ID: {item_id}】「{title}」"
            
            if 'primary_genre' in movie and pd.notna(movie.get('primary_genre')):
                genre = movie['primary_genre']
                sample_text += f"\n  ジャンル: {genre}"
            
            if 'release_year' in movie and pd.notna(movie.get('release_year')):
                year = float(movie['release_year'])
                sample_text += f"\n  公開年: {year:.0f}年"
            
            if 'description' in movie and pd.notna(movie.get('description')):
                desc = str(movie['description'])[:200]
                sample_text += f"\n  あらすじ: {desc}..."
            
            if 'factor_value' in movie and pd.notna(movie.get('factor_value')):
                factor_val = float(movie['factor_value'])
                sample_text += f"\n  因子値: {factor_val:.4f}"
            
            for key in ['option1', 'option3']:
                if key in movie and pd.notna(movie.get(key)):
                    value = str(movie[key])[:50]
                    sample_text += f"\n  {key}: {value}"
            
            formatted_samples.append(sample_text)
            
    except Exception:
        return "Format error"
    
    if not formatted_samples:
        return "No data"
    
    return "\n\n".join(formatted_samples)

def load_netflix_tags():
    """Load Netflix tags from NFtag.txt"""
    try:
        # カレントディレクトリからNFtag.txtを読み込み
        with open('NFtag.txt', 'r', encoding='utf-8') as f:
            content = f.read()
        
        tags = []
        current_category = None
        
        for line in content.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # カテゴリー行（数字で終わる）
            if line.endswith(')') and '(' in line:
                current_category = line.split('(')[0].strip()
                tags.append(current_category)
            # サブカテゴリー行（ハイフンを含む）
            elif '–' in line or '-' in line:
                tag_name = line.split('–')[0].split('-')[0].strip()
                if tag_name:
                    tags.append(tag_name)
        
        return tags
    except FileNotFoundError:
        st.warning("NFtag.txt file not found. Please place it in the same directory.")
        return []
    except Exception as e:
        st.warning(f"NFtag.txt loading error: {str(e)}")
        return []

def generate_bulk_factor_naming_prompt(factors_data):
    """Generate bulk factor naming prompt for all factors"""
    prompt = """あなたは映画推薦システムの専門家です。以下の因子（Factor）それぞれに適切な名前を付けてください。

各因子について、SHAP値が高い映画と低い映画のサンプルを提供します。これらのパターンから因子の特徴を分析し、端的で分かりやすい因子名を付けてください。

重要: 分析は客観的かつ一貫性を保ち、同じデータに対しては同じ結論を導いてください。

"""
    
    for i, (factor_name, data) in enumerate(factors_data.items(), 1):
        high_samples = data['high_samples']
        low_samples = data['low_samples']
        
        prompt += f"""
## 因子{i}: {factor_name}

**高い値の映画:** {high_samples}

**低い値の映画:** {low_samples}

"""
    
    prompt += """
以下の形式で必ず回答してください（形式を厳密に守ってください）：

**因子名の提案:**
1. [元の因子名] → [提案する因子名] - [理由（1行で簡潔に）]
2. [元の因子名] → [提案する因子名] - [理由（1行で簡潔に）]
...

**分析指針:**
- 因子名は日本語で、映画の特徴を表す分かりやすい名前にしてください
- 各因子の理由は1行で簡潔に説明してください
- ジャンル、年代、テーマ、スタイルなどの観点から特徴を捉えてください
- 客観的な事実に基づいて分析し、推測は最小限にしてください
- 同じパターンには一貫した命名規則を適用してください
"""
    
    return prompt

def bulk_factor_naming(shap_values_df, metadata_df, available_factors, provider, api_key):
    """Bulk factor naming for all factors"""
    st.subheader("Factor Naming")
    
    with st.spinner("Collecting data for all factors..."):
        factors_data = {}
        
        # 各ファクターについて高い/低いアイテムを取得
        max_factors = 10
        for factor in available_factors[:max_factors]:
            try:
                high_items = get_representative_items(factor, shap_values_df, metadata_df, top=True, n=10)
                low_items = get_representative_items(factor, shap_values_df, metadata_df, top=False, n=10)
                
                high_samples = format_movie_samples_compact(high_items, max_movies=5)
                low_samples = format_movie_samples_compact(low_items, max_movies=5)
                
                factors_data[factor] = {
                    'high_samples': high_samples,
                    'low_samples': low_samples
                }
            except Exception as e:
                st.warning(f"Data retrieval error for factor {factor}: {str(e)}")
                continue
    
    if not factors_data:
        st.error("Could not retrieve analyzable factor data.")
        return
    
    st.info(f"Analysis factor count: {len(factors_data)} (max {max_factors})")
    
    with st.spinner("Generating factor names with LLM..."):
        try:
            prompt = generate_bulk_factor_naming_prompt(factors_data)
            
            # プロンプト長をチェック
            st.write(f"Prompt length: {len(prompt)} characters")
            if len(prompt) > 25000:
                st.warning("Prompt too long, reducing factor count.")
                # ファクター数を半分に削減
                limited_factors = dict(list(factors_data.items())[:5])
                prompt = generate_bulk_factor_naming_prompt(limited_factors)
                st.write(f"Reduced prompt length: {len(prompt)} characters")
            
            naming_result = call_llm_api(prompt, provider, api_key)
            
            st.markdown("### Factor Name Suggestion Results")
            st.markdown(naming_result)
            
            # 因子名マッピングテーブルを表示
            st.markdown("### Analysis Data Summary")
            
            factor_summary = []
            for factor, data in factors_data.items():
                factor_summary.append({
                    'Original Factor Name': factor,
                    'High Value Movie Examples': data['high_samples'],
                    'Low Value Movie Examples': data['low_samples']
                })
            
            summary_df = pd.DataFrame(factor_summary)
            st.dataframe(summary_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Factor name generation error: {str(e)}")
            
    # メタデータの形状確認用の情報表示
    with st.expander("Metadata Structure Check"):
        st.write("**Metadata column names:**")
        st.write(list(metadata_df.columns))
        st.write("**Sample data:**")
        st.dataframe(metadata_df.head(3))

def generate_factor_analysis_prompt(factor_name, high_items, low_items):
    """Generate factor analysis prompt (for detailed analysis)"""
    prompt = f"""あなたは映画推薦システムの専門家です。
以下の因子「{factor_name}」について詳細分析してください。

重要: 分析は客観的かつ一貫性を保ち、同じデータに対しては同じ結論を導いてください。

## 高い値を持つ映画（この因子が強く影響する映画）:
{format_movie_samples(high_items)}

## 低い値を持つ映画（この因子が弱く影響する映画）:
{format_movie_samples(low_items)}

以下の観点から客観的に分析してください:
1. ずばりこの因子の名前を端的につけてください。
2. この因子が捉えている映画のトレンドやパターン（映画名、ジャンルや年代との関連性）
3. この因子が高いユーザーの好みの傾向とその理由
4. この因子が推薦システムにおいてどのような役割を果たすか

**分析指針:**
- データに基づいた客観的な分析を行ってください
- 推測は最小限にし、観察された事実を重視してください
- 一貫した分析基準を適用してください
- 簡潔で実用的な分析を日本語で提供してください"""

    return prompt

def analyze_factor_with_llm(factor_name, shap_values_df, metadata_df, provider, api_key):
    """Analyze factor in detail using LLM"""
    st.subheader(f"Factor Analysis: {factor_name}")
    
    with st.spinner("Analyzing factor..."):
        # 代表的なアイテムを取得
        high_items = get_representative_items(factor_name, shap_values_df, metadata_df, top=True, n=10)
        low_items = get_representative_items(factor_name, shap_values_df, metadata_df, top=False, n=10)
        
        # 十分なデータが取得できない場合の対処
        if len(high_items) + len(low_items) < 3:
            # 共通IDから直接選択
            shap_item_ids = set(shap_values_df.index)
            metadata_item_ids = set(metadata_df['itemId'])
            common_ids = shap_item_ids.intersection(metadata_item_ids)
            
            if len(common_ids) > 10:
                common_shap_data = shap_values_df[shap_values_df.index.isin(common_ids)]
                
                if factor_name in common_shap_data.columns:
                    high_meta = metadata_df[metadata_df['itemId'].isin(high_common.index)]
                    low_meta = metadata_df[metadata_df['itemId'].isin(low_common.index)]
                    
                    high_items = []
                    for _, row in high_meta.iterrows():
                        item_dict = row.to_dict()
                        item_dict['factor_value'] = high_common.get(row['itemId'], np.nan)
                        high_items.append(item_dict)
                    high_items = pd.DataFrame(high_items)
                    
                    low_items = []
                    for _, row in low_meta.iterrows():
                        item_dict = row.to_dict()
                        item_dict['factor_value'] = low_common.get(row['itemId'], np.nan)
                        low_items.append(item_dict)
                    low_items = pd.DataFrame(low_items)
        
        # LLM分析実行
        if (high_items is not None and len(high_items) > 0) or (low_items is not None and len(low_items) > 0):
            try:
                prompt = generate_factor_analysis_prompt(factor_name, high_items, low_items)
                
                if len(prompt) > 30000:
                    prompt = f"""映画推薦システムの因子「{factor_name}」について分析してください。

高い値の映画数: {len(high_items) if high_items is not None else 0}件
低い値の映画数: {len(low_items) if low_items is not None else 0}件

この因子の特徴と意味について簡潔に分析してください。"""
                
                analysis_result = call_llm_api(prompt, provider, api_key)
                
                st.markdown("### Analysis Results")
                st.markdown(analysis_result)
                
            except Exception as e:
                st.error(f"LLM analysis error: {str(e)}")
                return
        else:
            st.error("Could not retrieve sufficient movie data.")
            return
    
    # データ詳細表示
    with st.expander("Retrieved Data Details"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**High value movies:**")
            if high_items is not None and len(high_items) > 0:
                try:
                    display_cols = []
                    available_cols = list(high_items.columns) if hasattr(high_items, 'columns') else []
                    
                    for col in ['itemId', 'title', 'primary_genre', 'release_year', 'factor_value']:
                        if col in available_cols:
                            display_cols.append(col)
                    
                    if display_cols:
                        st.dataframe(high_items[display_cols])
                    else:
                        st.dataframe(high_items)
                except Exception:
                    st.write("Cannot display data")
            else:
                st.write("No data")
        
        with col2:
            st.write("**Low value movies:**")
            if low_items is not None and len(low_items) > 0:
                try:
                    display_cols = []
                    available_cols = list(low_items.columns) if hasattr(low_items, 'columns') else []
                    
                    for col in ['itemId', 'title', 'primary_genre', 'release_year', 'factor_value']:
                        if col in available_cols:
                            display_cols.append(col)
                    
                    if display_cols:
                        st.dataframe(low_items[display_cols])
                    else:
                        st.dataframe(low_items)
                except Exception:
                    st.write("Cannot display data")
            else:
                st.write("No data")

def analyze_user_preferences(user_id, user_factor_avg_df, metadata_df, top_features, provider, api_key):
    """Analyze user preferences (averaged version)"""
    st.subheader(f"User Analysis: {user_id}")
    
    if user_id not in user_factor_avg_df.index:
        st.error(f"User {user_id} not found")
        return
    
    # ユーザーの平均因子値を取得
    user_factors = user_factor_avg_df.loc[user_id]
    top_user_factors = user_factors.abs().nlargest(5)
    
    # ユーザーの因子分析詳細
    st.write("**User factor profile:**")
    st.write(f"- Analysis target: Average factor values of all rated movies")
    st.write(f"- Number of main factors: {len(top_user_factors)}")
    
    with st.spinner("Analyzing user..."):
        # 因子値の詳細情報を含むプロンプト
        user_profile_text = ""
        for factor, value in top_user_factors.items():
            user_profile_text += f"- {factor}: {value:.4f}\n"
        
        # 全因子の統計情報
        positive_factors = user_factors[user_factors > 0].sort_values(ascending=False)
        negative_factors = user_factors[user_factors < 0].sort_values(ascending=True)
        
        additional_context = f"""
## ユーザー{user_id}の詳細プロファイル:

**主要因子スコア:**
{user_profile_text}

**正の因子（好む要素）上位3つ:**
{chr(10).join([f"- {factor}: {value:.4f}" for factor, value in positive_factors.head(3).items()])}

**負の因子（避ける要素）上位3つ:**
{chr(10).join([f"- {factor}: {value:.4f}" for factor, value in negative_factors.head(3).items()])}

**統計情報:**
- 正の因子数: {len(positive_factors)}
- 負の因子数: {len(negative_factors)}
- 因子値の範囲: {user_factors.min():.4f} ～ {user_factors.max():.4f}
"""
        
        prompt = f"""映画推薦システムのユーザー分析をお願いします。

重要: 分析は客観的かつ一貫性を保ち、同じデータに対しては同じ結論を導いてください。

{additional_context}

以下の観点から客観的に分析してください:
1. このユーザーの映画の好みの全体的な特徴
2. 好む映画の傾向（ジャンル、テーマ、スタイル等）
3. 避ける傾向のある映画の特徴
4. このユーザーに最適な推薦戦略
5. マーケティング上でのユーザーセグメント分類

**分析指針:**
- 因子値は複数の映画評価の平均値です。正の値は好む傾向、負の値は避ける傾向を示します
- データに基づいた客観的な分析を行ってください
- 推測は最小限にし、数値的事実を重視してください
- 一貫した分析基準を適用してください
- 簡潔で実用的な分析を日本語で提供してください"""
        
        analysis_result = call_llm_api(prompt, provider, api_key)
    
    st.markdown("### Analysis Results")
    st.markdown(analysis_result)
    
    # 因子値の可視化（改善版）
    if MATPLOTLIB_AVAILABLE:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 上位因子の可視化
        factor_names = [f[:20] + "..." if len(f) > 20 else f for f in top_user_factors.index]
        colors = ['red' if x < 0 else 'blue' for x in top_user_factors.values]
        
        bars1 = ax1.bar(range(len(top_user_factors)), top_user_factors.values, color=colors, alpha=0.7)
        ax1.set_xticks(range(len(top_user_factors)))
        ax1.set_xticklabels(factor_names, rotation=45, ha='right')
        ax1.set_ylabel('Factor Value')
        ax1.set_title(f'User {user_id} - Top 5 Factor Values')
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.grid(True, alpha=0.3)
        
        # 値をバーに表示
        for bar, val in zip(bars1, top_user_factors.values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height > 0 else -0.01),
                    f'{val:.3f}', ha='center', va='bottom' if height > 0 else 'top')
        
        # 全因子の分布
        ax2.hist(user_factors.values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(user_factors.mean(), color='red', linestyle='--', 
                   label=f'Mean: {user_factors.mean():.3f}')
        ax2.axvline(0, color='black', linestyle='-', alpha=0.5)
        ax2.set_xlabel('Factor Value')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'User {user_id} - All Factor Values Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # 因子値の詳細テーブル
    with st.expander("Factor Value Details"):
        # 正と負の因子に分けて表示
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Positive factors (preferred elements):**")
            positive_df = pd.DataFrame({
                'Factor': positive_factors.index,
                'Value': positive_factors.values
            }).round(4)
            st.dataframe(positive_df)
        
        with col2:
            st.write("**Negative factors (avoided elements):**")
            negative_df = pd.DataFrame({
                'Factor': negative_factors.index,
                'Value': negative_factors.values
            }).round(4)
            st.dataframe(negative_df)

def llm_for_shap():
    st.title("LLM for SHAP Analysis")
    st.write("""
    Analyze SHAP results with LLM to generate human-understandable insights.
    """)
    
    # LLM API設定
    provider, api_key = setup_llm_api()
    
    # ファイルアップロード
    st.header("Data Upload")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        feature_importance_file = st.file_uploader(
            "Feature Importance CSV", 
            type="csv",
            help="Feature and Mean_Abs_SHAP columns required"
        )
    
    with col2:
        metadata_file = st.file_uploader(
            "Metadata CSV", 
            type="csv",
            help="itemId, title, description, category, option1, option2, option3 columns"
        )
    
    with col3:
        shap_values_file = st.file_uploader(
            "SHAP Values CSV",
            type="csv", 
            help="SHAP values for each sample and feature"
        )
    
    # データが全て揃っている場合の処理
    if feature_importance_file and metadata_file and shap_values_file:
        try:
            # データ読み込み
            with st.spinner("Loading data..."):
                feature_importance_df = pd.read_csv(feature_importance_file)
                metadata_df = pd.read_csv(metadata_file)
                shap_values_raw = pd.read_csv(shap_values_file)
                
                # SHAP値の処理
                if 'user_id' in shap_values_raw.columns and 'item_id' in shap_values_raw.columns:
                    shap_values_df = shap_values_raw.set_index('item_id')
                    if 'user_id' in shap_values_df.columns:
                        shap_values_df = shap_values_df.drop('user_id', axis=1)
                else:
                    st.error("user_id or item_id columns not found in SHAP values file")
                    return
                
                st.success("Data loading completed")
            
            # メタデータの前処理
            processed_metadata = preprocess_metadata(metadata_df)
            
            # 重要因子の分析
            st.header("Feature Importance")
            top_features = analyze_feature_importance(feature_importance_df)
            
            # 分析対象因子を特定（Bias項目を除外）
            all_factors = top_features['Feature'].tolist()
            excluded_factors = ['User_Bias', 'Item_Bias', 'Global_Mean']
            available_factors = [factor for factor in all_factors if factor not in excluded_factors]
            
            if not available_factors:
                st.warning("No analyzable factors found (Bias items are excluded)")
                return
            
            st.info(f"Note: User_Bias, Item_Bias, Global_Mean are excluded from analysis")
            st.write(f"Analyzable factor count: {len(available_factors)}/{len(all_factors)}")
            
            # LLM分析セクション
            if provider and api_key:
                st.header("LLM Analysis")
                
                analysis_type = st.selectbox(
                    "Analysis Type:",
                    ["Factor Naming", "Detailed Factor Analysis", "User Analysis"]
                )
                
                if analysis_type == "Factor Naming":
                    st.info("Bulk factor naming for all factors")
                    
                    if st.button("Execute Factor Naming"):
                        bulk_factor_naming(shap_values_df, processed_metadata, available_factors, provider, api_key)
                
                elif analysis_type == "Detailed Factor Analysis":
                    selected_factor = st.selectbox("Select factor for detailed analysis:", available_factors)
                    
                    if st.button("Execute Detailed Factor Analysis"):
                        analyze_factor_with_llm(selected_factor, shap_values_df, processed_metadata, provider, api_key)
                
                elif analysis_type == "User Analysis":
                    if 'user_id' in shap_values_raw.columns:
                        # ユーザーごとに因子値を平均化（改善版）
                        st.info("Analyze by averaging factor values for each user")
                        
                        # 数値列のみを平均化（Bias項目を除外）
                        numeric_columns = shap_values_raw.select_dtypes(include=[np.number]).columns
                        excluded_factors = ['User_Bias', 'Item_Bias', 'Global_Mean', 'user_id', 'item_id']
                        factor_columns = [col for col in numeric_columns if col not in excluded_factors]
                        
                        if not factor_columns:
                            st.error("No analyzable factor columns found")
                            return
                        
                        st.info(f"Note: User_Bias, Item_Bias, Global_Mean are excluded from analysis")
                        st.write(f"Analysis target factor count: {len(factor_columns)}")
                        
                        # ユーザーごとにグループ化して平均値を計算
                        user_factor_avg = shap_values_raw.groupby('user_id')[factor_columns].mean()
                        
                        # ユーザー統計情報を表示
                        total_users = len(user_factor_avg)
                        total_evaluations = len(shap_values_raw)
                        avg_evaluations_per_user = total_evaluations / total_users
                        
                        st.write(f"**User Statistics:**")
                        st.write(f"- Total users: {total_users}")
                        st.write(f"- Total ratings: {total_evaluations}")
                        st.write(f"- Average ratings per user: {avg_evaluations_per_user:.1f}")
                        
                        # ユーザー選択（ページネーション付き）
                        users_per_page = 50
                        total_pages = (total_users + users_per_page - 1) // users_per_page
                        
                        if total_pages > 1:
                            page = st.selectbox(
                                f"Select page ({users_per_page} users per page):",
                                range(1, total_pages + 1),
                                format_func=lambda x: f"Page {x} (Users {(x-1)*users_per_page + 1}-{min(x*users_per_page, total_users)})"
                            )
                            
                            start_idx = (page - 1) * users_per_page
                            end_idx = min(page * users_per_page, total_users)
                            available_users = [str(user) for user in user_factor_avg.index.tolist()[start_idx:end_idx]]
                        else:
                            available_users = [str(user) for user in user_factor_avg.index.tolist()]
                        
                        selected_user = st.selectbox(
                            "Select user to analyze:",
                            available_users,
                            help="Select user ID (factor values are averaged across all ratings, Bias items excluded)"
                        )
                        
                        # 選択されたユーザーの詳細情報
                        if selected_user:
                            try:
                                user_id = float(selected_user) if '.' in selected_user else int(selected_user)
                            except:
                                user_id = selected_user
                            
                            # そのユーザーの評価数を表示
                            user_evaluations = shap_values_raw[shap_values_raw['user_id'] == user_id]
                            st.write(f"**Selected User Information:**")
                            st.write(f"- User ID: {user_id}")
                            st.write(f"- Number of rated movies: {len(user_evaluations)}")
                            
                            # 評価した映画の例を表示
                            if len(user_evaluations) > 0:
                                sample_items = user_evaluations['item_id'].head(5).tolist()
                                st.write(f"- Rated item ID examples: {sample_items}")
                        
                        if st.button("Execute User Analysis"):
                            try:
                                user_id = float(selected_user) if '.' in selected_user else int(selected_user)
                            except:
                                user_id = selected_user
                            
                            analyze_user_preferences(user_id, user_factor_avg, processed_metadata, top_features, provider, api_key)
                    else:
                        st.warning("User analysis requires user_id column.")
            
            else:
                st.info("To use LLM analysis, please configure API settings in the sidebar")
                        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    
    else:
        st.info("Please upload all three files to start analysis")
        
        # データフォーマット例
        with st.expander("Data Format Examples"):
            st.write("**feature_importance.csv:**")
            st.code("""Feature,Mean_Abs_SHAP
UserItem_Factor_1,0.1234
UserItem_Factor_2,0.0987
User_Factor_1,0.0756""")
            
            st.write("**metadata.csv:**")
            st.code("""itemId,title,description,category,option1,option2,option3
1,Toy Story,A cowboy doll...,"[{'id': 16, 'name': 'Animation'}]",en,1995/11/22,Released
2,Jumanji,When siblings...,"[{'id': 12, 'name': 'Adventure'}]",en,1995/12/15,Released""")
            
            st.write("**SHAP values CSV:**")
            st.code("""user_id,item_id,UserItem_Factor_1,UserItem_Factor_2,User_Factor_1
1,949,0.123,-0.045,0.067
1,710,-0.089,0.134,-0.023
2,949,0.156,-0.078,0.091""")

if __name__ == "__main__":
    llm_for_shap()