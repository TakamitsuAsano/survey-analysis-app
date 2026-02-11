import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from streamlit_gsheets import GSheetsConnection

# --- 日本語フォント設定（システムインストール版） ---
def setup_japanese_font():
    # packages.txt でインストールされたフォントの場所
    font_path = "/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf"
    
    # フォントファイルが存在するか確認して設定
    if os.path.exists(font_path):
        try:
            fm.fontManager.addfont(font_path)
            plt.rcParams['font.family'] = 'IPAexGothic'
        except Exception as e:
            st.error(f"フォント設定エラー: {e}")
    else:
        # 万が一見つからない場合のフォールバック（Mac/Windowsローカル実行用など）
        try:
            plt.rcParams['font.family'] = 'Hiragino Sans' # Mac用
        except:
            pass # 何もしない

setup_japanese_font()
# ---------------------------------------

st.set_page_config(page_title="アンケート分析 & ドライバー分析", layout="wide")
st.title("📊 アンケート自動集計 & ドライバー分析アプリ")

# --- サイドバー：データ読み込み ---
st.sidebar.header("データの読み込み")
input_method = st.sidebar.radio("データの種類を選択", ["ファイルアップロード", "Googleスプレッドシート"])

df = None

if input_method == "ファイルアップロード":
    uploaded_file = st.sidebar.file_uploader("ExcelまたはCSVファイルをアップロード", type=['xlsx', 'csv'])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.sidebar.success("ファイル読み込み成功！")
        except Exception as e:
            st.error(f"エラー: {e}")

elif input_method == "Googleスプレッドシート":
    st.sidebar.info("事前にスプレッドシートの「共有」にサービスアカウントのアドレスを追加してください。")
    sheet_url = st.sidebar.text_input("スプレッドシートのURLを入力")
    
    if sheet_url:
        try:
            conn = st.connection("gsheets", type=GSheetsConnection)
            df = conn.read(spreadsheet=sheet_url, ttl=0)
            st.sidebar.success("スプレッドシート接続成功！")
        except Exception as e:
            st.sidebar.error(f"接続エラー: Secretsの設定またはURLを確認してください。\n{e}")

# --- 分析メイン処理 ---
if df is not None:
    # データの前処理
    df = df.dropna(how='all')

    tab1, tab2, tab3, tab4 = st.tabs(["📋 データ確認", "📈 クロス集計", "🌳 決定木分析", "🚀 要因(ドライバー)分析"])

    # --- タブ1: データ確認 ---
    with tab1:
        st.subheader("データプレビュー")
        st.dataframe(df)
        st.info(f"データサイズ: {df.shape[0]} 行, {df.shape[1]} 列")

    # --- タブ2: クロス集計 ---
    with tab2:
        st.subheader("クロス集計と可視化")
        col1, col2 = st.columns(2)
        with col1:
            index_col = st.selectbox("行（Index）を選択", df.columns, index=0)
        with col2:
            default_col_idx = 1 if len(df.columns) > 1 else 0
            columns_col = st.selectbox("列（Column）を選択", df.columns, index=default_col_idx)

        if index_col == columns_col:
            st.warning("⚠️ 行と列には異なる項目を選択してください。")
        else:
            cross_tab = pd.crosstab(df[index_col], df[columns_col])
            
            st.write("##### 集計表")
            st.caption("💡 **表の中をクリック** して `Ctrl + A` (全選択) → `Ctrl + C` (コピー) でExcelに貼り付けられます。")
            
            # 【変更点】st.dataframe ではなく st.data_editor を使用
            st.data_editor(cross_tab)

            # (念のためダウンロードボタンも残しておきます)
            csv = cross_tab.to_csv().encode('utf-8_sig')
            st.download_button(
                label="📥 CSVでダウンロード",
                data=csv,
                file_name=f'cross_tab_{index_col}_{columns_col}.csv',
                mime='text/csv',
            )

            graph_type = st.radio("グラフの種類", ["ヒートマップ", "積み上げ棒グラフ"], horizontal=True)
            if graph_type == "ヒートマップ":
                fig = px.imshow(cross_tab, text_auto=True, aspect="auto", color_continuous_scale='Blues')
            else:
                val_name = "Count"
                if val_name == index_col or val_name == columns_col: val_name = "Frequency"
                cross_tab_reset = cross_tab.reset_index().melt(id_vars=index_col, var_name=columns_col, value_name=val_name)
                fig = px.bar(cross_tab_reset, x=index_col, y=val_name, color=columns_col, title=f"{index_col} × {columns_col}")
            st.plotly_chart(fig, use_container_width=True)

    # --- タブ3: 決定木分析 ---
    with tab3:
        st.subheader("決定木分析 (Decision Tree)")
        
        col1, col2 = st.columns(2)
        with col1:
            target_col_tree = st.selectbox("目的変数（結果）", df.columns, key="tree_target")
        with col2:
            feature_cols_tree = st.multiselect("説明変数（要因）", [c for c in df.columns if c != target_col_tree], default=[c for c in df.columns if c != target_col_tree][:3], key="tree_feature")

        if st.button("決定木分析を実行"):
            if not feature_cols_tree:
                st.warning("説明変数を1つ以上選択してください。")
            else:
                df_ml = df.copy()
                le = LabelEncoder()
                df_ml = df_ml[[target_col_tree] + feature_cols_tree].dropna()
                
                for col in df_ml.columns:
                    df_ml[col] = df_ml[col].astype(str)
                    df_ml[col] = le.fit_transform(df_ml[col])

                X = df_ml[feature_cols_tree]
                y = df_ml[target_col_tree]

                clf = DecisionTreeClassifier(max_depth=3, random_state=42)
                clf.fit(X, y)

                fig, ax = plt.subplots(figsize=(14, 7))
                plot_tree(clf, feature_names=feature_cols_tree, class_names=True, filled=True, ax=ax, fontsize=12)
                st.pyplot(fig)

    # --- タブ4: ドライバー分析 ---
    with tab4:
        st.subheader("🚀 要因（ドライバー）分析")
        st.markdown("目的変数に対して、どの要素がプラス/マイナスに影響しているかを分析します。")

        col1, col2 = st.columns(2)
        with col1:
            target_col_reg = st.selectbox("目的変数（分析したい結果）", df.columns, key="reg_target")
        with col2:
            feature_cols_reg = st.multiselect("説明変数（背景・要因と思われる項目）", [c for c in df.columns if c != target_col_reg], default=[c for c in df.columns if c != target_col_reg][:5], key="reg_feature")

        if st.button("ドライバー分析を実行"):
            if not feature_cols_reg:
                st.warning("説明変数を1つ以上選択してください。")
            else:
                try:
                    df_reg = df[[target_col_reg] + feature_cols_reg].dropna()
                    le_dict = {}
                    for col in df_reg.columns:
                        if df_reg[col].dtype == 'object':
                            le = LabelEncoder()
                            df_reg[col] = df_reg[col].astype(str)
                            df_reg[col] = le.fit_transform(df_reg[col])
                            le_dict[col] = le 

                    X = df_reg[feature_cols_reg]
                    y = df_reg[target_col_reg]

                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)

                    model = LogisticRegression(max_iter=1000)
                    model.fit(X_scaled, y)

                    if model.coef_.shape[0] > 1:
                        target_class_index = -1 
                        coefs = model.coef_[target_class_index]
                    else:
                        coefs = model.coef_[0]

                    res_df = pd.DataFrame({
                        "要因": feature_cols_reg,
                        "影響度(係数)": coefs
                    }).sort_values(by="影響度(係数)", ascending=True)

                    st.write(f"### 「{target_col_reg}」への影響度分析")
                    fig = px.bar(res_df, x="影響度(係数)", y="要因", orientation='h', 
                                 title=f"「{target_col_reg}」に対するドライバー要因",
                                 color="影響度(係数)", color_continuous_scale="RdBu_r")
                    fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="black")
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"分析中にエラーが発生しました: {e}")
                    st.warning("データに偏りがあるか、選択した項目数が少なすぎる可能性があります。")
else:
    st.info("👈 左側のサイドバーからデータを選択してください。")
