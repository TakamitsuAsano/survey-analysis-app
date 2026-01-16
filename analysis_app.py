import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# --- 日本語フォント設定（全環境対応版） ---
def setup_japanese_font():
    # フォントファイルの保存先
    font_path = "ipaexg.ttf"
    
    # フォントファイルがなければダウンロードする
    if not os.path.exists(font_path):
        import urllib.request
        # IPAexゴシック（標準的な日本語フォント）のダウンロードURL
        url = "https://github.com/minodisk/font-ipa/raw/master/fonts/ipaexg.ttf"
        try:
            with st.spinner("日本語フォントを準備中..."):
                urllib.request.urlretrieve(url, font_path)
        except Exception as e:
            st.error(f"フォントのダウンロードに失敗しました: {e}")
            return

    # フォントをmatplotlibに登録
    fm.fontManager.addfont(font_path)
    plt.rc('font', family='IPAexGothic')

# アプリ起動時にフォント設定を実行
setup_japanese_font()
# ---------------------------------------

# ページ設定
st.set_page_config(page_title="アンケート分析 & 決定木ツール", layout="wide")

st.title("📊 アンケート自動集計 & 決定木分析アプリ")
st.markdown("ExcelやCSVをアップロードして、クロス集計と決定木分析を自動化します。")

# 1. ファイルアップロード
st.sidebar.header("データのアップロード")
uploaded_file = st.sidebar.file_uploader("ExcelまたはCSVファイルをアップロード", type=['xlsx', 'csv'])

if uploaded_file is not None:
    # データの読み込み
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.sidebar.success("読み込み成功！")
    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
        st.stop()

    # タブで機能を分ける
    tab1, tab2, tab3 = st.tabs(["📋 データ確認", "📈 クロス集計", "🌳 決定木分析"])

    # --- タブ1: データ確認 ---
    with tab1:
        st.subheader("アップロードされたデータ")
        st.dataframe(df)
        st.info(f"データサイズ: {df.shape[0]} 行, {df.shape[1]} 列")

    # --- タブ2: クロス集計 ---
    with tab2:
        st.subheader("クロス集計と可視化")
        
        col1, col2 = st.columns(2)
        with col1:
            index_col = st.selectbox("行（Index）を選択", df.columns, index=0)
        with col2:
            columns_col = st.selectbox("列（Column）を選択", df.columns, index=min(1, len(df.columns)-1))

        # クロス集計の実行
        cross_tab = pd.crosstab(df[index_col], df[columns_col])
        
        # 表示
        st.write("##### 集計表")
        st.dataframe(cross_tab)

        # グラフ化
        graph_type = st.radio("グラフの種類", ["ヒートマップ", "積み上げ棒グラフ"], horizontal=True)
        
        if graph_type == "ヒートマップ":
            fig = px.imshow(cross_tab, text_auto=True, aspect="auto", color_continuous_scale='Blues')
        else:
            cross_tab_reset = cross_tab.reset_index().melt(id_vars=index_col, var_name=columns_col, value_name="Count")
            fig = px.bar(cross_tab_reset, x=index_col, y="Count", color=columns_col, title=f"{index_col} × {columns_col}")
        
        st.plotly_chart(fig, use_container_width=True)

    # --- タブ3: 決定木分析 ---
    with tab3:
        st.subheader("決定木分析 (Decision Tree)")
        
        col1, col2 = st.columns(2)
        with col1:
            target_col = st.selectbox("目的変数（予測したい結果）", df.columns)
        with col2:
            feature_cols = st.multiselect("説明変数（要因と思われる項目）", [c for c in df.columns if c != target_col], default=[c for c in df.columns if c != target_col][:3])

        if st.button("分析を実行する"):
            if not feature_cols:
                st.warning("説明変数を1つ以上選択してください。")
            else:
                df_ml = df.copy()
                le = LabelEncoder()
                df_ml = df_ml[[target_col] + feature_cols].dropna()
                
                for col in df_ml.columns:
                    if df_ml[col].dtype == 'object':
                        df_ml[col] = df_ml[col].astype(str)
                        df_ml[col] = le.fit_transform(df_ml[col])

                X = df_ml[feature_cols]
                y = df_ml[target_col]

                clf = DecisionTreeClassifier(max_depth=3, random_state=42)
                clf.fit(X, y)

                # フォント指定済みの設定で描画
                fig, ax = plt.subplots(figsize=(14, 7))
                plot_tree(clf, feature_names=feature_cols, class_names=True, filled=True, ax=ax, fontsize=12)
                st.pyplot(fig)
                st.success("分析完了！")
else:
    st.info("👈 左側のサイドバーからファイルをアップロードしてください。")
