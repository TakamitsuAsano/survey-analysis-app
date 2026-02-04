import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from streamlit_gsheets import GSheetsConnection

# --- 日本語フォント設定（高安定版） ---
def setup_japanese_font():
    font_file = "ipaexg.ttf"
    if not os.path.exists(font_file) or os.path.getsize(font_file) < 1000:
        import urllib.request
        url = "https://raw.githubusercontent.com/yutodama/japanize-matplotlib/master/japanize_matplotlib/fonts/ipaexg.ttf"
        try:
            with st.spinner("日本語フォント(IPAexゴシック)をダウンロード中..."):
                urllib.request.urlretrieve(url, font_file)
        except Exception as e:
            st.error(f"フォントのダウンロードに失敗しました: {e}")
            return

    try:
        fm.fontManager.addfont(font_file)
        font_prop = fm.FontProperties(fname=font_file)
        plt.rcParams['font.family'] = font_prop.get_name()
    except Exception as e:
        st.error(f"フォント設定エラー: {e}")

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
    # データの前処理（空行削除）
    df = df.dropna(how='all')

    # タブ構成
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
            st.dataframe(cross_tab)

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
        st.markdown("条件分岐によって、どのような組み合わせが結果につながるかを可視化します。")
        
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

    # --- タブ4: ドライバー分析 (重回帰/ロジスティック回帰) ---
    with tab4:
        st.subheader("🚀 要因（ドライバー）分析")
        st.markdown("""
        ある結果に対して、**「どの要素がプラスに働き、どの要素がマイナスに働いたか」**をランキング化します。
        （例：「満足」と答えた人にとって、最も重要だったのは「接客」なのか「価格」なのか？）
        """)

        

        col1, col2 = st.columns(2)
        with col1:
            target_col_reg = st.selectbox("目的変数（分析したい結果）", df.columns, key="reg_target")
        with col2:
            feature_cols_reg = st.multiselect("説明変数（背景・要因と思われる項目）", [c for c in df.columns if c != target_col_reg], default=[c for c in df.columns if c != target_col_reg][:5], key="reg_feature")

        st.info("💡 ヒント: 目的変数が「満足/不満」のような文字の場合、自動的に数値に変換して分析します。")

        if st.button("ドライバー分析を実行"):
            if not feature_cols_reg:
                st.warning("説明変数を1つ以上選択してください。")
            else:
                try:
                    # データ準備
                    df_reg = df[[target_col_reg] + feature_cols_reg].dropna()
                    
                    # 数値化処理（One-Hot EncodingではなくLabel Encodingで簡易化、または数値化）
                    # 今回は解釈しやすくするため、全て数値化（LabelEncoder）して相関を見ます
                    le_dict = {}
                    for col in df_reg.columns:
                        if df_reg[col].dtype == 'object':
                            le = LabelEncoder()
                            df_reg[col] = df_reg[col].astype(str)
                            df_reg[col] = le.fit_transform(df_reg[col])
                            le_dict[col] = le # ラベルの対応表を保存（後で使えるように）

                    X = df_reg[feature_cols_reg]
                    y = df_reg[target_col_reg]

                    # データの標準化（影響度の大きさを比較できるようにする）
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)

                    # モデル構築（ロジスティック回帰を使用）
                    # 目的変数が多値の場合でも動くように設定
                    model = LogisticRegression(max_iter=1000)
                    model.fit(X_scaled, y)

                    # 係数の取得（クラスごとの係数を見る）
                    # 多クラス分類の場合、model.coef_ は (クラス数, 特徴量数) になる
                    # ここでは「値が最も大きいクラス（例：満足度が高い）」に対する影響度を表示する簡易ロジック
                    
                    # ターゲットのクラス名を取得（LabelEncoderを使った場合）
                    if target_col_reg in le_dict:
                        classes = le_dict[target_col_reg].classes_
                        target_class_index = -1 # 一番最後のクラス（例：満足、好き）をターゲットにする
                        target_label = classes[target_class_index]
                    else:
                        target_label = "最大値"
                        target_class_index = -1

                    # 係数の抽出
                    if model.coef_.shape[0] > 1:
                        # 多クラスの場合、一番最後のクラス（通常はポジティブな回答）への係数を使用
                        coefs = model.coef_[target_class_index]
                    else:
                        # 2値分類の場合
                        coefs = model.coef_[0]

                    # 結果のデータフレーム作成
                    res_df = pd.DataFrame({
                        "要因": feature_cols_reg,
                        "影響度(係数)": coefs
                    }).sort_values(by="影響度(係数)", ascending=True)

                    # グラフ化
                    st.write(f"### 「{target_col_reg}」への影響度分析")
                    st.markdown(f"※ グラフが**右（プラス）**にあるほど、その要素は結果を**促進**しています。\n※ グラフが**左（マイナス）**にあるほど、その要素は結果を**抑制**しています。")
                    
                    fig = px.bar(res_df, x="影響度(係数)", y="要因", orientation='h', 
                                 title=f"「{target_col_reg}」に対するドライバー要因",
                                 color="影響度(係数)", color_continuous_scale="RdBu_r")
                    
                    # 中心線を追加
                    fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="black")
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"分析中にエラーが発生しました: {e}")
                    st.warning("データに偏りがあるか、選択した項目数が少なすぎる可能性があります。")

else:
    st.info("👈 左側のサイドバーからデータを選択してください。")
