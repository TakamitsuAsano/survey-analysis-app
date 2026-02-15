import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import graphviz # Graphviz用に追加
from streamlit_gsheets import GSheetsConnection
from st_copy_to_clipboard import st_copy_to_clipboard

# --- 日本語フォント設定 ---
def setup_japanese_font():
    font_path = "/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf"
    if os.path.exists(font_path):
        try:
            fm.fontManager.addfont(font_path)
            plt.rcParams['font.family'] = 'IPAexGothic'
        except Exception as e:
            st.error(f"フォント設定エラー: {e}")
    else:
        try:
            plt.rcParams['font.family'] = 'Hiragino Sans'
        except:
            pass

setup_japanese_font()
# ---------------------------------------

st.set_page_config(page_title="アンケート分析 & セグメンテーション", layout="wide")
st.title("📊 アンケート分析 & セグメンテーションツール")

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
    df = df.dropna(how='all')

    # タブ構成
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 データ確認", 
        "📈 クロス集計", 
        "🌳 決定木分析", 
        "🚀 要因(ドライバー)分析", 
        "🧩 クラスター分析",
        "📖 分析手法の解説(用語集)"
    ])

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
            
            copy_text = cross_tab.to_csv(sep='\t')
            st_copy_to_clipboard(copy_text, "📋 表をコピー (ヘッダー付)", "✅ コピーしました！")
            
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

    # --- タブ3: 決定木分析 (Graphviz版) ---
    with tab3:
        st.subheader("決定木分析")
        st.caption("💡 図はマウスホイールで**拡大・縮小**、ドラッグで**移動**ができます。")
        
        col1, col2 = st.columns(2)
        with col1:
            target_col_tree = st.selectbox("目的変数（結果）", df.columns, key="tree_target")
        with col2:
            feature_cols_tree = st.multiselect("説明変数（要因）", [c for c in df.columns if c != target_col_tree], default=[c for c in df.columns if c != target_col_tree][:3], key="tree_feature")

        if st.button("決定木分析を実行"):
            if not feature_cols_tree:
                st.warning("説明変数を1つ以上選択してください。")
            else:
                try:
                    df_ml = df.copy()
                    
                    # 数値化マッピング
                    label_mappings = {}
                    class_names_list = None
                    
                    # 目的変数の処理
                    if df_ml[target_col_tree].dtype == 'object':
                        le_target = LabelEncoder()
                        df_ml[target_col_tree] = df_ml[target_col_tree].astype(str)
                        df_ml[target_col_tree] = le_target.fit_transform(df_ml[target_col_tree])
                        class_names_list = le_target.classes_.astype(str).tolist()
                    else:
                        class_names_list = sorted(df_ml[target_col_tree].unique().astype(str).tolist())

                    # 説明変数の処理
                    for col in feature_cols_tree:
                        if df_ml[col].dtype == 'object':
                            df_ml[col] = df_ml[col].astype(str)
                            le = LabelEncoder()
                            df_ml[col] = le.fit_transform(df_ml[col])

                    df_ml = df_ml.dropna(subset=[target_col_tree] + feature_cols_tree)
                    
                    X = df_ml[feature_cols_tree]
                    y = df_ml[target_col_tree]

                    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
                    clf.fit(X, y)

                    # --- 分岐ルールのテキスト生成 ---
                    tree_rules = export_text(clf, feature_names=feature_cols_tree)
                    st.write("##### 📋 分岐条件のテキスト詳細")
                    st_copy_to_clipboard(tree_rules, "📋 分岐ルールをコピー", "✅ コピーしました")
                    st.code(tree_rules)

                    # --- Graphvizによる描画 (ここを変更) ---
                    # 日本語フォントを指定してDOTデータを生成
                    dot_data = export_graphviz(
                        clf,
                        out_file=None,
                        feature_names=feature_cols_tree,
                        class_names=class_names_list,
                        filled=True,
                        rounded=True,
                        special_characters=True,
                        fontname="IPAexGothic" # 日本語フォント指定
                    )
                    
                    st.graphviz_chart(dot_data)
                    
                    # --- 画像ダウンロードボタン ---
                    # Graphvizを使ってPNGデータをメモリ上で生成
                    try:
                        graph = graphviz.Source(dot_data)
                        # png形式のバイナリを取得
                        png_bytes = graph.pipe(format='png')
                        
                        st.download_button(
                            label="📥 決定木画像をダウンロード (高画質PNG)",
                            data=png_bytes,
                            file_name="decision_tree.png",
                            mime="image/png"
                        )
                    except Exception as e:
                        st.warning(f"画像ダウンロード準備中にエラー: {e} (表示には影響ありません)")

                except Exception as e:
                    st.error(f"分析エラー: {e}")

    # --- タブ4: 要因(ドライバー)分析 ---
    with tab4:
        st.subheader("🚀 要因（ドライバー）分析：オッズ比")
        
        st.info("""
        **💡 数値の見方（オッズ比）**
        * **1.0 より大きい**: その要因が結果を**促進**します。（例：2.0なら、その要因があると結果が2倍起こりやすい）
        * **1.0 より小さい**: その要因が結果を**抑制**します。（例：0.5なら、その要因があると結果が半分しか起こらない）
        """)

        col1, col2 = st.columns(2)
        with col1:
            target_col_reg = st.selectbox("目的変数（分析したい結果）", df.columns, key="reg_target")
        with col2:
            feature_cols_reg = st.multiselect("説明変数（背景・要因と思われる項目）", [c for c in df.columns if c != target_col_reg], default=[c for c in df.columns if c != target_col_reg][:5], key="reg_feature")

        if st.button("要因分析を実行"):
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
                        coefs = model.coef_[-1]
                    else:
                        coefs = model.coef_[0]
                    
                    odds_ratios = np.exp(coefs)

                    res_df = pd.DataFrame({
                        "要因": feature_cols_reg,
                        "オッズ比": odds_ratios
                    }).sort_values(by="オッズ比", ascending=True)

                    st.write(f"### 「{target_col_reg}」への影響度（オッズ比）")
                    fig = px.bar(res_df, x="オッズ比", y="要因", orientation='h', 
                                 title=f"「{target_col_reg}」に対するオッズ比（1.0が基準）",
                                 color="オッズ比", 
                                 color_continuous_scale="RdBu_r",
                                 color_continuous_midpoint=1.0)
                    
                    fig.add_vline(x=1.0, line_width=2, line_dash="dash", line_color="black", annotation_text="基準(1.0)")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.write("##### 詳細データ")
                    st_copy_to_clipboard(res_df.to_csv(sep='\t'), "📋 数値をコピー", "✅ コピーしました")
                    st.dataframe(res_df)

                except Exception as e:
                    st.error(f"分析エラー: {e}")

    # --- タブ5: クラスター分析 ---
    with tab5:
        st.subheader("🧩 クラスター分析（セグメンテーション）")
        st.markdown("""
        要因分析で重要だとわかった項目を使って、ユーザーをグループ分けします。
        """)

        cluster_features = st.multiselect(
            "クラスター分析に使う変数を選択してください",
            df.columns,
            default=df.columns[:5],
            key="cluster_features"
        )
        
        n_clusters = st.slider("分類するグループ数（クラスター数）", 2, 10, 4)

        if st.button("クラスター分析を実行"):
            if not cluster_features:
                st.warning("変数を選択してください")
            else:
                try:
                    df_cluster = df[cluster_features].dropna()
                    
                    for col in df_cluster.columns:
                        if df_cluster[col].dtype == 'object':
                            le = LabelEncoder()
                            df_cluster[col] = df_cluster[col].astype(str)
                            df_cluster[col] = le.fit_transform(df_cluster[col])

                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(df_cluster)

                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    clusters = kmeans.fit_predict(X_scaled)
                    
                    df['Cluster'] = clusters
                    df['Cluster_Name'] = df['Cluster'].apply(lambda x: f"グループ {x+1}")

                    st.success(f"{n_clusters}つのグループに分類しました！")

                    # ヒートマップ
                    df_cluster['Cluster_Name'] = df['Cluster'].apply(lambda x: f"グループ {x+1}")
                    cluster_means_numeric = df_cluster.groupby('Cluster_Name').mean()

                    st.write("##### グループごとの回答傾向（ヒートマップ）")
                    fig = px.imshow(cluster_means_numeric, 
                                    text_auto=".2f", 
                                    aspect="auto",
                                    color_continuous_scale="Viridis",
                                    title="クラスターごとの特徴比較")
                    st.plotly_chart(fig, use_container_width=True)

                    st.write("##### 分類結果付きデータのダウンロード")
                    csv = df.to_csv(index=False).encode('utf-8_sig')
                    st.download_button(
                        label="📥 分類結果付きCSVをダウンロード",
                        data=csv,
                        file_name='clustered_data.csv',
                        mime='text/csv',
                    )
                    st.write("##### グループごとの人数")
                    st.dataframe(df['Cluster_Name'].value_counts().reset_index().rename(columns={'index':'Group', 'Cluster_Name':'Count'}))

                except Exception as e:
                    st.error(f"分析エラー: {e}")

    # --- タブ6: 分析手法の解説(用語集) ---
    with tab6:
        st.header("📖 統計分析手法の解説ガイド")
        st.markdown("""
        このアプリで使用している分析手法について、「専門的な説明（Technical）」と「わかりやすい説明（Plain）」を併記しています。
        報告書作成やプレゼンテーションの際にご活用ください。
        """)
        
        st.divider()

        # --- 1. クロス集計 ---
        st.subheader("1. クロス集計 (Cross Tabulation)")
        with st.expander("詳細を見る"):
            st.markdown("""
            #### 🛠 使用メソッド・原理
            * **手法**: 分割表 (Contingency Table) の作成
            * **統計的背景**: 2つの変数（質問項目）の間に「関連があるか」を見るために使用します。厳密には「カイ二乗検定 (Chi-square test)」を用いて、その偏りが偶然かどうかを判断することが一般的です。

            #### 💡 わかりやすい説明
            * **これ何？**: 「年代別 × 満足度」のように、2つの質問を掛け合わせて表にする最も基本的な分析です。
            * **目的**: 全体だけでは見えない、特定の属性（男女、年代など）ごとの違いを発見します。
            * **「有意差（意味のある差）」の目安**:
                * 一般的に、比較したいグループ間で **10%以上の差** があれば、「差がある」と見なして良いケースが多いです（マーケティング現場レベル）。
                * サンプル数が少ない（n=30以下など）場合は、たまたまの誤差である可能性が高いため、20〜30%の差がないと信頼できません。
            """)

        # --- 2. 決定木分析 ---
        st.subheader("2. 決定木分析 (Decision Tree)")
        with st.expander("詳細を見る"):
            st.markdown("""
            #### 🛠 使用メソッド・原理
            * **手法**: **CART法 (Classification and Regression Trees)**
            * **アルゴリズム**: scikit-learnの `DecisionTreeClassifier` を使用。
            * **統計的背景**: データを分割する際、**「Gini不純度 (Gini Impurity)」** という指標を使っています。これは「どれだけ綺麗にYes/Noが分かれたか」を計算するもので、この不純度が最も低くなる条件を探して自動的に分岐を作っています。

            #### 💡 わかりやすい説明
            * **これ何？**: 

[Image of decision tree diagram example]
 「もしAならB、そうでなければC」というように、結果に至る条件をツリー状に分解する手法です。
            * **目的**: 複雑な要因を整理し、**「一番影響力が大きい条件は何か？」**を視覚的に見つけるために使います。
            * **見方**:
                * **一番上の分岐**: これが**最も重要な要因**です。ここを見るだけで、結果を左右する最大のポイントがわかります。
            """)

        # --- 3. ドライバー分析 ---
        st.subheader("3. ドライバー分析 / 要因分析")
        with st.expander("詳細を見る"):
            st.markdown("""
            #### 🛠 使用メソッド・原理
            * **手法**: **ロジスティック回帰分析 (Logistic Regression)**
            * **アルゴリズム**: scikit-learnの `LogisticRegression` を使用。
            * **統計的背景**: 結果が「Yes/No（買った/買わない）」のような2値データの場合、通常の回帰分析は使えません。そこで「確率」を予測するロジスティック回帰を用います。
            * **係数の変換**: 算出された「偏回帰係数」を、指数変換（$e^x$）することで **「オッズ比 (Odds Ratio)」** に変換して表示しています。

            #### 💡 わかりやすい説明
            * **これ何？**:  ある結果（例：商品を買った）に対して、どの要因がどれくらい影響したかを数値化する手法です。
            * **目的**: 施策の優先順位を決めるためです。「これを改善すれば、結果がこれだけ伸びる」というレバーを見つけます。
            * **「オッズ比」とは？**:
                * 結果の**「起こりやすさ」が何倍になるか**を表す数値です。
                * **1.0**: 影響なし（プラマイゼロ）。
                * **2.0**: その要素があると、結果が **2倍** 起こりやすくなる（強い促進要因）。
                * **0.5**: その要素があると、結果が **半分** しか起きなくなる（強い阻害要因）。
            """)

        # --- 4. クラスター分析 ---
        st.subheader("4. クラスター分析 (Cluster Analysis)")
        with st.expander("詳細を見る"):
            st.markdown("""
            #### 🛠 使用メソッド・原理
            * **手法**: **K-Means法 (K-平均法 / 非階層クラスター分析)**
            * **アルゴリズム**: scikit-learnの `KMeans` を使用。
            * **統計的背景**: データを $k$ 個のグループに分ける際、各グループの中心（重心）からの距離が最小になるように計算します。教師なし学習（正解データがいらない分析）の一種です。

            #### 💡 わかりやすい説明
            * **これ何？**:  回答パターンが似ている人を集めて、自動的にグループ（チーム）を作る手法です。
            * **目的**: 「セグメンテーション（顧客分類）」を行うためです。性別や年代だけでなく、「意識」や「行動」で分類することで、より刺さるメッセージを作れます。
            * **使い方のコツ**: 
                * 要因分析で「重要だ」とわかった項目を使ってクラスターを作ると、意味のあるグループができやすいです。
                * グループ名は、ヒートマップを見て人間が考えます（例：「価格重視派」「品質重視派」など）。
            """)

else:
    st.info("👈 左側のサイドバーからデータを選択してください。")
