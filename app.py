import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import shap
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="🏠 Housing Price Predictor", page_icon="🏠",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 100%); }
    .block-container { padding: 2rem 2.5rem 4rem; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #111827 100%);
        border-right: 1px solid rgba(124,131,253,0.15);
    }
    .hero {
        background: linear-gradient(135deg, rgba(124,131,253,0.12), rgba(168,85,247,0.08));
        border: 1px solid rgba(124,131,253,0.25); border-radius: 20px;
        padding: 2.5rem 2rem; margin-bottom: 2rem; text-align: center;
    }
    .hero-title {
        font-size: 2.6rem; font-weight: 800; margin: 0;
        background: linear-gradient(135deg, #a78bfa, #7c83fd, #60a5fa);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero-sub { color: rgba(226,232,240,0.55); font-size: 0.95rem; margin-top: 0.6rem; }
    .hero-badges { margin-top: 1rem; display: flex; gap: 0.5rem; justify-content: center; flex-wrap: wrap; }
    .badge {
        background: rgba(124,131,253,0.12); border: 1px solid rgba(124,131,253,0.3);
        border-radius: 20px; padding: 4px 14px; font-size: 0.75rem;
        color: #a78bfa; font-weight: 600;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
        border: 1px solid rgba(124,131,253,0.2); border-radius: 16px;
        padding: 1.5rem 1rem; text-align: center; position: relative; overflow: hidden;
    }
    .metric-card::after {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, #7c83fd, #a78bfa, #60a5fa);
    }
    .metric-value {
        font-size: 1.8rem; font-weight: 800;
        background: linear-gradient(135deg, #7c83fd, #a78bfa);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .metric-label { font-size: 0.72rem; color: rgba(226,232,240,0.45); margin-top: 6px; text-transform: uppercase; letter-spacing: 0.06em; }
    .metric-icon { font-size: 1.4rem; margin-bottom: 4px; }
    .section-title {
        font-size: 1.05rem; font-weight: 700; color: #e2e8f0;
        margin: 1.5rem 0 1rem; display: flex; align-items: center; gap: 10px;
    }
    .section-title::after { content: ''; flex: 1; height: 1px; background: linear-gradient(90deg, rgba(124,131,253,0.3), transparent); }
    .card {
        background: linear-gradient(135deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
        border: 1px solid rgba(124,131,253,0.15); border-radius: 16px;
        padding: 1.25rem; margin-bottom: 1rem;
    }
    .pred-result {
        background: linear-gradient(135deg, rgba(124,131,253,0.1), rgba(168,85,247,0.07));
        border: 1px solid rgba(124,131,253,0.3); border-radius: 20px;
        padding: 2.5rem; text-align: center; margin-top: 1.5rem; position: relative;
    }
    .pred-result::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #7c83fd, #a78bfa, #f472b6);
    }
    .pred-label { font-size: 0.8rem; color: rgba(226,232,240,0.45); text-transform: uppercase; letter-spacing: 0.1em; }
    .pred-price { font-size: 3rem; font-weight: 800; margin: 0.5rem 0;
        background: linear-gradient(135deg, #7c83fd, #a78bfa);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .pred-juta { font-size: 1.2rem; color: #fbbf24; font-weight: 600; }
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.03); border-radius: 12px; padding: 4px; gap: 4px;
        border: 1px solid rgba(124,131,253,0.1);
    }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; font-weight: 600; font-size: 0.82rem; }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(124,131,253,0.2), rgba(168,85,247,0.12)) !important;
        color: #a78bfa !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #7c83fd, #a78bfa) !important;
        border: none !important; border-radius: 12px !important;
        font-weight: 700 !important; color: white !important;
        box-shadow: 0 4px 20px rgba(124,131,253,0.3) !important;
    }
    hr { border-color: rgba(124,131,253,0.1) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("Housing.csv")

@st.cache_data
def preprocess(df):
    df2 = df.copy()
    for col in ['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea']:
        df2[col] = df2[col].map({'yes':1,'no':0})
    df2['furnishingstatus'] = df2['furnishingstatus'].map({'furnished':2,'semi-furnished':1,'unfurnished':0})
    return df2

df = load_data()
df_processed = preprocess(df)
X = df_processed.drop('price', axis=1)
y = df_processed['price']

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 0.5rem;">
        <div style="font-size:3rem;">🏠</div>
        <div style="font-size:1.2rem;font-weight:800;background:linear-gradient(135deg,#a78bfa,#7c83fd);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;">Housing Price</div>
        <div style="font-size:0.75rem;color:rgba(226,232,240,0.4);margin-top:4px;">ML Dashboard</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown("""
    <div style="padding:0.5rem 0;">
        <div style="color:rgba(226,232,240,0.5);font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.75rem;">Dataset Info</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"🏘️ &nbsp; **{len(df)}** rumah tercatat")
    st.markdown(f"📋 &nbsp; **{len(df.columns)}** fitur tersedia")
    st.markdown(f"✅ &nbsp; **0** missing values")
    st.markdown(f"💰 &nbsp; Rp **{df['price'].min()/1e6:.1f}M** – **{df['price'].max()/1e6:.1f}M**")
    st.divider()
    st.markdown("""
    <div style="color:rgba(226,232,240,0.35);font-size:0.72rem;text-align:center;line-height:1.8;">
        Python · Pandas · Scikit-learn<br>Plotly · SHAP · Streamlit
    </div>
    """, unsafe_allow_html=True)

st.title("🏠 Housing Price Prediction Dashboard")
st.caption("EDA · Model Comparison · SHAP · Simulation · Leaderboard · AI Chat · Upload CSV")
st.divider()

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "📊 EDA", "🤖 Model & Evaluasi", "📈 Perbandingan Model",
    "🔮 Prediksi Harga", "🎯 SHAP Explainer", "🗺️ Simulasi What-If", "🏆 Leaderboard",
    "📁 Upload CSV", "📋 Auto EDA Report", "🤖 Chat dengan Data"
])

# ══════════════════════════════════════════════
# TAB 1 — EDA
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">📌 Overview</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><div class="metric-icon">🏘️</div><div class="metric-value">{len(df)}</div><div class="metric-label">Total Rumah</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-icon">💰</div><div class="metric-value">Rp {df["price"].mean()/1e6:.1f}M</div><div class="metric-label">Rata-rata Harga</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-icon">📐</div><div class="metric-value">{int(df["area"].mean()):,}</div><div class="metric-label">Rata-rata Luas</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><div class="metric-icon">🏆</div><div class="metric-value">Rp {df["price"].max()/1e6:.1f}M</div><div class="metric-label">Harga Tertinggi</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">1️⃣ Distribusi Harga</div>', unsafe_allow_html=True)
    col1,col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x='price', nbins=40, color_discrete_sequence=['#7c83fd'], title="Histogram Harga")
        fig.update_layout(template='plotly_dark', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.box(df, y='price', color_discrete_sequence=['#f6c90e'], title="Boxplot Harga")
        fig.update_layout(template='plotly_dark', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">2️⃣ Area vs Harga</div>', unsafe_allow_html=True)
    fig = px.scatter(df, x='area', y='price', color='furnishingstatus', title="Area vs Price",
                     color_discrete_sequence=['#7c83fd','#f6c90e','#ff6b6b'])
    fig.update_layout(template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">3️⃣ Pengaruh Fitur Kategorik</div>', unsafe_allow_html=True)
    sel = st.selectbox("Pilih fitur:", ['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea','furnishingstatus'])
    fig = px.box(df, x=sel, y='price', color=sel, color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_layout(template='plotly_dark', showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">4️⃣ Correlation Heatmap</div>', unsafe_allow_html=True)
    fig = px.imshow(df_processed.corr(), text_auto='.2f', color_continuous_scale='RdBu_r')
    fig.update_layout(template='plotly_dark', height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">📋 Raw Data</div>', unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True, height=300)

# ══════════════════════════════════════════════
# TAB 2 — MODEL & EVALUASI
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">🤖 Training Model</div>', unsafe_allow_html=True)
    col1,col2 = st.columns([1,2])
    with col1:
        model_choice = st.selectbox("Model:", ["Linear Regression","Random Forest","Gradient Boosting"])
        test_size = st.slider("Test Size:", 0.1, 0.4, 0.2, 0.05)

    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    if model_choice == "Linear Regression": m = LinearRegression()
    elif model_choice == "Random Forest": m = RandomForestRegressor(n_estimators=100, random_state=42)
    else: m = GradientBoostingRegressor(random_state=42)
    m.fit(X_train, y_train); y_pred = m.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test-y_pred)/y_test))*100

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        m1,m2,m3 = st.columns(3)
        m1.metric("R² Score", f"{r2:.3f}")
        m2.metric("MAE", f"Rp {mae/1e6:.2f}M")
        m3.metric("MAPE", f"{mape:.1f}%")

    st.divider()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test.values, y=y_pred, mode='markers',
                             marker=dict(color='#7c83fd', opacity=0.6, size=7), name='Prediksi'))
    fig.add_trace(go.Scatter(x=[y_test.min(),y_test.max()], y=[y_test.min(),y_test.max()],
                             mode='lines', line=dict(color='#ff6b6b',dash='dash'), name='Ideal'))
    fig.update_layout(template='plotly_dark', title="Actual vs Predicted")
    st.plotly_chart(fig, use_container_width=True)

    if model_choice != "Linear Regression":
        imp_df = pd.DataFrame({'Feature':X.columns,'Importance':m.feature_importances_}).sort_values('Importance', ascending=True)
        fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                     color='Importance', color_continuous_scale='Blues', title="Feature Importance")
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 3 — PERBANDINGAN MODEL
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">📈 Perbandingan Semua Model</div>', unsafe_allow_html=True)

    @st.cache_data
    def compare_models():
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
            "SVR": SVR(kernel='rbf', C=1e6),
        }
        X_tr,X_te,y_tr,y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr); X_te_s = scaler.transform(X_te)
        results = []
        for name, mdl in models.items():
            if name == "SVR": mdl.fit(X_tr_s,y_tr); preds = mdl.predict(X_te_s)
            else: mdl.fit(X_tr,y_tr); preds = mdl.predict(X_te)
            results.append({"Model":name, "R²":round(r2_score(y_te,preds),4),
                            "MAE (Juta)":round(mean_absolute_error(y_te,preds)/1e6,3),
                            "MAPE (%)":round(np.mean(np.abs((y_te-preds)/y_te))*100,2)})
        return pd.DataFrame(results).sort_values("R²", ascending=False)

    with st.spinner("⚙️ Training semua model..."):
        result_df = compare_models()

    st.dataframe(result_df, use_container_width=True)
    st.success(f"🏆 Model Terbaik: **{result_df.iloc[0]['Model']}** (R² = {result_df.iloc[0]['R²']})")

    c1,c2,c3 = st.columns(3)
    with c1:
        fig = px.bar(result_df, x='Model', y='R²', color='R²', color_continuous_scale='Greens', title="R² (↑ lebih baik)")
        fig.update_layout(template='plotly_dark', showlegend=False); st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(result_df, x='Model', y='MAE (Juta)', color='MAE (Juta)', color_continuous_scale='Reds_r', title="MAE (↓ lebih baik)")
        fig.update_layout(template='plotly_dark', showlegend=False); st.plotly_chart(fig, use_container_width=True)
    with c3:
        fig = px.bar(result_df, x='Model', y='MAPE (%)', color='MAPE (%)', color_continuous_scale='Oranges_r', title="MAPE (↓ lebih baik)")
        fig.update_layout(template='plotly_dark', showlegend=False); st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">🕸️ Radar Chart</div>', unsafe_allow_html=True)
    fig_radar = go.Figure()
    cats = ['R²','MAE (inv)','MAPE (inv)']
    for _,row in result_df.iterrows():
        vals = [row['R²'], 1-(row['MAE (Juta)']/result_df['MAE (Juta)'].max()),
                1-(row['MAPE (%)']/result_df['MAPE (%)'].max())]
        fig_radar.add_trace(go.Scatterpolar(r=vals+[vals[0]], theta=cats+[cats[0]],
                                             fill='toself', name=row['Model'], opacity=0.7))
    fig_radar.update_layout(template='plotly_dark', polar=dict(radialaxis=dict(range=[0,1])),
                             title="Radar Perbandingan Model")
    st.plotly_chart(fig_radar, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 4 — PREDIKSI
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">🔮 Prediksi Harga Rumah</div>', unsafe_allow_html=True)

    @st.cache_resource
    def get_pred_model():
        mdl = RandomForestRegressor(n_estimators=100, random_state=42)
        mdl.fit(X, y); return mdl
    pred_model = get_pred_model()

    col1,col2 = st.columns(2)
    with col1:
        area = st.number_input("📐 Luas (sqft)", 500, 20000, 5000, 100, key="p_area")
        bedrooms = st.slider("🛏️ Kamar Tidur", 1, 6, 3, key="p_bed")
        bathrooms = st.slider("🚿 Kamar Mandi", 1, 4, 2, key="p_bath")
        stories = st.slider("🏢 Lantai", 1, 4, 2, key="p_stor")
        parking = st.slider("🚗 Parkir", 0, 3, 1, key="p_park")
        furnishing = st.selectbox("🛋️ Furnitur", ["furnished","semi-furnished","unfurnished"], key="p_furn")
    with col2:
        mainroad = st.radio("🛣️ Jalan Utama?", ["yes","no"], horizontal=True, key="p_road")
        guestroom = st.radio("🛎️ Kamar Tamu?", ["yes","no"], horizontal=True, key="p_guest")
        basement = st.radio("🏚️ Basement?", ["yes","no"], horizontal=True, key="p_base")
        hotwater = st.radio("🔥 Pemanas Air?", ["yes","no"], horizontal=True, key="p_hot")
        aircon = st.radio("❄️ AC?", ["yes","no"], horizontal=True, key="p_ac")
        prefarea = st.radio("⭐ Area Premium?", ["yes","no"], horizontal=True, key="p_pref")

    st.divider()
    if st.button("🔮 Prediksi!", use_container_width=True, type="primary"):
        inp = pd.DataFrame([{'area':area,'bedrooms':bedrooms,'bathrooms':bathrooms,'stories':stories,
                             'mainroad':1 if mainroad=='yes' else 0,'guestroom':1 if guestroom=='yes' else 0,
                             'basement':1 if basement=='yes' else 0,'hotwaterheating':1 if hotwater=='yes' else 0,
                             'airconditioning':1 if aircon=='yes' else 0,'parking':parking,
                             'prefarea':1 if prefarea=='yes' else 0,
                             'furnishingstatus':{'furnished':2,'semi-furnished':1,'unfurnished':0}[furnishing]}])
        price = pred_model.predict(inp)[0]
        avg = df['price'].mean(); diff = price-avg; pct = (diff/avg)*100
        st.markdown(f"""<div class="pred-result">
            <div class="pred-label">✨ Estimasi Harga Rumah</div>
            <div class="pred-price">Rp {price:,.0f}</div>
            <div class="pred-juta">≈ Rp {price/1e6:.2f} Juta</div>
        </div>""", unsafe_allow_html=True)
        ca,cb = st.columns(2)
        ca.metric("Rata-rata pasar", f"Rp {avg/1e6:.2f}M")
        cb.metric("Selisih", f"Rp {abs(diff)/1e6:.2f}M", delta=f"{pct:+.1f}%")
        st.session_state['last_pred'] = {'price': price, 'avg': avg, 'area': area,
                                          'bedrooms': bedrooms, 'ac': aircon, 'furnishing': furnishing}

# ══════════════════════════════════════════════
# TAB 5 — SHAP
# ══════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-title">🎯 SHAP Explainer</div>', unsafe_allow_html=True)

    @st.cache_resource
    def get_shap():
        mdl = RandomForestRegressor(n_estimators=100, random_state=42)
        mdl.fit(X, y)
        exp = shap.TreeExplainer(mdl)
        sv = exp.shap_values(X)
        return mdl, exp, sv
    with st.spinner("Menghitung SHAP values..."):
        shap_model, exp, sv = get_shap()

    st.markdown('<div class="section-title">1️⃣ Global Feature Importance</div>', unsafe_allow_html=True)
    ms = np.abs(sv).mean(axis=0)
    sdf = pd.DataFrame({'Feature':X.columns,'Mean |SHAP|':ms}).sort_values('Mean |SHAP|', ascending=True)
    fig = px.bar(sdf, x='Mean |SHAP|', y='Feature', orientation='h', color='Mean |SHAP|', color_continuous_scale='Viridis')
    fig.update_layout(template='plotly_dark', showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.info(f"💡 Fitur **`{sdf.iloc[-1]['Feature']}`** paling berpengaruh!")

    st.markdown('<div class="section-title">2️⃣ Penjelasan Per Rumah</div>', unsafe_allow_html=True)
    sidx = st.slider("Pilih rumah:", 0, len(df)-1, 0)
    sshap = sv[sidx]
    spred = shap_model.predict(X.iloc[[sidx]])[0]

    st.markdown(f"""<div class="card">
        <b>🏠 Rumah #{sidx}</b><br>
        Area: {df.iloc[sidx]['area']:,} sqft | Kamar: {df.iloc[sidx]['bedrooms']} | AC: {df.iloc[sidx]['airconditioning']}<br>
        💰 Aktual: <b style="color:#f6c90e">Rp {df.iloc[sidx]['price']:,}</b> |
        🤖 Prediksi: <b style="color:#7c83fd">Rp {spred:,.0f}</b></div>""", unsafe_allow_html=True)

    contribs = sorted(zip(X.columns, sshap), key=lambda x: abs(x[1]))
    fts = [c[0] for c in contribs]; vls = [c[1] for c in contribs]
    fig = go.Figure(go.Bar(x=vls, y=fts, orientation='h',
                           marker_color=['#2ecc71' if v>0 else '#e74c3c' for v in vls],
                           text=[f"+Rp{v/1e6:.2f}M" if v>0 else f"Rp{v/1e6:.2f}M" for v in vls],
                           textposition='outside'))
    fig.update_layout(title=f"Kontribusi Fitur — Rumah #{sidx}", template='plotly_dark', height=420)
    st.plotly_chart(fig, use_container_width=True)

    tp = max(zip(X.columns,sshap), key=lambda x:x[1])
    tn = min(zip(X.columns,sshap), key=lambda x:x[1])
    st.success(f"✅ Pendorong naik: **`{tp[0]}`** (+Rp {tp[1]/1e6:.2f}M)")
    if tn[1] < 0: st.error(f"⬇️ Penekan harga: **`{tn[0]}`** (Rp {tn[1]/1e6:.2f}M)")

    st.markdown('<div class="section-title">3️⃣ Dependence Plot</div>', unsafe_allow_html=True)
    sf = st.selectbox("Fitur:", X.columns.tolist())
    fi = list(X.columns).index(sf)
    fig = px.scatter(x=X[sf], y=sv[:,fi], color=sv[:,fi], color_continuous_scale='RdBu',
                     labels={'x':sf,'y':'SHAP Value'}, title=f"SHAP Dependence: {sf}")
    fig.add_hline(y=0, line_dash='dash', line_color='white', opacity=0.4)
    fig.update_layout(template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 6 — SIMULASI WHAT-IF
# ══════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-title">🗺️ Simulasi What-If</div>', unsafe_allow_html=True)
    st.caption("Ubah fitur rumah dan lihat langsung dampaknya terhadap harga prediksi!")

    @st.cache_resource
    def get_sim_model():
        mdl = RandomForestRegressor(n_estimators=100, random_state=42)
        mdl.fit(X, y); return mdl
    sim_model = get_sim_model()

    col1,col2 = st.columns([1,2])
    with col1:
        bidx = st.number_input("Index rumah base:", 0, len(df)-1, 0)
        base_row = df_processed.iloc[bidx].copy()
        base_price = sim_model.predict(X.iloc[[bidx]])[0]
        st.markdown(f"""<div class="card"><b>Rumah #{bidx} (Original)</b><br>
            Area: {df.iloc[bidx]['area']:,} sqft<br>
            Aktual: <b style="color:#f6c90e">Rp {df.iloc[bidx]['price']:,}</b><br>
            Prediksi Base: <b style="color:#7c83fd">Rp {base_price:,.0f}</b></div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("**✏️ Ubah fitur:**")
        sim_area = st.slider("📐 Luas", 1650, 16200, int(df.iloc[bidx]['area']), 100, key="s_area")
        sim_bed = st.slider("🛏️ Kamar Tidur", 1, 6, int(df.iloc[bidx]['bedrooms']), key="s_bed")
        sim_bath = st.slider("🚿 Kamar Mandi", 1, 4, int(df.iloc[bidx]['bathrooms']), key="s_bath")
        sim_stor = st.slider("🏢 Lantai", 1, 4, int(df.iloc[bidx]['stories']), key="s_stor")
        sim_park = st.slider("🚗 Parkir", 0, 3, int(df.iloc[bidx]['parking']), key="s_park")
        sim_ac = st.radio("❄️ AC?", ["yes","no"], index=0 if df.iloc[bidx]['airconditioning']=='yes' else 1, horizontal=True, key="s_ac")
        sim_pref = st.radio("⭐ Premium?", ["yes","no"], index=0 if df.iloc[bidx]['prefarea']=='yes' else 1, horizontal=True, key="s_pref")
        sim_furn = st.selectbox("🛋️ Furnitur", ["furnished","semi-furnished","unfurnished"],
                                 index=['furnished','semi-furnished','unfurnished'].index(df.iloc[bidx]['furnishingstatus']), key="s_furn")

    sim_inp = base_row.copy()
    sim_inp['area']=sim_area; sim_inp['bedrooms']=sim_bed; sim_inp['bathrooms']=sim_bath
    sim_inp['stories']=sim_stor; sim_inp['parking']=sim_park
    sim_inp['airconditioning']=1 if sim_ac=='yes' else 0
    sim_inp['prefarea']=1 if sim_pref=='yes' else 0
    sim_inp['furnishingstatus']={'furnished':2,'semi-furnished':1,'unfurnished':0}[sim_furn]
    sim_price = sim_model.predict(pd.DataFrame([sim_inp.drop('price')]))[0]
    delta = sim_price - base_price; dpct = (delta/base_price)*100

    st.divider()
    r1,r2,r3 = st.columns(3)
    r1.metric("🏠 Base", f"Rp {base_price/1e6:.2f}M")
    r2.metric("🔮 Simulasi", f"Rp {sim_price/1e6:.2f}M")
    r3.metric("📊 Perubahan", f"Rp {abs(delta)/1e6:.2f}M", delta=f"{dpct:+.1f}%")

    fig = go.Figure(go.Bar(x=["Base","Simulasi"], y=[base_price,sim_price],
                           marker_color=['#7c83fd','#2ecc71' if delta>=0 else '#e74c3c'],
                           text=[f"Rp {base_price/1e6:.2f}M", f"Rp {sim_price/1e6:.2f}M"],
                           textposition='outside'))
    fig.update_layout(template='plotly_dark', title="Base vs Simulasi", height=350)
    st.plotly_chart(fig, use_container_width=True)

    if delta>0: st.success(f"📈 Harga naik **Rp {delta/1e6:.2f}M** ({dpct:+.1f}%)")
    elif delta<0: st.error(f"📉 Harga turun **Rp {abs(delta)/1e6:.2f}M** ({dpct:+.1f}%)")
    else: st.info("Tidak ada perubahan signifikan.")

# ══════════════════════════════════════════════
# TAB 7 — LEADERBOARD
# ══════════════════════════════════════════════
with tab7:
    st.markdown('<div class="section-title">🏆 Leaderboard Harga Rumah</div>', unsafe_allow_html=True)

    col1,col2 = st.columns([1,3])
    with col1:
        top_n = st.slider("Tampilkan Top:", 5, 50, 10)
        sort_order = st.radio("Urutan:", ["Termahal","Termurah"])
        filter_ac = st.radio("Filter AC:", ["Semua","Hanya AC","Tanpa AC"])

    filtered = df.copy()
    if filter_ac=="Hanya AC": filtered = filtered[filtered['airconditioning']=='yes']
    elif filter_ac=="Tanpa AC": filtered = filtered[filtered['airconditioning']=='no']
    top_df = filtered.sort_values('price', ascending=(sort_order=="Termurah")).head(top_n).reset_index(drop=True)
    top_df.index += 1

    fig = px.bar(top_df.reset_index(), x='index', y='price', color='price',
                 color_continuous_scale='Plasma' if sort_order=="Termahal" else 'Blues',
                 hover_data=['area','bedrooms','furnishingstatus'],
                 labels={'index':'Ranking','price':'Harga (Rp)'},
                 title=f"Top {top_n} Rumah {sort_order}")
    fig.update_layout(template='plotly_dark', showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    disp = top_df[['price','area','bedrooms','bathrooms','stories','airconditioning','furnishingstatus']].copy()
    disp['price'] = disp['price'].apply(lambda x: f"Rp {x:,.0f}")
    disp['area'] = disp['area'].apply(lambda x: f"{x:,} sqft")
    disp.columns = ['💰 Harga','📐 Luas','🛏️ Kamar','🚿 Mandi','🏢 Lantai','❄️ AC','🛋️ Furnitur']
    st.dataframe(disp, use_container_width=True)

    st.divider()
    s1,s2,s3,s4 = st.columns(4)
    s1.metric("Tertinggi", f"Rp {top_df['price'].max()/1e6:.1f}M")
    s2.metric("Terendah", f"Rp {top_df['price'].min()/1e6:.1f}M")
    s3.metric("Rata-rata", f"Rp {top_df['price'].mean()/1e6:.1f}M")
    s4.metric("% Ber-AC", f"{(top_df['airconditioning']=='yes').sum()/len(top_df)*100:.0f}%")

    fc = top_df['furnishingstatus'].value_counts()
    fig2 = px.pie(values=fc.values, names=fc.index, title=f"Furnishing — Top {top_n}",
                  color_discrete_sequence=['#7c83fd','#f6c90e','#ff6b6b'])
    fig2.update_layout(template='plotly_dark')
    st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 8 — UPLOAD CSV
# ══════════════════════════════════════════════
with tab8:
    st.markdown('<div class="section-title">📁 Upload Dataset Kamu Sendiri</div>', unsafe_allow_html=True)
    st.caption("Upload file CSV apapun — app akan otomatis analisis dan buat model prediksi!")

    uploaded = st.file_uploader("Upload file CSV", type=['csv'])

    if uploaded:
        try:
            udf = pd.read_csv(uploaded)
            st.success(f"✅ File berhasil diupload: **{uploaded.name}** ({len(udf)} baris, {len(udf.columns)} kolom)")

            st.markdown('<div class="section-title">👀 Preview Data</div>', unsafe_allow_html=True)
            st.dataframe(udf.head(10), use_container_width=True)

            st.markdown('<div class="section-title">📊 Info Dataset</div>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Baris", len(udf))
            c2.metric("Total Kolom", len(udf.columns))
            c3.metric("Missing Values", udf.isnull().sum().sum())
            c4.metric("Duplikat", udf.duplicated().sum())

            st.markdown('<div class="section-title">🔢 Tipe Data</div>', unsafe_allow_html=True)
            dtype_df = pd.DataFrame({'Kolom': udf.dtypes.index, 'Tipe': udf.dtypes.values.astype(str),
                                     'Missing': udf.isnull().sum().values,
                                     'Unique': [udf[c].nunique() for c in udf.columns]})
            st.dataframe(dtype_df, use_container_width=True)

            num_cols = udf.select_dtypes(include=np.number).columns.tolist()
            if len(num_cols) >= 2:
                st.markdown('<div class="section-title">📈 Distribusi Kolom Numerik</div>', unsafe_allow_html=True)
                sel_col = st.selectbox("Pilih kolom:", num_cols, key="up_col")
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.histogram(udf, x=sel_col, color_discrete_sequence=['#7c83fd'], title=f"Histogram {sel_col}")
                    fig.update_layout(template='plotly_dark'); st.plotly_chart(fig, use_container_width=True)
                with col2:
                    fig = px.box(udf, y=sel_col, color_discrete_sequence=['#f6c90e'], title=f"Boxplot {sel_col}")
                    fig.update_layout(template='plotly_dark'); st.plotly_chart(fig, use_container_width=True)

                st.markdown('<div class="section-title">🌡️ Correlation Heatmap</div>', unsafe_allow_html=True)
                fig = px.imshow(udf[num_cols].corr(), text_auto='.2f', color_continuous_scale='RdBu_r')
                fig.update_layout(template='plotly_dark', height=450)
                st.plotly_chart(fig, use_container_width=True)

                st.markdown('<div class="section-title">🤖 Quick ML Prediction</div>', unsafe_allow_html=True)
                target_col = st.selectbox("Pilih kolom TARGET (yang mau diprediksi):", num_cols, key="up_target")
                feature_cols = [c for c in num_cols if c != target_col]

                if len(feature_cols) > 0 and st.button("🚀 Train Model Otomatis!", key="up_train", type="primary"):
                    Xu = udf[feature_cols].dropna()
                    yu = udf[target_col].loc[Xu.index]
                    Xu_tr, Xu_te, yu_tr, yu_te = train_test_split(Xu, yu, test_size=0.2, random_state=42)
                    um = RandomForestRegressor(n_estimators=100, random_state=42)
                    um.fit(Xu_tr, yu_tr); yu_pred = um.predict(Xu_te)
                    ur2 = r2_score(yu_te, yu_pred)
                    umae = mean_absolute_error(yu_te, yu_pred)

                    a1, a2 = st.columns(2)
                    a1.metric("R² Score", f"{ur2:.3f}")
                    a2.metric("MAE", f"{umae:.2f}")

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=yu_te.values, y=yu_pred, mode='markers',
                                             marker=dict(color='#7c83fd', opacity=0.6)))
                    fig.add_trace(go.Scatter(x=[yu_te.min(), yu_te.max()], y=[yu_te.min(), yu_te.max()],
                                             mode='lines', line=dict(color='#ff6b6b', dash='dash')))
                    fig.update_layout(template='plotly_dark', title="Actual vs Predicted")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ Dataset perlu minimal 2 kolom numerik untuk analisis lebih lanjut.")
        except Exception as e:
            st.error(f"❌ Error membaca file: {e}")
    else:
        st.info("👆 Upload file CSV kamu di atas untuk mulai analisis otomatis!")
        st.markdown("""
        <div class="card">
        <b>📌 Format yang didukung:</b><br>
        • CSV dengan header kolom di baris pertama<br>
        • Kolom numerik untuk analisis & prediksi<br>
        • Boleh ada kolom kategorik (akan dideteksi otomatis)<br><br>
        <b>💡 Contoh dataset yang bisa dicoba:</b><br>
        • Kaggle: Titanic, Iris, Car Price, Student Score, dll
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 9 — AUTO EDA REPORT
# ══════════════════════════════════════════════
with tab9:
    st.markdown('<div class="section-title">📋 Auto EDA Report — 1 Klik!</div>', unsafe_allow_html=True)
    st.caption("Generate laporan analisis data lengkap dalam format yang bisa didownload.")

    if st.button("📋 Generate Report Sekarang!", type="primary", use_container_width=True, key="gen_report"):
        with st.spinner("⚙️ Membuat laporan..."):

            # Hitung semua statistik
            num_cols = df_processed.select_dtypes(include=np.number).columns.tolist()
            corr = df_processed.corr()
            top_corr = corr['price'].drop('price').sort_values(ascending=False)

            # Train model untuk summary
            Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(X, y, test_size=0.2, random_state=42)
            rm = RandomForestRegressor(n_estimators=100, random_state=42)
            rm.fit(Xr_tr, yr_tr); yr_pred = rm.predict(Xr_te)
            rr2 = r2_score(yr_te, yr_pred)
            rmae = mean_absolute_error(yr_te, yr_pred)
            rmape = np.mean(np.abs((yr_te - yr_pred) / yr_te)) * 100

            imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': rm.feature_importances_}).sort_values('Importance', ascending=False)

            report_html = f"""
<!DOCTYPE html><html lang="id"><head><meta charset="UTF-8">
<title>EDA Report — Housing Price Dataset</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; background:#0e1117; color:#e0e0e0; margin:0; padding:32px; }}
  h1 {{ color:#7c83fd; border-bottom:3px solid #7c83fd; padding-bottom:12px; }}
  h2 {{ color:#f6c90e; margin-top:40px; border-left:4px solid #f6c90e; padding-left:12px; }}
  h3 {{ color:#aaa; }}
  .card {{ background:#1e2130; border-radius:12px; padding:20px; margin:16px 0; border:1px solid #3d4570; }}
  .grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin:16px 0; }}
  .metric {{ background:#2d3250; border-radius:8px; padding:16px; text-align:center; }}
  .metric-val {{ font-size:1.8rem; font-weight:bold; color:#7c83fd; }}
  .metric-lbl {{ font-size:0.8rem; color:#aaa; }}
  table {{ width:100%; border-collapse:collapse; }}
  th {{ background:#2d3250; padding:10px; text-align:left; color:#7c83fd; }}
  td {{ padding:8px 10px; border-bottom:1px solid #2d3250; }}
  tr:hover {{ background:#1e2130; }}
  .pos {{ color:#2ecc71; }} .neg {{ color:#e74c3c; }} .neu {{ color:#f6c90e; }}
  .badge {{ display:inline-block; padding:4px 10px; border-radius:20px; font-size:0.8rem; font-weight:bold; }}
  .badge-green {{ background:#1a4731; color:#2ecc71; }}
  .badge-red {{ background:#4a1525; color:#e74c3c; }}
  footer {{ margin-top:60px; text-align:center; color:#555; font-size:0.85rem; }}
</style></head><body>
<h1>🏠 EDA Report — Housing Price Dataset</h1>
<p style="color:#aaa">Generated otomatis oleh Housing Price Prediction Dashboard</p>

<h2>📌 1. Overview Dataset</h2>
<div class="grid">
  <div class="metric"><div class="metric-val">{len(df)}</div><div class="metric-lbl">Total Rumah</div></div>
  <div class="metric"><div class="metric-val">{len(df.columns)}</div><div class="metric-lbl">Total Fitur</div></div>
  <div class="metric"><div class="metric-val">0</div><div class="metric-lbl">Missing Values</div></div>
  <div class="metric"><div class="metric-val">Rp {df['price'].mean()/1e6:.1f}M</div><div class="metric-lbl">Rata-rata Harga</div></div>
</div>

<h2>📊 2. Statistik Deskriptif</h2>
<div class="card">
<table>
<tr><th>Statistik</th><th>Price</th><th>Area</th><th>Bedrooms</th><th>Bathrooms</th></tr>
<tr><td>Min</td><td>Rp {df['price'].min():,}</td><td>{df['area'].min():,}</td><td>{df['bedrooms'].min()}</td><td>{df['bathrooms'].min()}</td></tr>
<tr><td>Max</td><td>Rp {df['price'].max():,}</td><td>{df['area'].max():,}</td><td>{df['bedrooms'].max()}</td><td>{df['bathrooms'].max()}</td></tr>
<tr><td>Mean</td><td>Rp {df['price'].mean():,.0f}</td><td>{df['area'].mean():,.0f}</td><td>{df['bedrooms'].mean():.1f}</td><td>{df['bathrooms'].mean():.1f}</td></tr>
<tr><td>Median</td><td>Rp {df['price'].median():,.0f}</td><td>{df['area'].median():,.0f}</td><td>{df['bedrooms'].median():.1f}</td><td>{df['bathrooms'].median():.1f}</td></tr>
<tr><td>Std Dev</td><td>Rp {df['price'].std():,.0f}</td><td>{df['area'].std():,.0f}</td><td>{df['bedrooms'].std():.2f}</td><td>{df['bathrooms'].std():.2f}</td></tr>
</table></div>

<h2>🌡️ 3. Korelasi dengan Harga</h2>
<div class="card"><table>
<tr><th>Fitur</th><th>Korelasi</th><th>Kekuatan</th></tr>
{"".join([f'<tr><td>{feat}</td><td class="{"pos" if val>0 else "neg"}">{val:+.3f}</td><td><span class="badge {"badge-green" if abs(val)>0.3 else "badge-red"}">{"Kuat" if abs(val)>0.3 else "Lemah"}</span></td></tr>' for feat, val in top_corr.items()])}
</table></div>

<h2>🤖 4. Performa Model (Random Forest)</h2>
<div class="grid">
  <div class="metric"><div class="metric-val">{rr2:.3f}</div><div class="metric-lbl">R² Score</div></div>
  <div class="metric"><div class="metric-val">Rp {rmae/1e6:.2f}M</div><div class="metric-lbl">MAE</div></div>
  <div class="metric"><div class="metric-val">{rmape:.1f}%</div><div class="metric-lbl">MAPE</div></div>
  <div class="metric"><div class="metric-val">80/20</div><div class="metric-lbl">Train/Test Split</div></div>
</div>

<h2>🏆 5. Feature Importance</h2>
<div class="card"><table>
<tr><th>Ranking</th><th>Fitur</th><th>Importance Score</th></tr>
{"".join([f'<tr><td>#{i+1}</td><td>{row["Feature"]}</td><td>{row["Importance"]:.4f}</td></tr>' for i, (_, row) in enumerate(imp_df.iterrows())])}
</table></div>

<h2>💡 6. Key Insights</h2>
<div class="card">
<ul>
  <li>Fitur <b style="color:#7c83fd">{imp_df.iloc[0]['Feature']}</b> adalah yang paling berpengaruh terhadap harga rumah</li>
  <li>Harga rumah berkisar antara <b>Rp {df['price'].min():,}</b> sampai <b>Rp {df['price'].max():,}</b></li>
  <li>Rata-rata luas rumah adalah <b>{df['area'].mean():,.0f} sqft</b></li>
  <li>Model Random Forest mencapai akurasi <b>R² = {rr2:.3f}</b> dengan error rata-rata <b>{rmape:.1f}%</b></li>
  <li>Rumah dengan AC cenderung lebih mahal: <b>Rp {df[df['airconditioning']=='yes']['price'].mean()/1e6:.1f}M</b> vs <b>Rp {df[df['airconditioning']=='no']['price'].mean()/1e6:.1f}M</b></li>
</ul>
</div>

<footer>📊 Auto-generated by Housing Price Dashboard | Data Science Portfolio Project</footer>
</body></html>"""

            st.success("✅ Report berhasil dibuat!")
            st.download_button(
                label="⬇️ Download Report (HTML)",
                data=report_html,
                file_name="housing_eda_report.html",
                mime="text/html",
                use_container_width=True
            )
            st.markdown("**Preview Report:**")
            st.components.v1.html(report_html, height=600, scrolling=True)
    else:
        st.markdown("""
        <div class="card">
        <b>📋 Report ini mencakup:</b><br><br>
        ✅ Overview & statistik dataset<br>
        ✅ Statistik deskriptif semua kolom<br>
        ✅ Analisis korelasi dengan harga<br>
        ✅ Performa model Machine Learning<br>
        ✅ Feature importance ranking<br>
        ✅ Key insights otomatis<br><br>
        <i>Klik tombol di atas untuk generate!</i>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 10 — CHAT DENGAN DATA (AI)
# ══════════════════════════════════════════════
with tab10:
    st.markdown('<div class="section-title">🤖 Chat dengan Data — Tanya Apapun!</div>', unsafe_allow_html=True)
    st.caption("Powered by Claude AI — tanya soal dataset, minta insight, atau minta penjelasan hasil analisis.")

    api_key = st.text_input("🔑 Masukkan Anthropic API Key:", type="password",
                             placeholder="sk-ant-...", key="api_key_input")
    st.caption("API key tidak disimpan. Dapatkan gratis di: console.anthropic.com")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Siapkan konteks dataset
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    corr_price = df_processed.corr()['price'].drop('price').sort_values(ascending=False)
    dataset_context = f"""
Kamu adalah data analyst expert yang membantu menganalisis Housing Price Dataset.

INFORMASI DATASET:
- Total data: {len(df)} rumah
- Kolom: {', '.join(df.columns.tolist())}
- Harga: Min Rp {df['price'].min():,} | Max Rp {df['price'].max():,} | Rata-rata Rp {df['price'].mean():,.0f}
- Luas: Min {df['area'].min()} | Max {df['area'].max()} | Rata-rata {df['area'].mean():.0f} sqft
- Kamar tidur: {df['bedrooms'].min()}-{df['bedrooms'].max()} (rata-rata {df['bedrooms'].mean():.1f})
- Kamar mandi: {df['bathrooms'].min()}-{df['bathrooms'].max()}
- Lantai: {df['stories'].min()}-{df['stories'].max()}
- {(df['airconditioning']=='yes').sum()} rumah ber-AC ({(df['airconditioning']=='yes').mean()*100:.0f}%)
- {(df['prefarea']=='yes').sum()} rumah di area premium

KORELASI FITUR DENGAN HARGA (tertinggi ke terendah):
{corr_price.to_string()}

STATISTIK HARGA PER KATEGORI:
- Rata-rata harga rumah AC: Rp {df[df['airconditioning']=='yes']['price'].mean():,.0f}
- Rata-rata harga rumah non-AC: Rp {df[df['airconditioning']=='no']['price'].mean():,.0f}
- Rata-rata harga furnished: Rp {df[df['furnishingstatus']=='furnished']['price'].mean():,.0f}
- Rata-rata harga unfurnished: Rp {df[df['furnishingstatus']=='unfurnished']['price'].mean():,.0f}

Jawab dalam Bahasa Indonesia, singkat dan informatif. Gunakan angka spesifik dari dataset.
"""

    # Tampilkan riwayat chat
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input chat
    if prompt := st.chat_input("Tanya tentang dataset... (contoh: 'fitur apa yang paling berpengaruh?')", key="chat_input"):
        if not api_key:
            st.warning("⚠️ Masukkan API Key dulu ya!")
        else:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("🤔 Berpikir..."):
                    try:
                        import requests
                        messages = [{"role": m["role"], "content": m["content"]}
                                    for m in st.session_state.chat_history]
                        response = requests.post(
                            "https://api.anthropic.com/v1/messages",
                            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                                     "content-type": "application/json"},
                            json={"model": "claude-sonnet-4-20250514", "max_tokens": 1000,
                                  "system": dataset_context, "messages": messages}
                        )
                        if response.status_code == 200:
                            reply = response.json()['content'][0]['text']
                            st.markdown(reply)
                            st.session_state.chat_history.append({"role": "assistant", "content": reply})
                        else:
                            err = response.json().get('error', {}).get('message', 'Unknown error')
                            st.error(f"❌ API Error: {err}")
                            st.session_state.chat_history.pop()
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
                        st.session_state.chat_history.pop()

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()

    if not api_key:
        st.markdown("""
        <div class="card">
        <b>💡 Contoh pertanyaan yang bisa kamu tanyakan:</b><br><br>
        🔹 "Fitur apa yang paling berpengaruh terhadap harga?"<br>
        🔹 "Berapa rata-rata harga rumah ber-AC vs tidak?"<br>
        🔹 "Apakah luas rumah selalu berbanding lurus dengan harga?"<br>
        🔹 "Rekomendasikan rumah dengan value terbaik"<br>
        🔹 "Jelaskan hasil model Random Forest"<br>
        🔹 "Insight apa yang paling menarik dari dataset ini?"
        </div>""", unsafe_allow_html=True)
