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
    .metric-card { background:linear-gradient(135deg,#1e2130,#2d3250); border-radius:12px;
                   padding:20px; text-align:center; border:1px solid #3d4570; }
    .metric-value { font-size:2rem; font-weight:bold; color:#7c83fd; }
    .metric-label { font-size:0.85rem; color:#aaa; margin-top:4px; }
    .section-title { font-size:1.4rem; font-weight:700; color:#7c83fd; margin-bottom:16px;
                     border-left:4px solid #7c83fd; padding-left:12px; }
    .card { background:linear-gradient(135deg,#1e2130,#2d3250); border-radius:12px;
            padding:16px; border:1px solid #3d4570; margin-bottom:12px; }
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
    st.image("https://img.icons8.com/fluency/96/home.png", width=70)
    st.title("🏠 Housing Price")
    st.caption("Prediksi Harga Rumah — ML Project")
    st.divider()
    st.markdown(f"- 📦 **{len(df)} rumah** tercatat")
    st.markdown(f"- 📋 **{len(df.columns)} fitur** tersedia")
    st.markdown(f"- ✅ **0 missing values**")
    st.divider()
    st.markdown("Python · Plotly · Scikit-learn · SHAP · Streamlit")

st.title("🏠 Housing Price Prediction Dashboard")
st.caption("EDA · Model Comparison · SHAP · Simulation · Leaderboard")
st.divider()

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 EDA", "🤖 Model & Evaluasi", "📈 Perbandingan Model",
    "🔮 Prediksi Harga", "🎯 SHAP Explainer", "🗺️ Simulasi What-If", "🏆 Leaderboard"
])

# ══════════════════════════════════════════════
# TAB 1 — EDA
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">📌 Overview</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><div class="metric-value">{len(df)}</div><div class="metric-label">Total Rumah</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-value">Rp {df["price"].mean()/1e6:.1f}M</div><div class="metric-label">Rata-rata Harga</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-value">{int(df["area"].mean()):,}</div><div class="metric-label">Rata-rata Luas</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><div class="metric-value">Rp {df["price"].max()/1e6:.1f}M</div><div class="metric-label">Harga Tertinggi</div></div>', unsafe_allow_html=True)

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
        area = st.number_input("📐 Luas (sqft)", 500, 20000, 5000, 100)
        bedrooms = st.slider("🛏️ Kamar Tidur", 1, 6, 3)
        bathrooms = st.slider("🚿 Kamar Mandi", 1, 4, 2)
        stories = st.slider("🏢 Lantai", 1, 4, 2)
        parking = st.slider("🚗 Parkir", 0, 3, 1)
        furnishing = st.selectbox("🛋️ Furnitur", ["furnished","semi-furnished","unfurnished"])
    with col2:
        mainroad = st.radio("🛣️ Jalan Utama?", ["yes","no"], horizontal=True)
        guestroom = st.radio("🛎️ Kamar Tamu?", ["yes","no"], horizontal=True)
        basement = st.radio("🏚️ Basement?", ["yes","no"], horizontal=True)
        hotwater = st.radio("🔥 Pemanas Air?", ["yes","no"], horizontal=True)
        aircon = st.radio("❄️ AC?", ["yes","no"], horizontal=True)
        prefarea = st.radio("⭐ Area Premium?", ["yes","no"], horizontal=True)

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
        st.markdown(f"""<div style="background:linear-gradient(135deg,#1e2130,#2d3250);border-radius:16px;
            padding:32px;text-align:center;border:2px solid #7c83fd;margin-top:16px;">
            <div style="font-size:1rem;color:#aaa;">Estimasi Harga</div>
            <div style="font-size:3rem;font-weight:bold;color:#7c83fd;">Rp {price:,.0f}</div>
            <div style="font-size:1.2rem;color:#f6c90e;">≈ Rp {price/1e6:.2f} Juta</div></div>""",
            unsafe_allow_html=True)
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
        sim_area = st.slider("📐 Luas", 1650, 16200, int(df.iloc[bidx]['area']), 100)
        sim_bed = st.slider("🛏️ Kamar Tidur", 1, 6, int(df.iloc[bidx]['bedrooms']))
        sim_bath = st.slider("🚿 Kamar Mandi", 1, 4, int(df.iloc[bidx]['bathrooms']))
        sim_stor = st.slider("🏢 Lantai", 1, 4, int(df.iloc[bidx]['stories']))
        sim_park = st.slider("🚗 Parkir", 0, 3, int(df.iloc[bidx]['parking']))
        sim_ac = st.radio("❄️ AC?", ["yes","no"], index=0 if df.iloc[bidx]['airconditioning']=='yes' else 1, horizontal=True)
        sim_pref = st.radio("⭐ Premium?", ["yes","no"], index=0 if df.iloc[bidx]['prefarea']=='yes' else 1, horizontal=True)
        sim_furn = st.selectbox("🛋️ Furnitur", ["furnished","semi-furnished","unfurnished"],
                                 index=['furnished','semi-furnished','unfurnished'].index(df.iloc[bidx]['furnishingstatus']))

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
