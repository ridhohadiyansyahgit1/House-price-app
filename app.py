import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🏠 Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #2d3250);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #3d4570;
    }
    .metric-value { font-size: 2rem; font-weight: bold; color: #7c83fd; }
    .metric-label { font-size: 0.85rem; color: #aaa; margin-top: 4px; }
    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #7c83fd;
        margin-bottom: 16px;
        border-left: 4px solid #7c83fd;
        padding-left: 12px;
    }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("Housing.csv")
    return df

@st.cache_data
def preprocess(df):
    df2 = df.copy()
    binary_cols = ['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea']
    for col in binary_cols:
        df2[col] = df2[col].map({'yes': 1, 'no': 0})
    df2['furnishingstatus'] = df2['furnishingstatus'].map({
        'furnished': 2, 'semi-furnished': 1, 'unfurnished': 0
    })
    return df2

df = load_data()
df_processed = preprocess(df)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/home.png", width=70)
    st.title("🏠 Housing Price")
    st.caption("Prediksi Harga Rumah — ML Project")
    st.divider()
    st.markdown("**Dataset Info**")
    st.markdown(f"- 📦 **{len(df)} rumah** tercatat")
    st.markdown(f"- 📋 **{len(df.columns)} fitur** tersedia")
    st.markdown(f"- ✅ **0 missing values**")
    st.divider()
    st.markdown("**Dibuat dengan:**")
    st.markdown("🐍 Python · 📊 Plotly · 🤖 Scikit-learn · 🌐 Streamlit")

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("🏠 Housing Price Prediction Dashboard")
st.caption("Exploratory Data Analysis + Machine Learning Model")
st.divider()

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 EDA", "🤖 Model & Evaluasi", "🔮 Prediksi Harga"])

# ══════════════════════════════════════════════
# TAB 1 — EDA
# ══════════════════════════════════════════════
with tab1:
    # KPI Cards
    st.markdown('<div class="section-title">📌 Overview Dataset</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{len(df)}</div>
            <div class="metric-label">Total Rumah</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">Rp {df['price'].mean()/1e6:.1f}M</div>
            <div class="metric-label">Rata-rata Harga</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{int(df['area'].mean()):,}</div>
            <div class="metric-label">Rata-rata Luas (sqft)</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">Rp {df['price'].max()/1e6:.1f}M</div>
            <div class="metric-label">Harga Tertinggi</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Distribusi Harga
    st.markdown('<div class="section-title">1️⃣ Distribusi Harga Rumah</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x='price', nbins=40, color_discrete_sequence=['#7c83fd'],
                           title="Histogram Harga")
        fig.update_layout(template='plotly_dark', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.box(df, y='price', color_discrete_sequence=['#f6c90e'],
                     title="Boxplot Harga (Outlier Detection)")
        fig.update_layout(template='plotly_dark', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Scatter Area vs Price
    st.markdown('<div class="section-title">2️⃣ Hubungan Luas vs Harga</div>', unsafe_allow_html=True)
    fig = px.scatter(df, x='area', y='price', color='furnishingstatus',
                     trendline='ols', title="Area vs Price (dengan trendline)",
                     color_discrete_sequence=['#7c83fd', '#f6c90e', '#ff6b6b'])
    fig.update_layout(template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)

    # Kategorik
    st.markdown('<div class="section-title">3️⃣ Pengaruh Fitur Kategorik terhadap Harga</div>', unsafe_allow_html=True)
    cat_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating',
                'airconditioning', 'prefarea', 'furnishingstatus']
    selected_cat = st.selectbox("Pilih fitur kategorik:", cat_cols)
    fig = px.box(df, x=selected_cat, y='price', color=selected_cat,
                 title=f"Distribusi Harga berdasarkan {selected_cat}",
                 color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_layout(template='plotly_dark', showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Heatmap Korelasi
    st.markdown('<div class="section-title">4️⃣ Correlation Heatmap</div>', unsafe_allow_html=True)
    corr = df_processed.corr()
    fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                    title="Korelasi Antar Fitur")
    fig.update_layout(template='plotly_dark', height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Raw Data
    st.markdown('<div class="section-title">📋 Raw Data</div>', unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True, height=300)

# ══════════════════════════════════════════════
# TAB 2 — MODEL & EVALUASI
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">🤖 Training Model</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        model_choice = st.selectbox("Pilih Model:", ["Linear Regression", "Random Forest"])
        test_size = st.slider("Test Size:", 0.1, 0.4, 0.2, 0.05)
        train_btn = st.button("🚀 Train Model", use_container_width=True, type="primary")

    X = df_processed.drop('price', axis=1)
    y = df_processed['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    if model_choice == "Linear Regression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("R² Score", f"{r2:.3f}", help="Semakin mendekati 1 = semakin bagus")
        m2.metric("MAE", f"Rp {mae/1e6:.2f}M", help="Rata-rata error prediksi")
        m3.metric("MAPE", f"{mape:.1f}%", help="Error dalam persen")

    st.divider()

    # Actual vs Predicted
    st.markdown('<div class="section-title">📈 Actual vs Predicted</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test.values, y=y_pred, mode='markers',
                             marker=dict(color='#7c83fd', opacity=0.6, size=7),
                             name='Prediksi'))
    fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()],
                             y=[y_test.min(), y_test.max()],
                             mode='lines', line=dict(color='#ff6b6b', dash='dash'),
                             name='Ideal (perfect prediction)'))
    fig.update_layout(template='plotly_dark', title="Actual vs Predicted Price",
                      xaxis_title="Harga Actual", yaxis_title="Harga Prediksi")
    st.plotly_chart(fig, use_container_width=True)

    # Feature Importance (Random Forest only)
    if model_choice == "Random Forest":
        st.markdown('<div class="section-title">🏆 Feature Importance</div>', unsafe_allow_html=True)
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)
        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                     color='Importance', color_continuous_scale='Blues',
                     title="Fitur Paling Berpengaruh")
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 3 — PREDIKSI
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">🔮 Prediksi Harga Rumah Kamu</div>', unsafe_allow_html=True)
    st.caption("Masukkan spesifikasi rumah untuk mendapatkan estimasi harga")

    col1, col2 = st.columns(2)

    with col1:
        area = st.number_input("📐 Luas Rumah (sqft)", min_value=500, max_value=20000, value=5000, step=100)
        bedrooms = st.slider("🛏️ Jumlah Kamar Tidur", 1, 6, 3)
        bathrooms = st.slider("🚿 Jumlah Kamar Mandi", 1, 4, 2)
        stories = st.slider("🏢 Jumlah Lantai", 1, 4, 2)
        parking = st.slider("🚗 Kapasitas Parkir", 0, 3, 1)
        furnishing = st.selectbox("🛋️ Kondisi Furnitur", ["furnished", "semi-furnished", "unfurnished"])

    with col2:
        mainroad = st.radio("🛣️ Akses Jalan Utama?", ["yes", "no"], horizontal=True)
        guestroom = st.radio("🛎️ Ada Kamar Tamu?", ["yes", "no"], horizontal=True)
        basement = st.radio("🏚️ Ada Basement?", ["yes", "no"], horizontal=True)
        hotwater = st.radio("🔥 Pemanas Air?", ["yes", "no"], horizontal=True)
        aircon = st.radio("❄️ Ada AC?", ["yes", "no"], horizontal=True)
        prefarea = st.radio("⭐ Area Premium?", ["yes", "no"], horizontal=True)

    st.divider()
    predict_btn = st.button("🔮 Prediksi Sekarang!", use_container_width=True, type="primary")

    if predict_btn:
        input_data = pd.DataFrame([{
            'area': area,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'stories': stories,
            'mainroad': 1 if mainroad == 'yes' else 0,
            'guestroom': 1 if guestroom == 'yes' else 0,
            'basement': 1 if basement == 'yes' else 0,
            'hotwaterheating': 1 if hotwater == 'yes' else 0,
            'airconditioning': 1 if aircon == 'yes' else 0,
            'parking': parking,
            'prefarea': 1 if prefarea == 'yes' else 0,
            'furnishingstatus': {'furnished': 2, 'semi-furnished': 1, 'unfurnished': 0}[furnishing]
        }])

        # Train model baru untuk prediksi
        X_full = df_processed.drop('price', axis=1)
        y_full = df_processed['price']
        pred_model = RandomForestRegressor(n_estimators=100, random_state=42)
        pred_model.fit(X_full, y_full)
        predicted_price = pred_model.predict(input_data)[0]

        st.success("✅ Prediksi berhasil!")
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1e2130, #2d3250);
                    border-radius: 16px; padding: 32px; text-align: center;
                    border: 2px solid #7c83fd; margin-top: 16px;">
            <div style="font-size: 1rem; color: #aaa;">Estimasi Harga Rumah</div>
            <div style="font-size: 3rem; font-weight: bold; color: #7c83fd; margin: 8px 0;">
                Rp {predicted_price:,.0f}
            </div>
            <div style="font-size: 1.2rem; color: #f6c90e;">
                ≈ Rp {predicted_price/1e6:.2f} Juta
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Perbandingan dengan rata-rata
        avg = df['price'].mean()
        diff = predicted_price - avg
        pct = (diff / avg) * 100
        st.markdown("<br>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        col_a.metric("Rata-rata pasar", f"Rp {avg/1e6:.2f}M")
        col_b.metric("Selisih dari rata-rata", f"Rp {abs(diff)/1e6:.2f}M",
                     delta=f"{pct:+.1f}%")
