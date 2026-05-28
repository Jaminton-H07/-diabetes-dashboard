import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json

st.set_page_config(
    page_title="DiabetesML — Pima Indians",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display:ital@0;1&display=swap');
html,body,[class*="css"]{ font-family:'DM Sans',sans-serif; }
.stApp{ background:#0B0F1A; color:#F0F4FF; }
[data-testid="stSidebar"]{ background:#111827; border-right:1px solid rgba(255,255,255,0.07); }
.hero-title{ font-family:'DM Serif Display',serif; font-size:2.4rem; line-height:1.1; letter-spacing:-0.03em; color:#F0F4FF; margin-bottom:0.3rem; }
.hero-title em{ color:#4ECDC4; font-style:italic; }
.hero-sub{ color:#8892A4; font-size:0.95rem; margin-bottom:1.2rem; }
.sec-eye{ font-size:11px; font-weight:600; letter-spacing:.1em; text-transform:uppercase; color:#4ECDC4; margin-bottom:4px; }
.sec-title{ font-family:'DM Serif Display',serif; font-size:1.6rem; letter-spacing:-0.02em; margin-bottom:0.4rem; }
.sec-sub{ color:#8892A4; font-size:14px; max-width:680px; line-height:1.7; margin-bottom:1.5rem; }
.metric-card{ background:#111827; border:1px solid rgba(255,255,255,0.07); border-radius:12px; padding:1rem 1.2rem; text-align:center; }
.metric-val{ font-family:'DM Serif Display',serif; font-size:1.9rem; color:#F0F4FF; line-height:1; margin-bottom:4px; }
.metric-lbl{ font-size:11px; color:#8892A4; text-transform:uppercase; letter-spacing:.06em; }
.metric-sub{ font-size:12px; color:#4ECDC4; margin-top:4px; font-weight:500; }
.theory-box{ background:rgba(78,205,196,0.06); border-left:3px solid #4ECDC4; border-radius:0 8px 8px 0; padding:1rem 1.2rem; margin-bottom:1.2rem; font-size:13px; color:#B0BAC8; line-height:1.7; }
.insight-box{ background:rgba(255,209,102,0.07); border-left:3px solid #FFD166; border-radius:0 8px 8px 0; padding:0.8rem 1rem; margin-top:0.8rem; font-size:13px; color:#B0BAC8; }
.conclusion-box{ background:rgba(6,214,160,0.07); border:1px solid rgba(6,214,160,0.2); border-radius:12px; padding:1.2rem 1.4rem; font-size:14px; color:#B0BAC8; line-height:1.8; }
.disclaimer{ background:rgba(255,255,255,0.04); border-left:3px solid #4ECDC4; padding:0.7rem 1rem; border-radius:0 8px 8px 0; font-size:12px; color:#8892A4; margin-top:1rem; }
hr.sec{ border:none; border-top:1px solid rgba(255,255,255,0.07); margin:2rem 0; }
.stButton>button{ background:#4ECDC4!important; color:#0B0F1A!important; font-weight:700!important; border:none!important; border-radius:8px!important; width:100%!important; }
.stTabs [data-baseweb="tab-list"]{ background:#111827; border-radius:8px; }
.stTabs [data-baseweb="tab"]{ color:#8892A4!important; }
.stTabs [aria-selected="true"]{ color:#4ECDC4!important; border-bottom-color:#4ECDC4!important; }
.stNumberInput input,.stTextInput input,.stSelectbox select{ background:#1a2235!important; border:1px solid rgba(255,255,255,0.12)!important; color:#F0F4FF!important; border-radius:8px!important; }
</style>
""", unsafe_allow_html=True)

# ─── Datos reales del notebook ────────────────────────────────────────────────
DATA = {
  "shapiro": {
    "Pregnancies":              {"W": 0.9364, "p": 0.0},
    "Glucose":                  {"W": 0.9885, "p": 0.0},
    "BloodPressure":            {"W": 0.8727, "p": 0.0},
    "SkinThickness":            {"W": 0.9194, "p": 0.0},
    "Insulin":                  {"W": 0.7702, "p": 0.0},
    "BMI":                      {"W": 0.9779, "p": 0.0},
    "DiabetesPedigreeFunction": {"W": 0.8278, "p": 0.0},
    "Age":                      {"W": 0.9303, "p": 0.0}
  },
  "nulos_pct": {"Pregnancies":0.0,"Glucose":0.654,"BloodPressure":4.557,"SkinThickness":29.608,"Insulin":48.697,"BMI":1.432,"DiabetesPedigreeFunction":0.0,"Age":0.0,"Outcome":0.0},
  "rachas": {
    "Glucose":       {"rachas":14,"Z":-0.107,"p":0.9149},
    "BloodPressure": {"rachas":61,"Z":0.694, "p":0.4878},
    "SkinThickness": {"rachas":200,"Z":1.062,"p":0.2882},
    "Insulin":       {"rachas":253,"Z":1.483,"p":0.1381},
    "BMI":           {"rachas":22,"Z":0.237, "p":0.8127}
  },
  "outliers": {
    "Pregnancies":              {"Q1":4.0,"Q3":12.0,"IQR":8.0,"li":-12.0,"ls":28.0,"n_out":0,"pct_out":0.0},
    "Glucose":                  {"Q1":99.0,"Q3":143.0,"IQR":44.0,"li":33.0,"ls":209.0,"n_out":0,"pct_out":0.0},
    "BloodPressure":            {"Q1":62.0,"Q3":80.0,"IQR":18.0,"li":35.0,"ls":107.0,"n_out":4,"pct_out":0.6},
    "SkinThickness":            {"Q1":20.0,"Q3":32.0,"IQR":12.0,"li":2.0,"ls":50.0,"n_out":1,"pct_out":0.2},
    "Insulin":                  {"Q1":121.5,"Q3":207.5,"IQR":86.0,"li":-7.5,"ls":336.5,"n_out":11,"pct_out":2.8},
    "BMI":                      {"Q1":27.5,"Q3":36.6,"IQR":9.1,"li":13.8,"ls":50.3,"n_out":4,"pct_out":0.7},
    "DiabetesPedigreeFunction": {"Q1":0.244,"Q3":0.626,"IQR":0.382,"li":-0.329,"ls":1.199,"n_out":29,"pct_out":3.8},
    "Age":                      {"Q1":27.0,"Q3":41.0,"IQR":14.0,"li":6.0,"ls":62.0,"n_out":4,"pct_out":0.5}
  },
  "medianas_imp": {"Glucose":117.0,"BloodPressure":72.0,"SkinThickness":23.0,"Insulin":125.0,"BMI":32.3},
  "pca": {"ve":[14.44,14.14,13.15,12.92,12.38,11.76,11.57,9.64],"va":[14.44,28.59,41.74,54.66,67.04,78.8,90.37,100.01],"n_opt":6},
  "corr": {"Glucose":0.4954,"Insulin":0.39,"BMI":0.3152,"Age":0.2387,"Pregnancies":0.2216,"SkinThickness":0.1737,"DiabetesPedigreeFunction":0.1628,"BloodPressure":0.0649},
  "metricas": {
    "Regresión Logística": {"accuracy":0.7597,"precision":0.6567,"recall":0.6111,"f1":0.633,"auc":0.8139,"tp":44,"fp":23,"tn":72,"fn":28,"fpr":[0.0,0.0,0.0,0.0,0.0104,0.0104,0.0208,0.0208,0.0208,0.0313,0.0313,0.0313,0.0313,0.0521,0.0521,0.0521,0.0521,0.0625,0.0625,0.0625,0.0729,0.0729,0.0938,0.0938,0.1042,0.1042,0.1146,0.1146,0.125,0.125,0.1354,0.1458,0.1458,0.1458,0.1563,0.1563,0.1667,0.1667,0.1875,0.1875,0.1875,0.2083,0.2083,0.2083,0.2188,0.2396,0.2396,0.2604,0.2604,0.2708,0.2708,0.2813,0.2813,0.2917,0.3021,0.3021,0.3021,0.3229,0.3229,0.3438,0.3438,0.3542,0.3542,0.3646,0.3646,0.375,0.375,0.3958,0.3958,0.4167,0.4167,0.4375,0.4375,0.4688,0.4688,0.4896,0.5104,0.5104,0.5208,0.5208,0.5417,0.5625,0.5625,0.5938,0.6042,0.625,0.6354,0.6458,0.6563,0.6771,0.6979,0.7083,0.7292,0.7396,0.7813,0.8021,0.8125,0.8333,0.8438,0.875,0.8854,0.9167,0.9271,0.9375,0.9583,1.0],"tpr":[0.0,0.0139,0.0278,0.0556,0.0556,0.0694,0.0694,0.0833,0.0972,0.0972,0.1111,0.125,0.1389,0.1389,0.1528,0.1806,0.1944,0.1944,0.2083,0.2222,0.2222,0.2361,0.2361,0.25,0.25,0.2778,0.2778,0.2917,0.2917,0.3194,0.3194,0.3194,0.3333,0.3472,0.3472,0.3611,0.3611,0.375,0.375,0.3889,0.4028,0.4028,0.4167,0.4444,0.4444,0.4444,0.4583,0.4583,0.4722,0.4722,0.4861,0.4861,0.5,0.5,0.5,0.5139,0.5278,0.5278,0.5417,0.5417,0.5556,0.5556,0.5694,0.5694,0.5694,0.5833,0.5972,0.6111,0.6111,0.625,0.625,0.6389,0.6528,0.6528,0.6667,0.6667,0.6806,0.6806,0.6944,0.7083,0.7222,0.7222,0.7361,0.75,0.75,0.7639,0.7639,0.7639,0.7778,0.7917,0.8056,0.8056,0.8194,0.8333,0.8472,0.8611,0.8611,0.875,0.8889,0.8889,0.9028,0.9028,0.9167,0.9306,1.0]},
    "Árbol de Decisión":   {"accuracy":0.7468,"precision":0.6286,"recall":0.7593,"f1":0.6875,"auc":0.7898,"tp":55,"fp":33,"tn":62,"fn":17,"fpr":[0.0,0.3438,0.3438,1.0],"tpr":[0.0,0.7593,1.0,1.0]},
    "KNN (k=11)":          {"accuracy":0.7338,"precision":0.6271,"recall":0.6389,"f1":0.6329,"auc":0.7948,"tp":46,"fp":28,"tn":67,"fn":26,"fpr":[0.0,0.0,0.0104,0.0104,0.0208,0.0208,0.0313,0.0313,0.0417,0.0417,0.0521,0.0625,0.0625,0.0729,0.0729,0.0833,0.0938,0.0938,0.1042,0.1042,0.1146,0.1146,0.125,0.125,0.1354,0.1354,0.1458,0.1563,0.1563,0.1667,0.1667,0.1667,0.1875,0.1875,0.2083,0.2083,0.2292,0.2292,0.25,0.25,0.2604,0.2604,0.2708,0.2813,0.2813,0.3021,0.3021,0.3125,0.3125,0.3229,0.3229,0.3438,0.3438,0.3646,0.3646,0.3958,0.3958,0.4167,0.4167,0.4375,0.4375,0.4583,0.4583,0.4792,0.4792,0.5,0.5,0.5208,0.5208,0.5417,0.5417,0.5625,0.5729,0.5729,0.5938,0.5938,0.625,0.625,0.6458,0.6563,0.6563,0.6771,0.6771,0.6979,0.6979,0.7292,0.7292,0.7604,0.7604,0.7917,0.8021,0.8125,0.8125,0.8438,0.8542,0.8646,0.8646,0.9167,0.9583,0.9583,1.0],"tpr":[0.0,0.0139,0.0139,0.0278,0.0278,0.0556,0.0556,0.0694,0.0694,0.0972,0.0972,0.0972,0.1111,0.1111,0.1389,0.1389,0.1389,0.1528,0.1528,0.1667,0.1667,0.1944,0.1944,0.2083,0.2083,0.2222,0.2222,0.2222,0.2361,0.2361,0.25,0.2639,0.2639,0.2778,0.2778,0.2917,0.2917,0.3056,0.3056,0.3194,0.3194,0.3333,0.3333,0.3333,0.3472,0.3472,0.3611,0.3611,0.375,0.375,0.3889,0.3889,0.4167,0.4167,0.4306,0.4306,0.4583,0.4583,0.4722,0.4722,0.5,0.5,0.5139,0.5139,0.5278,0.5278,0.5417,0.5417,0.5556,0.5556,0.5694,0.5694,0.5833,0.5833,0.5972,0.5972,0.6111,0.6111,0.625,0.625,0.6389,0.6389,0.6528,0.6528,0.6667,0.6667,0.6806,0.6806,0.6944,0.6944,0.7083,0.7222,0.7222,0.7361,0.7361,0.7639,0.7639,0.7778,1.0]}
  },
  "mejor_auc": "Regresión Logística",
  "mejor_recall": "Árbol de Decisión",
  "desc": {
    "Pregnancies":              {"media":3.845,"mediana":3.0,"std":3.37,"min":0.0,"max":17.0,"skew":0.902,"kurt":0.159},
    "Glucose":                  {"media":120.895,"mediana":117.0,"std":31.973,"min":0.0,"max":199.0,"skew":0.174,"kurt":-0.431},
    "BloodPressure":            {"media":69.105,"mediana":72.0,"std":19.356,"min":0.0,"max":122.0,"skew":-1.844,"kurt":5.18},
    "SkinThickness":            {"media":20.536,"mediana":23.0,"std":15.952,"min":0.0,"max":99.0,"skew":0.109,"kurt":-0.52},
    "Insulin":                  {"media":79.799,"mediana":30.5,"std":115.244,"min":0.0,"max":846.0,"skew":2.272,"kurt":7.214},
    "BMI":                      {"media":31.993,"mediana":32.0,"std":7.884,"min":0.0,"max":67.1,"skew":0.596,"kurt":3.29},
    "DiabetesPedigreeFunction": {"media":0.472,"mediana":0.3725,"std":0.331,"min":0.078,"max":2.42,"skew":1.92,"kurt":5.595},
    "Age":                      {"media":33.241,"mediana":29.0,"std":11.76,"min":21.0,"max":81.0,"skew":1.13,"kurt":0.643}
  },
  "imputacion_stats": {
    "original_mean":125.81,"original_median":125.0,"original_std":89.56,
    "mediana_mean":125.45,"mediana_median":125.0,"mediana_std":79.38,
    "mice_mean":123.97,"mice_median":117.42,"mice_std":84.21
  }
}

COLORES = {'Regresión Logística':'#4ECDC4','Árbol de Decisión':'#FFD166','KNN (k=11)':'#FF6B6B'}
BG = '#111827'; GRID = 'rgba(255,255,255,0.06)'; MUTED = '#8892A4'

def plotly_base(h=280):
    return dict(paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color='#F0F4FF',size=12),
                height=h, margin=dict(l=10,r=10,t=30,b=10),
                legend=dict(font=dict(color=MUTED),bgcolor='rgba(0,0,0,0)'),
                xaxis=dict(gridcolor=GRID), yaxis=dict(gridcolor=GRID))

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 0.5rem;'>
        <div style='font-family:"DM Serif Display",serif;font-size:1.4rem;color:#4ECDC4;'>DiabetesML</div>
        <div style='font-size:12px;color:#8892A4;'>Seminario de Ciencia de los Datos</div>
        <div style='font-size:12px;color:#8892A4;'>I.U. Pascual Bravo</div>
    </div>
    <hr style='border:none;border-top:1px solid rgba(255,255,255,0.07);margin:.8rem 0;'>
    """, unsafe_allow_html=True)

    pagina = st.radio("", [
        "🏠 Inicio",
        "I. Análisis Descriptivo",
        "II. Distribuciones y Normalidad",
        "III. Datos Faltantes",
        "IV. Tratamiento de Outliers",
        "V. Imputación Comparativa",
        "VI. PCA",
        "VII. Selección de Características",
        "VIII. Modelamiento",
        "IX. Conclusiones",
        "🩺 Calculadora de Riesgo",
    ], label_visibility="collapsed")

    st.markdown("""
    <hr style='border:none;border-top:1px solid rgba(255,255,255,0.07);margin:1rem 0;'>
    <div style='font-size:11px;color:#8892A4;line-height:1.8;'>
    <b style='color:#F0F4FF;'>Dataset:</b> Pima Indians Diabetes<br>
    <b style='color:#F0F4FF;'>Fuente:</b> UCI ML Repository<br>
    <b style='color:#F0F4FF;'>Registros:</b> 768 · <b style='color:#F0F4FF;'>Variables:</b> 9<br>
    <b style='color:#F0F4FF;'>Metodología:</b> CRISP-DM
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# INICIO
# ══════════════════════════════════════════════════════════════════════════════
if pagina == "🏠 Inicio":
    st.markdown('<div class="hero-title">Impacto de factores clínicos<br>en la <em>presencia de diabetes</em></div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Proyecto Final · Seminario de Ciencia de los Datos · Pima Indians Diabetes Dataset (UCI ML Repository) · Metodología CRISP-DM</div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    for col, val, lbl, sub in zip([c1,c2,c3,c4],
        ["768","9","34.9%",f"{DATA['metricas'][DATA['mejor_auc']]['auc']:.3f}"],
        ["Registros","Variables","Prevalencia DM","Mejor AUC"],
        ["dataset completo","8 predictoras + Outcome","desbalance 1.87:1","Regresión Logística"]):
        col.markdown(f'<div class="metric-card"><div class="metric-val">{val}</div><div class="metric-lbl">{lbl}</div><div class="metric-sub">{sub}</div></div>', unsafe_allow_html=True)

    st.markdown('<hr class="sec">', unsafe_allow_html=True)
    st.markdown('<div class="sec-eye">Metodología</div><div class="sec-title">Pipeline CRISP-DM — 9 secciones</div>', unsafe_allow_html=True)

    pasos = [
        ("01","Análisis Descriptivo","Estadísticos, distribución de clases, tipos de datos"),
        ("02","Normalidad","Shapiro-Wilk, Q-Q plots"),
        ("03","Datos Faltantes","Ceros inválidos → NaN, prueba de rachas MCAR"),
        ("04","Outliers","Método IQR, capping/winsorización"),
        ("05","Imputación","Mediana vs MICE, comparación KDE"),
        ("06","PCA","Reducción dimensional, Scree Plot, biplot"),
        ("07","Características","Correlación, top 5 variables"),
        ("08","Modelamiento","Reg. Logística, Árbol, KNN"),
        ("09","Conclusiones","Hallazgos, limitaciones, mejoras futuras"),
    ]
    cols = st.columns(3)
    for i,(num,titulo,desc) in enumerate(pasos):
        with cols[i%3]:
            st.markdown(f"""
            <div style='background:#111827;border:1px solid rgba(255,255,255,0.07);border-radius:10px;padding:1rem;margin-bottom:12px;'>
                <div style='font-size:11px;color:#4ECDC4;font-weight:600;margin-bottom:4px;'>{num}</div>
                <div style='font-size:14px;font-weight:600;margin-bottom:4px;'>{titulo}</div>
                <div style='font-size:12px;color:#8892A4;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# I. ANÁLISIS DESCRIPTIVO
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "I. Análisis Descriptivo":
    st.markdown('<div class="sec-eye">Sección I</div><div class="sec-title">Análisis Descriptivo y de Calidad</div>', unsafe_allow_html=True)
    st.markdown('<div class="theory-box"><b>Marco Teórico:</b> El EDA (Análisis Exploratorio de Datos) es la primera fase del proceso CRISP-DM. Su objetivo es comprender la estructura, calidad y propiedades estadísticas del dataset antes de aplicar técnicas avanzadas. En datos clínicos es especialmente crítico porque los errores de medición y los valores fisiológicamente imposibles son frecuentes (Tukey, 1977).</div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    for col,v,l,s in zip([c1,c2,c3,c4],["768","268","500","8"],
        ["Registros totales","Con diabetes","Sin diabetes","Variables predictoras"],
        ["mujeres Pima ≥21 años","34.9% del dataset","65.1% del dataset","+ 1 variable objetivo"]):
        col.markdown(f'<div class="metric-card"><div class="metric-val">{v}</div><div class="metric-lbl">{l}</div><div class="metric-sub">{s}</div></div>', unsafe_allow_html=True)

    st.markdown('<hr class="sec">', unsafe_allow_html=True)

    col_t, col_c = st.columns([1.4, 1])
    with col_t:
        st.markdown("**Estadísticos descriptivos completos**")
        rows = []
        for col_name in ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']:
            d = DATA['desc'][col_name]
            rows.append({'Variable':col_name,'Media':d['media'],'Mediana':d['mediana'],
                         'Desv.Est.':d['std'],'Mín':d['min'],'Máx':d['max'],'Asimetría':d['skew'],'Curtosis':d['kurt']})
        df_show = pd.DataFrame(rows)
        st.dataframe(df_show, use_container_width=True, hide_index=True)
        st.markdown('<div class="insight-box">⚠️ <b>Insulin</b> tiene media=79.8 pero mediana=30.5 → muchos ceros (datos faltantes enmascarados). <b>SkinThickness</b> y otras 3 variables presentan el mismo problema.</div>', unsafe_allow_html=True)

    with col_c:
        st.markdown("**Distribución de la variable objetivo**")
        fig = go.Figure(go.Pie(
            labels=["Sin diabetes (65.1%)","Con diabetes (34.9%)"],
            values=[500,268], hole=0.6,
            marker=dict(colors=["#4ECDC4","#FF6B6B"],line=dict(color="#0B0F1A",width=2)),
        ))
        fig.update_layout(**plotly_base(260))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight-box">Ratio de desbalance: <b>1.87:1</b>. La clase mayoritaria representa el 65.1%. Esto se consideró al evaluar los modelos.</div>', unsafe_allow_html=True)

    st.markdown('<hr class="sec">', unsafe_allow_html=True)
    st.markdown("**Descripción clínica de las variables**")
    vars_info = [
        ["Pregnancies","Número de embarazos previos","Entera / conteo","0% (válido = 0)"],
        ["Glucose","Glucosa plasmática 2h post-carga (OGTT)","Entera / mg/dL","0.65% nulos"],
        ["BloodPressure","Presión arterial diastólica","Entera / mm Hg","4.56% nulos"],
        ["SkinThickness","Grosor pliegue cutáneo tríceps","Entera / mm","29.61% nulos"],
        ["Insulin","Insulina sérica 2h post-carga","Entera / mu U/ml","48.70% nulos ⚠️"],
        ["BMI","Índice de Masa Corporal","Decimal / kg/m²","1.43% nulos"],
        ["DiabetesPedigreeFunction","Índice de riesgo genético familiar","Decimal / índice","0%"],
        ["Age","Edad de la paciente","Entera / años","0%"],
        ["Outcome","Diagnóstico de diabetes (objetivo)","Binaria (0/1)","—"],
    ]
    df_vars = pd.DataFrame(vars_info, columns=["Variable","Descripción","Tipo / Unidad","Nulos"])
    st.dataframe(df_vars, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# II. DISTRIBUCIONES Y NORMALIDAD
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "II. Distribuciones y Normalidad":
    st.markdown('<div class="sec-eye">Sección II</div><div class="sec-title">Distribuciones y Normalidad</div>', unsafe_allow_html=True)
    st.markdown('<div class="theory-box"><b>Marco Teórico:</b> La prueba de <b>Shapiro-Wilk</b> evalúa si los datos siguen una distribución normal gaussiana. H₀: los datos son normales. Si p &lt; 0.05 → rechazamos H₀ → datos NO normales. Este resultado es clave porque determina qué métodos de preprocesamiento son válidos: si no hay normalidad, debemos usar la <b>mediana</b> (no la media) para imputar y el método <b>IQR</b> (no Z-score) para outliers.</div>', unsafe_allow_html=True)

    st.markdown("**Resultados Shapiro-Wilk — ninguna variable es normal (p < 0.001)**")
    sw_rows = []
    for col_name, v in DATA['shapiro'].items():
        d = DATA['desc'][col_name]
        sw_rows.append({'Variable':col_name,'Estadístico W':v['W'],'p-valor':'< 0.001','¿Normal?':'No ✗','Asimetría':d['skew'],'Implicación metodológica':
            'Mediana para imputar' if col_name in ['Glucose','Insulin','BMI','SkinThickness','BloodPressure'] else
            'Cola derecha severa' if d['skew'] > 1.5 else 'Variable discreta' if col_name in ['Pregnancies','Age'] else 'Método IQR para outliers'})
    st.dataframe(pd.DataFrame(sw_rows), use_container_width=True, hide_index=True)

    st.markdown('<hr class="sec">', unsafe_allow_html=True)
    st.markdown("**Asimetría (skewness) por variable**")
    cols_names = list(DATA['desc'].keys())
    skew_vals  = [DATA['desc'][c]['skew'] for c in cols_names]
    colores_sk = ['#FF6B6B' if abs(v)>1 else '#FFD166' if abs(v)>0.5 else '#4ECDC4' for v in skew_vals]
    fig = go.Figure(go.Bar(x=cols_names, y=skew_vals, marker=dict(color=colores_sk,cornerradius=4),
                           text=[f"{v:.2f}" for v in skew_vals], textposition='outside', textfont=dict(color='#F0F4FF')))
    fig.add_hline(y=0, line_color='white', line_width=1, opacity=0.3)
    fig.update_layout(**plotly_base(260), yaxis_title="Skewness")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="insight-box">🔴 Rojo (|skew|>1): distribución muy sesgada → usar mediana. 🟡 Amarillo (0.5-1): sesgo moderado. 🟢 Verde (&lt;0.5): aproximadamente simétrica.</div>', unsafe_allow_html=True)

    st.markdown('<hr class="sec">', unsafe_allow_html=True)
    st.markdown("**Conclusión de normalidad**")
    st.markdown('<div class="conclusion-box">✅ <b>Ninguna de las 8 variables numéricas sigue una distribución normal</b> según la prueba de Shapiro-Wilk (p &lt; 0.001 en todos los casos). Esto es esperado en datos biomédicos reales. Las consecuencias metodológicas son:<br><br>• <b>Imputación:</b> se usará la <b>mediana</b> (robusta ante outliers y asimetría) en lugar de la media<br>• <b>Outliers:</b> se usará el método <b>IQR</b> (no asume normalidad) en lugar del Z-score<br>• <b>Modelos:</b> se usan modelos con regularización adecuada o no paramétricos</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# III. DATOS FALTANTES
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "III. Datos Faltantes":
    st.markdown('<div class="sec-eye">Sección III</div><div class="sec-title">Detección y Tratamiento de Ausentes</div>', unsafe_allow_html=True)
    st.markdown('<div class="theory-box"><b>Marco Teórico:</b> En el dataset Pima Indians, los valores <b>cero</b> en variables fisiológicas como glucosa, insulina o presión arterial son <b>biológicamente imposibles</b> en un paciente vivo. Representan datos faltantes enmascarados como ceros. La <b>Prueba de Rachas de Wald-Wolfowitz</b> evalúa si estos nulos son completamente aleatorios (MCAR). Si p > 0.05 → nulos aleatorios → imputación estándar válida.</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Porcentaje de valores faltantes por variable**")
        nulos_vars = {k:v for k,v in DATA['nulos_pct'].items() if v > 0 and k != 'Outcome'}
        nulos_sorted = dict(sorted(nulos_vars.items(), key=lambda x: x[1]))
        colores_n = ['#FF6B6B' if v>30 else '#FFD166' if v>5 else '#4ECDC4' for v in nulos_sorted.values()]
        fig = go.Figure(go.Bar(x=list(nulos_sorted.values()), y=list(nulos_sorted.keys()),
                               orientation='h', marker=dict(color=colores_n,cornerradius=4),
                               text=[f"{v:.1f}%" for v in nulos_sorted.values()],
                               textposition='outside', textfont=dict(color='#F0F4FF')))
        fig.update_layout(**plotly_base(240))
        fig.update_xaxes(range=[0,60], gridcolor=GRID, title='% de valores faltantes', color=MUTED)
        fig.update_yaxes(gridcolor=GRID, color=MUTED)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("**Prueba de Rachas — Aleatoriedad de nulos (MCAR)**")
        r_rows = []
        for col_name, v in DATA['rachas'].items():
            r_rows.append({'Variable':col_name,'Rachas':v['rachas'],'Z':v['Z'],'p-valor':v['p'],
                           'Conclusión':'MCAR ✅' if v['p'] and v['p']>0.05 else 'NO aleatorio ⚠️'})
        st.dataframe(pd.DataFrame(r_rows), use_container_width=True, hide_index=True)
        st.markdown('<div class="insight-box"><b>Resultado:</b> p > 0.05 en todas las variables → nulos completamente aleatorios (MCAR). Esto <b>valida el uso de imputación estándar</b> sin necesidad de técnicas más complejas.</div>', unsafe_allow_html=True)

    st.markdown('<hr class="sec">', unsafe_allow_html=True)
    st.markdown("**Resumen del tratamiento**")
    c1,c2,c3 = st.columns(3)
    c1.markdown('<div class="metric-card"><div class="metric-val">652</div><div class="metric-lbl">Valores faltantes totales</div><div class="metric-sub">5 variables afectadas</div></div>', unsafe_allow_html=True)
    c2.markdown('<div class="metric-card"><div class="metric-val">48.7%</div><div class="metric-lbl">Máx. nulos (Insulin)</div><div class="metric-sub">Variable más crítica</div></div>', unsafe_allow_html=True)
    c3.markdown('<div class="metric-card"><div class="metric-val">MCAR</div><div class="metric-lbl">Mecanismo confirmado</div><div class="metric-sub">Imputación estándar válida</div></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# IV. OUTLIERS
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "IV. Tratamiento de Outliers":
    st.markdown('<div class="sec-eye">Sección IV</div><div class="sec-title">Tratamiento de Outliers</div>', unsafe_allow_html=True)
    st.markdown('<div class="theory-box"><b>Marco Teórico:</b> El método <b>IQR (Rango Intercuartílico)</b> identifica outliers fuera del intervalo [Q1 − 1.5·IQR, Q3 + 1.5·IQR]. Es preferible al Z-score para datos no normales porque usa la mediana como referencia. Se eligió <b>capping (winsorización)</b> sobre eliminación para preservar los 768 registros y mantener la información clínica de los casos extremos (Tukey, 1977).</div>', unsafe_allow_html=True)

    st.markdown("**Detección estadística de outliers — Método IQR**")
    out_rows = []
    for col_name, v in DATA['outliers'].items():
        out_rows.append({'Variable':col_name,'Q1':v['Q1'],'Q3':v['Q3'],'IQR':v['IQR'],
                         'Lím. Inf.':v['li'],'Lím. Sup.':v['ls'],'Outliers (n)':v['n_out'],'Outliers (%)':f"{v['pct_out']}%"})
    st.dataframe(pd.DataFrame(out_rows), use_container_width=True, hide_index=True)

    st.markdown('<hr class="sec">', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Outliers detectados por variable**")
        out_n = {k:v['n_out'] for k,v in DATA['outliers'].items() if v['n_out']>0}
        fig = go.Figure(go.Bar(x=list(out_n.keys()), y=list(out_n.values()),
                               marker=dict(color='#FFD166',cornerradius=4),
                               text=list(out_n.values()), textposition='outside', textfont=dict(color='#F0F4FF')))
        fig.update_layout(**plotly_base(250), yaxis_title='Número de outliers')
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("**Justificación del Capping**")
        justif = [
            ("✅ Preserva el tamaño muestral","768 registros son pocos. Eliminar filas reduce el poder predictivo de los modelos."),
            ("✅ Relevancia clínica","Un nivel de insulina extremo puede representar resistencia insulínica severa — un caso real de interés médico."),
            ("✅ Conserva la estructura","El capping mantiene que 'este valor era extremo' sin introducir valores artificiales."),
            ("❌ Eliminación descartada","Eliminar registros introduce sesgo de selección en datos clínicos."),
        ]
        for titulo, desc in justif:
            st.markdown(f"""
            <div style='background:#111827;border:1px solid rgba(255,255,255,0.07);border-radius:8px;padding:0.8rem 1rem;margin-bottom:8px;'>
                <div style='font-size:13px;font-weight:600;margin-bottom:3px;'>{titulo}</div>
                <div style='font-size:12px;color:#8892A4;'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="conclusion-box">✅ Resultado del capping: <b>0 registros eliminados</b>. Los valores fuera de los límites IQR × 1.5 fueron reemplazados por el valor del límite correspondiente. El dataset conserva sus 768 registros completos.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# V. IMPUTACIÓN
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "V. Imputación Comparativa":
    st.markdown('<div class="sec-eye">Sección V</div><div class="sec-title">Imputación Comparativa</div>', unsafe_allow_html=True)
    st.markdown('<div class="theory-box"><b>Marco Teórico:</b> La <b>imputación simple por mediana</b> reemplaza todos los nulos con el mismo valor central. La <b>imputación iterativa MICE</b> (Multiple Imputation by Chained Equations) modela cada variable faltante como función de las demás, estimando un valor individualizado por registro. MICE preserva mejor las correlaciones entre variables (van Buuren, 2018).</div>', unsafe_allow_html=True)

    st.markdown("**Comparación de estadísticos — Variable: Insulin (48.7% de nulos)**")
    imp = DATA['imputacion_stats']
    imp_df = pd.DataFrame({
        'Estadístico':  ['Media','Mediana','Desv. Estándar'],
        'Original (presentes)': [imp['original_mean'], imp['original_median'], imp['original_std']],
        'Imputación Mediana':    [imp['mediana_mean'],  imp['mediana_median'],  imp['mediana_std']],
        'Imputación MICE':       [imp['mice_mean'],     imp['mice_median'],     imp['mice_std']],
    })
    st.dataframe(imp_df, use_container_width=True, hide_index=True)

    st.markdown('<hr class="sec">', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Diferencia entre métodos — Media e Insulin**")
        categorias = ['Media','Mediana','Desv. Estándar']
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Original', x=categorias, y=[imp['original_mean'],imp['original_median'],imp['original_std']], marker=dict(color='#8892A4',cornerradius=4)))
        fig.add_trace(go.Bar(name='Mediana',  x=categorias, y=[imp['mediana_mean'], imp['mediana_median'], imp['mediana_std']],  marker=dict(color='#4ECDC4',cornerradius=4)))
        fig.add_trace(go.Bar(name='MICE',     x=categorias, y=[imp['mice_mean'],    imp['mice_median'],    imp['mice_std']],     marker=dict(color='#FFD166',cornerradius=4)))
        fig.update_layout(**plotly_base(280), barmode='group')
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("**¿Por qué MICE es mejor?**")
        razones = [
            ("Valores individualizados", "Cada nulo recibe un valor estimado según su glucosa, BMI, edad. No todos reciben el mismo valor."),
            ("Preserva la varianza","La desviación estándar post-MICE se aproxima más a la original que la mediana."),
            ("Aprovecha correlaciones","Insulin está correlacionada con Glucose (r=0.39) y BMI (r=0.27). MICE usa esas relaciones."),
            ("Estándar biomédico","Van Buuren (2018) recomienda MICE cuando la proporción de nulos supera el 20%."),
        ]
        for titulo, desc in razones:
            st.markdown(f"""
            <div style='background:#111827;border:1px solid rgba(78,205,196,0.2);border-radius:8px;padding:0.8rem 1rem;margin-bottom:8px;'>
                <div style='font-size:13px;font-weight:600;color:#4ECDC4;margin-bottom:3px;'>{titulo}</div>
                <div style='font-size:12px;color:#8892A4;'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="conclusion-box">✅ <b>Decisión:</b> Se usa la Imputación Iterativa (MICE) para el Entregable 2. La mediana crea un pico artificial al dar el mismo valor a los 374 nulos de Insulin. MICE distribuye los valores de forma más realista aprovechando las correlaciones entre variables.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# VI. PCA
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "VI. PCA":
    st.markdown('<div class="sec-eye">Sección VI</div><div class="sec-title">Análisis de Componentes Principales (PCA)</div>', unsafe_allow_html=True)
    st.markdown('<div class="theory-box"><b>Marco Teórico:</b> El PCA transforma las variables originales correlacionadas en componentes principales no correlacionados, ordenados por varianza explicada. <b>Requisito fundamental:</b> estandarizar las variables antes (media=0, desv=1) para que ninguna domine por escala. Se busca el número mínimo de componentes que explique ≥ 80% de la varianza (Jolliffe, 2002).</div>', unsafe_allow_html=True)

    pca = DATA['pca']
    c1,c2,c3 = st.columns(3)
    c1.markdown(f'<div class="metric-card"><div class="metric-val">{pca["n_opt"]}</div><div class="metric-lbl">Componentes óptimos</div><div class="metric-sub">para capturar ≥80% varianza</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-val">{pca["va"][pca["n_opt"]-1]:.1f}%</div><div class="metric-lbl">Varianza acumulada</div><div class="metric-sub">con {pca["n_opt"]} componentes</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-val">{int((1-pca["n_opt"]/8)*100)}%</div><div class="metric-lbl">Reducción dimensional</div><div class="metric-sub">de 8 → {pca["n_opt"]} dimensiones</div></div>', unsafe_allow_html=True)

    st.markdown('<hr class="sec">', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Scree Plot — Varianza explicada por componente**")
        cps = [f'CP{i+1}' for i in range(8)]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=cps, y=pca['ve'], marker=dict(color='#4ECDC4',cornerradius=4), name='Varianza'))
        fig.add_trace(go.Scatter(x=cps, y=pca['ve'], mode='lines+markers', line=dict(color='white',width=2), marker=dict(size=7), name='Tendencia'))
        fig.update_layout(**plotly_base(270), yaxis_title='Varianza explicada (%)')
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("**Varianza acumulada — umbral 80%**")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cps, y=pca['va'], mode='lines+markers', fill='tozeroy',
                                 fillcolor='rgba(78,205,196,0.1)', line=dict(color='#4ECDC4',width=2.5),
                                 marker=dict(size=8,color='#4ECDC4'), name='Acumulada'))
        fig.add_hline(y=80, line_dash='dot', line_color='#FFD166', annotation_text='80%', annotation_font_color='#FFD166')
        fig.add_hline(y=90, line_dash='dot', line_color='#FF6B6B', annotation_text='90%', annotation_font_color='#FF6B6B')
        fig.update_layout(**plotly_base(270))
        fig.update_xaxes(gridcolor=GRID, color=MUTED)
        fig.update_yaxes(range=[0,105], gridcolor=GRID, title='Varianza acumulada (%)', color=MUTED)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<hr class="sec">', unsafe_allow_html=True)
    st.markdown("**Varianza explicada por componente — tabla completa**")
    pca_table = pd.DataFrame({'Componente':cps, 'Varianza (%)':pca['ve'], 'Acumulada (%)':pca['va']})
    pca_table['Interpretación'] = ['Factores metabólicos (glucosa, insulina)','Edad y embarazos','Presión y piel',
                                    'Info. complementaria','Variabilidad de insulina','Varianza residual','Ruido menor','Ruido']
    st.dataframe(pca_table, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# VII. CARACTERÍSTICAS
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "VII. Selección de Características":
    st.markdown('<div class="sec-eye">Sección VII</div><div class="sec-title">Selección de Características</div>', unsafe_allow_html=True)
    st.markdown('<div class="theory-box"><b>Marco Teórico:</b> El método de filtro por <b>correlación de Pearson</b> evalúa la relación lineal entre cada predictor y la variable objetivo. A diferencia del PCA, conserva las variables originales facilitando la interpretabilidad clínica. Se identifican las 5 variables con mayor correlación absoluta con Outcome.</div>', unsafe_allow_html=True)

    corr = DATA['corr']
    col_a, col_b = st.columns([1.2, 1])

    with col_a:
        st.markdown("**Ranking de correlación con Outcome (valor absoluto)**")
        corr_sorted = dict(sorted(corr.items(), key=lambda x: x[1], reverse=True))
        colores_c = ['#4ECDC4' if v>0.3 else '#FFD166' if v>0.15 else '#8892A4' for v in corr_sorted.values()]
        fig = go.Figure(go.Bar(x=list(corr_sorted.values()), y=list(corr_sorted.keys()),
                               orientation='h', marker=dict(color=colores_c,cornerradius=4),
                               text=[f"r = {v:.4f}" for v in corr_sorted.values()],
                               textposition='outside', textfont=dict(color='#F0F4FF')))
        fig.update_layout(**plotly_base(300))
        fig.update_xaxes(range=[0,0.6], gridcolor=GRID, title='Correlación de Pearson', color=MUTED)
        fig.update_yaxes(gridcolor=GRID, color=MUTED)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("**Top 5 variables — relevancia clínica**")
        top5_info = {
            'Glucose':                  ('r=0.495','Principal marcador diagnóstico OMS (≥126 mg/dL)'),
            'Insulin':                  ('r=0.390','Resistencia insulínica: mecanismo central de DM2'),
            'BMI':                      ('r=0.315','Sobrepeso: factor de riesgo comprobado para DM2'),
            'Age':                      ('r=0.239','Prevalencia de DM2 aumenta con la edad'),
            'Pregnancies':              ('r=0.222','Diabetes gestacional previa aumenta riesgo a largo plazo'),
        }
        for i,(var,(corr_val,desc)) in enumerate(top5_info.items(),1):
            st.markdown(f"""
            <div style='background:#111827;border:1px solid rgba(78,205,196,0.2);border-radius:8px;padding:0.8rem 1rem;margin-bottom:8px;'>
                <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:3px;'>
                    <span style='font-size:14px;font-weight:600;'>{i}. {var}</span>
                    <span style='font-size:13px;font-weight:700;color:#4ECDC4;font-family:monospace;'>{corr_val}</span>
                </div>
                <div style='font-size:12px;color:#8892A4;'>{desc}</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# VIII. MODELAMIENTO
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "VIII. Modelamiento":
    st.markdown('<div class="sec-eye">Sección VIII</div><div class="sec-title">Modelamiento Supervisado</div>', unsafe_allow_html=True)
    st.markdown('<div class="theory-box"><b>Marco Teórico:</b> Se comparan 3 modelos: <b>Regresión Logística</b> (lineal, interpretable), <b>Árbol de Decisión</b> (max_depth=5, evita sobreajuste) y <b>KNN k=11</b> (basado en vecinos más cercanos). División 80/20 estratificada. En diagnóstico clínico el <b>Recall</b> es la métrica crítica: un falso negativo (diabética no detectada) tiene consecuencias médicas graves.</div>', unsafe_allow_html=True)

    met = DATA['metricas']
    mejor_auc = DATA['mejor_auc']; mejor_rec = DATA['mejor_recall']

    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><div class="metric-val">{met[mejor_auc]["auc"]:.3f}</div><div class="metric-lbl">Mejor ROC-AUC</div><div class="metric-sub">{mejor_auc}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-val">{met[mejor_rec]["recall"]:.3f}</div><div class="metric-lbl">Mejor Recall</div><div class="metric-sub">{mejor_rec}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-val">80/20</div><div class="metric-lbl">División train/test</div><div class="metric-sub">Estratificada por clase</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><div class="metric-val">614</div><div class="metric-lbl">Registros train</div><div class="metric-sub">154 en test</div></div>', unsafe_allow_html=True)

    st.markdown('<hr class="sec">', unsafe_allow_html=True)

    # Tabla comparativa
    st.markdown("**Comparación de métricas — todos los modelos**")
    met_rows = []
    for nombre, m in met.items():
        met_rows.append({'Modelo':nombre,'Accuracy':m['accuracy'],'Precision':m['precision'],
                         'Recall':m['recall'],'F1-Score':m['f1'],'ROC-AUC':m['auc'],
                         'TP':m['tp'],'FN':m['fn'],'TN':m['tn'],'FP':m['fp']})
    st.dataframe(pd.DataFrame(met_rows), use_container_width=True, hide_index=True)
    st.markdown(f'<div class="insight-box">⚕ Perspectiva clínica: el <b>Árbol de Decisión</b> tiene el mejor Recall ({met[mejor_rec]["recall"]:.3f}) → detecta más diabéticas. La <b>Regresión Logística</b> tiene el mejor AUC ({met[mejor_auc]["auc"]:.3f}) → mejor discriminación general.</div>', unsafe_allow_html=True)

    st.markdown('<hr class="sec">', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Curvas ROC comparativas**")
        fig = go.Figure()
        for nombre, color in COLORES.items():
            m = met[nombre]
            fig.add_trace(go.Scatter(x=m['fpr'], y=m['tpr'], mode='lines',
                                     name=f"{nombre} (AUC={m['auc']:.3f})",
                                     line=dict(color=color,width=2.5)))
        fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',name='Aleatorio (0.5)',
                                 line=dict(color=MUTED,dash='dot',width=1.5)))
        fig.update_layout(**plotly_base(310), xaxis_title='Tasa Falsos Positivos', yaxis_title='Sensibilidad (Recall)')
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("**Comparación de métricas por modelo**")
        metricas_names = ['accuracy','precision','recall','f1','auc']
        metricas_labels = ['Accuracy','Precision','Recall','F1','AUC']
        fig = go.Figure()
        for nombre, color in COLORES.items():
            m = met[nombre]
            fig.add_trace(go.Bar(name=nombre, x=metricas_labels,
                                 y=[m[k] for k in metricas_names],
                                 marker=dict(color=color,opacity=0.85,cornerradius=4),
                                 text=[f"{m[k]:.3f}" for k in metricas_names],
                                 textposition='outside', textfont=dict(size=10,color='#F0F4FF')))
        fig.update_layout(**plotly_base(310), barmode='group')
        fig.update_xaxes(gridcolor=GRID, color=MUTED)
        fig.update_yaxes(range=[0,1.05], gridcolor=GRID, color=MUTED)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<hr class="sec">', unsafe_allow_html=True)
    st.markdown("**Matrices de confusión**")
    cols_cm = st.columns(3)
    for col_cm, (nombre, color) in zip(cols_cm, COLORES.items()):
        m = met[nombre]
        with col_cm:
            fig = go.Figure(go.Heatmap(
                z=[[m['tn'],m['fp']],[m['fn'],m['tp']]],
                x=['Pred: No DM','Pred: DM'], y=['Real: No DM','Real: DM'],
                text=[[f"TN={m['tn']}",f"FP={m['fp']}"],[f"FN={m['fn']} ⚠️",f"TP={m['tp']}"]],
                texttemplate='%{text}', colorscale=[[0,'#111827'],[1,color]],
                showscale=False
            ))
            fig.update_layout(**plotly_base(220), title=dict(text=f"{nombre}<br>AUC={m['auc']:.3f}", font=dict(size=12,color='#F0F4FF')))
            st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# IX. CONCLUSIONES
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "IX. Conclusiones":
    st.markdown('<div class="sec-eye">Sección IX</div><div class="sec-title">Conclusiones y Reflexión Final</div>', unsafe_allow_html=True)

    conclusiones = [
        ("📊 Hallazgos del EDA",
         f"El dataset tiene 768 registros con prevalencia de diabetes del 34.9% (ratio 1.87:1). Ninguna variable sigue distribución normal (Shapiro-Wilk p < 0.001). Las variables con mayor asimetría fueron Insulin (skew=2.27) y DiabetesPedigreeFunction (skew=1.92)."),
        ("🔧 Impacto del preprocesamiento",
         "Se identificaron 652 valores faltantes enmascarados como ceros en 5 variables. La prueba de rachas confirmó MCAR en todas. El capping IQR preservó los 768 registros. La imputación MICE preservó mejor la distribución que la mediana simple."),
        ("📉 Reducción dimensional (PCA)",
         f"Con {DATA['pca']['n_opt']} componentes principales se captura el {DATA['pca']['va'][DATA['pca']['n_opt']-1]:.1f}% de la varianza. Reducción de 8 → {DATA['pca']['n_opt']} dimensiones. Glucose y BMI contribuyen más al CP1 (eje metabólico)."),
        ("🎯 Variables más importantes",
         f"Glucose (r={DATA['corr']['Glucose']:.3f}) es el predictor más potente, consistente con los criterios diagnósticos de la OMS. Le siguen Insulin (r={DATA['corr']['Insulin']:.3f}), BMI (r={DATA['corr']['BMI']:.3f}) y Age (r={DATA['corr']['Age']:.3f})."),
        ("🤖 Rendimiento de modelos",
         f"La Regresión Logística obtuvo el mejor AUC ({DATA['metricas']['Regresión Logística']['auc']:.3f}). El Árbol de Decisión obtuvo el mejor Recall ({DATA['metricas']['Árbol de Decisión']['recall']:.3f}), detectando más casos reales de diabetes. Desde perspectiva clínica, el Recall es la métrica prioritaria."),
    ]
    for titulo, texto in conclusiones:
        st.markdown(f"""
        <div class="conclusion-box" style='margin-bottom:1rem;'>
            <div style='font-size:15px;font-weight:600;color:#F0F4FF;margin-bottom:6px;'>{titulo}</div>
            {texto}
        </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="sec">', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Limitaciones**")
        lims = ["Solo mujeres Pima > 21 años → baja generalización a otras poblaciones",
                "48.7% de nulos en Insulin introduce incertidumbre en la imputación",
                "Desbalance de clases no tratado con SMOTE o class_weight",
                "Sin variables adicionales como HbA1c o colesterol"]
        for l in lims:
            st.markdown(f"<div style='font-size:13px;color:#8892A4;padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.04);'>⚠️ {l}</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown("**Mejoras futuras**")
        mejs = ["Modelos ensemble: Random Forest, XGBoost → mayor AUC esperado",
                "Balanceo de clases: SMOTE o class_weight → mejor Recall",
                "Validación cruzada k-fold estratificada → estimaciones más robustas",
                "Más variables clínicas: HbA1c, colesterol, glucosa en ayunas"]
        for m in mejs:
            st.markdown(f"<div style='font-size:13px;color:#8892A4;padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.04);'>🚀 {m}</div>", unsafe_allow_html=True)

    st.markdown('<hr class="sec">', unsafe_allow_html=True)
    st.markdown('<div class="conclusion-box" style="text-align:center;font-size:15px;">"El preprocesamiento es tan importante como el modelamiento.<br>Un dataset mal limpiado produce modelos malos sin importar cuán sofisticado sea el algoritmo."<br><br><b style="color:#4ECDC4;">— Garbage in, garbage out</b></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CALCULADORA
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "🩺 Calculadora de Riesgo":
    st.markdown('<div class="sec-eye">Herramienta interactiva</div><div class="sec-title">Calculadora de riesgo de <em style="color:#4ECDC4;font-style:italic;">diabetes</em></div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Basada en la Regresión Logística entrenada con el dataset Pima Indians Diabetes. Ingresa tus datos y obtén una estimación de probabilidad.</div>', unsafe_allow_html=True)

    col_form, col_res = st.columns([1, 1.1], gap="large")

    with col_form:
        st.markdown("**Datos personales**")
        nombre = st.text_input("Nombre", placeholder="Tu nombre")
        c1,c2 = st.columns(2)
        with c1: edad = st.number_input("Edad (años)", 21, 90, 33)
        with c2: sexo = st.selectbox("Sexo", ["Femenino","Masculino"])
        c3,c4 = st.columns(2)
        with c3: peso = st.number_input("Peso (kg)", 30.0, 200.0, 70.0, 0.5)
        with c4: altura = st.number_input("Altura (cm)", 100, 220, 160)

        bmi = peso / ((altura/100)**2)
        bmi_cat = "Normal ✓" if bmi<25 else "Sobrepeso" if bmi<30 else "Obesidad"
        bmi_col = "#4ECDC4" if bmi<25 else "#FFD166" if bmi<30 else "#FF6B6B"
        st.markdown(f'<div style="background:#1a2235;border:1px solid rgba(255,255,255,0.07);border-radius:8px;padding:8px 14px;display:flex;align-items:center;gap:12px;margin-bottom:1rem;"><span style="font-size:12px;color:#8892A4;">IMC</span><span style="font-size:1.3rem;font-weight:700;">{bmi:.1f}</span><span style="font-size:12px;font-weight:600;color:{bmi_col};">{bmi_cat}</span></div>', unsafe_allow_html=True)

        st.markdown("**Factores clínicos**")
        glucosa    = st.slider("Glucosa (mg/dL)", 60, 200, 100)
        presion    = st.slider("Presión arterial diastólica (mm Hg)", 40, 130, 72)
        c5,c6 = st.columns(2)
        with c5: embarazos = st.number_input("Embarazos previos", 0, 15, 0)
        with c6: pedigree  = st.slider("Antecedentes fam. (0–3)", 0.0, 3.0, 0.3, 0.1)
        insulina = st.slider("Insulina sérica (mu U/ml)", 0, 400, 80)

        calcular = st.button("Calcular probabilidad →")

    with col_res:
        if calcular:
            score = (-8.0 + 0.123*embarazos + 0.035*glucosa - 0.013*presion
                     + 0.090*bmi + 0.945*pedigree + 0.015*edad + 0.001*insulina)
            prob = round(1/(1+np.exp(-score))*100, 1)
            nivel = "Riesgo bajo" if prob<30 else "Riesgo moderado" if prob<60 else "Riesgo alto"
            color = "#06D6A0" if prob<30 else "#FFD166" if prob<60 else "#FF6B6B"
            css   = "risk-low" if prob<30 else "risk-mid" if prob<60 else "risk-high"

            st.markdown(f"""
            <div style="background:{'rgba(6,214,160,0.1)' if prob<30 else 'rgba(255,209,102,0.1)' if prob<60 else 'rgba(255,107,107,0.1)'};
                        border:1px solid {'rgba(6,214,160,0.3)' if prob<30 else 'rgba(255,209,102,0.3)' if prob<60 else 'rgba(255,107,107,0.3)'};
                        border-radius:12px;padding:1.2rem;margin-bottom:1rem;">
                <div style="font-size:12px;color:#8892A4;margin-bottom:4px;">Análisis para: {nombre if nombre else 'el paciente'}</div>
                <div style="font-family:'DM Serif Display',serif;font-size:2rem;color:{color};">{nivel}</div>
                <div style="font-size:14px;color:#8892A4;">Probabilidad estimada: <b style="color:{color};">{prob}%</b></div>
            </div>""", unsafe_allow_html=True)

            fig = go.Figure(go.Indicator(mode="gauge+number",value=prob,
                number=dict(suffix="%",font=dict(size=36,color=color)),
                gauge=dict(axis=dict(range=[0,100]),bar=dict(color=color,thickness=0.3),
                           bgcolor="#1a2235",
                           steps=[{"range":[0,30],"color":"rgba(6,214,160,0.1)"},
                                  {"range":[30,60],"color":"rgba(255,209,102,0.1)"},
                                  {"range":[60,100],"color":"rgba(255,107,107,0.1)"}])))
            fig.update_layout(paper_bgcolor="#111827",height=200,margin=dict(l=20,r=20,t=20,b=10),font=dict(color="#F0F4FF"))
            st.plotly_chart(fig, use_container_width=True)

            chips = ""
            for label,ok,warn in [
                (f"Glucosa: {glucosa}",glucosa<100,glucosa<126),
                (f"IMC: {bmi:.1f}",bmi<25,bmi<30),
                (f"Edad: {edad}",edad<35,edad<50),
                (f"Presión: {presion}",presion<80,presion<90),
                (f"Pedigree: {pedigree:.1f}",pedigree<0.5,pedigree<1.2),
            ]:
                c = ("f-ok" if ok else "f-warn" if warn else "f-bad")
                bg = ("rgba(6,214,160,0.1)" if ok else "rgba(255,209,102,0.1)" if warn else "rgba(255,107,107,0.1)")
                fc = ("#06D6A0" if ok else "#FFD166" if warn else "#FF6B6B")
                bc = ("rgba(6,214,160,0.3)" if ok else "rgba(255,209,102,0.3)" if warn else "rgba(255,107,107,0.3)")
                chips += f'<span style="display:inline-block;padding:3px 10px;border-radius:999px;font-size:11px;font-weight:500;background:{bg};color:{fc};border:1px solid {bc};margin:3px;">{label}</span>'
            st.markdown(chips, unsafe_allow_html=True)

            st.markdown('<div class="disclaimer">⚕ Resultado orientativo. No reemplaza una consulta médica. Basado en modelo de Regresión Logística entrenado con el dataset Pima Indians Diabetes (UCI ML Repository, Smith et al., 1988).</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#111827;border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:3rem;text-align:center;margin-top:2rem;">
                <div style="font-size:3rem;margin-bottom:1rem;">🩺</div>
                <div style="font-family:'DM Serif Display',serif;font-size:1.4rem;margin-bottom:0.5rem;">Completa el formulario</div>
                <div style="color:#8892A4;font-size:14px;">Llena tus datos y presiona<br><b style="color:#4ECDC4;">Calcular probabilidad</b></div>
            </div>""", unsafe_allow_html=True)
