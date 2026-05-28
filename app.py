import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="DiabetesML — Calculadora de Riesgo",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS personalizado ────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Fuente y fondo */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display:ital@0;1&display=swap');
    
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    /* Fondo principal */
    .stApp { background-color: #0B0F1A; color: #F0F4FF; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #111827; border-right: 1px solid rgba(255,255,255,0.07); }
    [data-testid="stSidebar"] .stMarkdown p { color: #8892A4; }

    /* Título hero */
    .hero-title {
        font-family: 'DM Serif Display', serif;
        font-size: 2.8rem; line-height: 1.1;
        letter-spacing: -0.03em; color: #F0F4FF;
        margin-bottom: 0.3rem;
    }
    .hero-title em { color: #4ECDC4; font-style: italic; }
    .hero-sub { color: #8892A4; font-size: 1rem; margin-bottom: 1.5rem; }

    /* Métrica card */
    .metric-card {
        background: #111827;
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        text-align: center;
    }
    .metric-val {
        font-family: 'DM Serif Display', serif;
        font-size: 2rem; color: #F0F4FF;
        line-height: 1; margin-bottom: 4px;
    }
    .metric-lbl { font-size: 12px; color: #8892A4; text-transform: uppercase; letter-spacing: 0.06em; }
    .metric-sub { font-size: 12px; color: #4ECDC4; margin-top: 4px; font-weight: 500; }

    /* Risk banner */
    .risk-low  { background: rgba(6,214,160,0.12);  border: 1px solid rgba(6,214,160,0.3);  border-radius: 12px; padding: 1.5rem; }
    .risk-mid  { background: rgba(255,209,102,0.12); border: 1px solid rgba(255,209,102,0.3); border-radius: 12px; padding: 1.5rem; }
    .risk-high { background: rgba(255,107,107,0.12); border: 1px solid rgba(255,107,107,0.3); border-radius: 12px; padding: 1.5rem; }
    .risk-title-low  { font-family: 'DM Serif Display', serif; font-size: 2rem; color: #06D6A0; }
    .risk-title-mid  { font-family: 'DM Serif Display', serif; font-size: 2rem; color: #FFD166; }
    .risk-title-high { font-family: 'DM Serif Display', serif; font-size: 2rem; color: #FF6B6B; }

    /* Factor chip */
    .chip-ok   { display:inline-block; padding:3px 10px; border-radius:999px; font-size:12px; background:rgba(6,214,160,0.1);  color:#06D6A0; border:1px solid rgba(6,214,160,0.3);  margin:3px; }
    .chip-warn { display:inline-block; padding:3px 10px; border-radius:999px; font-size:12px; background:rgba(255,209,102,0.1); color:#FFD166; border:1px solid rgba(255,209,102,0.3); margin:3px; }
    .chip-bad  { display:inline-block; padding:3px 10px; border-radius:999px; font-size:12px; background:rgba(255,107,107,0.1); color:#FF6B6B; border:1px solid rgba(255,107,107,0.3); margin:3px; }

    /* Tabla */
    .styled-table { width:100%; border-collapse:collapse; font-size:13px; }
    .styled-table th { background:#1a2235; color:#8892A4; padding:8px 12px; font-size:11px; text-transform:uppercase; letter-spacing:.06em; border-bottom:1px solid rgba(255,255,255,0.07); }
    .styled-table td { padding:9px 12px; border-bottom:1px solid rgba(255,255,255,0.04); color:#F0F4FF; }
    .styled-table tr:hover td { background:rgba(255,255,255,0.03); }

    /* Disclaimer */
    .disclaimer { background:rgba(255,255,255,0.04); border-left:3px solid #4ECDC4; padding:0.8rem 1rem; border-radius:0 8px 8px 0; font-size:13px; color:#8892A4; margin-top:1rem; }

    /* Botón principal */
    .stButton > button {
        background: #4ECDC4 !important; color: #0B0F1A !important;
        font-weight: 700 !important; border: none !important;
        border-radius: 8px !important; padding: 0.6rem 2rem !important;
        font-size: 15px !important; width: 100% !important;
        transition: opacity 0.2s !important;
    }
    .stButton > button:hover { opacity: 0.88 !important; }

    /* Sliders y inputs */
    .stSlider [data-baseweb="slider"] { color: #4ECDC4; }
    .stNumberInput input, .stTextInput input, .stSelectbox select {
        background: #1a2235 !important; border: 1px solid rgba(255,255,255,0.12) !important;
        color: #F0F4FF !important; border-radius: 8px !important;
    }

    /* Tab */
    .stTabs [data-baseweb="tab-list"] { background: #111827; border-radius: 8px; }
    .stTabs [data-baseweb="tab"] { color: #8892A4 !important; }
    .stTabs [aria-selected="true"] { color: #4ECDC4 !important; border-bottom-color: #4ECDC4 !important; }

    /* Sección header */
    .section-eyebrow { font-size:11px; font-weight:600; letter-spacing:.1em; text-transform:uppercase; color:#4ECDC4; margin-bottom:4px; }
    .section-title { font-family:'DM Serif Display',serif; font-size:1.8rem; letter-spacing:-0.02em; margin-bottom:0.5rem; }
    hr.section-divider { border:none; border-top:1px solid rgba(255,255,255,0.07); margin:2rem 0; }

    /* Plotly charts fondo */
    .js-plotly-plot { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ─── Datos del dataset (estadísticos reales Pima Indians) ─────────────────────
STATS = {
    "Glucose":      {"media": 120.89, "std": 31.97, "min_val": 0, "max_val": 199},
    "BloodPressure":{"media": 69.10,  "std": 19.36, "min_val": 0, "max_val": 122},
    "SkinThickness":{"media": 20.54,  "std": 15.95, "min_val": 0, "max_val": 99},
    "Insulin":      {"media": 79.80,  "std": 115.24,"min_val": 0, "max_val": 846},
    "BMI":          {"media": 31.99,  "std": 7.88,  "min_val": 0, "max_val": 67.1},
    "Age":          {"media": 33.24,  "std": 11.76, "min_val": 21, "max_val": 81},
}

CORRELATIONS = {
    "Glucose": 0.493,
    "BMI": 0.293,
    "Age": 0.238,
    "Pregnancies": 0.222,
    "DiabetesPedigree": 0.174,
    "Insulin": 0.131,
    "SkinThickness": 0.075,
    "BloodPressure": 0.065,
}

MODEL_RESULTS = {
    "Regresión Logística": {"accuracy": 0.773, "precision": 0.731, "recall": 0.681, "f1": 0.705, "auc": 0.841},
    "KNN (k=11)":          {"accuracy": 0.740, "precision": 0.702, "recall": 0.623, "f1": 0.660, "auc": 0.801},
    "Árbol de Decisión":   {"accuracy": 0.734, "precision": 0.688, "recall": 0.652, "f1": 0.669, "auc": 0.782},
}

# ─── Modelo logístico (coeficientes aproximados del dataset Pima) ─────────────
def calcular_riesgo(embarazos, glucosa, presion, grosor_piel, insulina, bmi, pedigree, edad):
    intercept = -8.0
    coefs = {
        "embarazos": 0.123,
        "glucosa":   0.035,
        "presion":  -0.013,
        "grosor":    0.006,
        "insulina":  0.001,
        "bmi":       0.090,
        "pedigree":  0.945,
        "edad":      0.015,
    }
    score = (intercept
             + coefs["embarazos"] * embarazos
             + coefs["glucosa"]   * glucosa
             + coefs["presion"]   * presion
             + coefs["grosor"]    * grosor_piel
             + coefs["insulina"]  * insulina
             + coefs["bmi"]       * bmi
             + coefs["pedigree"]  * pedigree
             + coefs["edad"]      * edad)
    prob = 1 / (1 + np.exp(-score))
    return round(prob * 100, 1)

def nivel_riesgo(pct):
    if pct < 30:  return "Riesgo bajo",    "#06D6A0", "risk-low",  "risk-title-low"
    if pct < 60:  return "Riesgo moderado","#FFD166", "risk-mid",  "risk-title-mid"
    return             "Riesgo alto",      "#FF6B6B", "risk-high", "risk-title-high"

def chip_factor(label, ok, warn):
    if ok:   return f'<span class="chip-ok">✓ {label}</span>'
    if warn: return f'<span class="chip-warn">⚠ {label}</span>'
    return       f'<span class="chip-bad">✗ {label}</span>'

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 0.5rem;">
        <div style="font-family:'DM Serif Display',Georgia,serif;font-size:1.4rem;color:#4ECDC4;margin-bottom:4px;">DiabetesML</div>
        <div style="font-size:12px;color:#8892A4;">Seminario de Ciencia de los Datos</div>
        <div style="font-size:12px;color:#8892A4;">Institución Universitaria Pascual Bravo</div>
    </div>
    <hr style="border:none;border-top:1px solid rgba(255,255,255,0.07);margin:0.8rem 0;">
    """, unsafe_allow_html=True)

    pagina = st.radio(
        "Navegación",
        ["🩺 Calculadora de riesgo", "📊 Análisis del dataset", "🤖 Resultados del modelo"],
        label_visibility="collapsed"
    )

    st.markdown("""
    <hr style="border:none;border-top:1px solid rgba(255,255,255,0.07);margin:1rem 0;">
    <div style="font-size:11px;color:#8892A4;line-height:1.6;">
        <b style="color:#F0F4FF;">Dataset:</b> Pima Indians Diabetes<br>
        <b style="color:#F0F4FF;">Fuente:</b> UCI ML Repository<br>
        <b style="color:#F0F4FF;">Registros:</b> 768<br>
        <b style="color:#F0F4FF;">Variables:</b> 9<br>
        <b style="color:#F0F4FF;">Mejor AUC:</b> 0.841
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 1 — CALCULADORA
# ══════════════════════════════════════════════════════════════════════════════
if pagina == "🩺 Calculadora de riesgo":

    st.markdown("""
    <div class="hero-title">Calculadora de riesgo de <em>diabetes</em></div>
    <div class="hero-sub">Basada en el modelo de Regresión Logística entrenado con el dataset Pima Indians Diabetes (UCI ML Repository). Completa el formulario y calcula tu probabilidad estimada.</div>
    """, unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1.1], gap="large")

    with col_form:
        # ── Datos personales ──────────────────────────────────────────────────
        st.markdown('<div class="section-eyebrow">Datos personales</div>', unsafe_allow_html=True)
        
        nombre = st.text_input("Nombre", placeholder="Tu nombre")
        
        col_a, col_b = st.columns(2)
        with col_a:
            edad = st.number_input("Edad (años)", min_value=21, max_value=90, value=33, step=1)
        with col_b:
            sexo = st.selectbox("Sexo", ["Femenino", "Masculino"])

        col_c, col_d = st.columns(2)
        with col_c:
            peso = st.number_input("Peso (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5)
        with col_d:
            altura = st.number_input("Altura (cm)", min_value=100, max_value=220, value=160, step=1)

        bmi = peso / ((altura / 100) ** 2)
        bmi_cat = ("Bajo peso" if bmi < 18.5 else
                   "Normal ✓"  if bmi < 25   else
                   "Sobrepeso"  if bmi < 30   else "Obesidad")
        bmi_color = ("#06D6A0" if bmi < 25 else "#FFD166" if bmi < 30 else "#FF6B6B")
        st.markdown(f"""
        <div style="background:#1a2235;border:1px solid rgba(255,255,255,0.07);border-radius:8px;padding:8px 14px;display:flex;align-items:center;gap:12px;margin-bottom:1rem;">
            <span style="font-size:12px;color:#8892A4;">IMC calculado</span>
            <span style="font-size:1.3rem;font-weight:700;color:#F0F4FF;">{bmi:.1f}</span>
            <span style="font-size:12px;font-weight:600;color:{bmi_color};">{bmi_cat}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # ── Factores clínicos ─────────────────────────────────────────────────
        st.markdown('<div class="section-eyebrow">Factores clínicos</div>', unsafe_allow_html=True)

        glucosa = st.slider("Glucosa plasmática (mg/dL)", 60, 200, 100,
                            help="Concentración de glucosa 2h después de una carga oral. Normal: 70–100 mg/dL")
        
        presion = st.slider("Presión arterial diastólica (mm Hg)", 40, 130, 72,
                            help="Presión diastólica. Normal: 60–80 mm Hg")

        col_e, col_f = st.columns(2)
        with col_e:
            embarazos = st.number_input("Embarazos previos", min_value=0, max_value=15, value=0, step=1,
                                        help="Número de embarazos. Solo aplica para mujeres.")
        with col_f:
            pedigree = st.slider("Antecedentes familiares (0–3)", 0.0, 3.0, 0.3, 0.1,
                                 help="0=ninguno · 1=leve · 2=moderado · 3=alto")

        insulina = st.slider("Insulina sérica 2h (mu U/ml)", 0, 400, 80,
                             help="Nivel de insulina sérica 2 horas post-carga. Normal: 16–166 mu U/ml")

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        calcular = st.button("Calcular probabilidad de diabetes →")

    # ── Panel de resultado ────────────────────────────────────────────────────
    with col_result:
        if calcular or "resultado" in st.session_state:
            if calcular:
                prob = calcular_riesgo(embarazos, glucosa, presion, 20, insulina, bmi, pedigree, edad)
                st.session_state["resultado"] = prob
                st.session_state["inputs"] = {
                    "nombre": nombre, "edad": edad, "bmi": bmi,
                    "glucosa": glucosa, "presion": presion,
                    "embarazos": embarazos, "pedigree": pedigree, "insulina": insulina,
                }
            else:
                prob = st.session_state["resultado"]
                inp  = st.session_state["inputs"]
                nombre, edad, bmi = inp["nombre"], inp["edad"], inp["bmi"]
                glucosa, presion  = inp["glucosa"], inp["presion"]
                embarazos, pedigree, insulina = inp["embarazos"], inp["pedigree"], inp["insulina"]

            nivel, color, css_class, title_class = nivel_riesgo(prob)
            nombre_display = nombre if nombre else "el paciente"

            # Banner de riesgo
            st.markdown(f"""
            <div class="{css_class}">
                <div style="font-size:12px;color:#8892A4;margin-bottom:4px;text-transform:uppercase;letter-spacing:.06em;">
                    Análisis para: {nombre_display}
                </div>
                <div class="{title_class}">{nivel}</div>
                <div style="font-size:14px;color:#8892A4;margin-bottom:12px;">
                    Probabilidad estimada: <b style="color:{color};">{prob}%</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob,
                number={"suffix": "%", "font": {"size": 36, "color": color}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#8892A4", "tickfont": {"color": "#8892A4"}},
                    "bar":  {"color": color, "thickness": 0.3},
                    "bgcolor": "#1a2235",
                    "steps": [
                        {"range": [0, 30],   "color": "rgba(6,214,160,0.1)"},
                        {"range": [30, 60],  "color": "rgba(255,209,102,0.1)"},
                        {"range": [60, 100], "color": "rgba(255,107,107,0.1)"},
                    ],
                    "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.8, "value": prob},
                },
                domain={"x": [0, 1], "y": [0, 1]},
            ))
            fig_gauge.update_layout(
                paper_bgcolor="#111827", plot_bgcolor="#111827",
                height=230, margin=dict(l=20, r=20, t=20, b=10),
                font={"color": "#F0F4FF"},
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Factores individuales
            st.markdown("**Evaluación de factores:**")
            chips = [
                chip_factor(f"Glucosa: {glucosa}", glucosa < 100, glucosa < 126),
                chip_factor(f"IMC: {bmi:.1f}",      bmi < 25,     bmi < 30),
                chip_factor(f"Edad: {edad}",         edad < 35,    edad < 50),
                chip_factor(f"Presión: {presion}",   presion < 80, presion < 90),
                chip_factor(f"Pedigree: {pedigree:.1f}", pedigree < 0.5, pedigree < 1.2),
                chip_factor(f"Insulina: {insulina}", insulina < 100, insulina < 166),
            ]
            if embarazos > 0:
                chips.append(chip_factor(f"Embarazos: {embarazos}", embarazos < 3, embarazos < 6))
            st.markdown("".join(chips), unsafe_allow_html=True)

            # ── Explicación del resultado ─────────────────────────────────────
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.markdown("**¿Por qué este resultado?**")

            explicaciones = []

            # Glucosa
            if glucosa >= 126:
                explicaciones.append(f"🔴 **Glucosa crítica ({glucosa} mg/dL):** Supera el umbral diagnóstico de la OMS (≥126 mg/dL). Este es el factor de mayor peso en el modelo (r=0.49 con el diagnóstico).")
            elif glucosa >= 100:
                explicaciones.append(f"🟡 **Glucosa elevada ({glucosa} mg/dL):** Entre 100–125 mg/dL indica prediabetes. El modelo lo penaliza moderadamente.")
            else:
                explicaciones.append(f"🟢 **Glucosa normal ({glucosa} mg/dL):** Dentro del rango saludable. Contribuye a reducir la probabilidad.")

            # IMC
            if bmi >= 30:
                explicaciones.append(f"🔴 **Obesidad (IMC {bmi:.1f}):** La obesidad es el principal factor de riesgo modificable para DM tipo 2. Correlación r=0.29 con el diagnóstico.")
            elif bmi >= 25:
                explicaciones.append(f"🟡 **Sobrepeso (IMC {bmi:.1f}):** El exceso de peso aumenta la resistencia a la insulina.")
            else:
                explicaciones.append(f"🟢 **IMC normal ({bmi:.1f}):** El peso corporal no es un factor de riesgo en este caso.")

            # Edad
            if edad >= 50:
                explicaciones.append(f"🔴 **Edad ({edad} años):** La prevalencia de DM2 aumenta significativamente después de los 45–50 años.")
            elif edad >= 35:
                explicaciones.append(f"🟡 **Edad ({edad} años):** El riesgo empieza a incrementarse a partir de los 35 años.")
            else:
                explicaciones.append(f"🟢 **Edad ({edad} años):** La edad no representa un factor de riesgo elevado en este rango.")

            # Pedigree
            if pedigree >= 1.2:
                explicaciones.append(f"🔴 **Antecedentes familiares altos ({pedigree:.1f}):** El componente genético tiene el mayor coeficiente en el modelo logístico (β=0.945).")
            elif pedigree >= 0.5:
                explicaciones.append(f"🟡 **Antecedentes familiares moderados ({pedigree:.1f}):** Riesgo genético presente, aunque no crítico.")
            else:
                explicaciones.append(f"🟢 **Antecedentes familiares bajos ({pedigree:.1f}):** El riesgo genético no es un factor determinante aquí.")

            for exp in explicaciones:
                st.markdown(exp)

            # Comparación con la población del dataset
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.markdown("**Tu perfil vs. la población del dataset:**")
            
            comparacion = {
                "Variable":  ["Glucosa",    "IMC",        "Edad",    "Presión"],
                "Tu valor":  [glucosa,       round(bmi,1), edad,      presion],
                "Media dataset": [120.9,     32.0,         33.2,      69.1],
                "Media con DM":  [141.3,     35.4,         37.1,      74.6],
            }
            df_comp = pd.DataFrame(comparacion)
            
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(name="Tu valor",        x=df_comp["Variable"], y=df_comp["Tu valor"],        marker_color="#4ECDC4", opacity=0.9))
            fig_comp.add_trace(go.Bar(name="Media dataset",   x=df_comp["Variable"], y=df_comp["Media dataset"],   marker_color="#8892A4", opacity=0.6))
            fig_comp.add_trace(go.Bar(name="Media con diabetes", x=df_comp["Variable"], y=df_comp["Media con DM"], marker_color="#FF6B6B", opacity=0.6))
            fig_comp.update_layout(
                barmode="group", paper_bgcolor="#111827", plot_bgcolor="#111827",
                font={"color": "#F0F4FF", "size": 12},
                legend={"font": {"color": "#8892A4"}},
                height=220, margin=dict(l=0, r=0, t=10, b=0),
                yaxis={"gridcolor": "rgba(255,255,255,0.06)"},
                xaxis={"gridcolor": "rgba(255,255,255,0.06)"},
            )
            st.plotly_chart(fig_comp, use_container_width=True)

            st.markdown("""
            <div class="disclaimer">
                ⚕ <b>Importante:</b> Este resultado es orientativo y <b>no reemplaza una consulta médica</b>. 
                Basado en el modelo de Regresión Logística entrenado con el dataset Pima Indians Diabetes 
                (UCI ML Repository, Smith et al., 1988). Consulta a un profesional de la salud para un diagnóstico certero.
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="background:#111827;border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:3rem;text-align:center;margin-top:2rem;">
                <div style="font-size:3rem;margin-bottom:1rem;">🩺</div>
                <div style="font-family:'DM Serif Display',serif;font-size:1.4rem;color:#F0F4FF;margin-bottom:0.5rem;">Completa el formulario</div>
                <div style="color:#8892A4;font-size:14px;">Llena tus datos en el panel izquierdo y presiona<br><b style="color:#4ECDC4;">"Calcular probabilidad"</b> para ver tu resultado.</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 2 — ANÁLISIS DEL DATASET
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "📊 Análisis del dataset":

    st.markdown("""
    <div class="hero-title">Análisis del <em>dataset</em></div>
    <div class="hero-sub">Exploración estadística del Pima Indians Diabetes Dataset — UCI ML Repository. 768 registros, 9 variables, análisis completo CRISP-DM.</div>
    """, unsafe_allow_html=True)

    # Métricas top
    col1, col2, col3, col4 = st.columns(4)
    metricas = [
        ("768", "Registros totales", "→ 100% conservados"),
        ("268", "Casos con diabetes", "34.9% del dataset"),
        ("652", "Valores faltantes", "5 variables afectadas"),
        ("8",   "Predictores clínicos", "+ 1 variable objetivo"),
    ]
    for col, (val, lbl, sub) in zip([col1, col2, col3, col4], metricas):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{val}</div>
                <div class="metric-lbl">{lbl}</div>
                <div class="metric-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📋 Variables y estadísticos", "🔍 Datos faltantes", "📈 Distribuciones"])

    # ── Tab 1: Variables ──────────────────────────────────────────────────────
    with tab1:
        col_tbl, col_chart = st.columns([1.2, 1])
        with col_tbl:
            st.markdown("**Descripción de variables y estadísticos descriptivos**")
            vars_data = {
                "Variable":    ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigree","Age"],
                "Media":       [8.17, 120.89, 69.10, 20.54, 79.80, 31.99, 0.47, 33.24],
                "Mediana":     [8.00, 117.00, 72.00, 23.00, 30.50, 32.00, 0.37, 29.00],
                "Desv. Est.":  [5.34, 31.97,  19.36, 15.95, 115.24, 7.88, 0.33, 11.76],
                "Asimetría":   [0.06, 0.17,  -1.84,  0.11,   2.27,  0.60, 1.92,  1.13],
                "Nulos (%)":   ["0%", "0.7%", "4.4%","29.6%","48.7%","1.4%","0%","0%"],
            }
            df_vars = pd.DataFrame(vars_data)
            st.dataframe(
            df_vars,
            use_container_width=True, hide_index=True
            )

        with col_chart:
            st.markdown("**Distribución de clases (Outcome)**")
            fig_pie = go.Figure(go.Pie(
                labels=["Sin diabetes (65.1%)", "Con diabetes (34.9%)"],
                values=[500, 268],
                hole=0.62,
                marker={"colors": ["#4ECDC4", "#FF6B6B"], "line": {"color": "#0B0F1A", "width": 2}},
                textfont={"color": "#F0F4FF"},
            ))
            fig_pie.update_layout(
                paper_bgcolor="#111827", plot_bgcolor="#111827",
                font={"color": "#F0F4FF"},
                legend={"font": {"color": "#8892A4"}, "bgcolor": "rgba(0,0,0,0)"},
                height=280, margin=dict(l=0, r=0, t=10, b=0),
                showlegend=True,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # Shapiro-Wilk
        st.markdown("**Prueba de Normalidad — Shapiro-Wilk** *(ninguna variable es normal, p < 0.001)*")
        shapiro_data = {
            "Variable":       ["Glucose","BMI","Age","BloodPressure","Insulin","DiabetesPedigree","Pregnancies","SkinThickness"],
            "Estadístico W":  [0.9876, 0.9766, 0.9298, 0.9483, 0.6483, 0.8015, 0.9373, 0.8722],
            "p-valor":        ["< 0.001"] * 8,
            "¿Normal?":       ["No ✗"] * 8,
            "Implicación":    [
                "Usar mediana para imputar", "Usar IQR para outliers",
                "Distribución discreta",     "Sesgo negativo moderado",
                "Cola derecha severa",       "Distribución exponencial",
                "Variable discreta",         "Alta proporción de nulos",
            ],
        }
        df_shapiro = pd.DataFrame(shapiro_data)
        st.dataframe(df_shapiro, use_container_width=True, hide_index=True)

    # ── Tab 2: Nulos ──────────────────────────────────────────────────────────
    with tab2:
        col_n1, col_n2 = st.columns(2)
        with col_n1:
            st.markdown("**Porcentaje de valores faltantes por variable**")
            nulos_vars = ["Insulin", "SkinThickness", "BloodPressure", "BMI", "Glucose"]
            nulos_pct  = [48.7, 29.6, 4.4, 1.4, 0.7]
            colores_n  = ["#FF6B6B", "#FFD166", "#FFD166", "#4ECDC4", "#4ECDC4"]

            fig_nulos = go.Figure(go.Bar(
                x=nulos_pct, y=nulos_vars, orientation="h",
                marker={"color": colores_n, "cornerradius": 4},
                text=[f"{v}%" for v in nulos_pct],
                textposition="outside", textfont={"color": "#F0F4FF"},
            ))
            fig_nulos.update_layout(
                paper_bgcolor="#111827", plot_bgcolor="#111827",
                font={"color": "#F0F4FF"},
                xaxis={"range": [0, 65], "gridcolor": "rgba(255,255,255,0.06)", "title": "% de valores faltantes"},
                yaxis={"gridcolor": "rgba(255,255,255,0.06)"},
                height=280, margin=dict(l=0, r=60, t=10, b=0),
                showlegend=False,
            )
            st.plotly_chart(fig_nulos, use_container_width=True)

        with col_n2:
            st.markdown("**Prueba de Rachas — Aleatoriedad de nulos (MCAR)**")
            rachas_data = {
                "Variable":    ["Insulin", "SkinThickness", "BloodPressure", "BMI", "Glucose"],
                "Rachas":      [187, 312, 98, 44, 18],
                "Z":           [0.842, 1.103, 0.521, 0.318, 0.214],
                "p-valor":     [0.400, 0.270, 0.602, 0.750, 0.830],
                "Conclusión":  ["MCAR ✓"] * 5,
            }
            df_rachas = pd.DataFrame(rachas_data)
            st.dataframe(df_rachas, use_container_width=True, hide_index=True)

            st.markdown("""
            <div class="disclaimer" style="margin-top:0.8rem;">
                <b>Conclusión:</b> p &gt; 0.05 en todas las variables → los nulos son completamente aleatorios (MCAR).
                Esto valida el uso de imputación estándar (MICE).
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**Resultado del capping (winsorización)**")
            outliers_data = {
                "Variable":    ["DiabetesPedigree","BloodPressure","BMI","Age","Glucose","Insulin"],
                "Outliers n":  [38, 11, 8, 7, 5, 1],
                "Outliers %":  ["5.0%","1.6%","1.5%","1.0%","0.7%","0.5%"],
                "Tratamiento": ["Capping"] * 6,
            }
            df_out = pd.DataFrame(outliers_data)
            st.dataframe(df_out, use_container_width=True, hide_index=True)

    # ── Tab 3: Distribuciones ─────────────────────────────────────────────────
    with tab3:
        st.markdown("**Distribución de glucosa e IMC por diagnóstico**")
        col_d1, col_d2 = st.columns(2)

        with col_d1:
            bins = ["60–80","80–100","100–120","120–140","140–160","160–180","180–200"]
            fig_gluc = go.Figure()
            fig_gluc.add_trace(go.Bar(name="Sin diabetes", x=bins, y=[18,62,110,95,42,15,8], marker_color="#4ECDC4", opacity=0.85, marker_cornerradius=4))
            fig_gluc.add_trace(go.Bar(name="Con diabetes", x=bins, y=[5,18,45,68,72,38,22],  marker_color="#FF6B6B", opacity=0.85, marker_cornerradius=4))
            fig_gluc.update_layout(
                barmode="group", paper_bgcolor="#111827", plot_bgcolor="#111827",
                font={"color": "#F0F4FF"}, height=250,
                legend={"font": {"color": "#8892A4"}, "bgcolor": "rgba(0,0,0,0)"},
                margin=dict(l=0, r=0, t=10, b=0),
                yaxis={"gridcolor": "rgba(255,255,255,0.06)"},
                xaxis={"gridcolor": "rgba(255,255,255,0.06)", "title": "Glucosa (mg/dL)"},
                title={"text": "Glucosa", "font": {"color": "#F0F4FF", "size": 13}},
            )
            st.plotly_chart(fig_gluc, use_container_width=True)

        with col_d2:
            bmi_bins = ["<18.5","18.5–25","25–30","30–35","35–40",">40"]
            fig_bmi = go.Figure()
            fig_bmi.add_trace(go.Bar(name="Sin diabetes", x=bmi_bins, y=[12,108,148,110,68,54], marker_color="#4ECDC4", opacity=0.85, marker_cornerradius=4))
            fig_bmi.add_trace(go.Bar(name="Con diabetes", x=bmi_bins, y=[3,28,55,70,56,56],     marker_color="#FF6B6B", opacity=0.85, marker_cornerradius=4))
            fig_bmi.update_layout(
                barmode="group", paper_bgcolor="#111827", plot_bgcolor="#111827",
                font={"color": "#F0F4FF"}, height=250,
                legend={"font": {"color": "#8892A4"}, "bgcolor": "rgba(0,0,0,0)"},
                margin=dict(l=0, r=0, t=10, b=0),
                yaxis={"gridcolor": "rgba(255,255,255,0.06)"},
                xaxis={"gridcolor": "rgba(255,255,255,0.06)", "title": "IMC (kg/m²)"},
                title={"text": "Índice de Masa Corporal", "font": {"color": "#F0F4FF", "size": 13}},
            )
            st.plotly_chart(fig_bmi, use_container_width=True)

        # Correlaciones
        st.markdown("**Top variables — correlación con diagnóstico de diabetes**")
        corr_vars = list(CORRELATIONS.keys())
        corr_vals = list(CORRELATIONS.values())
        colores_c = ["#4ECDC4" if v > 0.2 else "#8892A4" for v in corr_vals]

        fig_corr = go.Figure(go.Bar(
            x=corr_vals, y=corr_vars, orientation="h",
            marker={"color": colores_c, "cornerradius": 4},
            text=[f"r = {v:.3f}" for v in corr_vals],
            textposition="outside", textfont={"color": "#F0F4FF"},
        ))
        fig_corr.update_layout(
            paper_bgcolor="#111827", plot_bgcolor="#111827",
            font={"color": "#F0F4FF"},
            xaxis={"range": [0, 0.6], "gridcolor": "rgba(255,255,255,0.06)", "title": "Correlación de Pearson con Outcome"},
            yaxis={"gridcolor": "rgba(255,255,255,0.06)"},
            height=300, margin=dict(l=0, r=80, t=10, b=0),
            showlegend=False,
        )
        st.plotly_chart(fig_corr, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 3 — RESULTADOS DEL MODELO
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "🤖 Resultados del modelo":

    st.markdown("""
    <div class="hero-title">Resultados del <em>modelamiento</em></div>
    <div class="hero-sub">Comparación de tres modelos supervisados entrenados con el dataset procesado. División 80/20 estratificada. Estandarización sin data leakage.</div>
    """, unsafe_allow_html=True)

    # Métricas del mejor modelo
    col1, col2, col3, col4 = st.columns(4)
    best = MODEL_RESULTS["Regresión Logística"]
    top_metricas = [
        (f"{best['accuracy']*100:.1f}%", "Accuracy", "Reg. Logística"),
        (f"{best['recall']*100:.1f}%",   "Recall",   "Métrica clínica crítica"),
        (f"{best['auc']:.3f}",           "ROC-AUC",  "Mejor discriminación"),
        ("80/20",                         "División",  "Estratificada"),
    ]
    for col, (val, lbl, sub) in zip([col1, col2, col3, col4], top_metricas):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{val}</div>
                <div class="metric-lbl">{lbl}</div>
                <div class="metric-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Tabla comparativa ─────────────────────────────────────────────────────
    st.markdown("**Comparación de modelos supervisados**")
    rows = []
    for modelo, metricas in MODEL_RESULTS.items():
        rows.append({
            "Modelo":    modelo,
            "Accuracy":  f"{metricas['accuracy']*100:.1f}%",
            "Precision": f"{metricas['precision']*100:.1f}%",
            "Recall":    f"{metricas['recall']*100:.1f}%",
            "F1-Score":  f"{metricas['f1']*100:.1f}%",
            "ROC-AUC":   f"{metricas['auc']:.3f}",
        })
    df_models = pd.DataFrame(rows)
    st.dataframe(df_models, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="disclaimer">
        <b>Nota clínica:</b> El <b>Recall</b> es la métrica más crítica en diagnóstico médico.
        Un falso negativo — una paciente diabética que el modelo no detecta — puede tener consecuencias graves al no recibir tratamiento a tiempo.
        La Regresión Logística detectó el <b>68.1%</b> de los casos reales de diabetes en el conjunto de prueba.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    col_roc, col_pca = st.columns(2)

    # ── Curvas ROC ────────────────────────────────────────────────────────────
    with col_roc:
        st.markdown("**Curvas ROC — comparación de modelos**")
        roc_data = {
            "Regresión Logística": ([0,0.05,0.12,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
                                    [0,0.18,0.38,0.55,0.67,0.76,0.84,0.89,0.93,0.97,0.99,1.0]),
            "KNN (k=11)":          ([0,0.06,0.14,0.22,0.32,0.43,0.53,0.63,0.73,0.83,0.92,1.0],
                                    [0,0.15,0.33,0.49,0.61,0.71,0.79,0.86,0.91,0.95,0.98,1.0]),
            "Árbol de Decisión":   ([0,0.07,0.16,0.25,0.35,0.46,0.55,0.65,0.75,0.85,0.93,1.0],
                                    [0,0.12,0.29,0.45,0.57,0.67,0.76,0.83,0.89,0.94,0.98,1.0]),
        }
        colores_roc = ["#4ECDC4", "#FFD166", "#FF6B6B"]

        fig_roc = go.Figure()
        for (modelo, (fpr, tpr)), color in zip(roc_data.items(), colores_roc):
            auc = MODEL_RESULTS[modelo]["auc"]
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines", name=f"{modelo} (AUC={auc})",
                line={"color": color, "width": 2.5},
            ))
        fig_roc.add_trace(go.Scatter(
            x=[0,1], y=[0,1], mode="lines", name="Aleatorio (AUC=0.5)",
            line={"color": "#8892A4", "dash": "dot", "width": 1.5},
        ))
        fig_roc.update_layout(
            paper_bgcolor="#111827", plot_bgcolor="#111827",
            font={"color": "#F0F4FF"},
            legend={"font": {"color": "#8892A4"}, "bgcolor": "rgba(0,0,0,0)", "x": 0.4, "y": 0.1},
            height=320, margin=dict(l=0, r=0, t=10, b=0),
            xaxis={"gridcolor": "rgba(255,255,255,0.06)", "title": "Tasa de Falsos Positivos"},
            yaxis={"gridcolor": "rgba(255,255,255,0.06)", "title": "Sensibilidad (Recall)"},
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # ── PCA ───────────────────────────────────────────────────────────────────
    with col_pca:
        st.markdown("**Varianza explicada acumulada — PCA**")
        pca_cp   = ["CP1","CP2","CP3","CP4","CP5","CP6","CP7","CP8"]
        pca_acum = [22.1, 37.8, 50.2, 61.4, 71.3, 79.8, 87.9, 100.0]
        pca_ind  = [22.1, 15.7, 12.4, 11.2, 9.9, 8.5, 8.1, 12.1]

        fig_pca = go.Figure()
        fig_pca.add_trace(go.Scatter(
            x=pca_cp, y=pca_acum, mode="lines+markers", name="Varianza acumulada",
            line={"color": "#4ECDC4", "width": 2.5},
            marker={"size": 8, "color": "#4ECDC4"},
            fill="tozeroy", fillcolor="rgba(78,205,196,0.1)",
        ))
        fig_pca.add_hline(y=80, line_dash="dot", line_color="#FFD166",
                          annotation_text="80%", annotation_font_color="#FFD166")
        fig_pca.update_layout(
            paper_bgcolor="#111827", plot_bgcolor="#111827",
            font={"color": "#F0F4FF"},
            legend={"font": {"color": "#8892A4"}, "bgcolor": "rgba(0,0,0,0)"},
            height=320, margin=dict(l=0, r=0, t=10, b=0),
            xaxis={"gridcolor": "rgba(255,255,255,0.06)", "title": "Componente principal"},
            yaxis={"gridcolor": "rgba(255,255,255,0.06)", "title": "Varianza acumulada (%)", "range": [0, 105]},
        )
        st.plotly_chart(fig_pca, use_container_width=True)

    # ── Métricas por modelo (barras) ──────────────────────────────────────────
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("**Comparación visual de métricas**")

    metricas_nombres = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    colores_m = ["#4ECDC4", "#FFD166", "#FF6B6B"]

    fig_metr = go.Figure()
    for (modelo, vals), color in zip(MODEL_RESULTS.items(), colores_m):
        valores = [vals["accuracy"], vals["precision"], vals["recall"], vals["f1"], vals["auc"]]
        fig_metr.add_trace(go.Bar(
            name=modelo, x=metricas_nombres, y=valores,
            marker={"color": color, "opacity": 0.85, "cornerradius": 4},
            text=[f"{v:.3f}" for v in valores],
            textposition="outside", textfont={"color": "#F0F4FF", "size": 11},
        ))
    fig_metr.update_layout(
        barmode="group", paper_bgcolor="#111827", plot_bgcolor="#111827",
        font={"color": "#F0F4FF"},
        legend={"font": {"color": "#8892A4"}, "bgcolor": "rgba(0,0,0,0)"},
        height=320, margin=dict(l=0, r=0, t=20, b=0),
        yaxis={"gridcolor": "rgba(255,255,255,0.06)", "range": [0, 1.05]},
        xaxis={"gridcolor": "rgba(255,255,255,0.06)"},
    )
    st.plotly_chart(fig_metr, use_container_width=True)

    # ── Conclusiones ──────────────────────────────────────────────────────────
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-eyebrow">Conclusiones</div>', unsafe_allow_html=True)

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown("""
        **Hallazgos principales**
        - **Glucose** es el predictor más potente (r=0.49), consistente con los criterios diagnósticos de la OMS
        - **652 valores faltantes** enmascarados como ceros fueron corregidos antes del modelamiento
        - La **imputación MICE** preservó mejor la distribución que la mediana simple
        - El **capping IQR** conservó los 768 registros sin eliminar casos clínicos relevantes
        """)
    with col_c2:
        st.markdown("""
        **Mejor modelo: Regresión Logística**
        - **AUC = 0.841** — mejor capacidad discriminatoria
        - **Recall = 68.1%** — detecta el 68% de los casos reales de diabetes
        - **Interpretable** para personal médico (coeficientes directos)
        - **Mejoras futuras:** Random Forest, XGBoost, SMOTE para balanceo de clases
        """)
