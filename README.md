# DiabetesML — Dashboard Streamlit

Dashboard interactivo para calcular el riesgo de diabetes, basado en el análisis del dataset **Pima Indians Diabetes (UCI ML Repository)**.

## Proyecto

**Seminario de Ciencia de los Datos — Institución Universitaria Pascual Bravo**  
Temática: Impacto de factores clínicos y demográficos en la presencia de diabetes.

## Secciones del dashboard

- **🩺 Calculadora de riesgo** — formulario con datos personales y clínicos, resultado con gauge visual, explicación por factor y comparación con la población del dataset
- **📊 Análisis del dataset** — estadísticos descriptivos, prueba Shapiro-Wilk, datos faltantes, prueba de rachas, distribuciones por diagnóstico
- **🤖 Resultados del modelo** — comparación de 3 modelos ML (Regresión Logística, Árbol de Decisión, KNN), curvas ROC, PCA, métricas completas

## Instalación local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy en Streamlit Cloud (gratis)

1. Sube este repositorio a GitHub (con `app.py` y `requirements.txt`)
2. Ve a [share.streamlit.io](https://share.streamlit.io)
3. Conecta tu cuenta de GitHub
4. Selecciona el repositorio y el archivo `app.py`
5. Clic en **Deploy** — en 2 minutos tienes una URL pública

## Dataset

- **Fuente:** UCI Machine Learning Repository  
- **Nombre:** Pima Indians Diabetes Database  
- **Referencia:** Smith et al. (1988)  
- **Registros:** 768 · **Variables:** 9 · **Prevalencia diabetes:** 34.9%

## Modelo utilizado

Regresión Logística entrenada con el dataset procesado (imputación MICE, capping IQR).  
**ROC-AUC: 0.841 · Recall: 68.1% · Accuracy: 77.3%**

> ⚕ Los resultados de la calculadora son orientativos y no reemplazan una consulta médica.
