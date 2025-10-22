import streamlit as st

st.title("📘 Metodología y Datos (resumen)")
st.markdown(
    """
**Fuentes:** CNMC Open Data (CKAN) – datasets anual, trimestral, mensual, provincial e infraestructuras.

**Proceso:** descarga y limpieza automática → unificación de nombres de columnas → coerción numérica →
normalización Min-Max opcional → analítica avanzada (KPIs, mapas, correlaciones, TOPSIS).

**Gobernanza y calidad:** control de rangos temporales, filtros por operador/servicio, agregaciones por
territorio y verificación de métricas (pearson/spearman/kendall) antes de extraer conclusiones.

**Limitaciones:** datos provisionales y revisiones históricas (fusiones, adquisiciones) pueden introducir
saltos; se recomienda contrastar con notas metodológicas de CNMC para cada serie.
"""
)
