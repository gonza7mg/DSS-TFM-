import streamlit as st

st.title("📊 Resultados y Mapas")
st.markdown(
    """
En la pestaña principal encontrarás un panel completo con:

* **KPIs ejecutivos** por operador, servicio o paquete.
* **Mapas provinciales** y desglose por CCAA con datos agregados dinámicamente.
* **Matriz de correlaciones** y dispersión para detectar dependencias clave.
* **Ranking TOPSIS** configurable para priorizar operadores, tecnologías o segmentos.

Utiliza los filtros de la barra lateral (años, servicios, operadores…) para ajustar el contexto
de negocio y después navega por las pestañas del panel para generar las visualizaciones en vivo.
"""
)
