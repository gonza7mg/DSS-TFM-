# app.py  — DSS Telecom CNMC (descarga, limpieza, persistencia y análisis)
import io, zipfile, base64, datetime as dt
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import streamlit as st

# --- utilidades del proyecto ---
from utils.mcdm import topsis_rank
from utils.cnmc_ckan import fetch_resource
from utils.data_prep import (
    unify_columns_lower,
    clean_strings,
    normalize_minmax,
)

# ------------------------------------------------------------
# Config general
# ------------------------------------------------------------
st.set_page_config(page_title="DSS Telecomunicaciones – CNMC", layout="wide", page_icon="📶")
st.title("📶 DSS Telecomunicaciones – CNMC – Panel Directivo")

# Recursos CKAN que vamos a “congelar” como CSV limpios
RESOURCES = {
    "anual_datos_generales": "5e2d8f37-2385-4774-82ec-365cd83d65bd",
    "anual_mercados": "7afbf769-655d-4b43-b49f-95c2919ec1fe",
    "mensual": "3632297f-07d8-480c-aca5-c987dcde0ccb",
    "provinciales": "1efe6d64-72a8-4f45-a36c-691054f3e277",
    "trimestrales": "5da45f2f-e596-4940-b682-eab18e85288a",
    "infraestructuras": "baab2a5e-cc52-4704-a799-a28b19223a3b",
}

CSV_PATHS = {
    "Anual – Datos generales": "data/clean/anual_datos_generales_clean.csv",
    "Anual – Mercados": "data/clean/anual_mercados_clean.csv",
    "Mensual": "data/clean/mensual_clean.csv",
    "Provinciales": "data/clean/provinciales_clean.csv",
    "Trimestrales": "data/clean/trimestrales_clean.csv",
    "Infraestructuras": "data/clean/infraestructuras_clean.csv",
}

FILTER_CANDIDATES = [
    "servicio",
    "concepto",
    "operador",
    "segmento",
    "tipo_de_paquete",
    "tipo_de_ingreso",
    "tipo_de_cliente",
    "tipo_de_trafico",
    "tipo_de_trafico_de_mensaje",
    "tipo_de_contrato",
    "tipo_de_tarifa",
    "tipo_de_mercado",
    "tipo_de_oferta",
    "tipo_de_acceso",
    "tecnologia_de_acceso",
    "tecnologia_de_acceso_baf",
    "ccaa",
    "provincia",
]

DIMENSION_CANDIDATES = [
    "operador",
    "servicio",
    "concepto",
    "segmento",
    "ccaa",
    "provincia",
    "tipo_de_paquete",
    "tipo_de_ingreso",
    "tecnologia_de_acceso",
]

VALUE_EXCLUDE = {
    "_id",
    "anno",
    "mes",
    "trimestre",
    "mes_num",
    "trimestre_num",
}

@st.cache_data(show_spinner=False)
def load_province_centroids() -> pd.DataFrame:
    path = Path("data/reference_province_centroids.csv")
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=["provincia", "lat", "lon"])


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_object_dtype(out[col]):
            series = out[col].astype(str).str.strip()
            series = series.replace({"nan": np.nan, "None": np.nan, "": np.nan})
            series = series.str.replace("%", "", regex=False).str.replace(",", ".", regex=False)
            out[col] = pd.to_numeric(series, errors="ignore")
    return out


def append_period_column(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    out = df.copy()
    if "mes" in out.columns:
        out["__period__"] = pd.to_datetime(out["mes"], format="%Y-%m", errors="coerce")
        return out, "__period__"
    if "trimestre" in out.columns:
        try:
            period_idx = pd.PeriodIndex(out["trimestre"].astype(str), freq="Q")
            out["__period__"] = period_idx.to_timestamp()
        except Exception:
            out["__period__"] = pd.NaT
        return out, "__period__"
    if "anno" in out.columns:
        out["__period__"] = pd.to_datetime(out["anno"].astype(str), format="%Y", errors="coerce")
        return out, "__period__"
    return out, None


def candidate_value_columns(df: pd.DataFrame) -> list[str]:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    values = []
    for col in numeric_cols:
        if col in VALUE_EXCLUDE or col.endswith("_id"):
            continue
        if df[col].dropna().nunique() <= 1:
            continue
        values.append(col)
    return values


def prettify_label(text: str) -> str:
    return text.replace("_", " ").title()


def format_metric(value: float) -> str:
    if value is None or pd.isna(value):
        return "-"
    magnitude = abs(value)
    if magnitude >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f} B"
    if magnitude >= 1_000_000:
        return f"{value / 1_000_000:.2f} M"
    if magnitude >= 1_000:
        return f"{value / 1_000:.2f} k"
    return f"{value:,.0f}".replace(",", ".")


def compute_time_series(df: pd.DataFrame, metric: str, period_col: str | None) -> pd.Series | None:
    if not period_col or metric not in df.columns:
        return None
    series = df[[period_col, metric]].dropna()
    if series.empty:
        return None
    grouped = series.groupby(series[period_col]).sum().sort_index()
    return grouped.iloc[:, 0]


def compute_delta(series: pd.Series | None) -> float | None:
    if series is None or len(series) < 2:
        return None
    last, previous = series.iloc[-1], series.iloc[-2]
    if previous == 0:
        return None
    return (last - previous) / abs(previous)


def aggregate_dimension(df: pd.DataFrame, dimension: str, metric: str, top_n: int = 15) -> pd.DataFrame:
    data = df.copy()
    if dimension not in data.columns or metric not in data.columns:
        return pd.DataFrame(columns=[dimension, metric])
    data[dimension] = data[dimension].fillna("Sin dato")
    grouped = data.groupby(dimension, dropna=False)[metric].sum().sort_values(ascending=False).head(top_n)
    return grouped.reset_index()

# ------------------------------------------------------------
# Funciones de ETL (Descargar → Limpiar → (Opcional) Normalizar)
# ------------------------------------------------------------
def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza mínima común a todos los recursos."""
    df = unify_columns_lower(df)
    text_cols = [
        "servicio","concepto","operador","tipo_de_paquete","tipo_de_ingreso",
        "provincia","ccaa","tecnología_de_acceso","tipo_de_ba_mayorista",
        "tipo_de_estaciones_base","unidades"
    ]
    text_cols = [c for c in text_cols if c in df.columns]
    df = clean_strings(df, text_cols)
    return df

def download_and_clean_all() -> dict[str, bytes]:
    """Descarga y limpia todos los recursos. Devuelve {nombre_csv: bytes_csv}."""
    out: dict[str, bytes] = {}
    for name, rid in RESOURCES.items():
        df = fetch_resource(rid)
        dfc = basic_clean(df)

        # --- Si quieres normalización 0–1 rápida descomenta: ---
        # num_cols = [c for c in dfc.columns if pd.api.types.is_numeric_dtype(dfc[c])]
        # if num_cols:
        #     rng = (dfc[num_cols].max() - dfc[num_cols].min()).replace(0, 1)
        #     dfc[num_cols] = (dfc[num_cols] - dfc[num_cols].min()) / rng

        csv_bytes = dfc.to_csv(index=False).encode("utf-8")
        out[f"{name}_clean.csv"] = csv_bytes
    return out

def make_zip(files_dict: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname, content in files_dict.items():
            zf.writestr(fname, content)
    return buf.getvalue()

# ------------------------------------------------------------
# Guardado automático en GitHub (opcional, con secrets)
# ------------------------------------------------------------
def github_put_file(owner, repo, branch, path, content_bytes, token):
    """Crea/actualiza archivo en GitHub (PUT /repos/:owner/:repo/contents/:path)."""
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"

    # comprobar si existe para obtener sha
    r = requests.get(url, params={"ref": branch}, headers={"Authorization": f"Bearer {token}"})
    sha = r.json().get("sha") if r.status_code == 200 else None

    payload = {
        "message": f"Update {path}",
        "content": base64.b64encode(content_bytes).decode("utf-8"),
        "branch": branch,
    }
    if sha:
        payload["sha"] = sha

    r = requests.put(url, headers={"Authorization": f"Bearer {token}"}, json=payload)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"GitHub PUT failed for {path}: {r.status_code} {r.text}")

def push_all_to_github(files_dict: dict[str, bytes], subdir: str = "data/clean/"):
    """Empuja todos los CSV a tu repo usando secretos en Streamlit Cloud."""
    owner_repo = st.secrets["GITHUB_REPO"]           # ej: "usuario/DSS-TFM-"
    branch     = st.secrets.get("GITHUB_BRANCH", "main")
    token      = st.secrets["GITHUB_TOKEN"]
    owner, repo = owner_repo.split("/", 1)
    for fname, content in files_dict.items():
        github_put_file(owner, repo, branch, f"{subdir}{fname}", content, token)

# ------------------------------------------------------------
# Sidebar: control de datos persistentes
# ------------------------------------------------------------
st.sidebar.header("Fuente de datos")
modo = st.sidebar.radio("Selecciona", ["CSV (repositorio)", "A futuro: API CNMC"], index=0)

st.sidebar.subheader("Persistencia de datos (CNMC → CSV)")
c1, c2 = st.sidebar.columns(2)
btn_fetch = c1.button("⬇️ Descargar+limpiar")
btn_zip   = c2.button("💾 Generar ZIP")
btn_push  = st.sidebar.button("⬆️ Guardar en GitHub (data/clean/)")  # requiere secrets

if "CNMC_FILES" not in st.session_state:
    st.session_state["CNMC_FILES"] = None

if btn_fetch:
    with st.spinner("Descargando y limpiando datasets CNMC…"):
        files = download_and_clean_all()
        st.session_state["CNMC_FILES"] = files
        st.success(f"Listo: {len(files)} CSV limpios preparados.")

if st.session_state["CNMC_FILES"] and btn_zip:
    zbytes = make_zip(st.session_state["CNMC_FILES"])
    st.download_button(
        "Descargar data_clean.zip",
        data=zbytes,
        file_name="data_clean.zip",
        mime="application/zip",
        use_container_width=True
    )

if st.session_state["CNMC_FILES"] and btn_push:
    try:
        push_all_to_github(st.session_state["CNMC_FILES"], subdir="data/clean/")
        st.success("CSV guardados en tu repositorio en data/clean/ ✅")
    except Exception as e:
        st.error(f"No se pudieron subir a GitHub: {e}")
        st.info("Configura secrets: GITHUB_REPO, GITHUB_TOKEN (y opcional GITHUB_BRANCH).")

# ------------------------------------------------------------
# Carga de CSV “congelados” desde el repo
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv(path: str):
    return pd.read_csv(path)

dataset_name = st.sidebar.selectbox("Dataset", list(CSV_PATHS.keys()))
path = CSV_PATHS[dataset_name]

df = None
if modo.startswith("CSV"):
    try:
        df = load_csv(path)
        st.success(f"{dataset_name}: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    except Exception as e:
        st.error(f"No se pudo cargar {path}. Ejecuta la descarga y sube los CSV. Detalle: {e}")
else:
    st.info("La carga directa por API se activará en la siguiente iteración.")


# ------------------------------------------------------------
# Analítica interactiva y tablero de mando
# ------------------------------------------------------------
if df is not None:
    df = coerce_numeric(df)
    df, period_col = append_period_column(df)
    raw_df = df.copy()

    with st.expander("📋 Vista previa y esquema de columnas"):
        st.write("Columnas disponibles: " + ", ".join(raw_df.columns))
        st.dataframe(raw_df.head(50), use_container_width=True)

    df_filtered = raw_df.copy()

    # Ventanas temporales según granularidad del dataset
    if "mes" in raw_df.columns:
        period_values = pd.to_datetime(raw_df["__period__"].dropna().sort_values().unique())
        if len(period_values):
            start_default = pd.Timestamp(period_values[0]).to_pydatetime()
            end_default = pd.Timestamp(period_values[-1]).to_pydatetime()
            start_date, end_date = st.sidebar.slider(
                "Ventana temporal",
                min_value=start_default,
                max_value=end_default,
                value=(start_default, end_default),
                format="YYYY-MM",
            )
            mask = (df_filtered["__period__"] >= start_date) & (df_filtered["__period__"] <= end_date)
            df_filtered = df_filtered[mask]
    elif "trimestre" in raw_df.columns:
        period_values = pd.to_datetime(raw_df["__period__"].dropna().sort_values().unique())
        if len(period_values):
            start_default = pd.Timestamp(period_values[0]).to_pydatetime()
            end_default = pd.Timestamp(period_values[-1]).to_pydatetime()
            start_date, end_date = st.sidebar.slider(
                "Ventana temporal",
                min_value=start_default,
                max_value=end_default,
                value=(start_default, end_default),
                format="YYYY-MM",
            )
            mask = (df_filtered["__period__"] >= start_date) & (df_filtered["__period__"] <= end_date)
            df_filtered = df_filtered[mask]
    elif "anno" in raw_df.columns:
        years = sorted(raw_df["anno"].dropna().astype(int).unique())
        if years:
            start_year, end_year = st.sidebar.slider(
                "Rango de años",
                min_value=int(years[0]),
                max_value=int(years[-1]),
                value=(int(years[0]), int(years[-1])),
            )
            df_filtered = df_filtered[(df_filtered["anno"] >= start_year) & (df_filtered["anno"] <= end_year)]

    # Filtros categóricos dinámicos
    with st.sidebar.expander("Filtros analíticos", expanded=False):
        for col in FILTER_CANDIDATES:
            if col in raw_df.columns:
                options = raw_df[col].dropna().astype(str).unique()
                if len(options) == 0:
                    continue
                options = sorted(options)
                selection = st.multiselect(prettify_label(col), options, key=f"flt_{col}")
                if selection:
                    df_filtered = df_filtered[df_filtered[col].astype(str).isin(selection)]

    registros, columnas = df_filtered.shape
    if registros == 0:
        st.warning("No hay registros tras aplicar los filtros. Ajusta los parámetros para continuar.")
    else:
        value_cols_raw = candidate_value_columns(df_filtered)
        norm_cols = value_cols_raw.copy()
        df_normalized = normalize_minmax(df_filtered, norm_cols) if norm_cols else df_filtered
        use_norm = False
        if norm_cols:
            use_norm = st.sidebar.toggle(
                "Trabajar con métricas normalizadas (0-1)",
                value=False,
                help="Normaliza dinámicamente las métricas numéricas (Min-Max) para análisis comparables.",
            )
        df_analysis = df_normalized if use_norm and norm_cols else df_filtered
        value_cols_analysis = candidate_value_columns(df_analysis)
        dims_available = [
            d
            for d in DIMENSION_CANDIDATES
            if d in df_filtered.columns and df_filtered[d].notna().nunique() > 0
        ]

        st.markdown("### Resumen del dataset filtrado")
        st.write(
            f"**Registros activos:** {registros:,} · "
            f"**Columnas:** {columnas} · "
            f"**Métricas cuantitativas (sin normalizar):** {len(value_cols_raw)}"
        )

        tab_kpi, tab_map, tab_corr, tab_topsis, tab_detail = st.tabs([
            "📊 KPIs y tendencias",
            "🗺️ Inteligencia territorial",
            "🔗 Correlaciones",
            "⚖️ Ranking TOPSIS",
            "📑 Detalle y descarga",
        ])

        with tab_kpi:
            st.subheader("Indicadores ejecutivos")
            if value_cols_raw:
                ordered_metrics = sorted(
                    value_cols_raw,
                    key=lambda c: df_filtered[c].fillna(0).sum(),
                    reverse=True,
                )
                top_metrics = ordered_metrics[: min(4, len(ordered_metrics))]
                kpi_columns = st.columns(len(top_metrics)) if top_metrics else []
                for container, metric in zip(kpi_columns, top_metrics):
                    total = df_filtered[metric].sum()
                    ts = compute_time_series(df_filtered, metric, period_col)
                    delta = compute_delta(ts)
                    delta_text = f"{delta * 100:+.1f}% vs periodo previo" if delta is not None else "—"
                    with container:
                        st.metric(
                            label=prettify_label(metric),
                            value=format_metric(total),
                            delta=delta_text,
                        )

                trend_metric = st.selectbox(
                    "Serie temporal a monitorizar",
                    ordered_metrics,
                    index=0,
                    key="trend_metric",
                )
                ts = compute_time_series(df_filtered, trend_metric, period_col)
                if ts is not None and len(ts) > 0:
                    ts_df = ts.reset_index(name=prettify_label(trend_metric))
                    time_col_name = period_col or ts_df.columns[0]
                    ts_df = ts_df.rename(columns={time_col_name: "Periodo"})
                    fig_trend = px.line(
                        ts_df,
                        x="Periodo",
                        y=prettify_label(trend_metric),
                        markers=True,
                        title=f"Evolución de {prettify_label(trend_metric)}",
                    )
                    fig_trend.update_layout(xaxis_title="Periodo", yaxis_title=prettify_label(trend_metric))
                    st.plotly_chart(fig_trend, use_container_width=True)
                else:
                    st.info("No hay serie temporal disponible con la granularidad actual.")

                if dims_available:
                    c1, c2 = st.columns(2)
                    with c1:
                        dim_choice = st.selectbox(
                            "Desglose estratégico",
                            dims_available,
                            index=0,
                            key="dim_breakdown",
                        )
                    with c2:
                        metric_dim = st.selectbox(
                            "Métrica asociada",
                            ordered_metrics,
                            index=0,
                            key="metric_breakdown",
                        )
                    breakdown_df = aggregate_dimension(df_filtered, dim_choice, metric_dim)
                    if not breakdown_df.empty:
                        fig_breakdown = px.bar(
                            breakdown_df,
                            x=metric_dim,
                            y=dim_choice,
                            orientation="h",
                            text=metric_dim,
                            title=f"Top {dim_choice} por {prettify_label(metric_dim)}",
                        )
                        fig_breakdown.update_layout(
                            yaxis=dict(categoryorder="total ascending"),
                            xaxis_title=prettify_label(metric_dim),
                            yaxis_title=prettify_label(dim_choice),
                        )
                        st.plotly_chart(fig_breakdown, use_container_width=True)
                    else:
                        st.info("No hay datos suficientes para el desglose seleccionado.")
            else:
                st.info("No hay métricas numéricas disponibles tras los filtros aplicados.")

        with tab_map:
            st.subheader("Cobertura e impacto territorial")
            if "provincia" in df_filtered.columns and value_cols_raw:
                metric_map = st.selectbox(
                    "Variable a cartografiar",
                    value_cols_raw,
                    index=0,
                    key="metric_map",
                )
                coords = load_province_centroids()
                map_df = aggregate_dimension(df_filtered, "provincia", metric_map, top_n=1000)
                map_df = map_df.merge(coords, on="provincia", how="left")
                map_df = map_df.dropna(subset=["lat", "lon"])
                if not map_df.empty:
                    fig_map = px.scatter_mapbox(
                        map_df,
                        lat="lat",
                        lon="lon",
                        size=metric_map,
                        color=metric_map,
                        hover_name="provincia",
                        hover_data={metric_map: ':,.0f'},
                        size_max=45,
                        color_continuous_scale="teal",
                        zoom=4.5,
                        title=f"Impacto provincial de {prettify_label(metric_map)}",
                    )
                    fig_map.update_layout(mapbox_style="carto-positron", margin=dict(l=0, r=0, t=60, b=0))
                    st.plotly_chart(fig_map, use_container_width=True)
                    st.dataframe(map_df.sort_values(metric_map, ascending=False).head(25), use_container_width=True)
                else:
                    st.warning("No hay coordenadas disponibles para las provincias filtradas.")
            elif "ccaa" in df_filtered.columns and value_cols_raw:
                st.info("El conjunto no incluye provincias; se muestra el desglose por comunidad autónoma.")
                ccaa_metric = st.selectbox(
                    "Variable a analizar",
                    value_cols_raw,
                    index=0,
                    key="metric_ccaa",
                )
                ccaa_df = aggregate_dimension(df_filtered, "ccaa", ccaa_metric, top_n=19)
                if not ccaa_df.empty:
                    fig_ccaa = px.bar(
                        ccaa_df,
                        x=ccaa_metric,
                        y="ccaa",
                        orientation="h",
                        title=f"Impacto por CCAA – {prettify_label(ccaa_metric)}",
                    )
                    fig_ccaa.update_layout(yaxis=dict(categoryorder="total ascending"))
                    st.plotly_chart(fig_ccaa, use_container_width=True)
                    st.dataframe(ccaa_df, use_container_width=True)
            else:
                st.info("No hay información territorial disponible en este dataset.")

        with tab_corr:
            st.subheader("Correlaciones y relaciones clave")
            if len(value_cols_analysis) >= 2:
                method = st.selectbox(
                    "Método de correlación",
                    ["pearson", "spearman", "kendall"],
                    index=0,
                    key="corr_method",
                )
                corr_df = df_analysis[value_cols_analysis].corr(method=method)
                fig_corr = px.imshow(
                    corr_df,
                    text_auto=True,
                    color_continuous_scale="RdBu_r",
                    zmin=-1,
                    zmax=1,
                    title=f"Matriz de correlación ({method.title()})",
                )
                fig_corr.update_layout(margin=dict(l=0, r=0, t=60, b=0))
                st.plotly_chart(fig_corr, use_container_width=True)

                c1, c2 = st.columns(2)
                with c1:
                    x_col = st.selectbox("Eje X", value_cols_analysis, index=0, key="scatter_x")
                with c2:
                    y_candidates = [c for c in value_cols_analysis if c != x_col] or value_cols_analysis
                    y_col = st.selectbox("Eje Y", y_candidates, index=0, key="scatter_y")
                scatter_df = df_analysis[[x_col, y_col]].dropna()
                if not scatter_df.empty:
                    fig_scatter = px.scatter(
                        scatter_df,
                        x=x_col,
                        y=y_col,
                        title=f"{prettify_label(y_col)} vs {prettify_label(x_col)}",
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.info("No hay suficientes datos para el diagrama de dispersión seleccionado.")
            else:
                st.info("Selecciona filtros con al menos dos métricas numéricas para calcular correlaciones.")

        with tab_topsis:
            st.subheader("Ranking multicriterio (TOPSIS)")
            if dims_available and value_cols_analysis:
                entity = st.selectbox(
                    "Unidad de evaluación",
                    dims_available,
                    index=0,
                    key="topsis_dim",
                )
                aggregated = df_analysis.copy()
                aggregated[entity] = aggregated[entity].fillna("Sin dato")
                aggregated = aggregated.groupby(entity)[value_cols_analysis].sum().reset_index()
                st.caption("Los criterios se agregan mediante suma sobre la unidad seleccionada.")
                default_criteria = value_cols_analysis[: min(4, len(value_cols_analysis))]
                criteria = st.multiselect(
                    "Criterios numéricos",
                    value_cols_analysis,
                    default=default_criteria,
                    key="topsis_criteria",
                )
                if criteria:
                    weights, benefit_flags = [], []
                    st.markdown("Configura pesos y tipo de criterio:")
                    for crit in criteria:
                        c1, c2, c3 = st.columns([2, 1, 2])
                        with c1:
                            st.write(f"**{prettify_label(crit)}**")
                        with c2:
                            benefit_flags.append(st.toggle("Beneficio", True, key=f"topsis_b_{crit}"))
                        with c3:
                            weights.append(
                                st.slider(
                                    "Peso relativo",
                                    0.0,
                                    1.0,
                                    1.0 / max(len(criteria), 1),
                                    0.01,
                                    key=f"topsis_w_{crit}",
                                )
                            )
                    try:
                        ranking = topsis_rank(aggregated, criteria, weights, benefit_flags, index_col=entity)
                        st.dataframe(ranking, use_container_width=True)
                        fig_rank = px.bar(
                            ranking.head(15),
                            x="score_topsis",
                            y=entity,
                            orientation="h",
                            text="score_topsis",
                            title="Top 15 según TOPSIS",
                        )
                        fig_rank.update_layout(yaxis=dict(categoryorder="total ascending"))
                        st.plotly_chart(fig_rank, use_container_width=True)
                        csv_ranking = ranking.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "Descargar ranking TOPSIS (CSV)",
                            csv_ranking,
                            file_name="ranking_topsis.csv",
                            mime="text/csv",
                        )
                    except Exception as e:
                        st.warning(f"No se pudo calcular TOPSIS con los criterios seleccionados: {e}")
                else:
                    st.info("Selecciona al menos un criterio numérico para construir el ranking.")
            else:
                st.info("No hay suficientes dimensiones o métricas numéricas para construir el ranking.")

        with tab_detail:
            st.subheader("Detalle operativo y descarga")
            st.dataframe(df_filtered, use_container_width=True)
            csv_data = df_filtered.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Descargar datos filtrados (CSV)",
                csv_data,
                file_name="datos_filtrados.csv",
                mime="text/csv",
            )

st.caption(
    "Panel avanzado: ingestión CNMC, normalización opcional, KPIs dinámicos, mapas, correlaciones y TOPSIS listo para cuadros de mando del TFM."
)
