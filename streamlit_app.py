from pathlib import Path
import json
import unicodedata

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "mx-renta-por-colonia-2026-11.csv"
METRICS_PATH = ROOT / "data" / "metrics.json"

st.set_page_config(page_title="RentaSmart", page_icon="🏠", layout="wide")


def normalize(value):
    value = unicodedata.normalize("NFKD", str(value))
    return "".join(ch for ch in value if not unicodedata.combining(ch)).strip().lower()


@st.cache_data(show_spinner="Cargando y preparando los datos...")
def load_data():
    df = pd.read_csv(DATA_PATH)
    for col in ["avisos", "mediana", "p25", "p75", "mediana_m2", "mediana_area_m2"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["ciudad", "zona", "operacion", "tipo", "mediana", "avisos"])
    df = df[df["mediana"] > 0].copy()
    return df


@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return None


st.title("🏠 RentaSmart")
st.subheader("Explorador de rentas anunciadas por zona y tipo de vivienda")
st.caption("Fuente: Aptología (2026), corte 22/11/2026 · licencia CC BY 4.0 · cifras agregadas, no anuncios individuales.")

if not DATA_PATH.exists():
    st.error("No se encontró data/mx-renta-por-colonia-2026-11.csv. Sube la carpeta data junto con streamlit_app.py al repositorio.")
    st.stop()

df = load_data()

with st.sidebar:
    st.header("Filtros")
    cities = sorted(df["ciudad"].dropna().unique().tolist())
    city_default = next((c for c in cities if normalize(c) == "ciudad de mexico"), cities[0])
    city = st.selectbox("Ciudad", cities, index=cities.index(city_default))

    city_df = df[df["ciudad"] == city]
    operations = sorted(city_df["operacion"].dropna().unique().tolist())
    op_default = "alquiler" if "alquiler" in operations else operations[0]
    operation = st.selectbox("Operación", operations, index=operations.index(op_default))

    base = city_df[city_df["operacion"] == operation]
    types = sorted(base["tipo"].dropna().unique().tolist())
    recommended = [t for t in ["casa", "departamento"] if t in types]
    selected_types = st.multiselect("Tipo de inmueble", types, default=recommended or types)

    base = base[base["tipo"].isin(selected_types)]
    min_ads_max = max(1, int(base["avisos"].quantile(.95))) if not base.empty else 10
    min_ads = st.slider("Mínimo de anuncios representados", min_value=1, max_value=min_ads_max, value=min(5, min_ads_max))
    zone_query = st.text_input("Buscar zona", placeholder="Escribe parte del nombre")

    rent_base = base[base["avisos"] >= min_ads]
    rent_base = rent_base[rent_base["mediana"].notna()]
    if not rent_base.empty:
        lower = int(rent_base["mediana"].min())
        upper = int(rent_base["mediana"].max())
        if lower < upper:
            rent_range = st.slider("Rango de renta mensual (MXN)", lower, upper, (lower, upper), step=max(1, (upper-lower)//200))
        else:
            rent_range = (lower, upper)
            st.caption(f"Única renta disponible: ${lower:,.0f} MXN")
    else:
        rent_range = (0, 0)

filtered = base[base["avisos"] >= min_ads].copy()
if zone_query.strip():
    filtered = filtered[filtered["zona"].astype(str).str.contains(zone_query.strip(), case=False, na=False)]
if not filtered.empty:
    filtered = filtered[filtered["mediana"].between(*rent_range)]

if filtered.empty:
    st.warning("No hay registros con esos filtros. Amplía el rango de renta o cambia la selección.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Celdas zona-tipo", f"{len(filtered):,}")
col2.metric("Zonas", f"{filtered['zona'].nunique():,}")
col3.metric("Renta mediana", f"${filtered['mediana'].median():,.0f} MXN")
col4.metric("Anuncios representados", f"{filtered['avisos'].sum():,.0f}")

st.info("Cada fila resume un segmento de zona y tipo construido a partir de anuncios. No representa una vivienda concreta ni un precio de cierre.")

tab_overview, tab_explore, tab_model, tab_data = st.tabs(["Resumen", "Exploración", "Modelo", "Datos"])

with tab_overview:
    left, right = st.columns(2)
    with left:
        by_type = filtered.groupby("tipo", as_index=False).agg(
            renta_mediana=("mediana", "median"), zonas=("zona", "nunique"), anuncios=("avisos", "sum")
        ).sort_values("renta_mediana", ascending=False)
        fig = px.bar(by_type, x="tipo", y="renta_mediana", color="tipo", text="renta_mediana",
                     title="Renta mediana por tipo", labels={"tipo":"Tipo", "renta_mediana":"Renta mensual (MXN)"},
                     hover_data=["zonas", "anuncios"])
        fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.histogram(filtered, x="mediana", color="tipo", nbins=30, marginal="box",
                           title="Distribución de rentas medianas",
                           labels={"mediana":"Renta mensual mediana (MXN)", "tipo":"Tipo"})
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Cómo leerlo:** compara las medianas junto con el número de zonas y anuncios representados. La distribución muestra variación, no una tarifa recomendada.")
    st.dataframe(by_type, use_container_width=True, hide_index=True)

with tab_explore:
    left, right = st.columns(2)
    with left:
        if "mediana_area_m2" in filtered and filtered["mediana_area_m2"].notna().any():
            fig = px.scatter(filtered.dropna(subset=["mediana_area_m2"]), x="mediana_area_m2", y="mediana",
                             color="tipo", size="avisos", hover_name="zona", hover_data=["p25", "p75"],
                             title="Renta frente a superficie mediana",
                             labels={"mediana_area_m2":"Superficie mediana (m²)", "mediana":"Renta mensual (MXN)", "tipo":"Tipo"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No hay datos de superficie para los filtros seleccionados.")
    with right:
        fig = px.box(filtered, x="tipo", y="mediana", color="tipo", points="outliers",
                     title="Dispersión de rentas por tipo", labels={"tipo":"Tipo", "mediana":"Renta mensual (MXN)"},
                     hover_data=["zona", "avisos"])
        st.plotly_chart(fig, use_container_width=True)

    ranking = (filtered.groupby("zona", as_index=False)
               .agg(filas=("tipo", "size"), anuncios=("avisos", "sum"), renta_mediana=("mediana", "median"),
                    renta_p25=("mediana", lambda x: x.quantile(.25)), renta_p75=("mediana", lambda x: x.quantile(.75)))
               .sort_values("renta_mediana", ascending=False))
    st.subheader("Zonas con mayor renta mediana en la selección")
    top = ranking.head(15)
    st.plotly_chart(px.bar(top.sort_values("renta_mediana"), x="renta_mediana", y="zona", color="renta_mediana",
                           orientation="h", title="15 zonas principales", labels={"renta_mediana":"Renta mensual (MXN)"},
                           hover_data=["filas", "anuncios", "renta_p25", "renta_p75"]), use_container_width=True)
    st.dataframe(ranking.head(30), use_container_width=True, hide_index=True)

with tab_model:
    st.subheader("Evaluación experimental del modelo")
    st.write("Resultados fijos para Ciudad de México, alquiler, casas y departamentos. La prueba reserva colonias completas; no cambia con los filtros del explorador.")
    metrics = load_metrics()
    if metrics:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("R²", f"{metrics['R2']:.3f}")
        m2.metric("MAE", f"${metrics['MAE_MXN']:,.0f} MXN")
        m3.metric("RMSE", f"${metrics['RMSE_MXN']:,.0f} MXN")
        m4.metric("MAE línea base", f"${metrics['baseline_median_MAE_MXN']:,.0f} MXN")
        improvement = (1 - metrics["MAE_MXN"] / metrics["baseline_median_MAE_MXN"]) * 100
        st.write(f"En esa partición, el MAE fue {improvement:.1f}% menor que la línea base.")
        st.caption(f"Prueba: {metrics['test_rows']} celdas agregadas de {metrics['test_zones']} zonas reservadas. El modelo usa log(superficie mediana) y tipo de inmueble.")
    else:
        st.warning("No se encontró data/metrics.json; vuelve a ejecutar src/pipeline.py antes de publicar.")
    st.warning("Estas métricas no equivalen a precisión sobre viviendas individuales. El prototipo no debe usarse para fijar rentas, avalúos ni decisiones de acceso a vivienda.")

with tab_data:
    show_cols = [c for c in ["ciudad", "zona", "operacion", "tipo", "avisos", "mediana", "p25", "p75", "mediana_area_m2", "fecha_corte"] if c in filtered]
    st.subheader(f"Registros seleccionados ({len(filtered):,})")
    st.dataframe(filtered[show_cols].sort_values("mediana", ascending=False), use_container_width=True, hide_index=True)
    st.download_button("Descargar selección CSV", filtered[show_cols].to_csv(index=False).encode("utf-8-sig"),
                       file_name="rentasmart_seleccion.csv", mime="text/csv")

st.divider()
st.caption("RentaSmart · prototipo académico · datos agregados de Aptología (CC BY 4.0).")
