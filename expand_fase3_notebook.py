"""Build the expanded, interactive RentaSmart analysis notebook."""
from pathlib import Path
import json
import uuid

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "RentaSmart_Fase3.ipynb"

def md(text):
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {}, "source": text.splitlines(keepends=True)}

def code(text):
    return {"cell_type": "code", "id": uuid.uuid4().hex[:8], "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}

cells = [
md("""# RentaSmart · Fase 3: exploración, filtros y modelado

Notebook reproducible para explorar rentas anunciadas en México y evaluar un modelo de referencia para **Ciudad de México, alquiler, casas y departamentos**. La fuente original contiene agregados por ciudad, zona, operación y tipo: **cada fila no equivale a una vivienda individual**.

**Fuente:** Aptología (2026), datos abiertos, licencia CC BY 4.0, corte 22/11/2026. Este análisis es exploratorio y no constituye avalúo ni recomendación de renta.

> Ejecuta las celdas en orden. Para usar los filtros interactivos, abre el notebook en Jupyter o Google Colab y ejecuta `pip install -r requirements.txt`."""),
code("""from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from IPython.display import display, Markdown
import plotly.express as px
import plotly.graph_objects as go

pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 30)
pd.options.display.float_format = '{:,.2f}'.format

ROOT = Path.cwd()
if not (ROOT / 'data/mx-renta-por-colonia-2026-11.csv').exists():
    ROOT = ROOT.parent
DATA_PATH = ROOT / 'data/mx-renta-por-colonia-2026-11.csv'
if not DATA_PATH.exists():
    raise FileNotFoundError('No encuentro data/mx-renta-por-colonia-2026-11.csv. Ejecuta el notebook desde la carpeta del proyecto.')

raw = pd.read_csv(DATA_PATH)
required = {'pais','ciudad','zona','operacion','tipo','avisos','mediana','p25','p75','mediana_area_m2'}
missing_columns = required - set(raw.columns)
if missing_columns:
    raise ValueError(f'Faltan columnas necesarias: {sorted(missing_columns)}')
print(f'Archivo cargado: {DATA_PATH.name} · {len(raw):,} filas · {raw.shape[1]} columnas')"""),
md("""## 1. Calidad y preparación

Primero inspeccionamos faltantes, duplicados y tipos. Los valores atípicos se **marcan** con la regla 1.5×IQR para investigarlos; se conservan porque las filas son medianas de segmentos y sin los anuncios fuente no es posible afirmar que sean errores."""),
code("""numeric = ['avisos','mediana','p25','p75','mediana_m2','mediana_area_m2']
for col in numeric:
    if col in raw:
        raw[col] = pd.to_numeric(raw[col], errors='coerce')

quality = pd.DataFrame({
    'filas_con_faltante': raw.isna().sum(),
    'porcentaje_faltante': (raw.isna().mean() * 100).round(2),
    'tipo_dato': raw.dtypes.astype(str)
}).sort_values('filas_con_faltante', ascending=False)
display(quality)

key = ['pais','ciudad','zona','operacion','tipo']
duplicate_rows = int(raw.duplicated(key, keep='first').sum())
print(f'Registros duplicados según {key}: {duplicate_rows:,}')
display(raw.describe(include='all').T)"""),
md("""## 2. Filtros exploratorios

Los selectores filtran ciudad, operación, tipo, texto de zona, mínimo de anuncios y rango de renta. La visualización EDA y las tablas de esta sección se recalculan con la selección. Los filtros se aplican sobre la fuente completa; el modelado posterior mantiene el alcance fijo de Fase 2 para que las métricas sean comparables."""),
code("""# Normalizamos textos y construimos una copia analítica sin eliminar todavía filas.
data = raw.copy()
for col in ['pais','ciudad','zona','operacion','tipo']:
    data[col] = data[col].astype('string').str.strip()
data['mediana'] = pd.to_numeric(data['mediana'], errors='coerce')
data['mediana_area_m2'] = pd.to_numeric(data['mediana_area_m2'], errors='coerce')
data['avisos'] = pd.to_numeric(data['avisos'], errors='coerce')
data['p25'] = pd.to_numeric(data['p25'], errors='coerce')
data['p75'] = pd.to_numeric(data['p75'], errors='coerce')
data = data.dropna(subset=['ciudad','zona','operacion','tipo','mediana','mediana_area_m2','avisos'])
data = data[(data['mediana'] > 0) & (data['mediana_area_m2'] > 0) & (data['avisos'] >= 1)].copy()

try:
    import ipywidgets as widgets
    from IPython.display import display as ipy_display
    all_option = 'Todos'
    cities = [all_option] + sorted(data['ciudad'].dropna().unique().tolist())
    operations = [all_option] + sorted(data['operacion'].dropna().unique().tolist())
    types = [all_option] + sorted(data['tipo'].dropna().unique().tolist())
    min_rent, max_rent = int(data.mediana.min()), int(data.mediana.max())
    import unicodedata
    city_match = next((c for c in cities if c != all_option and unicodedata.normalize('NFKD', str(c)).encode('ascii','ignore').decode('ascii').lower() == 'ciudad de mexico'), all_option)
    city_w = widgets.Dropdown(options=cities, value=city_match, description='Ciudad:', layout={'width':'360px'})
    operation_w = widgets.Dropdown(options=operations, value='alquiler' if 'alquiler' in operations else all_option, description='Operación:', layout={'width':'300px'})
    type_w = widgets.Dropdown(options=types, value=all_option, description='Tipo:', layout={'width':'300px'})
    zone_w = widgets.Text(value='', description='Zona contiene:', placeholder='Ej. Roma', layout={'width':'360px'})
    avisos_w = widgets.IntSlider(value=5, min=1, max=max(5, int(data.avisos.quantile(.95))), step=1, description='Mín. avisos:', continuous_update=False, layout={'width':'500px'})
    rent_w = widgets.IntRangeSlider(value=(min_rent, max_rent), min=min_rent, max=max_rent, step=max(1, (max_rent-min_rent)//200), description='Renta MXN:', continuous_update=False, layout={'width':'650px'})
    filter_out = widgets.Output()
    def apply_filters(_=None):
        selected = data.copy()
        if city_w.value != all_option: selected = selected[selected.ciudad == city_w.value]
        if operation_w.value != all_option: selected = selected[selected.operacion == operation_w.value]
        if type_w.value != all_option: selected = selected[selected.tipo == type_w.value]
        if zone_w.value.strip(): selected = selected[selected.zona.str.contains(zone_w.value.strip(), case=False, na=False)]
        selected = selected[(selected.avisos >= avisos_w.value) & selected.mediana.between(*rent_w.value)]
        globals()['filtered'] = selected.copy()
        with filter_out:
            filter_out.clear_output(wait=True)
            print(f'{len(selected):,} filas · {selected.zona.nunique():,} zonas · renta {rent_w.value[0]:,}–{rent_w.value[1]:,} MXN · mínimo {avisos_w.value} avisos')
            if selected.empty: print('No hay filas con esta combinación. Amplía el rango o cambia los filtros.')
    for w in [city_w, operation_w, type_w, zone_w, avisos_w, rent_w]:
        w.observe(apply_filters, names='value')
    ipy_display(widgets.VBox([widgets.HBox([city_w, operation_w]), widgets.HBox([type_w, zone_w]), avisos_w, rent_w, filter_out]))
    apply_filters()
except ImportError:
    # Fallback editable sin ipywidgets: cambia estos valores y vuelve a ejecutar esta celda.
    city_filter = 'Ciudad de Mexico'  # 'Todos' o nombre de ciudad
    operation_filter = 'alquiler'     # 'Todos', alquiler, venta, etc.
    type_filter = 'Todos'             # 'Todos', casa, departamento, etc.
    zone_contains = ''                # Ejemplo: 'Roma'
    min_ads, rent_min, rent_max = 5, 0, float('inf')
    filtered = data.copy()
    if city_filter != 'Todos': filtered = filtered[filtered.ciudad == city_filter]
    if operation_filter != 'Todos': filtered = filtered[filtered.operacion == operation_filter]
    if type_filter != 'Todos': filtered = filtered[filtered.tipo == type_filter]
    if zone_contains: filtered = filtered[filtered.zona.str.contains(zone_contains, case=False, na=False)]
    filtered = filtered[(filtered.avisos >= min_ads) & filtered.mediana.between(rent_min, rent_max)]
    print('ipywidgets no está instalado; ajusta las variables de filtro de esta celda.')
    print(f'{len(filtered):,} filas · {filtered.zona.nunique():,} zonas')"""),
md("""## 3. Resumen de la selección y tablas descriptivas

Lee las tablas con los filtros activos. Las columnas `n_filas` y `avisos` ayudan a no comparar medianas sin considerar cuánta cobertura hay detrás de cada segmento."""),
code("""if filtered.empty:
    print('La selección está vacía; cambia los filtros antes de continuar.')
else:
    kpis = pd.DataFrame([
        ['Filas agregadas', len(filtered)], ['Zonas', filtered.zona.nunique()],
        ['Tipos', filtered.tipo.nunique()], ['Anuncios representados (suma)', int(filtered.avisos.sum())],
        ['Renta mediana (MXN)', filtered.mediana.median()], ['Superficie mediana (m²)', filtered.mediana_area_m2.median()]
    ], columns=['Indicador','Valor'])
    display(kpis.style.format({'Valor': lambda x: f'{x:,.0f}' if isinstance(x,(int,float,np.integer,np.floating)) else x}).set_caption('Indicadores de la selección'))

    tabla_tipo = (filtered.groupby('tipo').agg(n_filas=('zona','size'), zonas=('zona','nunique'), avisos=('avisos','sum'),
        renta_mediana=('mediana','median'), renta_promedio=('mediana','mean'), renta_p25=('mediana','quantile'),
        superficie_mediana=('mediana_area_m2','median')).sort_values('renta_mediana', ascending=False).reset_index())
    display(tabla_tipo.style.format({'avisos':'{:,.0f}','renta_mediana':'${:,.0f}','renta_promedio':'${:,.0f}','renta_p25':'${:,.0f}','superficie_mediana':'{:,.1f}'}).set_table_styles([{'selector':'th','props':[('background-color','#17324D'),('color','white')]}]).set_caption('Resumen por tipo de inmueble'))

    tabla_zona = (filtered.groupby('zona').agg(n_filas=('tipo','size'), tipos=('tipo','nunique'), avisos=('avisos','sum'),
        renta_mediana=('mediana','median'), renta_p25=('mediana','quantile'), renta_p75=('mediana',lambda x:x.quantile(.75)),
        superficie_mediana=('mediana_area_m2','median')).sort_values('renta_mediana',ascending=False).reset_index())
    display(tabla_zona.head(15).style.format({'avisos':'{:,.0f}','renta_mediana':'${:,.0f}','renta_p25':'${:,.0f}','renta_p75':'${:,.0f}','superficie_mediana':'{:,.1f}'}).set_table_styles([{'selector':'th','props':[('background-color','#168AAD'),('color','white')]}]).set_caption('15 zonas con renta mediana más alta en la selección'))"""),
md("""## 4. Visualizaciones EDA

Los gráficos son interactivos (hover, zoom y leyendas). La distribución y los diagramas de caja muestran dispersión; el scatter compara medianas de zona y tipo. No se interpretan como relaciones causales."""),
code("""if not filtered.empty:
    fig = px.histogram(filtered, x='mediana', color='tipo', nbins=30, marginal='box',
        title='Distribución de rentas medianas por zona y tipo', labels={'mediana':'Renta mensual mediana (MXN)','count':'Número de celdas','tipo':'Tipo'}, opacity=.78)
    fig.update_layout(bargap=.08, legend_title_text='Tipo de inmueble'); fig.show()

    fig = px.box(filtered, x='tipo', y='mediana', points='outliers', color='tipo',
        title='Dispersión de la renta mediana por tipo', labels={'tipo':'Tipo','mediana':'Renta mensual mediana (MXN)'}, hover_data=['zona','avisos','mediana_area_m2'])
    fig.show()

    fig = px.scatter(filtered, x='mediana_area_m2', y='mediana', color='tipo', size='avisos', size_max=18,
        hover_name='zona', hover_data={'avisos':True,'p25':True,'p75':True,'mediana_area_m2':':.1f'},
        title='Renta y superficie medianas (tamaño del punto = anuncios representados)',
        labels={'mediana_area_m2':'Superficie mediana (m²)','mediana':'Renta mensual mediana (MXN)','tipo':'Tipo'})
    fig.show()

    by_type = filtered.groupby('tipo',as_index=False).agg(renta=('mediana','median'), superficie=('mediana_area_m2','median'), avisos=('avisos','sum'))
    fig = px.scatter(by_type, x='superficie', y='renta', size='avisos', color='tipo', text='tipo',
        title='Comparación multivariada de medianas por tipo', labels={'superficie':'Superficie mediana (m²)','renta':'Renta mediana (MXN)','avisos':'Anuncios'})
    fig.update_traces(textposition='top center'); fig.show()"""),
md("""## 5. Comparaciones, correlación y observaciones atípicas

La matriz de correlación resume asociaciones lineales entre variables numéricas de la selección. Las observaciones IQR se conservan y se muestran para revisión contextual."""),
code("""if not filtered.empty:
    corr_cols = [c for c in ['mediana','mediana_area_m2','avisos','p25','p75','mediana_m2'] if c in filtered and filtered[c].nunique()>1]
    corr = filtered[corr_cols].corr(numeric_only=True)
    fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r', zmin=-1, zmax=1, aspect='auto',
        title='Correlaciones entre medidas numéricas (Pearson)', labels={'color':'Correlación'})
    fig.show()

    q1,q3=filtered.mediana.quantile([.25,.75]); iqr=q3-q1; low=max(0,q1-1.5*iqr); high=q3+1.5*iqr
    reviewed=filtered[(filtered.mediana<low)|(filtered.mediana>high)].copy().sort_values('mediana',ascending=False)
    print(f'Regla 1.5×IQR en la selección: límites ${low:,.0f}–${high:,.0f}; filas señaladas {len(reviewed)} de {len(filtered)}. Se conservan, no se eliminan automáticamente.')
    if not reviewed.empty:
        display(reviewed[['ciudad','zona','operacion','tipo','avisos','mediana','p25','p75','mediana_area_m2']].head(30).style.format({'mediana':'${:,.0f}','p25':'${:,.0f}','p75':'${:,.0f}','mediana_area_m2':'{:,.1f}'}).set_caption('Filas señaladas para revisión contextual'))
    else:
        print('No hay filas fuera de los límites IQR con esta selección.')"""),
md("""## 6. Modelo analítico de referencia (alcance fijo de Fase 2)

Para evitar mezclar ciudades y operaciones, el modelo se restringe a Ciudad de México, alquiler, casa/departamento. Se excluyen faltantes indispensables, valores no positivos y segmentos con menos de cinco anuncios; se eliminan duplicados por zona/tipo/operación. Las zonas de prueba se reservan completas para reducir fuga territorial."""),
code("""city_norm = raw.ciudad.astype(str).str.normalize('NFKD').str.encode('ascii',errors='ignore').str.decode('ascii').str.lower()
model_raw = raw[(raw.pais == 'mx') & city_norm.eq('ciudad de mexico') & (raw.operacion == 'alquiler') & raw.tipo.isin(['casa','departamento'])].copy()
scope_initial = len(model_raw)
model = model_raw.dropna(subset=['zona','tipo','mediana','mediana_area_m2','avisos'])
model = model[(model.mediana>0)&(model.mediana_area_m2>0)&(model.avisos>=5)].drop_duplicates(['zona','tipo','operacion']).copy()
q1,q3=model.mediana.quantile([.25,.75]); lo=q1-1.5*(q3-q1); hi=q3+1.5*(q3-q1)
model['atipico_iqr']=(model.mediana<lo)|(model.mediana>hi)
model['log_area']=np.log(model.mediana_area_m2); model['log_renta']=np.log(model.mediana)
scrub_table = pd.DataFrame([
    ['Filas iniciales en alcance',scope_initial],['Filas limpias',len(model)],['Zonas limpias',model.zona.nunique()],
    ['Faltantes de superficie excluidos',int(model_raw.mediana_area_m2.isna().sum())],
    ['Duplicados de llave eliminados',int(model_raw.duplicated(['zona','tipo','operacion']).sum())],
    ['Valores IQR señalados y conservados',int(model.atipico_iqr.sum())]
],columns=['Regla/medida','Resultado'])
display(scrub_table)
display(model.groupby('tipo').agg(n=('zona','size'),zonas=('zona','nunique'),avisos=('avisos','sum'),renta_mediana=('mediana','median'),area_mediana=('mediana_area_m2','median')).reset_index())"""),
code("""zones=sorted(model.zona.unique())
test_z=set(zones[::5])
train=model[~model.zona.isin(test_z)].copy(); test=model[model.zona.isin(test_z)].copy()
mu=train.log_area.mean(); sd=train.log_area.std() or 1
def design(frame):
    return np.column_stack([np.ones(len(frame)),(frame.log_area.to_numpy()-mu)/sd,(frame.tipo.to_numpy()=='casa').astype(float)])
beta=np.linalg.lstsq(design(train),train.log_renta.to_numpy(),rcond=None)[0]
y=test.mediana.to_numpy(); pred=np.exp(design(test)@beta)
mae=float(np.mean(np.abs(y-pred))); rmse=float(np.sqrt(np.mean((y-pred)**2)))
r2=float(1-np.sum((y-pred)**2)/np.sum((y-y.mean())**2))
base=float(np.mean(np.abs(y-train.mediana.median())))
metrics=pd.DataFrame([
    ['R²',r2,'Proporción de variación en la prueba; puede ser sensible al tamaño y composición de la muestra.'],
    ['MAE (MXN)',mae,'Error absoluto medio frente a rentas medianas agregadas.'],
    ['RMSE (MXN)',rmse,'Penaliza más errores grandes.'],
    ['MAE base (MXN)',base,'Predicción constante con mediana del conjunto de entrenamiento.'],
    ['Mejora MAE vs base (%)',(1-mae/base)*100,'Comparación dentro de esta única partición de prueba.']
],columns=['Métrica','Valor','Interpretación'])
display(metrics.style.format({'Valor':lambda x:f'{x:.3f}' if abs(x)<1 else f'{x:,.0f}'}).set_caption(f'OLS · entrenamiento {len(train)} filas · prueba {len(test)} filas · zonas de prueba {len(test_z)}'))

pred_test=test[['zona','tipo','mediana','mediana_area_m2','avisos']].copy()
pred_test['prediccion_mxn']=pred; pred_test['error_mxn']=pred_test.mediana-pred_test.prediccion_mxn
display(pred_test.assign(error_abs=pred_test.error_mxn.abs()).sort_values('error_abs',ascending=False).head(15).style.format({'mediana':'${:,.0f}','prediccion_mxn':'${:,.0f}','error_mxn':'${:,.0f}','error_abs':'${:,.0f}','mediana_area_m2':'{:,.1f}'}).set_caption('15 errores absolutos más altos en prueba; revisar como diagnóstico'))"""),
code("""fig = px.scatter(pred_test, x='mediana', y='prediccion_mxn', color='tipo', size='avisos', hover_name='zona',
    title='Rentas observadas y predichas en zonas reservadas', labels={'mediana':'Renta observada (MXN)','prediccion_mxn':'Predicción del modelo (MXN)','tipo':'Tipo'})
lo=min(pred_test.mediana.min(),pred_test.prediccion_mxn.min()); hi=max(pred_test.mediana.max(),pred_test.prediccion_mxn.max())
fig.add_trace(go.Scatter(x=[lo,hi],y=[lo,hi],mode='lines',name='Referencia ideal',line={'dash':'dash','color':'gray'})); fig.show()

pred_test['residuo']=pred_test.mediana-pred_test.prediccion_mxn
fig=px.scatter(pred_test,x='prediccion_mxn',y='residuo',color='tipo',hover_name='zona',
    title='Diagnóstico de residuos',labels={'prediccion_mxn':'Renta predicha (MXN)','residuo':'Residuo: observada − predicha (MXN)'})
fig.add_hline(y=0,line_dash='dash',line_color='gray'); fig.show()"""),
md("""## 7. Lectura de resultados, límites y siguientes pasos

El modelo se ajusta a medianas de segmentos zona-tipo, no a viviendas individuales. R², MAE y RMSE corresponden a una sola partición por colonia; no garantizan desempeño futuro ni en zonas con otra cobertura. Las rentas son precios anunciados, no cierres. Antes de usar cualquier cifra para decisiones, haría falta ampliar y verificar datos, validar por periodo y territorio y contar con revisión humana.

### Evidencias que debes comentar

1. Indica los filtros activos y cuántas filas/zonas quedaron.
2. Describe mediana y rango intercuartílico por tipo junto con anuncios representados.
3. Explica qué patrones muestran histograma, caja, dispersión y tabla de zonas; evita afirmar causalidad.
4. Compara el modelo con la línea base y presenta MAE/RMSE en MXN.
5. Reporta como limitación el origen de portales, los agregados, el tamaño de prueba y el dashboard aún local.

**Referencia de datos:** Aptología. (2026). *Datos abiertos de rentas en México* (corte 22 de noviembre de 2026) [Conjunto de datos, CC BY 4.0]. https://www.aptologia.com/mx/datos-abiertos"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}
OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Notebook guardado: {OUT} ({len(cells)} celdas)")
