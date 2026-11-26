# RentaSmart · Fase 3

## Alcance ejecutado
Estimación de medianas de renta mensual anunciada por zona y tipo de inmueble para Ciudad de México. La fuente contiene agregados por colonia/tipo con al menos cinco avisos; no contiene anuncios individuales. Por ello, esta ejecución aproxima medianas de segmentos y no predice el precio de una vivienda individual.

## Fuente de datos
Aptología (2026), *Precios de renta y venta de vivienda por colonia en México, noviembre de 2026*, corte 22/11/2026. Licencia CC BY 4.0. Dataset original: `data/mx-renta-por-colonia-2026-11.csv`. Página y método: https://www.aptologia.com/mx/datos-abiertos

Cita: Aptología (2026). *Precios de renta y venta de vivienda por colonia en México (Aptología, noviembre de 2026)* [dataset]. https://www.aptologia.com/mx/datos-abiertos. CC BY 4.0.

## Estructura
- `data/`: fuente original, datos depurados, resumen para el dashboard y métricas.
- `notebooks/RentaSmart_Fase3.ipynb`: notebook ampliado con filtros interactivos, tablas analíticas, visualizaciones y evaluación del modelo.
- `src/pipeline.py`: carga, filtrado, depuración, split por colonia, regresión y métricas.
- `src/build_assets.py`: prepara los datos y la página local del dashboard.
- `src/expand_fase3_notebook.py`: reconstruye el notebook ampliado.
- `dashboard/index.html`: tablero interactivo local. Abrir en navegador; no está publicado en línea.

## Reproducir
Requiere Python 3.10 o posterior. Las dependencias de análisis, gráficos e interacción están en `requirements.txt`.

```powershell
python -m pip install -r requirements.txt
python src/pipeline.py
python src/build_assets.py
```

Abra `dashboard/index.html` en un navegador. Para el notebook, abra `notebooks/RentaSmart_Fase3.ipynb` desde la raíz del proyecto y ejecute las celdas en orden. Los filtros permiten seleccionar ciudad, operación, tipo de inmueble, texto de zona, mínimo de anuncios y rango de renta. Si el entorno no muestra controles interactivos, edite los valores de respaldo en la celda de filtros y vuelva a ejecutarla.

## Resultados de esta ejecución
248 filas válidas; 197 zonas; 200 filas de entrenamiento y 48 de prueba. Separamos por zona para que ninguna aparezca en ambos conjuntos. El OLS usa log(superficie mediana) y tipo de inmueble para explicar log(renta mediana). Resultado en prueba: R²=0.796, MAE=8,907 MXN, RMSE=13,139 MXN. La línea base de mediana obtiene MAE=20,130 MXN. Estos valores evalúan celdas agregadas, no anuncios individuales.

Se identificaron 8 filas sin superficie mediana y se excluyeron por ser necesaria para el modelo. No hubo duplicados según zona/tipo/operación. Se marcaron 17 casos fuera de los límites IQR y se conservaron por no existir evidencia para considerarlos errores.

## Límites para interpretación
- Son precios anunciados, no precios de cierre ni avalúos.
- La fuente cubre varios portales y sus anuncios disponibles; no representa todo el mercado ni corrige la menor presencia digital de ciertas zonas.
- La fila analítica es zona × tipo y se publica solo donde hay al menos cinco anuncios subyacentes.
- El modelo es experimental, no causal y no se recomienda usar para fijar precios ni decisiones de acceso a vivienda.
- La app está preparada localmente. Falta publicarla en Streamlit u otro servicio para obtener un enlace público.
- Falta subir este directorio a GitHub y otorgar acceso al docente si el repositorio queda privado.

## Nota de equipo
La Fase 2 registra dos integrantes, mientras la guía de Fase 3 indica equipos de 3 a 4. Confirmen la composición antes de entregar.

