import json
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, KeepTogether
from reportlab.pdfbase.pdfmetrics import stringWidth
from PIL import Image as PILImage, ImageDraw, ImageFont
import numpy as np
ROOT=Path.cwd(); OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)
d=json.loads((ROOT/'data/dashboard_data.json').read_text(encoding='utf-8')); m=d['metrics']; rows=d['records']; by=d['by_type']
font_path=r'C:\Windows\Fonts\arial.ttf'; bold_path=r'C:\Windows\Fonts\arialbd.ttf'
try: f=ImageFont.truetype(font_path,14); fb=ImageFont.truetype(bold_path,19); fs=ImageFont.truetype(font_path,11)
except: f=fb=fs=ImageFont.load_default()
def draw_chart(path, title, xs, ys, xlabel, ylabel):
 W,H=980,490; im=PILImage.new('RGB',(W,H),'white'); dr=ImageDraw.Draw(im); L,T,R,B=90,65,25,72; pw=W-L-R; ph=H-T-B; xmax=max(xs)*1.05; ymax=max(ys)*1.1
 dr.text((W/2,T/2),title,font=fb,fill='#16324f',anchor='mm')
 for i in range(5):
  y=T+ph-i*ph/4; val=ymax*i/4; dr.line((L,y,L+pw,y),fill='#d8e1e8',width=1); dr.text((L-10,y),f'${val/1000:.0f}k',font=fs,fill='#536a7f',anchor='rm')
 dr.line((L,T,L,T+ph),fill='#536a7f',width=2); dr.line((L,T+ph,L+pw,T+ph),fill='#536a7f',width=2)
 for x,y in zip(xs,ys):
  px=L+x/xmax*pw; py=T+ph-y/ymax*ph; dr.ellipse((px-4,py-4,px+4,py+4),fill='#168aad')
 dr.text((L+pw/2,H-22),xlabel,font=fs,fill='#536a7f',anchor='mm')
 im.save(path)
chart1=ROOT/'work'/'eda_scatter.png'; draw_chart(chart1,'Renta mediana y superficie mediana',[float(r['mediana_area_m2']) for r in rows],[float(r['mediana']) for r in rows],'Superficie mediana (m²)','Renta mensual (MXN)')
# Type plot
W,H=800,310; im=PILImage.new('RGB',(W,H),'white'); dr=ImageDraw.Draw(im); dr.text((W/2,28),'Renta mediana por tipo de inmueble',font=fb,fill='#16324f',anchor='mm'); mx=max(r['mediana'] for r in by)
for i,r in enumerate(by):
 y=100+i*90; dr.text((24,y+17),r['tipo'].capitalize(),font=f,fill='#16324f'); w=475*r['mediana']/mx; dr.rounded_rectangle((160,y,160+w,y+45),radius=7,fill='#168aad'); dr.text((172+w,y+15),f"${r['mediana']:,.0f}  (n={r['n']})",font=f,fill='#263f54')
chart2=ROOT/'work'/'eda_tipo.png'; im.save(chart2)
styles=getSampleStyleSheet(); styles.add(ParagraphStyle(name='CoverX',parent=styles['Title'],fontName='Helvetica-Bold',fontSize=23,leading=29,textColor=colors.HexColor('#16324F'),alignment=TA_CENTER,spaceAfter=12)); styles.add(ParagraphStyle(name='H2X',parent=styles['Heading2'],fontName='Helvetica-Bold',fontSize=14,leading=18,textColor=colors.HexColor('#16324F'),spaceBefore=8,spaceAfter=6)); styles.add(ParagraphStyle(name='BX',parent=styles['BodyText'],fontSize=9.2,leading=13,spaceAfter=6)); styles.add(ParagraphStyle(name='SX',parent=styles['BodyText'],fontSize=8,leading=10)); styles.add(ParagraphStyle(name='AlertX',parent=styles['BodyText'],fontSize=9.2,leading=14,textColor=colors.HexColor('#674a07'),backColor=colors.HexColor('#fff4d6'),borderColor=colors.HexColor('#e1be53'),borderWidth=.6,borderPadding=8,spaceBefore=6,spaceAfter=8))
P=lambda x,s='BX':Paragraph(x,styles[s]); story=[Spacer(1,.22*inch),P('Instituto Tecnológico de Tláhuac 3 · Ingeniería en Ciencia de Datos','SX'),Spacer(1,.12*inch),P('Informe de la Fase 3','CoverX'),P('Ejecución, modelado y visualización','SX'),Spacer(1,.12*inch),P('RentaSmart: Estimador de Precios de Vivienda','H2X'),P('Equipo 09  |  Ciudad de México  |  2026','SX'),Spacer(1,8)]
story.append(P('<b>Alcance ejecutado.</b> Se analizaron medianas de renta anunciada por zona y tipo de inmueble. La fuente publica celdas agregadas a partir de anuncios, no registros individuales; por ello el modelo estima medianas de segmentos y no el precio de una vivienda específica.','AlertX'))
story += [P('Fuente y método','H2X'),P('Aptología (2026), corte al 22 de noviembre de 2026, licencia CC BY 4.0. El archivo contiene 10,179 celdas agregadas de México. El alcance de este análisis filtra Ciudad de México, alquiler y tipos casa/departamento. Se excluyeron ocho filas sin superficie mediana; se revisaron duplicados y no se encontraron bajo la llave zona/tipo/operación. Quedaron 248 filas analíticas para 197 zonas. La fuente solo incorpora celdas con al menos cinco anuncios y no publica datos personales.'),P('Depuración y EDA','H2X'),P('Se requirió precio y superficie mayores que cero y un mínimo de cinco anuncios. Los valores fuera de 1.5 IQR se marcaron (17 filas) y se conservaron: no hay registros individuales para confirmar que sean errores. La dispersión muestra asociación entre superficie y renta mediana, con amplitud por zona y tipo; esta asociación no implica causalidad.'),Image(str(chart1),width=6.9*inch,height=3.45*inch),P('Dispersión de medianas por colonia y tipo. Cada punto representa una celda agregada; no un anuncio individual.','SX'),PageBreak()]
story += [P('Análisis exploratorio','H2X'),Image(str(chart2),width=6.1*inch,height=2.35*inch),P('La comparación por tipo muestra medianas distintas y tamaños de muestra diferentes. El tablero permite filtrar el tipo y buscar colonias; los intervalos p25-p75 ayudan a mostrar la dispersión dentro de cada celda.','BX'),P('Modelo y evaluación','H2X')]
modeldata=[[P('<b>Métrica</b>','SX'),P('<b>Prueba por colonia</b>','SX'),P('<b>Interpretación</b>','SX')],[P('R²','SX'),P(f'{m["R2"]:.3f}','SX'),P('El modelo explica cerca del 79.6% de la variación observada en estas medianas de prueba. No equivale a precisión para anuncios individuales.','SX')],[P('MAE','SX'),P(f'${m["MAE_MXN"]:,.0f} MXN','SX'),P('Error absoluto medio frente a medianas agregadas.','SX')],[P('RMSE','SX'),P(f'${m["RMSE_MXN"]:,.0f} MXN','SX'),P('Penaliza más las diferencias grandes.','SX')],[P('Línea base MAE','SX'),P(f'${m["baseline_median_MAE_MXN"]:,.0f} MXN','SX'),P('Predicción constante con la mediana de entrenamiento.','SX')]]
t=Table(modeldata,colWidths=[1.15*inch,1.5*inch,4.15*inch],repeatRows=1); t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dce8f2')),('GRID',(0,0),(-1,-1),.5,colors.HexColor('#d9d9d9')),('VALIGN',(0,0),(-1,-1),'TOP'),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f3f7fa')]),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)])); story.append(t)
story += [Spacer(1,8),P(f'El modelo usa regresión lineal sobre log(renta mediana), con log(superficie mediana) y tipo de inmueble. La separación se hizo por zona: {m["train_rows"]} filas de entrenamiento y {m["test_rows"]} filas de prueba distribuidas en {m["test_zones"]} zonas no vistas. El MAE de la línea base es mayor, pero la muestra pequeña y agregada limita la generalización. Se debe validar con más cortes temporales y cobertura local antes de cualquier uso real.'),P('Lectura responsable','H2X'),P('Estos son precios solicitados, no precios de cierre ni avalúos. La cobertura de portales puede ser desigual por zona. No interpretar como precio justo, recomendación para fijar renta o evidencia de asequibilidad. No utilizar para decidir quién puede acceder a vivienda.','AlertX'),PageBreak()]
story += [P('Dashboard y entregables','H2X'),P('Se incluye un tablero HTML interactivo local con filtros por tipo y colonia, puntos explorables con tooltip, KPIs y una calculadora experimental construida con el modelo. El tablero no tiene URL pública ni está desplegado; la carpeta dashboard incluye el archivo fuente y los datos agregados para ejecutarlo sin conexión a servicios externos.'),P('Accesos locales dentro del paquete','H2X'),P('<b>Notebook ejecutado:</b> notebooks/RentaSmart_Fase3.ipynb<br/><b>Dashboard:</b> dashboard/index.html<br/><b>Pipeline:</b> src/pipeline.py<br/><b>Datos crudos y limpios:</b> data/mx-renta-por-colonia-2026-11.csv y data/cdmx_renta_limpia.csv<br/><b>README y dependencias:</b> README.md y requirements.txt'),P('Equipo registrado en Fase 2','H2X')]
team=[[P('<b>Integrante</b>','SX'),P('<b>Correo institucional</b>','SX'),P('<b>Rol</b>','SX')],[P('Gerardo Calet Ibarra Jiménez (líder)','SX'),P('L261120362@Tlahuac3.tecnm.mx','SX'),P('Líder de Proyecto / Ingeniero de Datos','SX')],[P('Jonathan Salazar Sánchez','SX'),P('L261120305@Tlahuac3.tecnm.mx','SX'),P('Científico de Datos / Visualización','SX')]]
t2=Table(team,colWidths=[2.2*inch,2.2*inch,2.4*inch]); t2.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dce8f2')),('GRID',(0,0),(-1,-1),.5,colors.HexColor('#d9d9d9')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)])); story.append(t2)
story += [P('Pendiente para entrega final','H2X'),P('El registro de Fase 2 muestra dos integrantes, mientras la guía de Fase 3 solicita equipos de 3 a 4; confirmen la composición del equipo. El código fuente, notebook y dashboard están en el paquete descargable. Debe subirse a GitHub para obtener un enlace de repositorio y publicarse el tablero en Streamlit u otro servicio si se requiere enlace público. Si el tablero se entrega como archivo fuente, agregar capturas de todas las vistas. El líder deberá entregar en Classroom y agregar los nombres y correos institucionales en el comentario privado.','AlertX'),P('Fuente: Aptología (2026). <i>Precios de renta y venta de vivienda por colonia en México</i>, noviembre 2026, CC BY 4.0. https://www.aptologia.com/mx/datos-abiertos','SX')]
def footer(c,doc): c.saveState(); c.setFont('Helvetica',8); c.setFillColor(colors.HexColor('#68798a')); c.drawString(.68*inch,.38*inch,'RentaSmart · Fase 3'); c.drawRightString(7.82*inch,.38*inch,str(doc.page)); c.restoreState()
doc=SimpleDocTemplate(str(OUT/'Equipo09_Fase3_RentaSmart.pdf'),pagesize=letter,rightMargin=.68*inch,leftMargin=.68*inch,topMargin=.58*inch,bottomMargin=.62*inch,title='RentaSmart Fase 3'); doc.build(story,onFirstPage=footer,onLaterPages=footer)
print('pdf created')





