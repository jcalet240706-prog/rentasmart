import json, math, re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/mx-renta-por-colonia-2026-11.csv'
OUT=ROOT/'data'
raw=pd.read_csv(DATA)
cdmx=raw[(raw.pais=='mx') & (raw.ciudad.astype(str).str.normalize('NFKD').str.encode('ascii',errors='ignore').str.decode('ascii').str.lower().eq('ciudad de mexico')) & (raw.operacion=='alquiler') & (raw.tipo.isin(['casa','departamento']))].copy()
before=len(cdmx); miss=cdmx[['zona','tipo','avisos','mediana','mediana_area_m2']].isna().sum().to_dict(); dup=int(cdmx.duplicated(['zona','tipo','operacion']).sum())
cdmx['mediana']=pd.to_numeric(cdmx.mediana,errors='coerce'); cdmx['mediana_area_m2']=pd.to_numeric(cdmx.mediana_area_m2,errors='coerce'); cdmx['avisos']=pd.to_numeric(cdmx.avisos,errors='coerce')
cdmx=cdmx.dropna(subset=['zona','tipo','mediana','mediana_area_m2','avisos'])
cdmx=cdmx[(cdmx.mediana>0)&(cdmx.mediana_area_m2>0)&(cdmx.avisos>=5)]
cdmx=cdmx.drop_duplicates(['zona','tipo','operacion']).copy()
# Extreme-value diagnostic uses 1.5 IQR fences on rent. Flag, retain, as these are aggregates of valid market observations.
q1,q3=cdmx.mediana.quantile([.25,.75]); lo,hi=q1-1.5*(q3-q1),q3+1.5*(q3-q1)
cdmx['atipico_iqr']=(cdmx.mediana<lo)|(cdmx.mediana>hi)
cdmx['log_area']=np.log(cdmx.mediana_area_m2); cdmx['log_renta']=np.log(cdmx.mediana)
cdmx.to_csv(OUT/'cdmx_renta_limpia.csv',index=False,encoding='utf-8-sig')
# Group split by colonia to prevent same-area categories straddling sets.
zones=sorted(cdmx.zona.unique()); test_z=set(zones[::5]); train=cdmx[~cdmx.zona.isin(test_z)]; test=cdmx[cdmx.zona.isin(test_z)]
# OLS on log rent, log median area and property type. Use training-only centering/scaling.
mu=train.log_area.mean(); sd=train.log_area.std() or 1

def design(df):
 x=(df.log_area.to_numpy()-mu)/sd
 is_house=(df.tipo.to_numpy()=='casa').astype(float)
 return np.column_stack([np.ones(len(df)),x,is_house])
coef=np.linalg.lstsq(design(train),train.log_renta.to_numpy(),rcond=None)[0]
pred=np.exp(design(test)@coef); actual=test.mediana.to_numpy()
mae=float(np.mean(np.abs(actual-pred))); rmse=float(np.sqrt(np.mean((actual-pred)**2))); r2=float(1-np.sum((actual-pred)**2)/np.sum((actual-actual.mean())**2))
med_base=float(train.mediana.median()); mae_base=float(np.mean(np.abs(actual-med_base)))
metrics={'rows_raw_scope':int(before),'rows_clean':int(len(cdmx)),'zones':int(cdmx.zona.nunique()),'train_rows':int(len(train)),'test_rows':int(len(test)),'test_zones':int(len(test_z)),'target':'Mediana de renta mensual anunciada por colonia y tipo (MXN)','model':'OLS en log(renta) con log(superficie mediana) y tipo de inmueble; split por colonia','R2':r2,'MAE_MXN':mae,'RMSE_MXN':rmse,'baseline_median_MAE_MXN':mae_base,'outlier_fences_MXN':[float(lo),float(hi)],'outliers_flagged_retained':int(cdmx.atipico_iqr.sum()),'missing_counts_raw_scope':{k:int(v) for k,v in miss.items()},'duplicates_removed':dup,'coefficients':coef.tolist(),'log_area_mean':float(mu),'log_area_sd':float(sd),'train_rent_median_MXN':med_base,'test_actual_median_MXN':float(np.median(actual)),'test_prediction_median_MXN':float(np.median(pred))}
(OUT/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
# summarize for plots and dashboard
roll=cdmx.groupby('tipo').agg(n=('mediana','size'),mediana=('mediana','median'),area=('mediana_area_m2','median')).reset_index()
summary={'metrics':metrics,'by_type':roll.to_dict(orient='records'),'zones':sorted(cdmx.zona.unique()),'records':cdmx[['zona','tipo','mediana','p25','p75','mediana_area_m2','avisos']].to_dict(orient='records')}
(OUT/'dashboard_data.json').write_text(json.dumps(summary,ensure_ascii=False),encoding='utf-8')
# SVG charts for notebook/PDF.
def chart(title, rows, xkey, ykey, xlabel, ylabel, filename):
 W,H=820,430; L,R,T,B=72,22,58,65; pw=W-L-R; ph=H-T-B
 xx=np.array([float(r[xkey]) for r in rows]); yy=np.array([float(r[ykey]) for r in rows]); xmax=xx.max()*1.05; ymax=yy.max()*1.1
 pts=[]
 for x,y in zip(xx,yy):
  px=L+(x/xmax)*pw; py=T+ph-(y/ymax)*ph; pts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="#168AAD" opacity=".62"><title>{x:.0f} m², ${y:,.0f}</title></circle>')
 grid=[]
 for i in range(5):
  y=T+ph-i*ph/4; grid.append(f'<line x1="{L}" y1="{y}" x2="{L+pw}" y2="{y}" stroke="#d8e1e8"/><text x="{L-8}" y="{y+4}" text-anchor="end" font-size="11" fill="#526579">${ymax*i/4:,.0f}</text>')
 svg=f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><rect width="100%" height="100%" fill="white"/><text x="{W/2}" y="28" text-anchor="middle" font-family="Arial" font-size="18" font-weight="bold" fill="#16324F">{title}</text>{''.join(grid)}<line x1="{L}" y1="{T}" x2="{L}" y2="{T+ph}" stroke="#526579"/><line x1="{L}" y1="{T+ph}" x2="{L+pw}" y2="{T+ph}" stroke="#526579"/>{''.join(pts)}<text x="{L+pw/2}" y="{H-18}" text-anchor="middle" font-family="Arial" font-size="12" fill="#526579">{xlabel}</text><text transform="translate(18 {T+ph/2}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="12" fill="#526579">{ylabel}</text></svg>'''
 (ROOT/'notebooks'/filename).write_text(svg,encoding='utf-8')
chart('Renta mediana frente a superficie mediana',summary['records'],'mediana_area_m2','mediana','Superficie mediana (m²)','Renta mensual mediana (MXN)','eda_area_renta.svg')
# type distribution bar SVG
rows=roll.to_dict(orient='records'); W,H=720,300; mx=max(float(r['mediana']) for r in rows); bars=''
for i,r in enumerate(rows):
 y=75+i*90; width=500*float(r['mediana'])/mx
 bars+=f'<text x="20" y="{y+22}" font-family="Arial" font-size="14">{r["tipo"]}</text><rect x="140" y="{y}" width="{width:.1f}" height="38" fill="#168AAD"/><text x="{150+width:.1f}" y="{y+24}" font-family="Arial" font-size="13">${r["mediana"]:,.0f}  (n={r["n"]})</text>'
(ROOT/'notebooks'/'eda_tipo.svg').write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}"><rect width="100%" height="100%" fill="white"/><text x="360" y="30" text-anchor="middle" font-family="Arial" font-size="18" font-weight="bold" fill="#16324F">Mediana de renta mensual por tipo</text>{bars}</svg>',encoding='utf-8')
print(json.dumps(metrics,ensure_ascii=False,indent=2))
