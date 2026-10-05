"""JB2008 via unmodified SET JB2008.for and a separate thin input/output wrapper.
No SET source or index files are redistributed with the results.
"""
from pathlib import Path
from datetime import date, timedelta
import subprocess
import numpy as np
import pandas as pd
from math import pi
from sys import path
import argparse
path.insert(0,str(Path('../code').resolve()))
from orbit_env import sun_eci, jd_from_dt64, gmst, F_WGS, RE_EQ
BASE=Path('jb_sources')
solar={};dtc={}
for line in (BASE/'SOLFSMY.TXT').open():
 if line.startswith('#'):continue
 q=line.split()
 try:
  y,doy=int(q[0]),int(q[1]);vals=np.array([float(v) for v in q[3:11]])
  if len(vals)==8:solar[date(y,1,1)+timedelta(days=doy-1)]=vals
 except (IndexError,ValueError):pass
for line in (BASE/'DTCFILE.TXT').open():
 q=line.split()
 if len(q)<27 or q[0]!='DTC':continue
 d=date(int(q[1]),1,1)+timedelta(days=int(q[2])-1)
 dtc[d]=np.array([float(x) for x in q[3:27]])
parser=argparse.ArgumentParser()
parser.add_argument('--acc',action='store_true',help='compare Swarm accelerometer product rather than POD')
args=parser.parse_args()
base=Path('observed_acc_results' if args.acc else 'observed_results')
rows=[]
for file in sorted(base.glob('*_aligned.csv')):
 df=pd.read_csv(file)
 df=df[np.isfinite(df.density)&(df.density>0)&(df.density<1e-9)].copy()
 if args.acc:
  t=pd.to_datetime(df.Timestamp,utc=True)
  df=df[(t.dt.minute%5==0)&(t.dt.second==0)].copy()
 else:
  df=df.iloc[::10].copy() # regular 30-s POD cadence -> 5 min
 if len(df)==0: continue
 ts=pd.to_datetime(df.Timestamp,utc=True)
 lon=np.deg2rad(df.Longitude_GD.to_numpy(float));lat=np.deg2rad(df.Latitude_GD.to_numpy(float))
 h=df.Height_GD.to_numpy(float)
 e2=F_WGS*(2-F_WGS);N=RE_EQ/np.sqrt(1-e2*np.sin(lat)**2)
 glat=np.arctan2((N*(1-e2)+h)*np.sin(lat),(N+h)*np.cos(lat))
 dts=ts.to_numpy(dtype='datetime64[ns]');jd=jd_from_dt64(dts)
 sun=sun_eci(jd);ra=np.arctan2(sun[:,1],sun[:,0]);dec=np.arcsin(sun[:,2]);satra=(gmst(jd)+lon)%(2*pi)
 input_lines=[]
 for i,t in enumerate(ts):
  day=t.date();F=solar[day-timedelta(days=1)];M=solar[day-timedelta(days=2)];Y=solar[day-timedelta(days=5)]
  hr=t.hour+t.minute/60+t.second/3600
  j=int(hr);f=hr-j
  ds=dtc[day];ds2=dtc.get(day+timedelta(days=1),ds)
  val=(1-f)*ds[j]+f*(ds[j+1] if j<23 else ds2[0]); dtemp=int(val+0.5)
  ix=[F[0],F[1],F[2],F[3],M[4],M[5],Y[6],Y[7]]
  fields=[jd[i]-2400000.5,ra[i],dec[i],satra[i],glat[i],h[i]/1000,*ix,dtemp]
  input_lines.append(' '.join(f'{x:.12g}' for x in fields))
 run=subprocess.run([str(BASE/'jb_call.exe')],input='\n'.join(input_lines)+'\n',text=True,capture_output=True,check=True)
 if run.stderr.strip():print('stderr',run.stderr[:500])
 out=np.array([float(x) for x in run.stdout.split()])
 assert len(out)==len(df) and np.all(np.isfinite(out)) and np.all(out>0)
 df['JB2008_kg_m3']=out
 name=file.stem
 df.to_csv(base/(name+'_5min_JB.csv'),index=False)
 obs=df.density.to_numpy()
 rec={'event':df.event.iloc[0],'satellite':df.satellite.iloc[0],'n':len(df),'alt_median_km':float(df.Height_GD.median()/1000)}
 for model,col in [('TIM','TIM_kg_m3'),('NRLMSISE00','MSIS00_kg_m3'),('JB2008','JB2008_kg_m3')]:
  q=df[col].to_numpy()/obs; l=np.log(q)
  rec[model+'_rms_log']=float(np.sqrt(np.mean(l*l)))
  rec[model+'_median_ratio']=float(np.median(q))
  rec[model+'_corr_log']=float(np.corrcoef(np.log(df[col]),np.log(obs))[0,1])
 rows.append(rec);print(rec,flush=True)
pd.DataFrame(rows).to_csv(base/('three_model_ACC_5min_comparison.csv' if args.acc else 'three_model_5min_comparison.csv'),index=False)
