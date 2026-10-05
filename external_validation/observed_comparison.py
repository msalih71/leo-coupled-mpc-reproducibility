"""Independent Swarm DNS POD comparison of the supplied TIM and NRLMSISE-00.
Run from external_validation/; download ESA VirES HAPI records locally.
"""
import sys, csv, io, pathlib
import numpy as np
import pandas as pd
import requests
sys.path.insert(0, str(pathlib.Path('../code').resolve()))
from orbit_env import jd_from_dt64, gmst, sun_eci, msis_density
from tim import TIM
EVENTS = {'Feb2022':'2022-02-02T00:00:00','Mar2015':'2015-03-16T00:00:00','May2024':'2024-05-09T12:00:00'}
API='https://vires.services/hapi/data'
OUT=pathlib.Path('observed_results'); OUT.mkdir(exist_ok=True)
summary=[]
for event,start in EVENTS.items():
 t0=pd.Timestamp(start,tz='UTC'); t1=t0+pd.Timedelta(hours=72)
 hist=t0-pd.Timedelta(hours=48)
 for sat in ('A','B'):
  coll=f'SW_OPER_DNS{sat}POD_2_'
  params={'id':coll,'time.min':hist.isoformat().replace('+00:00','Z'),
          'time.max':t1.isoformat().replace('+00:00','Z'),
          'parameters':'Timestamp,Latitude_GD,Longitude_GD,Height_GD,density,validity_flag'}
  response=requests.get(API,params=params,timeout=120); response.raise_for_status()
  df=pd.read_csv(io.StringIO(response.text),header=None,names=['Timestamp','Latitude_GD','Longitude_GD','Height_GD','density','validity_flag'])
  df['Timestamp']=pd.to_datetime(df['Timestamp'],utc=True)
  df=df.sort_values('Timestamp').drop_duplicates('Timestamp')
  raw=len(df)
  valid=(df['validity_flag']==0)&np.isfinite(df['density'])&(df['density']>0)&np.isfinite(df['Height_GD'])
  df=df[valid].copy()
  if len(df)<100: print('NO VALID',event,sat,raw,len(df),flush=True);continue
  dates=df['Timestamp'].to_numpy(dtype='datetime64[ns]')
  lat=np.deg2rad(df['Latitude_GD'].to_numpy(float));lon=np.deg2rad(df['Longitude_GD'].to_numpy(float))
  sun=sun_eci(jd_from_dt64(dates));dec=np.arcsin(sun[:,2]);ra=np.arctan2(sun[:,1],sun[:,0])
  hour_angle=(gmst(jd_from_dt64(dates))+lon-ra+np.pi)%(2*np.pi)-np.pi
  tr={'dt64':dates,'t':(dates-dates[0])/np.timedelta64(1,'s'),
      'lat':lat,'lon':lon,'alt':df['Height_GD'].to_numpy(float),
      'dec':dec,'hour_angle':hour_angle}
  medalt=np.median(tr['alt'])/1000
  href=400e3 if abs(medalt-400)<abs(medalt-550) else 550e3
  model=TIM.load(href/1e3)
  tim,_=model.hindcast(tr,href)
  msis=msis_density(tr,0)
  df['TIM_kg_m3']=tim;df['MSIS00_kg_m3']=msis
  df['h_ref_km']=int(href/1e3)
  df['event']=event;df['satellite']=sat
  df=df[(df['Timestamp']>=t0)&(df['Timestamp']<t1)]
  df.to_csv(OUT/f'{event}_Swarm{sat}_aligned.csv',index=False)
  obs=df['density'].to_numpy()
  row={'event':event,'satellite':sat,'n_raw_with_history':raw,'n_valid_storm':len(df),
       'alt_km_median':float(df['Height_GD'].median()/1000),'model_ref_km':int(href/1e3),
       'alt_offset_km':float(df['Height_GD'].median()/1000-href/1e3),
       'observed_median_kg_m3':float(np.median(obs))}
  for name,col in [('TIM','TIM_kg_m3'),('NRLMSISE00','MSIS00_kg_m3')]:
   pred=df[col].to_numpy();lr=np.log(pred/obs)
   row[name+'_rms_log']=float(np.sqrt(np.mean(lr**2)))
   row[name+'_median_ratio']=float(np.median(pred/obs))
   row[name+'_bias_log']=float(np.mean(lr))
   row[name+'_corr_log']=float(np.corrcoef(np.log(pred),np.log(obs))[0,1])
  summary.append(row);print(row,flush=True)
pd.DataFrame(summary).to_csv(OUT/'swarm_observed_summary.csv',index=False)
