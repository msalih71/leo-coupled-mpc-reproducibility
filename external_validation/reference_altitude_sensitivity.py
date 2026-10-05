"""Check both TIM calibrations on Swarm tracks after 24h in-event warm-up.
Requires six downloaded POD aligned CSVs and the corresponding JB 5-min files.
RMS log is evaluated on the remaining 48h; this is outside the 53-degree
calibration geometry and is not a re-fit of either TIM coefficient vector.
"""
from pathlib import Path
import sys
import pandas as pd
import numpy as np
sys.path.insert(0,str(Path('../code').resolve()))
from tim import TIM
from orbit_env import sun_eci,jd_from_dt64,gmst
rows=[]
for file in sorted(Path('observed_results').glob('*_aligned.csv')):
 df=pd.read_csv(file)
 times=pd.to_datetime(df.Timestamp,utc=True).to_numpy(dtype='datetime64[ns]')
 jd=jd_from_dt64(times);sun=sun_eci(jd)
 lat=np.deg2rad(df.Latitude_GD.to_numpy(float));lon=np.deg2rad(df.Longitude_GD.to_numpy(float))
 tr={'dt64':times,'t':(times-times[0])/np.timedelta64(1,'s'),'lat':lat,'lon':lon,
     'alt':df.Height_GD.to_numpy(float),'dec':np.arcsin(sun[:,2]),
     'hour_angle':(gmst(jd)+lon-np.arctan2(sun[:,1],sun[:,0])+np.pi)%(2*np.pi)-np.pi}
 jb=pd.read_csv(file.with_name(file.stem+'_5min_JB.csv'))
 idx=np.arange(24*120,len(df),10)
 assert len(jb)==len(df)//10 and np.array_equal(jb.Timestamp.to_numpy(),df.Timestamp.to_numpy()[::10])
 obs=df.density.to_numpy(float)[idx]
 rec={'event':df.event.iloc[0],'satellite':df.satellite.iloc[0],
      'n':len(idx),'alt_median_km':float(df.Height_GD.median()/1000)}
 for h in (400000,550000):
  pred,_=TIM.load(h/1000).hindcast(tr,h)
  rec[f'TIM_{h//1000}_rms_log']=float(np.sqrt(np.mean(np.log(pred[idx]/obs)**2)))
 rec['JB2008_rms_log']=float(np.sqrt(np.mean(np.log(jb.JB2008_kg_m3.to_numpy()[idx//10]/obs)**2)))
 rows.append(rec)
out=pd.DataFrame(rows);out.to_csv('reference_altitude_sensitivity.csv',index=False)
print(out.to_string(index=False,float_format=lambda v:f'{v:.3f}'))
