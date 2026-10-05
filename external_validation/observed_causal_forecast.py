"""Retrospective TIM forecast check on Swarm POD, with 24h in-event warmup.
At each 5-min issue time, predict actual future 5-min density windows at 5 min,
1 h, 3 h leads. The ap availability rule is causal; prior-day F10.7 comes
from a historical archive, not a saved real-time uplink log. No density
assimilation or control-state feedback is used.
"""
from pathlib import Path
import sys
import numpy as np,pandas as pd
sys.path.insert(0,str(Path('../code').resolve()))
from tim import TIM,TIMPredictor
from orbit_env import jd_from_dt64,sun_eci,gmst
rows=[]
for file in sorted(Path('observed_results').glob('*_aligned.csv')):
 df=pd.read_csv(file).iloc[::2].reset_index(drop=True) # 30 s -> 60 s
 dt=pd.to_datetime(df.Timestamp,utc=True).to_numpy(dtype='datetime64[ns]')
 jd=jd_from_dt64(dt);sun=sun_eci(jd);ra=np.arctan2(sun[:,1],sun[:,0]);dec=np.arcsin(sun[:,2])
 lat=np.deg2rad(df.Latitude_GD.to_numpy(float));lon=np.deg2rad(df.Longitude_GD.to_numpy(float))
 tr={'dt64':dt,'t':np.arange(len(dt),dtype=float)*60.,'lat':lat,'lon':lon,'alt':df.Height_GD.to_numpy(float),'dec':dec,
     'hour_angle':(gmst(jd)+lon-ra+np.pi)%(2*np.pi)-np.pi}
 href=int(df.h_ref_km.iloc[0])*1000;predictor=TIMPredictor(TIM.load(href/1000),tr,href,lat_s=3600)
 # Issue times 24h-69h; sample every 5 minutes. All future windows are available.
 issues=np.arange(24*60,69*60,5);truth=df.density.to_numpy(float)
 errs={5:[],60:[],180:[]}
 for i in issues:
  fut=predictor.predict_steps(i,36,5)
  for lead in errs:
   j=lead//5-1
   observed=np.mean(truth[i+j*5:i+(j+1)*5])
   errs[lead].append(np.log(fut[j]/observed))
 for lead,e in errs.items():
  e=np.array(e);rows.append({'event':df.event.iloc[0],'satellite':df.satellite.iloc[0],
   'lead_min':lead,'n_issues':len(e),'rms_log':float(np.sqrt(np.mean(e**2))),
   'median_model_over_obs':float(np.median(np.exp(e))),'mean_log_bias':float(e.mean())})
 print(file.stem, [(k,round(float(np.sqrt(np.mean(np.square(v)))),3)) for k,v in errs.items()],flush=True)
pd.DataFrame(rows).to_csv('observed_causal_forecast_summary.csv',index=False)
