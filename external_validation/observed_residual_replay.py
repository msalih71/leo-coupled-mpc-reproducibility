"""Controlled sensitivity study: transfer Swarm density residuals onto R3's 53-deg plant.
The Swarm measurements are on DIFFERENT orbits; this is not an actual Swarm
station-keeping demonstration or a direct replay of its measured drag.
"""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
sys.path.insert(0,str(Path('../code').resolve()))
from swmpc_sim import Scenario,run,metrics
from tim import SW
EVENTS={'Feb2022':'2022-02-02T00:00','Mar2015':'2015-03-16T00:00','May2024':'2024-05-09T12:00'}
sig=json.load(open('../data/sigma_profile.json'))
pid=json.load(open('../data/pid_tuning.json'))
def storm_window(t0):
 t=np.datetime64(t0)+np.arange(0,72*3600,10800)*np.timedelta64(1,'s');ap=SW.ap_at(t)
 ix=np.flatnonzero(ap>=39)
 return max(0.,ix[0]*3-3.),min(72.,ix[-1]*3+15.)
rows=[]
for event,t0 in EVENTS.items():
 for h,sat in [(400e3,'A'),(550e3,'B')]:
  df=pd.read_csv(f'observed_results/{event}_Swarm{sat}_aligned.csv')
  tt=pd.to_datetime(df.Timestamp,utc=True).to_numpy(dtype='datetime64[ns]').astype('int64')
  ratio=(df.density/df.MSIS00_kg_m3).to_numpy(float)
  scn=Scenario(event,t0,h)
  grid=scn.tr['dt64'].astype('datetime64[ns]').astype('int64')
  factor=np.interp(grid,tt,ratio,left=ratio[0],right=ratio[-1])
  original=scn.rho.copy();scn.rho=original*factor
  print(event,int(h/1000),sat,'ratio p05/p50/p95',np.quantile(factor,[.05,.5,.95]),flush=True)
  for ctrl in ('pid_ff','mpc_thrust','swmpc'):
   for seed in (0,1,2):
    out=run(scn,ctrl,seed=seed,sigma_prof=sig[f'{h/1000:.0f}km']['sigma_lam_p95'],pid_wn=pid[f'{h/1000:.0f}km']['wn'])
    m=metrics(out,storm=storm_window(t0))
    rows.append({'event':event,'alt_km':int(h/1000),'Swarm_residual_source':sat,'controller':ctrl,'seed':seed,
      'ratio_median':float(np.median(factor)),**m})
   print(' ',ctrl,'rms',np.mean([r['rms'] for r in rows[-3:]]),'fuel',np.mean([r['fuel_g'] for r in rows[-3:]]),flush=True)
pd.DataFrame(rows).to_csv('observed_residual_replay_metrics.csv',index=False)
