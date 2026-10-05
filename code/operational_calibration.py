"""Actual-KF horizon forecasts: fixed calibration/test split; strict solar persistence.
Density truth is used only after issuance to score forecasts. Calibration
policy uses the archived margin; test policy uses the newly calibrated profile.
The TIM itself retains its existing model calibration.
"""
import json,os
from pathlib import Path
from multiprocessing import Pool
from functools import lru_cache
import numpy as np,pandas as pd
from swmpc_sim import Scenario,run,metrics
from calib_sigma import storm_days
SIG=json.load(open('../data/sigma_profile.json'));PID=json.load(open('../data/pid_tuning.json'))
ORIGINAL={'Feb2022':'2022-02-02T00:00','Mar2015':'2015-03-16T00:00','May2024':'2024-05-09T12:00'}
CAL=[str(d)+'T00:00' for d in storm_days(8,start='2016-01-01',end='2020-01-01')]
HOLD=[str(d)+'T00:00' for d in storm_days(4,start='2020-01-01',end='2022-01-01')]
TEST={**ORIGINAL,**{f'Holdout{i+1}':t for i,t in enumerate(HOLD)}}
@lru_cache(None)
def scn(t,h):return Scenario(t,t,h)
def job(args):
 label,t,h,seed,sigma=args
 o=run(scn(t,h),'swmpc',seed=seed,sigma_prof=sigma,pid_wn=PID[f'{h/1000:.0f}km']['wn'],
       forecast_stride=3,strict_solar=True)
 f=o['forecast_trace'];pred=f['prediction'];truth=f['truth']; N=pred.shape[1]
 W=np.array([[.8**(k-j) if j<=k else 0 for j in range(N)] for k in range(N)])
 errs=(truth-pred)/pred;weighted=(truth-pred)@W.T/(pred@W.T)
 return dict(label=label,t=t,h_km=h/1000,seed=seed,metrics=metrics(o)),dict(step=errs,weighted=weighted,margin=f['margin'],t_h=f['t_h'])
if __name__=='__main__':
 jobs=[(f'Cal{i+1}',t,h,s,SIG[f'{h/1000:.0f}km']['sigma_lam_p95']) for i,t in enumerate(CAL) for h in (400e3,550e3) for s in range(3)]
 with Pool(2) as p:cal=p.map(job,jobs,chunksize=3)
 profiles={}
 for h in (400,550):
  a=np.concatenate([f['weighted'] for r,f in cal if r['h_km']==h])
  step=np.concatenate([f['step'] for r,f in cal if r['h_km']==h])
  profiles[f'{h}km']=dict(sigma_lam_p95=np.percentile(np.abs(a),95,axis=0).tolist(),sigma_step_p95=np.percentile(np.abs(step),95,axis=0).tolist(),n_issues=int(len(a)),calibration_dates=CAL,lam=.8,strict_solar=True,forecast_stride=3,n_seeds=3)
 Path('../data/operational_sigma_profile.json').write_text(json.dumps(profiles,indent=2))
 jobs=[(label,t,h,s,profiles[f'{h/1000:.0f}km']['sigma_lam_p95']) for label,t in TEST.items() for h in (400e3,550e3) for s in range(3)]
 with Pool(2) as p:test=p.map(job,jobs,chunksize=3)
 summary=[];arrays={}
 for phase,results in [('calibration',cal),('test',test)]:
  for i,(r,f) in enumerate(results):
   key=f'{phase}_{i}'
   arrays[key+'_weighted']=f['weighted'];arrays[key+'_step']=f['step'];arrays[key+'_margin']=f['margin'];arrays[key+'_times']=f['t_h']
   q=np.max(profiles[f"{r['h_km']:.0f}km"]['sigma_lam_p95'])
   row={**r,'phase':phase,'trace_key':key,'n_issues':len(f['t_h']),
        'coverage_base':float(np.mean(np.abs(f['weighted'])<=q)),
        'coverage_effective':float(np.mean(np.abs(f['weighted'])<=f['margin'])),
        'horizon_all_effective':float(np.mean(np.all(np.abs(f['weighted'])<=f['margin'],axis=1))),
        'weighted_p95':float(np.percentile(np.abs(f['weighted']),95)),
        'coverage_by_lead':np.mean(np.abs(f['weighted'])<=f['margin'],axis=0).tolist()}
   summary.append(row)
 np.savez_compressed('../data/operational_forecast_traces.npz',**arrays)
 Path('../data/operational_coverage.json').write_text(json.dumps(dict(calibration_dates=CAL,test_dates=TEST,rows=summary),indent=2,allow_nan=False))
 pd.DataFrame([{**{k:v for k,v in r.items() if k not in ('metrics','coverage_by_lead')},**r['metrics']} for r in summary]).to_csv('../data/operational_coverage_per_seed.csv',index=False)
 print('CAL',CAL,'TEST',TEST,flush=True)
 print('max profiles',{h:max(p['sigma_lam_p95']) for h,p in profiles.items()},flush=True)
 print(pd.DataFrame([{k:v for k,v in r.items() if k not in ('metrics','coverage_by_lead')} for r in summary]).groupby(['phase','label','h_km'])[['coverage_base','coverage_effective','horizon_all_effective']].mean().to_string())
