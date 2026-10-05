"""Predefined R5 resource/mismatch campaign; paired seeds 0--2; no retuning.
Future eclipse and scheduled payload loads are known commands/geometry.
Measured truth density is never exposed to control decisions.
"""
import json,os
from functools import lru_cache
from pathlib import Path
from multiprocessing import Pool
import numpy as np,pandas as pd
from swmpc_sim import Scenario,run,metrics,PAR
from orbit_env import cylindrical_sunlight
from operational_calibration import ORIGINAL,PID
@lru_cache(None)
def scenario(ev,h):return Scenario(ev,ORIGINAL[ev],h)
def job(a):
 case,ev,h,ctrl,seed=a;sc=scenario(ev,h);tag=f'{h/1000:.0f}km'
 sig=json.load(open('../data/operational_sigma_profile.json'))[tag]['sigma_lam_p95']
 par={};kw={};t=sc.tr['t']/3600-sc.spinup_h
 if case=='low_initial_energy':par['E0']=600.
 if case in ('eclipse_1500W_terminal','eclipse_payload_terminal'):kw['terminal_energy']=(72.,1450.)
 if case.startswith('eclipse'):
  light=cylindrical_sunlight(sc.tr);day_power=900. if case=='eclipse_900W' else 1500.
  kw['generation_W']=day_power*light
  if case in ('eclipse_payload','eclipse_payload_terminal'):kw['load_W']=600.+200.*((t>=30)&(t<36))
 if case=='scheduled_area':
  tc=np.arange(int(sc.dur_h*3600/PAR['dt']))*PAR['dt']/3600-sc.spinup_h
  kw['area_floor']=np.where((tc>=30)&(tc<36),8.5,PAR['A_min'])
 if case=='measurement_gap':kw['measurement_gap']=(24.,26.)
 if case=='index_outage':kw['index_outage']=(24.,30.)
 if case=='Cd_truth_1.8':kw['cd_truth']=1.8
 if case=='Cd_truth_2.6':kw['cd_truth']=2.6
 if case=='forced_solver_gap':kw['solver_gap']=(24.,24.5)
 o=run(sc,ctrl,seed=seed,par=par,sigma_prof=sig,pid_wn=PID[tag]['wn'],strict_solar=True,**kw)
 m=metrics(o)
 return dict(case=case,event=ev,h_km=h/1000,ctrl=ctrl,seed=seed,metrics=m,
             sunlight_fraction=float(cylindrical_sunlight(sc.tr)[(t>=0)&(t<72)].mean()) if case.startswith('eclipse') else 1.)
if __name__=='__main__':
 jobs=[('strict_nominal',ev,h,c,s) for ev in ORIGINAL for h in (400e3,550e3) for c in ('pid_ff','pid_lowdrag','swmpc') for s in range(3)]
 cases=['low_initial_energy','eclipse_900W','eclipse_1500W','eclipse_payload','scheduled_area','measurement_gap','index_outage','Cd_truth_1.8','Cd_truth_2.6']
 jobs += [(case,'May2024',400e3,c,s) for case in cases for c in ('pid_ff','pid_lowdrag','swmpc') for s in range(3)]
 jobs += [(case,'May2024',400e3,'pid_lowdrag_preview',s) for case in ('eclipse_900W','eclipse_1500W','eclipse_payload') for s in range(3)]
 jobs += [(case,'May2024',400e3,'swmpc',s) for case in ('eclipse_1500W_terminal','eclipse_payload_terminal') for s in range(3)]
 jobs += [('forced_solver_gap','May2024',400e3,'swmpc',s) for s in range(3)]
 with Pool(2) as p:r=p.map(job,jobs,chunksize=3)
 Path('../data/resource_campaign.json').write_text(json.dumps(r,indent=2,allow_nan=False))
 df=pd.DataFrame([{**{k:v for k,v in x.items() if k!='metrics'},**x['metrics']} for x in r])
 df.to_csv('../data/resource_campaign_per_seed.csv',index=False)
 df.groupby(['case','event','h_km','ctrl']).mean(numeric_only=True).to_csv('../data/resource_campaign_summary.csv')
 print('Finished',len(r),'runs');print(df.groupby(['case','event','h_km','ctrl'])[['rms','fuel_g','minE','n_floor_viol','n_energy_floor_viol','n_area_floor_viol','qp_failures_all','powered_regime_valid']].mean().to_string())
