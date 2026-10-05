"""R8 fixed one-factor protocol: retain all outcomes, no tuning."""
import hashlib,json,os
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import pandas as pd
from common_recovery_campaign import scenario,completion_hour,truncated_metrics
from operational_calibration import ORIGINAL,PID
from orbit_env import cylindrical_sunlight
from swmpc_sim import PAR,G0,run,metrics
CASES={'baseline':{},'delay6':{'delay':6},'delay12':{'delay':12},'delay24':{'delay':24},'area8':{'area':8},'area6':{'area':6},'gen1200':{'gen':1200},'gen1800':{'gen':1800},'load480':{'load':480},'load720':{'load':720},'cap1200':{'cap':1200},'cap1800':{'cap':1800},'eta04':{'eta':.4},'eta06':{'eta':.6}}
HASH=hashlib.sha256(Path('../R8_SENSITIVITY_PROTOCOL.md').read_bytes()).hexdigest()
def job(args):
 case,event,h,ctrl,seed=args;cfg=CASES[case];cap=cfg.get('cap',1500.);eta=cfg.get('eta',.5)
 par=dict(E_max=cap,E0=.9*cap,E_min=.2*cap,E_phys_min=cap/15,eta_T=eta,kp_power=G0*PAR['Isp']/(2*eta))
 sc=scenario(event,h,0.);sun=cylindrical_sunlight(sc.tr)
 sigma=json.loads(Path('../data/operational_sigma_profile.json').read_text())[f'{h/1000:.0f}km']['sigma_lam_p95']
 out=run(sc,ctrl,seed=seed,par=par,sigma_prof=sigma,pid_wn=PID[f'{h/1000:.0f}km']['wn'],strict_solar=True,generation_W=cfg.get('gen',1500.)*sun,load_W=np.full(len(sun),cfg.get('load',600.)),recovery_start_h=72.+cfg.get('delay',0.),recovery_area_max=cfg.get('area',10.))
 return dict(protocol_sha256=HASH,case=case,event=event,h_km=h/1000,ctrl=ctrl,seed=seed,capacity_Wh=cap,eta=eta,completion_event_h=completion_hour(event,h,0.),recovery_start_h=72.+cfg.get('delay',0.),recovery_area_max=cfg.get('area',10.),generation_W=cfg.get('gen',1500.),load_W=cfg.get('load',600.),full=metrics(out,t_from=-12.),event_window=truncated_metrics(out,72.,0.),after_event=metrics(out,t_from=72.))
def summarize(rows):
 flat=[]
 for r in rows:
  x={k:v for k,v in r.items() if k not in ('full','event_window','after_event')}
  for w in ('full','event_window','after_event'):x.update({w+'_'+k:v for k,v in r[w].items()})
  flat.append(x)
 pd.DataFrame(flat).to_csv('../data/resource_sensitivity_per_seed.csv',index=False)
 idx={(r['case'],r['event'],r['h_km'],r['ctrl'],r['seed']):r for r in rows};pairs=[]
 for r in rows:
  if r['ctrl']!='swmpc':continue
  pi=idx[(r['case'],r['event'],r['h_km'],'pid_lowdrag_preview',r['seed'])];a,b=pi['full'],r['full'];cap=r['capacity_Wh']
  valid=all(v['powered_regime_valid'] and v['n_floor_viol']==0 and v['n_energy_floor_viol']==0 and v['unserved_bus_Wh']==0 for v in (a,b))
  matched=valid and min(a['E_end_Wh'],b['E_end_Wh'])>=cap-5 and abs(a['E_end_Wh']-b['E_end_Wh'])<=5 and abs(a['x_end']-b['x_end'])<=1
  x={k:r[k] for k in ('case','event','h_km','seed','capacity_Wh','completion_event_h')};x.update(resource_matched=matched,raw_saving_pct=100*(a['fuel_g']-b['fuel_g'])/a['fuel_g'],E_difference_Wh=abs(a['E_end_Wh']-b['E_end_Wh']),x_difference_m=abs(a['x_end']-b['x_end']))
  for name,v in [('pi',pi),('mpc',r)]:
   for w in ('full','event_window','after_event'):
    for k in ('fuel_g','rms','minE','E_end_Wh','qp_failures_all','n_floor_viol','n_energy_floor_viol','unserved_bus_Wh'):x[name+'_'+w+'_'+k]=v[w][k]
  pairs.append(x)
 p=pd.DataFrame(pairs);p.to_csv('../data/resource_sensitivity_pairs.csv',index=False);report=[]
 for (case,event,h),g in p.groupby(['case','event','h_km'],sort=False):
  q=g[g.resource_matched];report.append(dict(case=case,event=event,h_km=h,n_pairs=len(g),n_matched=len(q),matched_saving_min=float(q.raw_saving_pct.min()) if len(q) else None,matched_saving_max=float(q.raw_saving_pct.max()) if len(q) else None,matched_saving_mean=float(q.raw_saving_pct.mean()) if len(q) else None,raw_saving_mean=float(g.raw_saving_pct.mean()),event_rms_ratio=float((g.mpc_event_window_rms/g.pi_event_window_rms).mean()),pi_min_SOC=float((g.pi_full_minE/g.capacity_Wh).min()),mpc_min_SOC=float((g.mpc_full_minE/g.capacity_Wh).min()),pi_endpoint_min=float(g.pi_full_E_end_Wh.min()),mpc_endpoint_min=float(g.mpc_full_E_end_Wh.min()),qp_failures=int(g.mpc_full_qp_failures_all.sum()),pi_floor_viol=int(g.pi_full_n_floor_viol.sum()),mpc_floor_viol=int(g.mpc_full_n_floor_viol.sum())))
 pd.DataFrame(report).to_csv('../data/resource_sensitivity_summary.csv',index=False)
 print('Runs',len(rows),'matched',int(p.resource_matched.sum()),'/',len(p),flush=True)
if __name__=='__main__':
 jobs=[(case,e,h,c,s) for case in CASES for e in ORIGINAL for h in (400e3,550e3) for c in ('pid_lowdrag_preview','swmpc') for s in range(3)]
 cp=Path('../data/resource_sensitivity_checkpoint.json');rows=json.loads(cp.read_text()) if cp.exists() else [];assert all(r['protocol_sha256']==HASH for r in rows)
 done={(r['case'],r['event'],r['h_km']*1000,r['ctrl'],r['seed']) for r in rows}
 with Pool(int(os.environ.get('NPROC','4'))) as pool:
  for r in pool.imap(job,[j for j in jobs if j not in done],chunksize=1):
   rows.append(r);cp.write_text(json.dumps(rows,allow_nan=False))
   if len(rows)%18==0:print('Completed',len(rows),'/',len(jobs),flush=True)
 rows.sort(key=lambda r:jobs.index((r['case'],r['event'],r['h_km']*1000,r['ctrl'],r['seed'])))
 Path('../data/resource_sensitivity_campaign.json').write_text(json.dumps(rows,indent=2,allow_nan=False));summarize(rows)
