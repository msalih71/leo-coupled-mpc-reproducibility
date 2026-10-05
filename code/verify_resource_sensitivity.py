"""Independent arithmetic, design and retained-baseline checks for R8."""
import hashlib,json,itertools
from pathlib import Path
import numpy as np
import pandas as pd
from swmpc_sim import G0,PAR

rows=json.loads(Path('../data/resource_sensitivity_campaign.json').read_text())
p=pd.read_csv('../data/resource_sensitivity_pairs.csv')
hash_=hashlib.sha256(Path('../R8_SENSITIVITY_PROTOCOL.md').read_bytes()).hexdigest()
cases={'baseline':{},'delay6':{'delay':6},'delay12':{'delay':12},'delay24':{'delay':24},'area8':{'area':8},'area6':{'area':6},'gen1200':{'gen':1200},'gen1800':{'gen':1800},'load480':{'load':480},'load720':{'load':720},'cap1200':{'cap':1200},'cap1800':{'cap':1800},'eta04':{'eta':.4},'eta06':{'eta':.6}}
keys={(r['case'],r['event'],r['h_km'],r['ctrl'],r['seed']) for r in rows}
expected=set(itertools.product(cases,('Feb2022','Mar2015','May2024'),(400.,550.),('pid_lowdrag_preview','swmpc'),range(3)))
assert keys==expected and len(rows)==504 and len(p)==252
idx={(r['case'],r['event'],r['h_km'],r['ctrl'],r['seed']):r for r in rows}
max_e=max_f=0.
for r in rows:
 cfg=cases[r['case']];cap=cfg.get('cap',1500.)
 assert r['protocol_sha256']==hash_ and r['capacity_Wh']==cap and r['eta']==cfg.get('eta',.5)
 assert r['recovery_start_h']==72+cfg.get('delay',0) and r['recovery_area_max']==cfg.get('area',10)
 assert r['generation_W']==cfg.get('gen',1500) and r['load_W']==cfg.get('load',600)
 for window in ('full','event_window','after_event'):
  m=r[window];de=m['E_end_Wh']-m['E_start_Wh']
  balance=m['total_generated_Wh']-m['total_load_Wh']-m['total_propulsion_Wh']-m['total_shunt']+m['total_unserved_Wh']
  fuel_energy=(G0*PAR['Isp'])**2/(7200*r['eta'])*m['fuel_g']/1000
  max_e=max(max_e,abs(de-balance));max_f=max(max_f,abs(fuel_energy-m['total_propulsion_Wh']))
 assert abs(r['full']['total_load_Wh']-r['load_W']*(r['completion_event_h']+12))<1e-6
 assert abs(r['event_window']['total_load_Wh']-r['load_W']*72)<1e-6
 assert abs(r['full']['E_start_Wh']-.9*cap)<1e-9
for pair in p.to_dict('records'):
 a=idx[(pair['case'],pair['event'],pair['h_km'],'pid_lowdrag_preview',pair['seed'])]['full']
 b=idx[(pair['case'],pair['event'],pair['h_km'],'swmpc',pair['seed'])]['full']
 valid=all(m['powered_regime_valid'] and m['n_floor_viol']==0 and m['n_energy_floor_viol']==0 and m['unserved_bus_Wh']==0 for m in (a,b))
 matched=valid and min(a['E_end_Wh'],b['E_end_Wh'])>=pair['capacity_Wh']-5 and abs(a['E_end_Wh']-b['E_end_Wh'])<=5 and abs(a['x_end']-b['x_end'])<=1
 assert bool(pair['resource_matched'])==matched
 assert abs(pair['raw_saving_pct']-100*(a['fuel_g']-b['fuel_g'])/a['fuel_g'])<1e-10
assert max_e<1e-8 and max_f<1e-8
legacy=json.loads(Path('../data/common_recovery_campaign.json').read_text());n=0
for old in legacy:
 if old['phase_deg']!=0 or old['payload'] or old['seed']>2 or old['ctrl']=='swmpc_nosw':continue
 new=idx[('baseline',old['event'],old['h_km'],old['ctrl'],old['seed'])]
 for w in ('full','event_window'):
  for k in ('fuel_g','rms','minE','E_end_Wh','x_end','total_generated_Wh','total_propulsion_Wh','qp_attempts_all','qp_failures_all'):
   assert abs(new[w][k]-old[w][k])<1e-9,(old['event'],k,new[w][k],old[w][k])
 n+=1
assert n==36
print('PASS: 504 unique runs,252 independently checked pairs,14 fixed cases;36 baseline regressions')
print('Energy closure max Wh',max_e,'propulsion/fuel closure max Wh',max_f)
print('Matched pairs',int(p.resource_matched.sum()))
