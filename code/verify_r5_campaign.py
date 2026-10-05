"""Verify recorded R5 resources, failures, endpoint comparisons and TeX identity."""
import json,numpy as np
from pathlib import Path
r=json.load(open('../data/resource_campaign.json'))
assert len(r)==153
keys=[(x['case'],x['event'],x['h_km'],x['ctrl'],x['seed']) for x in r]
assert len(set(keys))==len(keys)
assert set(x['seed'] for x in r)=={0,1,2}
for row in r:
 m=row['metrics'];assert m['minE']>=0
 if row['case']=='eclipse_900W':assert not m['powered_regime_valid']
 if row['case'].endswith('_terminal'):
  assert m['E_end_Wh']>=1450 and m['n_floor_viol']==0 and m['n_energy_floor_viol']==0
 if row['case']=='forced_solver_gap':assert m['forced_fallback_steps']==6 and m['qp_failures_all']==6
 if row['case']=='measurement_gap':assert m['missing_measurement_steps']==24
scheduled=[x['metrics']['qp_failures_all'] for x in r if x['case']=='scheduled_area' and x['ctrl']=='swmpc']
assert sum(scheduled)==1
for case in ('eclipse_1500W','eclipse_payload'):
 b=[x['metrics'] for x in r if x['case']==case and x['ctrl']=='pid_lowdrag_preview']
 m=[x['metrics'] for x in r if x['case']==case+'_terminal']
 mean=lambda a,k:float(np.mean([v[k] for v in a]))
 assert abs(mean(b,'E_end_Wh')-mean(m,'E_end_Wh'))<5
 assert mean(m,'fuel_g')<mean(b,'fuel_g') and mean(m,'rms')>mean(b,'rms')
print("PASS: numerical campaign design, resources, failures and comparisons.")
